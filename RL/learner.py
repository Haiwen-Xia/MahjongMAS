# 导入多进程模块，Learner 将作为一个独立的进程运行
from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F # 导入 PyTorch 函数库，通常用于损失函数、激活函数等
import os
import logging # 引入日志模块
from torch.utils.tensorboard import SummaryWriter # 引入 TensorBoard
import json # 用于打印配置

from utils import calculate_scheduled_lr # 导入动态学习率计算函数
from utils import setup_process_logging_and_tensorboard # 导入日志和 TensorBoard 设置函数


# 导入自定义模块
from replay_buffer import ReplayBuffer       # 经验回放缓冲区，用于存储 Actor 收集的数据并供 Learner 采样
from model_pool import ModelPoolServer        # 模型池服务器，用于管理和分发模型版本
# 假设您的模型定义在 model.py 中
from model import ResNet34AC, ResNet34 # , ResNetFeatureExtractor # 确保导入了需要的模型类
from collections import OrderedDict # 用于有序字典，保持参数加载顺序

# Learner 类，继承自 Process，负责模型的训练和更新
class Learner(Process):

    # 初始化函数
    def __init__(self, config, replay_buffer):
        """
        初始化 Learner 进程。

        Args:
            config (dict): 包含 Learner 配置参数的字典 (例如模型池大小/名称, 设备, 学习率, PPO超参数等)。
            replay_buffer (ReplayBuffer): Learner 从中采样数据进行训练的共享经验回放缓冲区。
        """
        super(Learner, self).__init__() # 调用父类 Process 的初始化方法
        self.replay_buffer = replay_buffer # 存储传入的 replay_buffer 实例
        self.config = config               # 存储配置字典
        # Learner 通常是单例，不需要像 Actor 那样通过 config 传递 name，可直接命名
        self.name = "Learner"
        self.logger = None # 初始化日志记录器为 None
        self.writer = None # 初始化 TensorBoard writer 为 None

    # 进程启动时执行的主函数
    def run(self):
        """
        Learner 进程的主要执行逻辑。包括：
        1. 初始化日志和 TensorBoard。
        2. 初始化模型池服务器。
        3. 初始化模型和优化器 (可能从 SL 模型加载)。
        4. 将初始模型推送到模型池。
        5. 进入无限循环，不断从 Replay Buffer 采样数据。
        6. 执行 PPO 算法更新模型参数。
        7. 将更新后的模型推送到模型池。
        8. 定期保存模型检查点。
        9. 记录日志和 TensorBoard 指标。
        """
        # --- 1. 初始化日志和 TensorBoard ---
        self.logger, self.writer = setup_process_logging_and_tensorboard(
            self.config['base_log_directory'], self.config['experiment_name'], self.name
        )


        # --- 2. 初始化模型池服务器 ---
        try:
            model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
            self.logger.info(f"Model Pool Server '{self.config['model_pool_name']}' initialized with size {self.config['model_pool_size']}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Model Pool Server: {e}. Learner exiting.")
            if self.writer: self.writer.close()
            return

        # --- 3. 初始化模型和优化器 ---
        device = torch.device(self.config['device'])
        self.logger.info(f"Learner using device: {device}")
        self.logger.info("Creating RL model (ResNet34AC)...")
        try:
            model = ResNet34AC(in_channels=self.config['in_channels']) # 使用 config 中的通道数
        except Exception as e:
             self.logger.error(f"Failed to create RL model instance: {e}. Learner exiting.")
             if self.writer: self.writer.close()
             return

        # --- 3.1 (可选) 加载监督学习预训练权重 ---
        # (此处嵌入之前优化过的加载代码)
        sl_model_path = self.config.get('supervised_model_path')
        if sl_model_path and os.path.isfile(sl_model_path):
            self.logger.info(f"Attempting to load pre-trained SL weights from: {sl_model_path}")
            try:
                sl_state_dict = torch.load(sl_model_path, map_location='cpu')
                self.logger.info(f"Successfully loaded supervised state_dict from {sl_model_path} to CPU.")
                rl_state_dict = model.state_dict()
                state_dict_to_load = OrderedDict()
                loaded_keys_count = 0
                skipped_keys_mismatch = []
                ignored_keys_sl_only = []
                # ... (遍历 sl_state_dict 并填充 state_dict_to_load 的逻辑，同上一版本) ...
                for name_sl, param_sl in sl_state_dict.items():
                    target_name_rl = None
                    if name_sl.startswith('feature_extractor.'):
                        target_name_rl = name_sl
                    elif name_sl.startswith('fc.'):
                        target_name_rl = name_sl.replace('fc.', 'actor.', 1)
                    else:
                        ignored_keys_sl_only.append(name_sl)
                        continue
                    if target_name_rl in rl_state_dict:
                        param_rl = rl_state_dict[target_name_rl]
                        if param_rl.shape == param_sl.shape:
                            state_dict_to_load[target_name_rl] = param_sl
                            loaded_keys_count += 1
                        else:
                            self.logger.warning(f"Shape mismatch for '{target_name_rl}' vs '{name_sl}'. Skipping.")
                            skipped_keys_mismatch.append(target_name_rl)
                    else:
                        self.logger.warning(f"Parameter '{target_name_rl}' not found in RL model. Skipping.")
                        ignored_keys_sl_only.append(name_sl)

                missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)
                # ... (详细的加载日志，同上一版本) ...
                self.logger.info(f"Parameter loading from SL model complete. Loaded {loaded_keys_count} tensors.")
                # Log skipped/missing keys info...

            except Exception as e:
                self.logger.error(f"Failed during loading/processing SL weights: {e}. Proceeding with initial RL model weights.")
        else:
            if sl_model_path:
                 self.logger.warning(f"Supervised model path specified but not found: {sl_model_path}. Using initial RL weights.")
            else:
                 self.logger.info("No supervised model path provided. Using initial RL model weights.")

        # 将最终模型移至设备
        model.to(device)
        self.logger.info(f"RL model moved to device: {device}")

        # --- 4. 推送初始模型到模型池 ---
        try:
            # 先移到 CPU 获取 state_dict
            initial_version_info = model_pool.push(model.to('cpu').state_dict())
            model.to(device) # 再移回训练设备
            initial_version_id = initial_version_info.get('id', 'N/A') if initial_version_info else 'N/A'
            self.logger.info(f"Pushed initial model version {initial_version_id} to Model Pool.")
        except Exception as e:
             self.logger.error(f"Failed to push initial model: {e}. Learner exiting.")
             if self.writer: self.writer.close()
             return

        # --- 5. 初始化优化器 ---
        base_lr = self.config['lr'] 
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr) # 初始时使用 base_lr，但会被调度器覆盖
        self.logger.info(f"Optimizer initialized (Adam, base_lr={base_lr}).")
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        # self.logger.info(f"Optimizer initialized (Adam, lr={self.config['lr']}).")

        # --- 5.1 初始化学习率调度器参数 (从config获取) ---
        use_lr_scheduler = self.config.get('use_lr_scheduler', False) # 新增配置项，是否启用调度器
        if use_lr_scheduler:
            warmup_iterations = self.config.get('warmup_iterations', 1000) # 例如预热1000次迭代
            # total_iterations_for_lr_decay 指的是从预热结束到LR降至min_lr所需的迭代次数
            # 例如，如果总训练迭代是1M，预热1k，那么衰减阶段可以是 1M - 1k。
            total_iterations_for_lr_decay = self.config.get('total_iterations_for_lr_decay', 500000) 
            min_lr = self.config.get('min_lr', 1e-6) # 最小学习率
            initial_lr_for_warmup = self.config.get('initial_lr_for_warmup', base_lr * 0.01) # 预热起始学习率，例如基础LR的1%
            
            self.self.logger.info(f"LR Scheduler enabled: Linear Warmup ({warmup_iterations} iters, from {initial_lr_for_warmup:.2e} to {base_lr:.2e}) "
                             f"then Cosine Decay ({total_iterations_for_lr_decay} iters to {min_lr:.2e}).")
        else:
            self.self.logger.info(f"LR Scheduler disabled. Using fixed LR: {base_lr:.2e}")

            
        # --- 6. 等待 Replay Buffer 数据 ---
        min_samples = self.config.get('min_sample_to_start_learner', 5000) # 使用新配置名
        self.logger.info(f"Waiting for Replay Buffer to have at least {min_samples} samples...")
        last_logged_size = 0
        while self.replay_buffer.size() < min_samples:
            current_size = self.replay_buffer.size()
            # 避免过于频繁地打印日志
            if current_size > last_logged_size and (current_size % (min_samples // 10 + 1) == 0 or current_size - last_logged_size > 1000):
                 self.logger.info(f"Replay buffer size: {current_size}/{min_samples}")
                 last_logged_size = current_size
            time.sleep(1) # 等待1秒再检查
        self.logger.info(f"Minimum samples ({min_samples}) reached. Starting training loop.")

        # --- 7. 主训练循环 ---
        cur_time_ckpt = time.time() # 用于记录上次保存检查点的时间
        cur_time_log = time.time() # 用于记录上次打印日志/计算速率的时间
        iterations = 0              # 训练迭代次数计数器 (全局步数 global_step)
        steps_processed_since_log = 0 # 用于计算速率

        while True:
            start_iter_time = time.time()

            # --- 7.0 更新学习率 (如果启用了调度器) ---
            if use_lr_scheduler:
                # 当前迭代次数是 'iterations'
                new_lr = calculate_scheduled_lr(
                    current_iter=iterations,
                    base_lr=base_lr,
                    warmup_iters=warmup_iterations,
                    total_iters_for_decay=total_iterations_for_lr_decay,
                    min_lr=min_lr,
                    initial_lr_for_warmup=initial_lr_for_warmup
                )
                # 应用新的学习率到优化器
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            else:
                new_lr = base_lr # 如果不使用调度器，保持基础学习率

            # --- 7.1. 采样和数据准备 ---
            try:
                 batch = self.replay_buffer.sample(self.config['batch_size'])
                 if batch is None:
                     self.logger.warning("Failed to sample batch or buffer empty, sleeping...")
                     time.sleep(0.5)
                     continue
            except Exception as e:
                 self.logger.error(f"Error sampling from replay buffer: {e}. Skipping iteration.")
                 time.sleep(1)
                 continue

            # 将 NumPy 数组转换为 PyTorch 张量，并移动到训练设备
            try:
                obs = torch.tensor(batch['state']['observation']).to(device, non_blocking=True) # non_blocking 可能提高效率
                mask = torch.tensor(batch['state']['action_mask']).to(device, non_blocking=True)
                states = {'obs': {'observation': obs, 'action_mask': mask}}
                actions = torch.tensor(batch['action'], dtype=torch.long).unsqueeze(-1).to(device, non_blocking=True)
                advs = torch.tensor(batch['adv']).to(device, non_blocking=True)
                targets = torch.tensor(batch['target']).to(device, non_blocking=True)
                # old_log_probs_batch = torch.tensor(batch['log_prob']).to(device, non_blocking=True) # PPO 需要从 buffer 获取
            except Exception as e:
                 self.logger.error(f"Error converting batch data to tensor or moving to device: {e}. Skipping iteration.")
                 continue

            # --- 7.2. PPO 更新 ---
            model.train() # 设置为训练模式

            # 存储每个 PPO epoch 的损失，用于平均
            policy_loss_epoch_avg = 0.0
            value_loss_epoch_avg = 0.0
            entropy_loss_epoch_avg = 0.0
            total_loss_epoch_avg = 0.0

            # PPO 核心逻辑: 计算旧 log prob (应该从 Replay Buffer 获取，这里仍然是重新计算作为示例)
            try:
                 with torch.no_grad():
                     old_logits, _ = model(states)
                     old_action_dist = torch.distributions.Categorical(logits=old_logits)
                     # ** PPO 标准实现应使用 Actor 记录的 log_prob: old_log_probs = old_log_probs_batch **
                     # 下面是重新计算的逻辑（假设模型与采样时一致或接近）
                     old_log_probs = old_action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1).detach()

                 ppo_epochs = self.config.get('epochs_per_batch', 5) # 使用新配置名
                 for ppo_epoch in range(ppo_epochs):
                     logits, values = model(states)
                     action_dist = torch.distributions.Categorical(logits=logits)
                     log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) # 计算新 log prob

                     # 计算 Ratio
                     ratio = torch.exp(log_probs - old_log_probs)

                     # Policy Loss (Clipped Surrogate Objective)
                     clip_eps = self.config['clip']
                     surr1 = ratio * advs
                     surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
                     policy_loss = -torch.mean(torch.min(surr1, surr2))

                     # Value Loss
                     value_loss = F.mse_loss(values.squeeze(-1), targets)

                     # Entropy Loss
                     entropy_loss = -torch.mean(action_dist.entropy())

                     # Total Loss
                     loss = (policy_loss +
                             self.config['value_coeff'] * value_loss +
                             self.config['entropy_coeff'] * entropy_loss)

                     # Optimization
                     optimizer.zero_grad()
                     loss.backward()
                     # Optional: Gradient Clipping
                     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.get('grad_clip_norm', 0.5))
                     optimizer.step()

                     # 累加损失用于计算平均值
                     policy_loss_epoch_avg += policy_loss.item()
                     value_loss_epoch_avg += value_loss.item()
                     entropy_loss_epoch_avg += entropy_loss.item()
                     total_loss_epoch_avg += loss.item()

                 # 计算 PPO 更新周期的平均损失
                 policy_loss_epoch_avg /= ppo_epochs
                 value_loss_epoch_avg /= ppo_epochs
                 entropy_loss_epoch_avg /= ppo_epochs
                 total_loss_epoch_avg /= ppo_epochs

            except Exception as e:
                 self.logger.error(f"Error during PPO update at iteration {iterations}: {e}", exc_info=True) # exc_info=True 记录 traceback
                 continue # 跳过这个 batch 的后续步骤

            # --- 7.3. 日志记录和 TensorBoard ---
            steps_processed_since_log += self.config['batch_size']
            current_time = time.time()
            log_interval = self.config.get('log_interval_learner', 100)

            if iterations % log_interval == 0 and iterations > 0:
                 # 计算耗时和速率
                 time_since_log = current_time - cur_time_log
                 iterations_per_sec = log_interval / time_since_log if time_since_log > 0 else float('inf')
                 samples_per_sec = steps_processed_since_log / time_since_log if time_since_log > 0 else float('inf')

                 # 获取 Replay Buffer 统计信息
                 buffer_size = self.replay_buffer.size()
                 buffer_stats = self.replay_buffer.stats # 假设 buffer 提供了统计信息
                 sample_in_rate = buffer_stats.get('sample_in_rate', 0) # 假设 buffer 计算速率
                 sample_out_rate = buffer_stats.get('sample_out_rate', 0)

                 # 获取当前学习率用于记录
                 current_optimizer_lr = optimizer.param_groups[0]['lr']

                 # 记录文本日志
                 log_msg = (
                     f"Iter: {iterations} | LR: {current_optimizer_lr:.2e} | Loss: {total_loss_epoch_avg:.4f} "
                     f"(P: {policy_loss_epoch_avg:.4f}, V: {value_loss_epoch_avg:.4f}, E: {entropy_loss_epoch_avg:.4f}) | "
                     f"Buffer: {buffer_size} (In/s: {sample_in_rate:.1f}, Out/s: {sample_out_rate:.1f}) | "
                     f"IPS: {iterations_per_sec:.2f} | SPS: {samples_per_sec:.1f}"
                     
                 )
                 self.logger.info(log_msg)

                 # 记录到 TensorBoard
                 if self.writer:
                     self.writer.add_scalar('Loss/Total', total_loss_epoch_avg, iterations)
                     self.writer.add_scalar('Loss/Policy', policy_loss_epoch_avg, iterations)
                     self.writer.add_scalar('Loss/Value', value_loss_epoch_avg, iterations)
                     self.writer.add_scalar('Loss/Entropy', entropy_loss_epoch_avg, iterations)
                     self.writer.add_scalar('ReplayBuffer/Size', buffer_size, iterations)
                     self.writer.add_scalar('Performance/IterationsPerSecond', iterations_per_sec, iterations)
                     self.writer.add_scalar('Performance/SamplesPerSecond', samples_per_sec, iterations)
                     self.writer.add_scalar('LearningRate', current_optimizer_lr, iterations) # 记录学习率到TB
                     # 其他可选指标:
                     # self.writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], iterations)
                     # self.writer.add_scalar('Advantage/Mean', advs.mean().item(), iterations)
                     # self.writer.add_scalar('ValueTarget/Mean', targets.mean().item(), iterations)
                     # self.writer.add_scalar('ValuePrediction/Mean', values.mean().item(), iterations) # Note: values is from last ppo epoch
                     # self.writer.add_scalar('PPO/ClipRatioFraction', approx_clipped_frac, iterations) # Need to calculate this
                     self.writer.flush() # 确保写入

                 # 重置计时器和计数器
                 cur_time_log = current_time
                 steps_processed_since_log = 0


            # --- 7.4. 推送模型到模型池 ---
            model_push_interval = self.config.get('model_push_interval', 10)
            if iterations % model_push_interval == 0:
                try:
                     model = model.to('cpu')
                     pushed_version_info = model_pool.push(model.state_dict())
                     pushed_version_id = pushed_version_info.get('id', 'N/A') if pushed_version_info else 'N/A'
                     model = model.to(device)
                     self.logger.info(f"Iteration {iterations}: Pushed updated model version {pushed_version_id} to Model Pool.")
                     if self.writer:
                          # 记录模型池最新版本 ID (如果服务器返回了有效信息)
                          if isinstance(pushed_version_id, (int, float)):
                                self.writer.add_scalar('ModelPool/PushedVersionID', pushed_version_id, iterations)
                except Exception as e:
                     self.logger.error(f"Failed to push model at iteration {iterations}: {e}")
                     model = model.to(device) # 确保模型仍在训练设备上

            # --- 7.5. 保存检查点 ---
            t_now_ckpt = time.time()
            ckpt_interval_sec = self.config.get('ckpt_save_interval_seconds', 600)
            if t_now_ckpt - cur_time_ckpt > ckpt_interval_sec:
                ckpt_dir = self.config['ckpt_save_path']
                os.makedirs(ckpt_dir, exist_ok=True)
                path = os.path.join(ckpt_dir, f'model_iter_{iterations}.pt')
                self.logger.info(f"Saving checkpoint at iteration {iterations} to {path}...")
                try:
                     # 保存模型状态字典即可，不需要优化器和调度器（因为是无限循环）
                     # 但如果需要恢复训练，保存 optimizer 状态也很重要
                     torch.save({
                         'iteration': iterations,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         # 可以添加其他需要恢复的状态
                     }, path)
                     cur_time_ckpt = t_now_ckpt
                except Exception as e:
                     self.logger.error(f"Failed to save checkpoint at iteration {iterations}: {e}")


            iterations += 1 # 增加迭代计数器

        # --- 循环结束 (理论上是无限循环，但可以添加退出条件) ---
        self.logger.info("Learner training loop finished (or was interrupted).")
        if self.writer:
             self.writer.close()