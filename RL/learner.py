# 导入多进程模块，Learner 将作为一个独立的进程运行
from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F # 导入 PyTorch 函数库，通常用于损失函数、激活函数等
import os
# import logging # logging is handled by setup_process_logging_and_tensorboard
from torch.utils.tensorboard import SummaryWriter # 引入 TensorBoard
import json # 用于打印配置

from utils import calculate_scheduled_lr # 导入动态学习率计算函数
from utils import setup_process_logging_and_tensorboard # 导入日志和 TensorBoard 设置函数


# 导入自定义模块
from replay_buffer import ReplayBuffer      # 经验回放缓冲区，用于存储 Actor 收集的数据并供 Learner 采样
from model_pool_extended import ModelPoolServer      # 模型池服务器，用于管理和分发模型版本
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
        self.model = None # Initialize model attribute

    def _set_requires_grad(self, module, requires_grad):
        """Helper function to set requires_grad for all parameters of a module."""
        if module is None:
            self.logger.warning(f"Attempted to set requires_grad on a None module.")
            return
            
        for param in module.parameters():
            param.requires_grad = requires_grad
        
        # Try to find the name of the module if it's a direct child of self.model
        component_name_list = [name for name, mod in self.model.named_children() if mod is module]
        component_name = component_name_list[0] if component_name_list else "Unnamed_Module"
        
        status = "TRAINABLE" if requires_grad else "FROZEN"
        self.logger.info(f"Component '{component_name}' is now {status}.")


    # 进程启动时执行的主函数
    def run(self):
        """
        Learner 进程的主要执行逻辑。包括：
        1. 初始化日志和 TensorBoard。
        2. 初始化模型池服务器。
        3. 初始化模型和优化器 (可能从 SL 模型加载, 并处理参数冻结)。
        4. 将初始模型推送到模型池。
        5. 进入无限循环，不断从 Replay Buffer 采样数据。
        6. 执行 PPO 算法更新模型参数 (根据迭代解冻参数)。
        7. 将更新后的模型推送到模型池。
        8. 定期保存模型检查点。
        9. 记录日志和 TensorBoard 指标。
        """
        # --- 1. 初始化日志和 TensorBoard ---
        # 确保 config 中有 'log_base_dir' 和 'experiment_name'
        log_base_dir = self.config.get('log_base_dir', './logs') # Default log base dir
        experiment_name = self.config.get('experiment_name', 'default_run')
        self.logger, self.writer = setup_process_logging_and_tensorboard(
            log_base_dir, experiment_name, self.name
        )
        if not self.logger:
            print(f"CRITICAL: Logger for {self.name} could not be initialized. Exiting.")
            if self.writer: self.writer.close()
            return
        self.logger.info(f"Learner process {self.name} started. Config: \n{json.dumps(self.config, indent=2, ensure_ascii=False)}")


        # --- 2. 初始化模型池服务器 ---
        try:
            model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
            self.logger.info(f"Model Pool Server '{self.config['model_pool_name']}' initialized with size {self.config['model_pool_size']}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Model Pool Server: {e}. Learner exiting.", exc_info=True)
            if self.writer: self.writer.close()
            return

        # --- 3. 初始化模型 ---
        device = torch.device(self.config['device'])
        self.logger.info(f"Learner using device: {device}")
        self.logger.info("Creating RL model (ResNet34AC)...")
        try:
            # Store model as self.model to access it in helper function _set_requires_grad
            self.model = ResNet34AC(in_channels=self.config['in_channels']) # 使用 config 中的通道数
        except Exception as e:
            self.logger.error(f"Failed to create RL model instance: {e}. Learner exiting.", exc_info=True)
            if self.writer: self.writer.close()
            return

        # --- 3.1 (可选) 加载监督学习预训练权重 ---
        sl_model_path = self.config.get('supervised_model_path')
        if sl_model_path and os.path.isfile(sl_model_path):
            self.logger.info(f"Attempting to load pre-trained SL weights from: {sl_model_path}")
            try:
                sl_state_dict = torch.load(sl_model_path, map_location='cpu')
                self.logger.info(f"Successfully loaded supervised state_dict from {sl_model_path} to CPU.")
                rl_state_dict = self.model.state_dict()
                state_dict_to_load = OrderedDict()
                loaded_keys_count = 0
                skipped_keys_mismatch = []
                ignored_keys_sl_only = []
                
                for name_sl, param_sl in sl_state_dict.items():
                    target_name_rl = None
                    # Assuming ResNet34AC has 'feature_extractor' and 'actor' (policy head)
                    if name_sl.startswith('feature_extractor.'):
                        target_name_rl = name_sl
                    elif name_sl.startswith('fc.'): # Assuming SL model's policy head is 'fc'
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
                            self.logger.warning(f"Shape mismatch for '{target_name_rl}' (RL: {param_rl.shape}) vs '{name_sl}' (SL: {param_sl.shape}). Skipping.")
                            skipped_keys_mismatch.append(target_name_rl)
                    else:
                        self.logger.warning(f"Parameter '{target_name_rl}' (derived from SL key '{name_sl}') not found in RL model. Skipping.")
                        ignored_keys_sl_only.append(name_sl) # Add original SL key

                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict_to_load, strict=False)
                self.logger.info(f"Parameter loading from SL model complete. Loaded {loaded_keys_count} tensors.")
                if skipped_keys_mismatch: self.logger.warning(f"Skipped {len(skipped_keys_mismatch)} SL keys due to shape mismatch: {skipped_keys_mismatch}")
                if ignored_keys_sl_only: self.logger.info(f"Ignored {len(ignored_keys_sl_only)} SL keys not targeted for RL model: {ignored_keys_sl_only}")
                if missing_keys: self.logger.warning(f"RL model keys not found in provided SL state_dict (after mapping): {missing_keys}")
                if unexpected_keys: self.logger.error(f"Unexpected keys found in state_dict_to_load: {unexpected_keys}")

            except Exception as e:
                self.logger.error(f"Failed during loading/processing SL weights: {e}. Proceeding with initial RL model weights.", exc_info=True)
        else:
            if sl_model_path:
                self.logger.warning(f"Supervised model path specified but not found: {sl_model_path}. Using initial RL weights.")
            else:
                self.logger.info("No supervised model path provided. Using initial RL model weights.")

        self.model.to(device)
        self.logger.info(f"RL model moved to device: {device}")

        # --- 3.2 参数冻结 (根据配置) ---
        # 默认情况下，价值函数 (critic) 是可训练的
        self._set_requires_grad(self.model.critic, True) # Value head is always trainable initially

        # 冻结策略头 (actor)
        unfreeze_policy_head_after_iters = self.config.get('unfreeze_policy_head_after_iters', 0)
        if unfreeze_policy_head_after_iters > 0:
            self._set_requires_grad(self.model.actor, False)
            # self.logger.info(f"Policy head (actor) will be frozen for the first {unfreeze_policy_head_after_iters} iterations.") # Logged in _set_requires_grad
        else:
            self._set_requires_grad(self.model.actor, True) # Train from start

        # 冻结特征提取器
        unfreeze_feature_extractor_after_iters = self.config.get('unfreeze_feature_extractor_after_iters', 0)
        if unfreeze_feature_extractor_after_iters > 0:
            self._set_requires_grad(self.model.feature_extractor, False)
            # self.logger.info(f"Feature extractor will be frozen for the first {unfreeze_feature_extractor_after_iters} iterations.") # Logged in _set_requires_grad
        else:
            self._set_requires_grad(self.model.feature_extractor, True) # Train from start


        # --- 4. 初始化优化器 (支持差分学习率和冻结) ---
        lr_value_head = self.config.get('lr_value_head', self.config.get('lr', 3e-4)) # Default to config['lr'] if specific not found
        lr_policy_head_finetune = self.config.get('lr_policy_head_finetune', 3e-5)
        lr_feature_extractor_finetune = self.config.get('lr_feature_extractor_finetune', 1e-5)

        param_groups = [
            {'params': self.model.critic.parameters(), 'lr': lr_value_head, 'name': 'critic'},
            {'params': self.model.actor.parameters(), 'lr': lr_policy_head_finetune, 'name': 'actor'},
            {'params': self.model.feature_extractor.parameters(), 'lr': lr_feature_extractor_finetune, 'name': 'feature_extractor'}
        ]
        
        optimizer = torch.optim.Adam(param_groups)
        self.logger.info(f"Optimizer initialized with parameter groups:")
        for pg_idx, pg in enumerate(optimizer.param_groups):
            num_params = sum(p.numel() for p in pg['params'] if p.requires_grad) # Count only trainable params in group
            total_params_in_group = sum(p.numel() for p in pg['params'])
            self.logger.info(f"  Group {pg_idx} ('{pg.get('name', 'Unnamed')}'): LR={pg['lr']:.2e}, Trainable_Params={num_params}/{total_params_in_group}")


        # --- 4.1 推送初始模型到模型池 ---
        try:
            initial_model_state_dict = self.model.to('cpu').state_dict()
            self.model.to(device) 
            initial_version_info = model_pool.push(initial_model_state_dict)
            initial_version_id = initial_version_info.get('id', 'N/A') if initial_version_info else 'N/A'
            self.logger.info(f"Pushed initial model version {initial_version_id} to Model Pool.")
        except Exception as e:
            self.logger.error(f"Failed to push initial model: {e}. Learner exiting.", exc_info=True)
            if self.writer: self.writer.close()
            self.model.to(device) 
            return
            
        # --- 5. 初始化学习率调度器参数 ---
        # base_lr for scheduler is the LR of the value head
        base_lr_for_scheduler = lr_value_head 
        use_lr_scheduler = self.config.get('use_lr_scheduler', False)
        if use_lr_scheduler:
            warmup_iterations = self.config.get('warmup_iterations', 1000) 
            total_iterations_for_lr_decay = self.config.get('total_iterations_for_lr_decay', 500000) 
            min_lr_scheduler = self.config.get('min_lr', 1e-6) # Min LR for the scheduled component (value head)
            initial_lr_for_warmup = self.config.get('initial_lr_for_warmup', base_lr_for_scheduler * 0.01)
            
            self.logger.info(f"LR Scheduler enabled for Value Head: Linear Warmup ({warmup_iterations} iters, from {initial_lr_for_warmup:.2e} to {base_lr_for_scheduler:.2e}) "
                             f"then Cosine Decay ({total_iterations_for_lr_decay} iters to {min_lr_scheduler:.2e}).")
            self.logger.info(f"Policy Head LR (fixed when unfrozen, unless also scheduled): {lr_policy_head_finetune:.2e}")
            self.logger.info(f"Feature Extractor LR (fixed when unfrozen, unless also scheduled): {lr_feature_extractor_finetune:.2e}")
        else:
            self.logger.info(f"LR Scheduler disabled. Using fixed LRs for components.")


        # --- 6. 等待 Replay Buffer 数据 ---
        min_samples = self.config.get('min_sample_to_start_learner', 5000)
        self.logger.info(f"Waiting for Replay Buffer to have at least {min_samples} samples...")
        last_logged_size = -1 # Ensure first log happens
        while self.replay_buffer.size() < min_samples:
            current_size = self.replay_buffer.size()
            if current_size > last_logged_size and (current_size % (min_samples // 10 + 1) == 0 or current_size - last_logged_size > 1000 or current_size == 0):
                self.logger.info(f"Replay buffer size: {current_size}/{min_samples}")
                last_logged_size = current_size
            time.sleep(1) 
        self.logger.info(f"Minimum samples ({min_samples}) reached. Starting training loop.")

        # --- 7. 主训练循环 ---
        cur_time_ckpt = time.time() 
        cur_time_log = time.time() 
        iterations = 0 
        steps_processed_since_log = 0 

        # Track unfreezing to only log once
        policy_head_unfrozen_logged = not (unfreeze_policy_head_after_iters > 0) # True if trainable from start
        feature_extractor_unfrozen_logged = not (unfreeze_feature_extractor_after_iters > 0) # True if trainable from start


        while True:
            start_iter_time = time.time()

            # --- 7.0 更新模型参数状态 (解冻) 和学习率 ---
            # 解冻策略头
            # Check a parameter within self.model.actor (e.g., its weight)
            if unfreeze_policy_head_after_iters > 0 and \
               not self.model.actor.weight.requires_grad and \
               iterations >= unfreeze_policy_head_after_iters:
                self._set_requires_grad(self.model.actor, True)
                # Update optimizer group info after unfreezing
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    if pg.get('name') == 'actor':
                        num_params = sum(p.numel() for p in pg['params'] if p.requires_grad)
                        total_params_in_group = sum(p.numel() for p in pg['params'])
                        self.logger.info(f"  Optimizer Group {pg_idx} ('actor') updated: Trainable_Params={num_params}/{total_params_in_group}")
                policy_head_unfrozen_logged = True # Will be logged by _set_requires_grad

            # 解冻特征提取器
            # Check a parameter within self.model.feature_extractor (e.g., layer1's weight)
            if unfreeze_feature_extractor_after_iters > 0 and \
               not self.model.feature_extractor.layer1.weight.requires_grad and \
               iterations >= unfreeze_feature_extractor_after_iters:
                self._set_requires_grad(self.model.feature_extractor, True)
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    if pg.get('name') == 'feature_extractor':
                        num_params = sum(p.numel() for p in pg['params'] if p.requires_grad)
                        total_params_in_group = sum(p.numel() for p in pg['params'])
                        self.logger.info(f"  Optimizer Group {pg_idx} ('feature_extractor') updated: Trainable_Params={num_params}/{total_params_in_group}")
                feature_extractor_unfrozen_logged = True # Will be logged by _set_requires_grad
            
            # 更新学习率 (仅针对价值函数头，如果启用了调度器)
            current_value_head_lr = lr_value_head # Default if no scheduler
            if use_lr_scheduler:
                new_scheduled_value_lr = calculate_scheduled_lr(
                    current_iter=iterations,
                    base_lr=base_lr_for_scheduler, # This is lr_value_head
                    warmup_iters=warmup_iterations,
                    total_iters_for_decay=total_iterations_for_lr_decay,
                    min_lr=min_lr_scheduler,
                    initial_lr_for_warmup=initial_lr_for_warmup
                )
                for pg in optimizer.param_groups:
                    if pg.get('name') == 'critic':
                        pg['lr'] = new_scheduled_value_lr
                        current_value_head_lr = new_scheduled_value_lr
                        break
            
            # --- 7.1. 采样和数据准备 ---
            try:
                batch = self.replay_buffer.sample(self.config['batch_size'])
                if batch is None:
                    self.logger.debug("Replay buffer sample returned None, possibly empty or too small. Sleeping...") # Changed to debug
                    time.sleep(0.5)
                    continue
            except Exception as e:
                self.logger.error(f"Error sampling from replay buffer: {e}. Skipping iteration.", exc_info=True)
                time.sleep(1)
                continue
            
            try:
                obs = torch.tensor(batch['state']['observation']).to(device, non_blocking=True) 
                mask = torch.tensor(batch['state']['action_mask']).to(device, non_blocking=True)
                states = {'obs': {'observation': obs, 'action_mask': mask}}
                actions = torch.tensor(batch['action'], dtype=torch.long).unsqueeze(-1).to(device, non_blocking=True)
                advs = torch.tensor(batch['adv']).to(device, non_blocking=True)
                targets = torch.tensor(batch['target']).to(device, non_blocking=True)
                old_log_probs_from_buffer = torch.tensor(batch['log_prob']).to(device, non_blocking=True) 
            except Exception as e:
                self.logger.error(f"Error converting batch data to tensor or moving to device: {e}. Skipping iteration.", exc_info=True)
                continue

            # --- 7.2. PPO 更新 ---
            self.model.train() 

            policy_loss_epoch_avg = 0.0
            value_loss_epoch_avg = 0.0
            entropy_loss_epoch_avg = 0.0
            total_loss_epoch_avg = 0.0 # This is the actual loss used for backward step
            
            try:
                old_log_probs = old_log_probs_from_buffer.detach() 

                ppo_epochs = self.config.get('epochs_per_batch', 5) 
                for ppo_epoch in range(ppo_epochs):
                    logits, values = self.model(states) 
                    action_dist = torch.distributions.Categorical(logits=logits)
                    log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) 

                    ratio = torch.exp(log_probs - old_log_probs)

                    clip_eps = self.config['clip']
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
                    
                    current_policy_loss_val = 0.0
                    current_entropy_loss_val = 0.0
                    
                    # Policy-related losses are only computed if actor is trainable
                    if self.model.actor.weight.requires_grad:
                        current_policy_loss = -torch.mean(torch.min(surr1, surr2))
                        current_entropy_loss = -torch.mean(action_dist.entropy())
                        current_policy_loss_val = current_policy_loss.item()
                        current_entropy_loss_val = current_entropy_loss.item()
                    else: # If actor is frozen, these losses are conceptually zero for the update
                        current_policy_loss = torch.tensor(0.0, device=device, requires_grad=False)
                        current_entropy_loss = torch.tensor(0.0, device=device, requires_grad=False)

                    current_value_loss = F.mse_loss(values.squeeze(-1), targets)
                    
                    # Construct the actual loss for backward pass
                    # Value loss is always included as critic is assumed trainable from start
                    loss = self.config['value_coeff'] * current_value_loss
                    if self.model.actor.weight.requires_grad: # Add policy components if actor is trainable
                        loss = loss + current_policy_loss + self.config['entropy_coeff'] * current_entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    if trainable_params:
                         grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.config.get('grad_clip_norm', 0.5))
                    
                    optimizer.step()

                    # Accumulate for logging (even if not used in loss, log the calculated value)
                    policy_loss_epoch_avg += current_policy_loss_val 
                    value_loss_epoch_avg += current_value_loss.item()
                    entropy_loss_epoch_avg += current_entropy_loss_val
                    total_loss_epoch_avg += loss.item() 

                policy_loss_epoch_avg /= ppo_epochs
                value_loss_epoch_avg /= ppo_epochs
                entropy_loss_epoch_avg /= ppo_epochs
                total_loss_epoch_avg /= ppo_epochs

            except Exception as e:
                self.logger.error(f"Error during PPO update at iteration {iterations}: {e}", exc_info=True) 
                continue 

            # --- 7.3. 日志记录和 TensorBoard ---
            steps_processed_since_log += self.config['batch_size']
            current_time = time.time()
            log_interval = self.config.get('log_interval_learner', 100)

            if iterations % log_interval == 0: 
                time_since_log = current_time - cur_time_log if cur_time_log and iterations > 0 else (current_time - start_iter_time if iterations == 0 else 1.0)
                iterations_per_sec = log_interval / time_since_log if time_since_log > 0 and iterations > 0 else \
                                     (1 / time_since_log if time_since_log > 0 and iterations == 0 else 0.0)
                samples_per_sec = steps_processed_since_log / time_since_log if time_since_log > 0 else 0.0

                buffer_size = self.replay_buffer.size()
                buffer_stats = self.replay_buffer.stats() if hasattr(self.replay_buffer, 'stats') and callable(getattr(self.replay_buffer, 'stats')) else {}
                sample_in_rate = buffer_stats.get('samples_in_per_second_smoothed', 0) 
                sample_out_rate = buffer_stats.get('samples_out_per_second_smoothed', 0)
                
                log_lr_critic, log_lr_actor, log_lr_fe = 0.0, 0.0, 0.0
                for pg in optimizer.param_groups: 
                    if pg.get('name') == 'critic': log_lr_critic = pg['lr']
                    if pg.get('name') == 'actor': log_lr_actor = pg['lr']
                    if pg.get('name') == 'feature_extractor': log_lr_fe = pg['lr']

                log_msg = (
                    f"Iter: {iterations} | "
                    f"LRs (C/A/FE): {log_lr_critic:.2e}/{log_lr_actor:.2e}/{log_lr_fe:.2e} | "
                    f"Loss(Actual): {total_loss_epoch_avg:.4f} " 
                    f"(P_contrib: {policy_loss_epoch_avg:.4f}, V_contrib: {value_loss_epoch_avg:.4f}, E_contrib: {entropy_loss_epoch_avg:.4f}) | "
                    f"Buffer: {buffer_size} (In/s: {sample_in_rate:.1f}, Out/s: {sample_out_rate:.1f}) | "
                    f"IPS: {iterations_per_sec:.2f} | SPS: {samples_per_sec:.1f}"
                )
                self.logger.info(log_msg)

                if self.writer:
                    self.writer.add_scalar('Loss/Total_Actual_Backward', total_loss_epoch_avg, iterations)
                    self.writer.add_scalar('Loss/Policy_Calculated', policy_loss_epoch_avg, iterations) # Calculated, not necessarily used in backward
                    self.writer.add_scalar('Loss/Value_Calculated', value_loss_epoch_avg, iterations)
                    self.writer.add_scalar('Loss/Entropy_Calculated', entropy_loss_epoch_avg, iterations)
                    self.writer.add_scalar('ReplayBuffer/Size', buffer_size, iterations)
                    self.writer.add_scalar('ReplayBuffer/RateInSmoothed', sample_in_rate, iterations)
                    self.writer.add_scalar('ReplayBuffer/RateOutSmoothed', sample_out_rate, iterations)
                    self.writer.add_scalar('Performance/IterationsPerSecond_Learner', iterations_per_sec, iterations)
                    self.writer.add_scalar('Performance/SamplesPerSecond_Learner', samples_per_sec, iterations)
                    
                    self.writer.add_scalar('LearningRate/Critic', log_lr_critic, iterations)
                    if self.model.actor.weight.requires_grad: 
                         self.writer.add_scalar('LearningRate/Actor', log_lr_actor, iterations)
                    if self.model.feature_extractor.layer1.weight.requires_grad: 
                         self.writer.add_scalar('LearningRate/FeatureExtractor', log_lr_fe, iterations)
                    self.writer.flush()

                cur_time_log = current_time
                steps_processed_since_log = 0
            
            # --- 7.4. 推送模型到模型池 ---
            model_push_interval = self.config.get('model_push_interval_iters', self.config.get('model_push_interval', 10))
            if iterations > 0 and iterations % model_push_interval == 0: 
                try:
                    model_state_to_push = self.model.to('cpu').state_dict()
                    self.model.to(device) 
                    pushed_version_info = model_pool.push(model_state_to_push)
                    pushed_version_id = pushed_version_info.get('id', 'N/A') if pushed_version_info else 'N/A'
                    self.logger.info(f"Iteration {iterations}: Pushed updated model version {pushed_version_id} to Model Pool.")
                    if self.writer:
                        if isinstance(pushed_version_id, (int, float)) and pushed_version_id != 'N/A':
                            try:
                                self.writer.add_scalar('ModelPool/PushedVersionID', float(pushed_version_id), iterations)
                            except ValueError:
                                self.logger.warning(f"Could not convert pushed_version_id '{pushed_version_id}' to float for TensorBoard.")
                except Exception as e:
                    self.logger.error(f"Failed to push model at iteration {iterations}: {e}", exc_info=True)
                    self.model.to(device) 

            # --- 7.5. 保存检查点 ---
            t_now_ckpt = time.time()
            ckpt_interval_sec = self.config.get('ckpt_save_interval_seconds', 600)
            default_ckpt_path = os.path.join(log_base_dir, experiment_name, 'checkpoints', self.name)
            ckpt_dir = self.config.get('ckpt_save_path', default_ckpt_path)

            if iterations > 0 and t_now_ckpt - cur_time_ckpt > ckpt_interval_sec : 
                os.makedirs(ckpt_dir, exist_ok=True)
                path = os.path.join(ckpt_dir, f'model_iter_{iterations}.pt')
                self.logger.info(f"Saving checkpoint at iteration {iterations} to {path}...")
                try:
                    # Ensure config is serializable (basic types)
                    serializable_config = {k: v for k, v in self.config.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
                    torch.save({
                        'iteration': iterations,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': serializable_config 
                    }, path)
                    cur_time_ckpt = t_now_ckpt
                except Exception as e:
                    self.logger.error(f"Failed to save checkpoint at iteration {iterations}: {e}", exc_info=True)

            iterations += 1

        # --- 循环结束 ---
        self.logger.info("Learner training loop finished (or was interrupted).")
        if self.writer:
            self.writer.close()
