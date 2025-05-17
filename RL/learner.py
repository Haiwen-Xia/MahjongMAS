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
        self.config = config                # 存储配置字典
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
        log_base_dir = self.config.get('log_base_dir', './logs') 
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
        self.logger.info("Creating RL model (ResNet34AC with separate FEs)...")
        try:
            self.model = ResNet34AC(in_channels=self.config['in_channels']) 
        except Exception as e:
            self.logger.error(f"Failed to create RL model instance: {e}. Learner exiting.", exc_info=True)
            if self.writer: self.writer.close()
            return

        # --- 3.1 (可选) 加载监督学习 (IL) 预训练权重到 Actor 相关组件 ---
        sl_model_path = self.config.get('supervised_model_path')
        sl_actor_fe_weights_loaded = False 
        sl_actor_head_weights_loaded = False

        if sl_model_path and os.path.isfile(sl_model_path):
            self.logger.info(f"Attempting to load pre-trained SL weights from: {sl_model_path} for Actor components.")
            try:
                sl_state_dict = torch.load(sl_model_path, map_location='cpu')
                self.logger.info(f"Successfully loaded supervised state_dict from {sl_model_path} (contains {len(sl_state_dict)} tensors).")
                                
                if not hasattr(self.model, 'feature_extractor_actor'):
                    self.logger.error("RL Model structure error: 'feature_extractor_actor' not found.")
                if not hasattr(self.model, 'actor_head'):
                    self.logger.error("RL Model structure error: 'actor_head' not found.")

                new_rl_state_dict = self.model.state_dict()
                loaded_keys_count_fe = 0
                loaded_keys_count_head = 0

                for sl_key_name, sl_param_tensor in sl_state_dict.items():
                    rl_target_key_name = None
                    
                    if sl_key_name.startswith('feature_extractor.'):
                        if hasattr(self.model, 'feature_extractor_actor'):
                            rl_target_key_name = sl_key_name.replace('feature_extractor.', 'feature_extractor_actor.', 1)
                            if rl_target_key_name in new_rl_state_dict and new_rl_state_dict[rl_target_key_name].shape == sl_param_tensor.shape:
                                new_rl_state_dict[rl_target_key_name] = sl_param_tensor
                                sl_actor_fe_weights_loaded = True
                                loaded_keys_count_fe +=1
                    elif sl_key_name.startswith('fc.'): 
                        if hasattr(self.model, 'actor_head'):
                            rl_target_key_name = sl_key_name.replace('fc.', 'actor_head.', 1)
                            if rl_target_key_name in new_rl_state_dict and new_rl_state_dict[rl_target_key_name].shape == sl_param_tensor.shape:
                                new_rl_state_dict[rl_target_key_name] = sl_param_tensor
                                sl_actor_head_weights_loaded = True
                                loaded_keys_count_head += 1
                
                if sl_actor_fe_weights_loaded or sl_actor_head_weights_loaded:
                    self.model.load_state_dict(new_rl_state_dict)
                    if sl_actor_fe_weights_loaded:
                        self.logger.info(f"Successfully loaded {loaded_keys_count_fe} tensors into 'feature_extractor_actor' from SL model.")
                    if sl_actor_head_weights_loaded:
                         self.logger.info(f"Successfully loaded {loaded_keys_count_head} tensors into 'actor_head' from SL model.")
                else:
                    self.logger.warning("No SL weights were matched or loaded into actor components (feature_extractor_actor, actor_head).")

                if hasattr(self.model, 'feature_extractor_actor') and not sl_actor_fe_weights_loaded :
                     self.logger.warning("Could not load any weights into 'feature_extractor_actor'. Check SL model keys/shapes (e.g. 'feature_extractor.*').")
                if hasattr(self.model, 'actor_head') and not sl_actor_head_weights_loaded:
                     self.logger.warning("Could not load any weights into 'actor_head'. Check SL model keys/shapes (e.g. 'fc.*').")
            except Exception as e:
                self.logger.error(f"Failed during loading or processing SL weights for Actor components: {e}. "
                                  "Actor components will use random initialization or current state.", exc_info=True)
                sl_actor_fe_weights_loaded = False 
                sl_actor_head_weights_loaded = False
        else: 
            if sl_model_path: 
                self.logger.warning(f"Supervised model path specified ('{sl_model_path}') but not found. "
                                    "Actor components will use random initialization.")
            else: 
                self.logger.info("No supervised model path provided. Actor components (feature_extractor_actor, actor_head) "
                                 "will use random initialization.")

        self.logger.info("Initializing feature_extractor_critic...")
        try:
            if hasattr(self.model, 'feature_extractor_actor') and hasattr(self.model, 'feature_extractor_critic'):
                actor_fe_state_dict = self.model.feature_extractor_actor.state_dict()
                self.model.feature_extractor_critic.load_state_dict(actor_fe_state_dict)
                source_description = "from SL-trained feature_extractor_actor" if sl_actor_fe_weights_loaded else \
                                     "from randomly initialized (or SL-load-attempted) feature_extractor_actor"
                self.logger.info(f"Successfully initialized feature_extractor_critic with weights {source_description}.")
            else:
                self.logger.error("Cannot initialize feature_extractor_critic: 'feature_extractor_actor' or "
                                  "'feature_extractor_critic' not found in the model structure.")
        except Exception as e:
            self.logger.error(f"Error during initialization of feature_extractor_critic from feature_extractor_actor: {e}", exc_info=True)
        
        self.logger.info("Critic_head will use its default (random) initialization.")
        self.model.to(device)
        self.logger.info(f"RL model (ResNet34AC) moved to device: {device}")

        # --- 3.2 参数冻结 (根据配置) ---
        # Critic head is always trainable from start to learn value function.
        if hasattr(self.model, 'critic_head'):
            self._set_requires_grad(self.model.critic_head, True)
        else:
            self.logger.error("Model has no 'critic_head'.")

        # Actor Head freezing logic
        unfreeze_actor_head_after_iters = self.config.get('unfreeze_actor_head_after_iters', 0)
        if hasattr(self.model, 'actor_head'):
            if unfreeze_actor_head_after_iters > 0:
                self._set_requires_grad(self.model.actor_head, False) # Freeze if unfreeze_iters > 0
            else:
                self._set_requires_grad(self.model.actor_head, True)  # Trainable from start
        else:
            self.logger.error("Model has no 'actor_head' for freezing logic.")

        # Actor Feature Extractor freezing logic
        # Critic Feature Extractor will use the SAME unfreeze iteration and logic as Actor Feature Extractor
        unfreeze_actor_fe_after_iters = self.config.get('unfreeze_actor_feature_extractor_after_iters', 0)

        if hasattr(self.model, 'feature_extractor_actor'):
            if unfreeze_actor_fe_after_iters > 0:
                self._set_requires_grad(self.model.feature_extractor_actor, False) # Freeze if unfreeze_iters > 0
            else:
                self._set_requires_grad(self.model.feature_extractor_actor, True)  # Trainable from start
        else:
            self.logger.error("Model has no 'feature_extractor_actor' for freezing logic.")

        # MODIFICATION: Critic Feature Extractor freezing logic (tied to actor_fe)
        if hasattr(self.model, 'feature_extractor_critic'):
            if unfreeze_actor_fe_after_iters > 0: # Use same condition as actor_fe
                self._set_requires_grad(self.model.feature_extractor_critic, False) # Initially FROZEN
            else:
                self._set_requires_grad(self.model.feature_extractor_critic, True)  # Trainable from start if actor_fe is
        else:
            self.logger.error("Model has no 'feature_extractor_critic' for freezing logic.")
        # MODIFICATION END

        # --- 4. 初始化优化器 (支持差分学习率和冻结) ---
        lr_critic_head = self.config.get('lr_critic_head', self.config.get('lr', 3e-4)) 
        lr_critic_fe = self.config.get('lr_critic_feature_extractor', lr_critic_head) 
        lr_actor_head = self.config.get('lr_actor_head_finetune', 3e-5)
        lr_actor_fe = self.config.get('lr_actor_feature_extractor_finetune', 1e-5)

        param_groups = []
        if hasattr(self.model, 'critic_head'):
            param_groups.append({'params': self.model.critic_head.parameters(), 'lr': lr_critic_head, 'name': 'critic_head'})
        if hasattr(self.model, 'feature_extractor_critic'):
            param_groups.append({'params': self.model.feature_extractor_critic.parameters(), 'lr': lr_critic_fe, 'name': 'critic_fe'})
        if hasattr(self.model, 'actor_head'):
            param_groups.append({'params': self.model.actor_head.parameters(), 'lr': lr_actor_head, 'name': 'actor_head'})
        if hasattr(self.model, 'feature_extractor_actor'):
            param_groups.append({'params': self.model.feature_extractor_actor.parameters(), 'lr': lr_actor_fe, 'name': 'actor_fe'})
        
        if not param_groups:
            self.logger.error("No parameter groups created for the optimizer. Check model attributes. Learner exiting.")
            if self.writer: self.writer.close()
            return

        optimizer = torch.optim.Adam(param_groups)
        self.logger.info(f"Optimizer initialized with parameter groups:")
        for pg_idx, pg in enumerate(optimizer.param_groups):
            num_params_total = sum(p.numel() for p in pg['params'])
            num_params_trainable = sum(p.numel() for p in pg['params'] if p.requires_grad) 
            self.logger.info(f"  Group {pg_idx} ('{pg.get('name', 'Unnamed')}'): LR={pg['lr']:.2e}, Trainable_Params={num_params_trainable}/{num_params_total}")

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
        base_lr_for_scheduler = lr_critic_head 
        use_lr_scheduler = self.config.get('use_lr_scheduler', False)
        if use_lr_scheduler:
            warmup_iterations = self.config.get('warmup_iterations', 1000) 
            total_iterations_for_lr_decay = self.config.get('total_iterations_for_lr_decay', 500000) 
            min_lr_scheduler_critic = self.config.get('min_lr_critic_schedule', 1e-6) 
            initial_lr_for_warmup_critic = self.config.get('initial_lr_warmup_critic', base_lr_for_scheduler * 0.01)
            
            self.logger.info(f"LR Scheduler enabled for Critic Head AND Critic FE. Both will follow the same schedule derived from Critic Head's settings:")
            self.logger.info(f"  Schedule: Linear Warmup ({warmup_iterations} iters, from {initial_lr_for_warmup_critic:.2e} to {base_lr_for_scheduler:.2e}) "
                             f"then Cosine Decay ({total_iterations_for_lr_decay} iters to {min_lr_scheduler_critic:.2e}).")
            self.logger.info(f"  Initial LRs for Actor components (fixed unless scheduled separately, and may be initially frozen): "
                             f"ActorHead={lr_actor_head:.2e}, ActorFE={lr_actor_fe:.2e}")
        else:
            self.logger.info(f"LR Scheduler disabled. Using fixed LRs for all components based on initial optimizer settings.")


        # --- 6. 等待 Replay Buffer 数据 ---
        min_samples = self.config.get('min_sample_to_start_learner', 20000) 
        self.logger.info(f"Waiting for Replay Buffer to have at least {min_samples} samples...")
        last_logged_size = -1 
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


        while True:
            start_iter_time = time.time()

            # --- 7.0 更新模型参数状态 (解冻) 和学习率 ---
            # 解冻 Actor head
            if hasattr(self.model, 'actor_head') and \
               unfreeze_actor_head_after_iters > 0 and \
               hasattr(self.model.actor_head, 'weight') and \
               not self.model.actor_head.weight.requires_grad and \
               iterations >= unfreeze_actor_head_after_iters:
                self._set_requires_grad(self.model.actor_head, True)
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    if pg.get('name') == 'actor_head':
                        num_params = sum(p.numel() for p in pg['params'] if p.requires_grad)
                        total_params_in_group = sum(p.numel() for p in pg['params'])
                        self.logger.info(f"  Optimizer Group {pg_idx} ('actor_head') updated: Trainable_Params={num_params}/{total_params_in_group}")

            # 解冻 Actor feature extractor 和 Critic feature extractor (使用相同的 unfreeze_actor_fe_after_iters)
            if iterations >= unfreeze_actor_fe_after_iters and unfreeze_actor_fe_after_iters > 0 : #  Ensure unfreeze_actor_fe_after_iters > 0 to trigger unfreeze
                # Unfreeze Actor Feature Extractor
                if hasattr(self.model, 'feature_extractor_actor') and \
                   hasattr(self.model.feature_extractor_actor, 'layer1') and \
                   not self.model.feature_extractor_actor.layer1.weight.requires_grad: 
                    self._set_requires_grad(self.model.feature_extractor_actor, True)
                    for pg_idx, pg in enumerate(optimizer.param_groups):
                        if pg.get('name') == 'actor_fe': 
                            num_params = sum(p.numel() for p in pg['params'] if p.requires_grad)
                            total_params_in_group = sum(p.numel() for p in pg['params'])
                            self.logger.info(f"  Optimizer Group {pg_idx} ('actor_fe') updated: Trainable_Params={num_params}/{total_params_in_group}")
                
                # MODIFICATION START: Unfreeze Critic Feature Extractor
                if hasattr(self.model, 'feature_extractor_critic') and \
                   hasattr(self.model.feature_extractor_critic, 'layer1') and \
                   not self.model.feature_extractor_critic.layer1.weight.requires_grad: 
                    self._set_requires_grad(self.model.feature_extractor_critic, True)
                    for pg_idx, pg in enumerate(optimizer.param_groups):
                        if pg.get('name') == 'critic_fe': 
                            num_params = sum(p.numel() for p in pg['params'] if p.requires_grad)
                            total_params_in_group = sum(p.numel() for p in pg['params'])
                            self.logger.info(f"  Optimizer Group {pg_idx} ('critic_fe') updated: Trainable_Params={num_params}/{total_params_in_group}")
                # MODIFICATION END
            
            # 更新学习率 (针对 Critic Head 和 Critic FE)
            if use_lr_scheduler:
                scheduled_lr_for_critic_components = calculate_scheduled_lr(
                    current_iter=iterations,
                    base_lr=base_lr_for_scheduler, 
                    warmup_iters=warmup_iterations,
                    total_iters_for_decay=total_iterations_for_lr_decay,
                    min_lr=min_lr_scheduler_critic, 
                    initial_lr_for_warmup=initial_lr_for_warmup_critic 
                )
                for pg in optimizer.param_groups:
                    if pg.get('name') == 'critic_head' or pg.get('name') == 'critic_fe':
                        pg['lr'] = scheduled_lr_for_critic_components
            
            # --- 7.1. 采样和数据准备 ---
            try:
                batch = self.replay_buffer.sample(self.config['batch_size'])
                if batch is None:
                    self.logger.debug("Replay buffer sample returned None, possibly empty or too small. Sleeping...")
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
            critic_loss_epoch_avg = 0.0
            entropy_loss_epoch_avg = 0.0
            total_loss_epoch_avg = 0.0 
            
            try:
                old_log_probs = old_log_probs_from_buffer.detach() 
                ppo_epochs = self.config.get('epochs_per_batch', 5) 
                for ppo_epoch in range(ppo_epochs):
                    action_logits_masked, predicted_critic_values = self.model(states)
                    action_dist = torch.distributions.Categorical(logits=action_logits_masked)
                    log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) 
                    ratio = torch.exp(log_probs - old_log_probs)
                    clip_eps = self.config['clip']
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
                    
                    current_policy_loss_val = 0.0
                    current_entropy_loss_val = 0.0
                    
                    if hasattr(self.model, 'actor_head') and self.model.actor_head.weight.requires_grad:
                        current_policy_loss_term = -torch.mean(torch.min(surr1, surr2))
                        current_entropy_loss_term = -torch.mean(action_dist.entropy())
                        current_policy_loss_val = current_policy_loss_term.item()
                        current_entropy_loss_val = current_entropy_loss_term.item()
                    else: 
                        current_policy_loss_term = torch.tensor(0.0, device=device, requires_grad=False)
                        current_entropy_loss_term = torch.tensor(0.0, device=device, requires_grad=False)

                    current_critic_loss_term = F.mse_loss(predicted_critic_values.squeeze(-1), targets)
                    loss = self.config['value_coeff'] * current_critic_loss_term
                    
                    if hasattr(self.model, 'actor_head') and self.model.actor_head.weight.requires_grad: 
                        loss = loss + current_policy_loss_term + self.config['entropy_coeff'] * current_entropy_loss_term
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    if trainable_params:
                         grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.config.get('grad_clip_norm', 0.3))
                    
                    optimizer.step()

                    policy_loss_epoch_avg += current_policy_loss_val 
                    critic_loss_epoch_avg += current_critic_loss_term.item() 
                    entropy_loss_epoch_avg += current_entropy_loss_val
                    total_loss_epoch_avg += loss.item() 

                policy_loss_epoch_avg /= ppo_epochs
                critic_loss_epoch_avg /= ppo_epochs
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
                buffer_stats_dict = self.replay_buffer.stats() if hasattr(self.replay_buffer, 'stats') and callable(getattr(self.replay_buffer, 'stats')) else {}
                sample_in_rate = buffer_stats_dict.get('samples_in_per_second_smoothed', buffer_stats_dict.get('samples_in_per_second',0)) 
                sample_out_rate = buffer_stats_dict.get('samples_out_per_second_smoothed', buffer_stats_dict.get('samples_out_per_second',0))
                
                log_lr_critic_head, log_lr_critic_fe, log_lr_actor_head, log_lr_actor_fe = "N/A", "N/A", "N/A", "N/A"
                for pg in optimizer.param_groups: 
                    lr_val_str = f"{pg['lr']:.2e}"
                    if pg.get('name') == 'critic_head': log_lr_critic_head = lr_val_str
                    if pg.get('name') == 'critic_fe': log_lr_critic_fe = lr_val_str
                    if pg.get('name') == 'actor_head': log_lr_actor_head = lr_val_str
                    if pg.get('name') == 'actor_fe': log_lr_actor_fe = lr_val_str
                
                lr_log_str = f"LRs (CH/CF/AH/AF): {log_lr_critic_head}/{log_lr_critic_fe}/{log_lr_actor_head}/{log_lr_actor_fe}"
                log_msg = (
                    f"Iter: {iterations} | {lr_log_str} | "
                    f"Loss(Actual): {total_loss_epoch_avg:.4f} " 
                    f"(P_contrib: {policy_loss_epoch_avg:.4f}, C_contrib: {critic_loss_epoch_avg:.4f}, E_contrib: {entropy_loss_epoch_avg:.4f}) | "
                    f"Buffer: {buffer_size} (QueueEp: {buffer_stats_dict.get('queue_size',0)}, In/s: {sample_in_rate:.1f}, Out/s: {sample_out_rate:.1f}) | " # Added QueueEp
                    f"IPS: {iterations_per_sec:.2f} | SPS: {samples_per_sec:.1f}"
                )
                self.logger.info(log_msg)

                if self.writer:
                    self.writer.add_scalar('Loss/Total_Actual_Backward', total_loss_epoch_avg, iterations)
                    self.writer.add_scalar('Loss/Policy_Calculated', policy_loss_epoch_avg, iterations)
                    self.writer.add_scalar('Loss/Critic_Calculated', critic_loss_epoch_avg, iterations) 
                    self.writer.add_scalar('Loss/Entropy_Calculated', entropy_loss_epoch_avg, iterations)
                    self.writer.add_scalar('ReplayBuffer/SizeTimesteps', buffer_size, iterations)
                    self.writer.add_scalar('ReplayBuffer/RateIn', sample_in_rate, iterations)
                    self.writer.add_scalar('ReplayBuffer/RateOut', sample_out_rate, iterations)
                    self.writer.add_scalar('ReplayBuffer/QueueSizeEpisodes', buffer_stats_dict.get('queue_size',0), iterations)
                    self.writer.add_scalar('Performance/IterationsPerSecond_Learner', iterations_per_sec, iterations)
                    self.writer.add_scalar('Performance/SamplesPerSecond_Learner', samples_per_sec, iterations)
                    
                    for pg in optimizer.param_groups:
                        group_name = pg.get('name', 'UnnamedGroup')
                        component_name_map = {
                            'critic_head': 'CriticHead', 'critic_fe': 'CriticFE',
                            'actor_head': 'ActorHead', 'actor_fe': 'ActorFE'
                        }
                        tb_component_name = component_name_map.get(group_name)
                        if tb_component_name:
                            is_trainable_now = False
                            if tb_component_name == 'CriticHead' and hasattr(self.model, 'critic_head') : is_trainable_now = self.model.critic_head.weight.requires_grad
                            elif tb_component_name == 'CriticFE' and hasattr(self.model, 'feature_extractor_critic') and hasattr(self.model.feature_extractor_critic, 'layer1') : is_trainable_now = self.model.feature_extractor_critic.layer1.weight.requires_grad
                            elif tb_component_name == 'ActorHead' and hasattr(self.model, 'actor_head') : is_trainable_now = self.model.actor_head.weight.requires_grad
                            elif tb_component_name == 'ActorFE' and hasattr(self.model, 'feature_extractor_actor') and hasattr(self.model.feature_extractor_actor, 'layer1') : is_trainable_now = self.model.feature_extractor_actor.layer1.weight.requires_grad
                            
                            if is_trainable_now:
                                self.writer.add_scalar(f'LearningRate/{tb_component_name}', pg['lr'], iterations)
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