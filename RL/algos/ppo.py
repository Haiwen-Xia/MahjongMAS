import torch
from torch import nn
from torch.nn import functional as F
import logging
import numpy as np
from utils import calculate_scheduled_lr

class PPOAlgorithm:
    def __init__(self, config, actor: nn.Module, critic: nn.Module):
        """
        初始化PPO算法模块。

        Args:
            config (dict): 包含所有超参数的配置字典。
            model (nn.Module): 要训练的PyTorch模型实例 (例如 ResNet34AC)。
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.logger = logging.getLogger(f"PPOAlgorithm") # 假设日志系统已配置
        
        # 健壮性配置参数
        self.config.setdefault('verbose_batch_info', False)  # 是否打印详细的batch信息
        self.config.setdefault('use_obs_as_fallback', True)  # 当缺少global_obs时是否使用obs作为备选
        
        # 训练阶段解冻配置参数
        self.config.setdefault('stage1_iterations', 10000)  # 只训练critic头部的迭代数
        self.config.setdefault('stage2_iterations', 50000)  # 开始训练critic特征提取器的迭代数
        
        # 学习率调度配置参数
        self.config.setdefault('use_lr_scheduler', False)  # 是否使用学习率调度
        self.config.setdefault('warmup_iterations', 1000)   # 学习率预热迭代数
        self.config.setdefault('total_iterations_for_lr_decay', 500000)  # 学习率衰减总迭代数
        self.config.setdefault('min_lr_for_scheduled_components', 1e-6)  # 最小学习率
        
        # 解冻时学习率重置配置
        self.config.setdefault('reset_lr_on_unfreeze', True)  # 组件解冻时是否重置学习率计数
        
        self.actor = actor.to(self.device)  # 假设 actor 是一个 nn.Module
        self.critic = critic.to(self.device)  # 假设 critic 是一个 nn.Module
        
        # 在算法类内部创建优化器
        self._create_optimizer()

    def _create_optimizer(self):
        """根据配置为模型的不同部分创建优化器和参数组。"""
        # 从配置中读取各部分的学习率
        lr_actor = self.config.get('lr_actor', 3e-5)  # Actor整体学习率
        lr_critic_fe = self.config.get('lr_critic_feature_extractor', 3e-4)  # Critic特征提取器学习率
        lr_critic_head = self.config.get('lr_critic_head', 3e-4)  # Critic头部学习率

        param_groups = []
        
        # 参数组1: Actor (整个ResNet34)
        param_groups.append({
            'params': self.actor.parameters(), 
            'lr': lr_actor, 
            'name': 'actor'
        })
        
        # 参数组2: Critic特征提取器 (仅feature_extractor_obs，可以用IL权重初始化)
        if hasattr(self.critic, 'feature_extractor_obs'):
            param_groups.append({
                'params': self.critic.feature_extractor_obs.parameters(), 
                'lr': lr_critic_fe, 
                'name': 'critic_feature_extractor'
            })
        
        # 参数组3: Critic头部 (critic_head_mlp + feature_extractor_extra，都是随机初始化)
        critic_head_params = []
        if hasattr(self.critic, 'critic_head_mlp'):
            critic_head_params.extend(list(self.critic.critic_head_mlp.parameters()))
        if hasattr(self.critic, 'feature_extractor_extra'):
            critic_head_params.extend(list(self.critic.feature_extractor_extra.parameters()))
        
        if critic_head_params:
            param_groups.append({
                'params': critic_head_params, 
                'lr': lr_critic_head, 
                'name': 'critic_head'
            })
        
        if not param_groups:
            self.logger.critical("No parameter groups created for the optimizer. Check model attributes.")
            raise ValueError("Cannot create optimizer with no parameter groups.")

        self.optimizer = torch.optim.Adam(param_groups)
        self.logger.info(f"Optimizer initialized with {len(param_groups)} parameter groups:")
        for i, pg in enumerate(param_groups):
            num_params = sum(p.numel() for p in pg['params'])
            self.logger.info(f"  Group {i}: {pg['name']} - {num_params} parameters, lr={pg['lr']}")

    def _set_requires_grad(self, module, requires_grad):
        """一个辅助函数，用于设置模块所有参数的 requires_grad 属性。"""
        # (这个函数可以作为 PPOAlgorithm 的一个私有方法)
        if module is None: return
        for param in module.parameters():
            param.requires_grad = requires_grad
        
        # component_name = ... # (获取模块名称的逻辑)
        # self.logger.info(f"Component '{component_name}' grad is now {'ENABLED' if requires_grad else 'DISABLED'}.")

    def update_freezing_status(self, iterations: int):
        """
        根据当前的迭代次数，检查并解冻模型的各个部分。
        训练阶段：
        1. 阶段1 (0 - stage1_iters): 冻结actor和critic_fe_obs，只训练critic_head(包含critic_head_mlp和feature_extractor_extra)
        2. 阶段2 (stage1_iters - stage2_iters): 固定actor，训练critic_fe_obs和critic_head  
        3. 阶段3 (stage2_iters+): 联合训练所有组件
        """
        stage1_iters = self.config.get('stage1_iterations', 10000)  # 只训练critic头部的迭代数
        stage2_iters = self.config.get('stage2_iterations', 50000)  # 开始训练critic特征提取器的迭代数
        
        if iterations < stage1_iters:
            # 阶段1: 只训练critic头部(包含feature_extractor_extra和critic_head_mlp)
            if iterations == 0:  # 初始化时设置冻结状态
                self.logger.info(f"Stage 1 (iters 0-{stage1_iters}): Freezing actor and critic_fe_obs, training critic_head (including feature_extractor_extra)")
                self._set_requires_grad(self.actor, False)
                if hasattr(self.critic, 'feature_extractor_obs'):
                    self._set_requires_grad(self.critic.feature_extractor_obs, False)
                # critic_head组包含feature_extractor_extra和critic_head_mlp，都设为可训练
                if hasattr(self.critic, 'feature_extractor_extra'):
                    self._set_requires_grad(self.critic.feature_extractor_extra, True)
                if hasattr(self.critic, 'critic_head_mlp'):
                    self._set_requires_grad(self.critic.critic_head_mlp, True)
                    
        elif iterations == stage1_iters:
            # 进入阶段2: 解冻critic_fe_obs
            self.logger.info(f"Stage 2 (iters {stage1_iters}-{stage2_iters}): Unfreezing critic_fe_obs, keeping actor frozen")
            if hasattr(self.critic, 'feature_extractor_obs'):
                self._set_requires_grad(self.critic.feature_extractor_obs, True)
            # actor仍然冻结，critic_head组继续训练
            
        elif iterations == stage2_iters:
            # 进入阶段3: 解冻所有组件
            self.logger.info(f"Stage 3 (iters {stage2_iters}+): Unfreezing all components for joint training")
            self._set_requires_grad(self.actor, True)
            
        # 记录当前参数组状态
        if iterations in [0, stage1_iters, stage2_iters]:
            for pg_idx, pg in enumerate(self.optimizer.param_groups):
                name = pg.get('name', f'group_{pg_idx}')
                num_trainable = sum(p.numel() for p in pg['params'] if p.requires_grad)
                total_params = sum(p.numel() for p in pg['params'])
                self.logger.info(f"  Optimizer Group {pg_idx} ('{name}'): Trainable={num_trainable}/{total_params} params")


    def schedule_learning_rate(self, iterations: int):
        """
        根据当前的迭代次数，更新优化器中各个参数组的学习率。
        针对不同训练阶段的解冻组件进行学习率调整，避免冲突。
        """
        if not self.config.get('use_lr_scheduler', False):
            return

        # 获取基础学习率和调度参数
        lr_actor = self.config.get('lr_actor', 3e-5)
        lr_critic_fe = self.config.get('lr_critic_feature_extractor', 3e-4)
        lr_critic_head = self.config.get('lr_critic_head', 3e-4)

        # 获取训练阶段的迭代次数边界
        stage1_iters = self.config.get('stage1_iterations', 10000)  # 只训练critic头部的迭代数
        stage2_iters = self.config.get('stage2_iterations', 50000)  # 开始训练critic特征提取器的迭代数
        
        # 学习率调度参数
        warmup_iters = self.config.get('warmup_iterations', 1000)
        total_iters = self.config.get('total_iterations_for_lr_decay', 500000)
        min_lr = self.config.get('min_lr_for_scheduled_components', 1e-6)
        
        # 为各个组件计算新的学习率
        initial_lr_warmup_actor = self.config.get('initial_lr_warmup_actor', lr_actor * 0.01)
        initial_lr_warmup_critic_fe = self.config.get('initial_lr_warmup_critic_fe', lr_critic_fe * 0.01)
        initial_lr_warmup_critic_head = self.config.get('initial_lr_warmup_critic_head', lr_critic_head * 0.01)
        
        # 为不同组件分别计算学习率，考虑到解冻时间
        # 1. Actor的学习率：如果还未解冻，考虑将'iterations'重置为0，或较小值，保证解冻时有足够大的学习率
        actor_effective_iter = iterations
        if iterations <= stage2_iters:
            # Actor在stage2_iters之前被冻结，解冻后应该使用较高的学习率开始训练
            actor_effective_iter = max(0, iterations - stage2_iters)  # 相对于解冻时刻的迭代次数
            
        # 2. Critic特征提取器的学习率：类似地，考虑解冻时间
        critic_fe_effective_iter = iterations
        if iterations <= stage1_iters:
            # Critic FE在stage1_iters之前被冻结
            critic_fe_effective_iter = max(0, iterations - stage1_iters)  # 相对于解冻时刻的迭代次数
            
        # 3. Critic头部始终训练，使用正常迭代次数
        critic_head_effective_iter = iterations
        
        # 计算调整后的学习率
        new_lr_actor = calculate_scheduled_lr(actor_effective_iter, lr_actor, warmup_iters, total_iters, min_lr, initial_lr_warmup_actor)
        new_lr_critic_fe = calculate_scheduled_lr(critic_fe_effective_iter, lr_critic_fe, warmup_iters, total_iters, min_lr, initial_lr_warmup_critic_fe)
        new_lr_critic_head = calculate_scheduled_lr(critic_head_effective_iter, lr_critic_head, warmup_iters, total_iters, min_lr, initial_lr_warmup_critic_head)
        
        # 记录调整后的学习率（仅在值变化时）
        if iterations % 1000 == 0 or iterations < 100:
            self.logger.debug(f"学习率 - Actor: {new_lr_actor:.6f} (有效迭代: {actor_effective_iter}), "
                            f"Critic FE: {new_lr_critic_fe:.6f} (有效迭代: {critic_fe_effective_iter}), "
                            f"Critic头部: {new_lr_critic_head:.6f}")
        
        # 更新优化器中的学习率
        for pg in self.optimizer.param_groups:
            name = pg.get('name')
            if name == 'actor':
                pg['lr'] = new_lr_actor
            elif name == 'critic_feature_extractor':
                pg['lr'] = new_lr_critic_fe
            elif name == 'critic_head':
                pg['lr'] = new_lr_critic_head
        
        self.logger.debug(f"Iter {iterations}: LRs scheduled. Actor: {new_lr_actor:.2e}, CriticFE: {new_lr_critic_fe:.2e}, CriticHead: {new_lr_critic_head:.2e}")

    def train_on_batch(self, batch: dict) -> dict:
        """
        使用一个批次的数据执行一次完整的 PPO 更新（包括多个内部 epoch）。
        """
        # --- 1. 数据准备 ---
        try:
            # 打印批次中的主要键和形状，帮助调试
            if self.config.get('verbose_batch_info', False):
                self.logger.debug(f"Batch keys: {list(batch.keys())}")
                self.logger.debug(f"State keys: {list(batch['state'].keys()) if 'state' in batch else 'No state'}")
                for key in batch.keys():
                    if isinstance(batch[key], dict):
                        self.logger.debug(f"{key} is dict with keys: {list(batch[key].keys())}")
                    elif isinstance(batch[key], (list, np.ndarray)):
                        self.logger.debug(f"{key} shape: {np.array(batch[key]).shape}")
                    else:
                        self.logger.debug(f"{key} type: {type(batch[key])}")
            
            # 检查必要的键
            required_keys = ['state', 'action', 'adv', 'target', 'log_prob']
            for key in required_keys:
                if key not in batch:
                    self.logger.error(f"Missing required key '{key}' in batch")
                    return {}
            
            # 检查state中必要的键
            required_state_keys = ['observation', 'action_mask']
            for key in required_state_keys:
                if key not in batch['state']:
                    self.logger.error(f"Missing required key '{key}' in batch['state']")
                    return {}
            
            # 将批次数据转换为张量并移动到设备
            try:
                obs = torch.tensor(batch['state']['observation']).to(self.device, non_blocking=True)
                self.logger.debug(f"Obs shape: {obs.shape}")
            except Exception as e:
                self.logger.error(f"Error converting observation to tensor: {e}")
                self.logger.error(f"Observation type: {type(batch['state']['observation'])}")
                if isinstance(batch['state']['observation'], (list, np.ndarray)):
                    self.logger.error(f"Observation shape: {np.array(batch['state']['observation']).shape}")
                return {}
            
            try:
                mask = torch.tensor(batch['state']['action_mask']).to(self.device, non_blocking=True)
                self.logger.debug(f"Mask shape: {mask.shape}")
            except Exception as e:
                self.logger.error(f"Error converting action_mask to tensor: {e}")
                return {}
            
            states = {'obs': {'observation': obs, 'action_mask': mask}}
            
            try:
                actions = torch.tensor(batch['action'], dtype=torch.long).unsqueeze(-1).to(self.device, non_blocking=True)
                self.logger.debug(f"Actions shape: {actions.shape}")
            except Exception as e:
                self.logger.error(f"Error converting actions to tensor: {e}")
                return {}
            
            try:
                advs = torch.tensor(batch['adv']).to(self.device, non_blocking=True)
                self.logger.debug(f"Advantages shape: {advs.shape}")
            except Exception as e:
                self.logger.error(f"Error converting advantages to tensor: {e}")
                return {}
            
            try:
                targets = torch.tensor(batch['target']).to(self.device, non_blocking=True)
                self.logger.debug(f"Targets shape: {targets.shape}")
            except Exception as e:
                self.logger.error(f"Error converting targets to tensor: {e}")
                return {}
            
            try:
                old_log_probs_from_buffer = torch.tensor(batch['log_prob']).to(self.device, non_blocking=True)
                self.logger.debug(f"Log probs shape: {old_log_probs_from_buffer.shape}")
            except Exception as e:
                self.logger.error(f"Error converting log_prob to tensor: {e}")
                return {}
            
            # 提取global_obs用于critic (增强版检查)
            global_obs = None
            adv_norm_flag = self.config.get("normalize_adv",True)
            if adv_norm_flag:
                
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            else:
                self.logger.warning("Advantages normalization is disabled.")
            # 策略1：首先尝试使用'global_obs'
            if 'global_obs' in batch['state'] and batch['state']['global_obs'] is not None:
                try:
                    global_obs_data = batch['state']['global_obs']
                    if isinstance(global_obs_data, (list, np.ndarray)) and len(global_obs_data) > 0:
                        global_obs = torch.tensor(global_obs_data).to(self.device, non_blocking=True)
                        self.logger.debug(f"Using 'global_obs' for critic training. Shape: {global_obs.shape}")
                    else:
                        self.logger.warning(f"'global_obs' exists but has invalid structure: {type(global_obs_data)}")
                except Exception as e:
                    self.logger.warning(f"Error converting 'global_obs' to tensor: {e}")
            
        except Exception as e:
            self.logger.error(f"Error preparing batch for training: {e}", exc_info=True)
            self.logger.error(f"Batch keys: {list(batch.keys()) if batch else 'None'}")
            if batch and 'state' in batch:
                state_keys = list(batch['state'].keys())
                self.logger.error(f"State keys: {state_keys}")
                
            # 打印更多关于CUDA错误的可能信息
            if "CUDA" in str(e):
                self.logger.error("CUDA错误可能与形状不匹配或数据类型不兼容有关。请检查设备和张量形状。")
                
            return {}

        self.actor.train()
        self.critic.train()
        
        # --- 2. PPO 更新循环 ---
        policy_loss_epoch_avg = 0.0
        critic_loss_epoch_avg = 0.0
        entropy_loss_epoch_avg = 0.0
        total_loss_epoch_avg = 0.0
        
        try:
            old_log_probs = old_log_probs_from_buffer.detach() 
            ppo_epochs = self.config.get('epochs_per_batch', 2)
            
            for ppo_epoch in range(ppo_epochs):
                # a. Actor 前向传播和损失计算
                action_logits = self.actor(states)
                action_dist = torch.distributions.Categorical(logits=action_logits)
                log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
                ratio = torch.exp(log_probs - old_log_probs)
                clip_eps = self.config.get('clip', 0.2)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
                
                current_policy_loss_val = 0.0
                current_entropy_loss_val = 0.0
                
                # 检查actor是否可训练 (检查第一个参数的requires_grad)
                actor_trainable = False
                for param in self.actor.parameters():
                    if param.requires_grad:
                        actor_trainable = True
                        break
                
                if actor_trainable:
                    current_policy_loss_term = -torch.mean(torch.min(surr1, surr2))
                    current_entropy_loss_term = -torch.mean(action_dist.entropy())
                    current_policy_loss_val = current_policy_loss_term.item()
                    current_entropy_loss_val = current_entropy_loss_term.item()
                else: 
                    current_policy_loss_term = torch.tensor(0.0, device=self.device, requires_grad=False)
                    current_entropy_loss_term = torch.tensor(0.0, device=self.device, requires_grad=False)

                # b. Critic 前向传播和损失计算
                predicted_values = self.critic(states, global_obs)
                current_critic_loss_term = F.mse_loss(predicted_values.squeeze(-1), targets)
                
                # c. 总损失和反向传播
                total_loss = self.config.get('value_coeff', 0.5) * current_critic_loss_term
                
                if actor_trainable:
                    total_loss = total_loss + current_policy_loss_term + self.config.get('entropy_coeff', 0.01) * current_entropy_loss_term
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                trainable_params = []
                trainable_params.extend([p for p in self.actor.parameters() if p.requires_grad and p.grad is not None])
                trainable_params.extend([p for p in self.critic.parameters() if p.requires_grad and p.grad is not None])
                
                if trainable_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.config.get('grad_clip_norm', 0.5))
                
                self.optimizer.step()
                
                # 累计损失用于平均
                policy_loss_epoch_avg += current_policy_loss_val
                critic_loss_epoch_avg += current_critic_loss_term.item()
                entropy_loss_epoch_avg += current_entropy_loss_val
                total_loss_epoch_avg += total_loss.item()

            # 计算平均损失
            policy_loss_epoch_avg /= ppo_epochs
            critic_loss_epoch_avg /= ppo_epochs
            entropy_loss_epoch_avg /= ppo_epochs
            total_loss_epoch_avg /= ppo_epochs
            
        except Exception as e:
            self.logger.error(f"Error during PPO update: {e}", exc_info=True)
            return {}
            
        # 3. 返回训练指标
        return {
            'policy_loss': policy_loss_epoch_avg,
            'critic_loss': critic_loss_epoch_avg,
            'entropy_loss': entropy_loss_epoch_avg,
            'total_loss': total_loss_epoch_avg
        }

