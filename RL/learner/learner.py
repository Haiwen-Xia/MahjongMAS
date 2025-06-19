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

import signal # 用于异常处理时 gracefully exit
import sys # 用于 sys.exit

# 导入自定义模块
from replay_buffer import ReplayBuffer      # 经验回放缓冲区，用于存储 Actor 收集的数据并供 Learner 采样
# from model_pool_extended import ModelPoolServer      # 模型池服务器，用于管理和分发模型版本
# 假设您的模型定义在 model.py 中
# from model import ResNet34AC, ResNet34 # , ResNetFeatureExtractor # 确保导入了需要的模型类
from models.actor import ResNet34Actor
from models.critic import ResNet34CentralizedCritic
from collections import OrderedDict # 用于有序字典，保持参数加载顺序
from algos.ppo import PPOAlgorithm # 导入 PPO 算法实现

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

        self.inference_server_cmd_queue = config.get('inference_server_cmd_queue')
        if self.inference_server_cmd_queue is None:
            # 如果没有提供命令队列，Learner 将无法与 InferenceServer 通信
            # 根据您的设计，这可能是一个致命错误
            # 这里可以先记录一个警告，在 run 方法开始时再做更严格的检查
            print(f"Warning: Learner 未收到 'inference_server_cmd_queue'。将无法更新 InferenceServer。")
            # (在实际代码中，您可能希望在这里就抛出异常或退出)

        self.shutdown_event = config.get('shutdown_event')
        if self.shutdown_event is None:
            print(f"Warning: Learner 未收到 'shutdown_event'。可能无法优雅关闭。")

        self.shutting_down = False # 信号处理或关闭事件的标志
        
    def _signal_handler(self, signum, frame):
        """处理终止信号，确保资源被清理。"""
        if self.shutting_down: # 防止重入
            return
        self.shutting_down = True

        signal_name = signal.Signals(signum).name
        self.logger.warning(f"Learner process {self.name} (PID: {os.getpid()}) received signal {signal_name} ({signum}). Initiating graceful shutdown...")
        
        # 关闭 TensorBoard writer (如果存在)
        if self.writer:
            self.logger.info("Learner closing TensorBoard writer...")
            try:
                self.writer.close()
            except Exception as e_writer:
                self.logger.error(f"Error closing TensorBoard writer in signal handler: {e_writer}", exc_info=True)
        
        # 清理可能泄露的共享内存和CUDA资源
        self.logger.info("Cleaning up shared memory and CUDA resources...")
        self._cleanup_resources()

        self.logger.info(f"Learner {self.name} shutdown tasks complete. Exiting via signal handler.")
        # 在信号处理器中，通常建议重新抛出信号或以相应的退出码退出
        # sys.exit(128 + signum) # 常见的退出码约定
        # 或者，如果希望Python的默认信号处理（如打印KeyboardInterrupt）发生：
        # signal.signal(signum, signal.SIG_DFL) # 恢复默认处理器
        # os.kill(os.getpid(), signum) # 重新发送信号给自己
        # 简单起见，可以直接退出，但可能不会打印标准 KeyboardInterrupt 消 Messages
        sys.exit(0) # 或者一个表示异常终止的非零退出码
        
    def _set_requires_grad(self, module, requires_grad):
        """Helper function to set requires_grad for all parameters of a module."""
        if module is None:
            self.logger.warning(f"Attempted to set requires_grad on a None module.")
            return
            
        for param in module.parameters():
            param.requires_grad = requires_grad
        
        # 判断模块是属于actor还是critic
        component_name = "Unknown"
        if hasattr(self, 'actor'):
            # 检查是否是actor的一部分
            for name, mod in self.actor.named_children():
                if mod is module:
                    component_name = f"Actor.{name}"
                    break
        
        if component_name == "Unknown" and hasattr(self, 'critic'):
            # 检查是否是critic的一部分
            for name, mod in self.critic.named_children():
                if mod is module:
                    component_name = f"Critic.{name}"
                    break
        
        status = "TRAINABLE" if requires_grad else "FROZEN"
        self.logger.info(f"Component '{component_name}' is now {status}.")


    # 进程启动时执行的主函数
    def run(self):
        # --- 设置信号处理器 ---
        # 通常在进程的主要执行逻辑开始时设置
        # 注意：在 Windows 上，signal.SIGINT 可能只能被主线程捕获。
        # 对于多进程，更通用的方式可能是在主进程中捕获，然后通知子进程。
        # 但如果 Learner 是主导 ModelPoolServer 的进程，它自身处理信号是合理的。
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, self._signal_handler) # kill 命令
        except ValueError as e_signal: # 例如在非主线程中设置 SIGINT 可能会失败
            print(f"Warning: Could not set signal handlers for Learner: {e_signal}")
            # self.logger 可能还未初始化，所以用 print


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
        self.logger.info(f"Learner process {self.name} started. PID: {os.getpid()}.")
        
        # --- 修改部分：更安全地记录配置信息 ---
        config_to_log = {}
        known_unserializable_keys = ['shutdown_event', 'inference_server_cmd_queue'] # 以及其他可能的队列或事件对象
        
        for key, value in self.config.items():
            if key in known_unserializable_keys:
                config_to_log[key] = f"<{type(value).__name__} object at {hex(id(value))}>" # 或者 str(value)
            elif callable(value) and hasattr(value, '__name__'): # 例如模型类
                config_to_log[key] = f"<class '{value.__name__}'>"
            else:
                # 对于其他值，先尝试直接复制，json.dumps 的 default 会处理后续问题
                config_to_log[key] = value 
        
        try:
            # 使用 default=str 作为备用方案处理其他未预料到的不可序列化类型
            config_json_str = json.dumps(config_to_log, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Learner Config (serializable view): \n{config_json_str}")
        except Exception as e_json:
            self.logger.error(f"Failed to serialize config for logging: {e_json}. Logging config keys only.")
            # 如果上面仍然失败，只记录键名或更简单的表示
            config_keys_str = json.dumps(list(self.config.keys()), indent=2, ensure_ascii=False)
            self.logger.info(f"Learner Config Keys: \n{config_keys_str}")

        # self.logger.info(f"Learner process {self.name} started. Config: \n{json.dumps(self.config, indent=2, ensure_ascii=False)}")

        # 检查必要的队列是否存在
        if self.inference_server_cmd_queue is None:
            self.logger.critical("'inference_server_cmd_queue' not provided in config. Learner cannot communicate with InferenceServer. Exiting.")
            # ... (恢复信号处理器并退出) ...
            return
        if self.shutdown_event is None:
            self.logger.warning("'shutdown_event' not provided. Learner might not shut down gracefully via main process signal.")
            # 可以选择不退出，但这是一个潜在问题
        
        self.logger.info("Learner will communicate with an external InferenceServer.")

        # 后续逻辑全部放在 try 中, 使得我们手动终止程序时会自动清理 ModelPoolServer 实例
        try:
            # --- 1. 创建模型和算法实例 ---
            device = torch.device(self.config['device'])
            self.logger.info(f"Learner will train model on device: {device}")
            
            # a. 创建模型实例 (Actor 和 Critic)
            #    这里假设您有一个主模型类，它内部包含了 actor 和 critic
            #    如果它们是完全分离的，则分别创建
            self.actor = self.config['actor_class'](
                in_channels=self.config['in_channels'],
                out_channels=self.config['out_channels'],
            ).to(device)
            self.critic = self.config['critic_class'](
                in_channels_obs=self.config['in_channels'],
                in_channels_extra=self.config.get('critic_extra_in_channels', 0),
            ).to(device)

            self.logger.info("RL model instance created.")

            # b. 从 SL 模型加载初始权重
        
            # 从配置中获取初始 Actor 模型路径
            initial_actor_path = self.config.get('initial_actor_eval_path')
            state_dict = torch.load(initial_actor_path, map_location=device)
            self.actor.load_state_dict(state_dict)
        
            # 从配置中获取初始 Critic 模型路径
            initial_critic_path = self.config.get('initial_critic_eval_path')
            critic_state_dict = torch.load(initial_critic_path, map_location=device)
            self.critic.load_state_dict(critic_state_dict)

            self.logger.info("SL weights loaded and Critic FE initialized.")

            # c. 创建算法实例，它会接管模型、优化器和更新逻辑
            self.algorithm = PPOAlgorithm(self.config, self.actor, self.critic, self.logger)
            self.logger.info("PPOAlgorithm instance created.")

            # --- 6. 等待 Replay Buffer 数据 ---
            min_samples = self.config.get('min_sample_to_start_learner', 20000)
            
            self.logger.info(f"Waiting for Replay Buffer to have at least {min_samples} samples...")

            last_logged_size = -1

            should_exit = False
            
            # 等待ReplayBuffer循环
            while True:  # 无条件循环，内部检查退出条件
                if should_exit:
                    self.logger.info("should_exit标志已设置，退出等待循环")
                    break
                
                try:
                    # # 安全检查shutdown_event
                    # if self.shutdown_event is not None:
                    #     try:
                    #         if self.shutdown_event.is_set():
                    #             self.logger.info("检测到关闭事件已设置，退出等待循环")
                    #             should_exit = True
                    #             break
                    #     except Exception as e_event:
                    #         self.logger.error(f"访问shutdown_event时出错: {e_event}，假设未设置")
                    
                    # 获取当前buffer大小
                    current_buffer_size = self.replay_buffer.size()  # Calls _flush internally
                    
                    # 检查是否达到所需样本数
                    if current_buffer_size >= min_samples:
                        self.logger.info(f"已达到所需样本数: {current_buffer_size}/{min_samples}")
                        break
                    
                    # 定期日志记录
                    else:
                        self.logger.info(f"Replay Buffer size: {current_buffer_size}/{min_samples}. Waiting for more samples...")
                        time.sleep(5)
                        
                    
                except Exception as e_rb_wait:  # 捕获缓冲区检查过程中的错误
                    self.logger.error(f"Error checking replay buffer size while waiting: {e_rb_wait}", exc_info=True)
                    # 安全检查shutdown_event
                    try:
                        if self.shutdown_event is not None and self.shutdown_event.is_set():
                            self.logger.info("检测到关闭事件已设置，catch块中退出等待循环")
                            break
                    except Exception as e_event_catch:
                        self.logger.error(f"Catch块中访问shutdown_event时出错: {e_event_catch}，继续等待")
                    # 出错时等待较长时间
                    time.sleep(5)
            
            self.logger.info(f"Minimum samples ({min_samples}) reached in Replay Buffer. Starting training loop.")
            
            # --- 7. 主训练循环 ---
            cur_time_ckpt = time.time() 
            cur_time_log = time.time() 
            iterations = 0 
            steps_processed_since_log = 0 
                
            # 主循环开始前安全检查shutdown_event以避免崩溃
            shutdown_detected = False
            # try:
            #     if self.shutdown_event is not None and self.shutdown_event.is_set():
            #         shutdown_detected = True
            # except Exception as e_main_shutdown:
            #     self.logger.error(f"主循环前访问shutdown_event时出错: {e_main_shutdown}，假设未设置")
                
            while not shutdown_detected:  # 使用安全标志而不是直接访问shutdown_event
                start_iter_time = time.time()

                # --- 7.0 更新模型参数状态 (解冻) 和学习率 ---
                # a. 更新模型的可训练状态 (解冻)
                # 首先调用更新冻结状态，然后根据冻结状态调整学习率
                self.algorithm.update_freezing_status(iterations)
                
                # b. 根据冻结状态计算和更新学习率
                # 注意：schedule_learning_rate已经优化，考虑了组件解冻时间
                self.algorithm.schedule_learning_rate(iterations)
                
                # # c. 定期记录各组件的训练状态和学习率（便于调试）
                # log_freq = self.config.get('training_components_log_freq', 1000)
                # if iterations % log_freq == 0 or iterations < 10:
                #     # 检查和记录各组件的可训练状态
                #     actor_trainable = any(p.requires_grad for p in self.actor.parameters())
                #     critic_fe_trainable = False
                #     critic_head_trainable = False
                    
                #     if hasattr(self.critic, 'feature_extractor_obs'):
                #         critic_fe_trainable = any(p.requires_grad for p in self.critic.feature_extractor_obs.parameters())
                    
                #     head_components = []
                #     if hasattr(self.critic, 'feature_extractor_extra'):
                #         head_components.append(self.critic.feature_extractor_extra)
                #     if hasattr(self.critic, 'critic_head_mlp'):
                #         head_components.append(self.critic.critic_head_mlp)
                    
                #     critic_head_trainable = any(any(p.requires_grad for p in comp.parameters()) 
                #                                for comp in head_components if comp is not None)
                    
                #     # 获取当前学习率
                #     lr_actor, lr_critic_fe, lr_critic_head = "N/A", "N/A", "N/A"
                #     for pg in self.algorithm.optimizer.param_groups:
                #         if pg.get('name') == 'actor': lr_actor = f"{pg['lr']:.2e}"
                #         if pg.get('name') == 'critic_feature_extractor': lr_critic_fe = f"{pg['lr']:.2e}"
                #         if pg.get('name') == 'critic_head': lr_critic_head = f"{pg['lr']:.2e}"
                    
                #     self.logger.info(f"Iter {iterations} - 组件状态: Actor {'训练中' if actor_trainable else '冻结'} (LR={lr_actor}), "
                #                     f"CriticFE {'训练中' if critic_fe_trainable else '冻结'} (LR={lr_critic_fe}), "
                #                     f"CriticHead {'训练中' if critic_head_trainable else '冻结'} (LR={lr_critic_head})")

                # --- 7.1. 采样和数据准备 ---
                self.logger.debug(f"Iter {iterations}: Attempting to sample from Replay Buffer.")
                batch = None
                try:
                    batch = self.replay_buffer.sample(self.config['batch_size'])
                except Exception as e_sample:
                    self.logger.error(f"Iter {iterations}: Error during replay_buffer.sample(): {e_sample}", exc_info=True)
                    time.sleep(1)
                    continue

                if batch is None:
                    rb_size = self.replay_buffer.size() # This calls _flush
                    q_size = self.replay_buffer.queue.qsize() if hasattr(self.replay_buffer.queue, 'qsize') else 'N/A'
                    self.logger.info(f"Iter {iterations}: replay_buffer.sample() returned None. Buffer(timesteps): {rb_size}. InputQueue(episodes): {q_size}. Sleeping...")
                    time.sleep(self.config.get('learner_empty_batch_sleep_sec', 1.0)) 
                    continue
                self.logger.debug(f"Iter {iterations}: Batch sampled successfully.")
                
                # --- 7.2. PPO 更新 ---
                # d. 调用算法执行训练步骤
                train_metrics = self.algorithm.train_on_batch(batch)

                if not train_metrics:  # 如果训练失败，跳过这次迭代
                    self.logger.warning(f"Iter {iterations}: train_on_batch returned empty metrics. Skipping iteration.")
                    continue

                # --- 7.3. 日志记录和 TensorBoard ---
                steps_processed_since_log += self.config['batch_size']
                current_time = time.time()
                log_interval = self.config.get('log_interval', 100)

                if iterations % log_interval == 0: 
                    time_since_log = current_time - cur_time_log if cur_time_log and iterations > 0 else (current_time - start_iter_time if iterations == 0 else 1.0)
                    iterations_per_sec = log_interval / time_since_log if time_since_log > 0 and iterations > 0 else \
                                        (1 / time_since_log if time_since_log > 0 and iterations == 0 else 0.0)
                    samples_per_sec = steps_processed_since_log / time_since_log if time_since_log > 0 else 0.0

                    buffer_size = self.replay_buffer.size()
                    buffer_stats_dict = self.replay_buffer.stats() if hasattr(self.replay_buffer, 'stats') and callable(getattr(self.replay_buffer, 'stats')) else {}
                    sample_in_rate = buffer_stats_dict.get('samples_in_per_second_smoothed', buffer_stats_dict.get('samples_in_per_second',0)) 
                    sample_out_rate = buffer_stats_dict.get('samples_out_per_second_smoothed', buffer_stats_dict.get('samples_out_per_second',0))
                      # 获取当前学习率信息
                    log_lr_actor, log_lr_critic_fe, log_lr_critic_head = "N/A", "N/A", "N/A"
                    for pg in self.algorithm.optimizer.param_groups: 
                        lr_val_str = f"{pg['lr']:.2e}"
                        if pg.get('name') == 'actor': log_lr_actor = lr_val_str
                        if pg.get('name') == 'critic_feature_extractor': log_lr_critic_fe = lr_val_str
                        if pg.get('name') == 'critic_head': log_lr_critic_head = lr_val_str
                    
                    lr_log_str = f"LRs (A/CFE/CH): {log_lr_actor}/{log_lr_critic_fe}/{log_lr_critic_head}"
                    
                    # 从训练指标中获取损失值
                    policy_loss_epoch_avg = train_metrics.get('policy_loss', 0.0)
                    critic_loss_epoch_avg = train_metrics.get('critic_loss', 0.0)
                    entropy_loss_epoch_avg = train_metrics.get('entropy_loss', 0.0)
                    total_loss_epoch_avg = train_metrics.get('total_loss', 0.0)
                    
                    log_msg = (
                        f"Iter: {iterations} | {lr_log_str} | "
                        f"Loss(Actual): {total_loss_epoch_avg:.4f} " 
                        f"(P_contrib: {policy_loss_epoch_avg:.4f}, C_contrib: {critic_loss_epoch_avg:.4f}, E_contrib: {entropy_loss_epoch_avg:.4f}) | "
                        f"Buffer: {buffer_size} (QueueEp: {buffer_stats_dict.get('queue_size',0)}, In/s: {sample_in_rate:.1f}, Out/s: {sample_out_rate:.1f}) | "
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
                          # 记录各组件的学习率（仅当组件可训练时）
                        for pg in self.algorithm.optimizer.param_groups:
                            group_name = pg.get('name', 'UnnamedGroup')
                            component_name_map = {
                                'actor': 'Actor',
                                'critic_feature_extractor': 'CriticFE', 
                                'critic_head': 'CriticHead'
                            }
                            tb_component_name = component_name_map.get(group_name)
                            if tb_component_name:
                                is_trainable_now = False
                                if tb_component_name == 'Actor':
                                    # Check if any actor parameter is trainable
                                    is_trainable_now = any(p.requires_grad for p in self.actor.parameters())
                                elif tb_component_name == 'CriticFE' and hasattr(self.critic, 'feature_extractor_obs'):
                                    # Check if critic FE obs is trainable
                                    is_trainable_now = any(p.requires_grad for p in self.critic.feature_extractor_obs.parameters())
                                elif tb_component_name == 'CriticHead':
                                    # Check if any component in critic head group is trainable
                                    is_trainable_now = False
                                    if hasattr(self.critic, 'critic_head_mlp'):
                                        is_trainable_now = any(p.requires_grad for p in self.critic.critic_head_mlp.parameters())
                                    if not is_trainable_now and hasattr(self.critic, 'feature_extractor_extra'):
                                        is_trainable_now = any(p.requires_grad for p in self.critic.feature_extractor_extra.parameters())
                                
                                if is_trainable_now:
                                    self.writer.add_scalar(f'LearningRate/{tb_component_name}', pg['lr'], iterations)
                        self.writer.flush()

                    cur_time_log = current_time
                    steps_processed_since_log = 0

                # --- 7.4 更新 InferenceServer 中的模型 ---
                update_eval_model_interval = self.config.get('model_push_interval', 10)
                
                if iterations > 0 and iterations % update_eval_model_interval == 0: 
                    self.logger.debug(f"Iter {iterations}: Attempting to send updated model state to InferenceServer for its eval_model.")
                    try:
                        # 发送 actor 状态
                        actor_state_cpu = {k: v.cpu() for k, v in self.actor.state_dict().items()}
                        critic_state_cpu = {k: v.cpu() for k, v in self.critic.state_dict().items()}

                        if self.inference_server_cmd_queue:
                            # 添加超时机制，避免因队列满导致阻塞
                            queue_put_timeout = self.config.get('queue_put_timeout_sec', 5.0)
                            
                            # 尝试推送actor模型，添加超时
                            try:
                                self.inference_server_cmd_queue.put(("UPDATE_ACTOR_MODEL", actor_state_cpu), 
                                                                  timeout=queue_put_timeout)
                                self.logger.debug(f"Iter {iterations}: Command 'UPDATE_ACTOR_MODEL' sent to InferenceServer.")
                            except Exception as e_put_actor:
                                self.logger.error(f"Iter {iterations}: Failed to send 'UPDATE_ACTOR_MODEL' command: {e_put_actor}. "
                                                 f"Queue might be full or InferenceServer unresponsive.")
                            
                            # 尝试推送critic模型，添加超时
                            try:
                                self.inference_server_cmd_queue.put(("UPDATE_CRITIC_MODEL", critic_state_cpu),
                                                                  timeout=queue_put_timeout)
                                self.logger.debug(f"Iter {iterations}: Command 'UPDATE_CRITIC_MODEL' sent to InferenceServer.")
                            except Exception as e_put_critic:
                                self.logger.error(f"Iter {iterations}: Failed to send 'UPDATE_CRITIC_MODEL' command: {e_put_critic}. "
                                                 f"Queue might be full or InferenceServer unresponsive.")
                        else:
                            self.logger.error(f"Iter {iterations}: Cannot send model update to InferenceServer: command queue is None.")

                    except Exception as e_update_eval_model:
                        self.logger.error(f"Iter {iterations}: Failed to send model update commands: {e_update_eval_model}", exc_info=True)

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
                            'actor_state_dict': self.actor.state_dict(),
                            'critic_state_dict': self.critic.state_dict(),
                            'optimizer_state_dict': self.algorithm.optimizer.state_dict(),
                            'config': serializable_config 
                        }, path)
                        cur_time_ckpt = t_now_ckpt
                    except Exception as e:
                        self.logger.error(f"Failed to save checkpoint at iteration {iterations}: {e}", exc_info=True)

                iterations += 1
                
                # 安全检查shutdown_event，避免直接访问导致崩溃
                # try:
                #     if self.shutdown_event is not None and self.shutdown_event.is_set():
                #         self.logger.info(f"在第 {iterations} 次迭代结束时检测到关闭事件已设置，退出主循环")
                #         shutdown_detected = True
                #         break
                # except Exception as e_iter_shutdown:
                #     # 如果访问shutdown_event出错，继续执行但记录错误
                #     self.logger.error(f"第 {iterations} 次迭代检查shutdown_event时出错: {e_iter_shutdown}")
                #     # 每100次迭代检查一次self.shutting_down作为备用方式
                #     if iterations % 100 == 0 and self.shutting_down:
                #         self.logger.info(f"通过self.shutting_down检测到关闭信号，退出主循环")
                #         break
                
            self.logger.info("Learner training loop finished (or was interrupted).")
            if self.writer:
                self.writer.close()
        
        except KeyboardInterrupt: 
            self.logger.warning(f"Learner {self.name} (PID: {os.getpid()}) caught KeyboardInterrupt in main try block.")
            if not self.shutting_down: # 如果信号处理器还未运行
                self._signal_handler(signal.SIGINT, None)
        except Exception as e_main_run:
            self.logger.critical(f"Learner {self.name} (PID: {os.getpid()}) encountered an unhandled exception in main run logic: {e_main_run}", exc_info=True)
        finally:
            self.logger.info(f"Learner {self.name} (PID: {os.getpid()}) entering finally block of run method.")
            if not self.shutting_down: 
                self.shutting_down = True 
                self.logger.info("Learner run method finished or aborted without external signal. Performing final cleanup...")
                # Learner 不再拥有 ModelPoolServer，所以不在这里清理它
                if self.writer:
                    try: self.writer.close()
                    except Exception as e_final_writer: self.logger.error(f"Error during final TensorBoard writer close: {e_final_writer}", exc_info=True)
            
            # 无论如何都要清理共享内存和CUDA资源，防止内存泄漏
            self.logger.info("最终清理所有共享内存和CUDA资源...")
            try:
                self._cleanup_resources()
            except Exception as e_cleanup:
                self.logger.error(f"最终资源清理过程中出错: {e_cleanup}", exc_info=True)
                
            try:
                signal.signal(signal.SIGINT, original_sigint_handler)
                signal.signal(signal.SIGTERM, original_sigterm_handler)
            except Exception as e_restore_signal:
                 self.logger.warning(f"Could not restore original signal handlers in Learner: {e_restore_signal}")
            self.logger.info(f"Learner {self.name} (PID: {os.getpid()}) run method fully completed.")
    
    def _cleanup_resources(self):
        """清理共享内存和CUDA资源，防止内存泄漏"""
        if self.logger:
            self.logger.info("开始清理资源以防止内存泄漏...")
        else:
            print("开始清理资源以防止内存泄漏...")
            
        # 清理模型资源
        if hasattr(self, 'actor') and self.actor is not None:
            try:
                del self.actor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.logger:
                    self.logger.info("Actor模型资源已清理")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"清理Actor模型资源时出错: {e}", exc_info=True)
                else:
                    print(f"清理Actor模型资源时出错: {e}")
        
        if hasattr(self, 'critic') and self.critic is not None:
            try:
                del self.critic
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.logger:
                    self.logger.info("Critic模型资源已清理")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"清理Critic模型资源时出错: {e}", exc_info=True)
                else:
                    print(f"清理Critic模型资源时出错: {e}")
        
        # 清理优化器资源
        if hasattr(self, 'algorithm') and hasattr(self.algorithm, 'optimizer') and self.algorithm.optimizer is not None:
            try:
                del self.algorithm.optimizer
                if self.logger:
                    self.logger.info("优化器资源已清理")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"清理优化器资源时出错: {e}", exc_info=True)
                else:
                    print(f"清理优化器资源时出错: {e}")
        
        if hasattr(self, 'algorithm') and self.algorithm is not None:
            try:
                del self.algorithm
                if self.logger:
                    self.logger.info("算法实例资源已清理")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"清理算法实例资源时出错: {e}", exc_info=True)
                else:
                    print(f"清理算法实例资源时出错: {e}")
        
        # 强制进行一次垃圾回收
        import gc
        collected = gc.collect()
        if self.logger:
            self.logger.info(f"垃圾回收器回收了 {collected} 个对象")
        else:
            print(f"垃圾回收器回收了 {collected} 个对象")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.logger:
                self.logger.info("CUDA缓存已清空")
            else:
                print("CUDA缓存已清空")
                
        if self.logger:
            self.logger.info("资源清理完成")
        else:
            print("资源清理完成")