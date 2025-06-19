import torch
import torch.multiprocessing as mp # 为更好地处理CUDA（如果子进程也使用CUDA），使用torch的多进程
import os
import logging
import json
import time # 用于join超时和可能的sleep
import signal # 用于信号处理
import sys # 用于sys.exit

# 假设这些导入路径相对于您的项目结构是正确的
from replay_buffer import ReplayBuffer 
from actor.actor import Actor # Actor 类 
from learner.learner import Learner # Learner 类 
from inference_server.inference_server import InferenceServer # 导入新的 InferenceServer
from utils import setup_process_logging_and_tensorboard, save_experiment_config # 日志和 TensorBoard 设置工具

from models.actor import ResNet34Actor # 导入具体的 Actor 模型
from models.critic import ResNet34CentralizedCritic # 导入具体的 Critic 模型

from agent.feature import FeatureAgent # 导入 FeatureAgent 类

# --- 全局关闭事件和进程句柄，用于信号处理 ---
shutdown_event = mp.Event() # 用于通知所有子进程开始关闭的事件
g_learner_process = None    # Learner 进程的全局句柄
g_actor_processes = []      # Actor 进程列表的全局句柄
g_inference_server_process = None # InferenceServer 进程的全局句柄
g_main_logger = None        # 主进程的 logger
g_main_writer = None        # 主进程的 TensorBoard writer
# 队列也需要全局可访问，以便 cleanup 函数可以发送 SHUTDOWN 命令
g_learner_to_server_cmd_q = None


def cleanup_all_processes(signal_received=None, frame=None):
    """
    集中的清理函数，用于关闭所有子进程和释放相关资源。
    可以由信号处理器或主程序的 finally 块调用。
    """
    global shutdown_event, g_learner_process, g_actor_processes, g_inference_server_process
    global g_main_logger, g_main_writer, g_learner_to_server_cmd_q
    
    # 防止重入 (如果信号处理器被多次触发或 cleanup 被多次调用)
    if hasattr(cleanup_all_processes, 'is_shutting_down') and cleanup_all_processes.is_shutting_down:
        return
    cleanup_all_processes.is_shutting_down = True # 设置标志，表示正在关闭

    log_func = print # 默认使用 print，如果 logger 可用则使用 logger
    if g_main_logger:
        log_func = g_main_logger.warning if signal_received else g_main_logger.info

    if signal_received:
        log_func(f"Main process received signal {signal.Signals(signal_received).name}. Initiating graceful shutdown...")
    else:
        log_func("Main process initiating cleanup (e.g., from finally block or normal exit)...")

    log_func("Setting shutdown event for child processes...")
    shutdown_event.set() # 通知所有子进程

    # 1. 等待 Actor 进程退出 (它们可能依赖 InferenceServer，但应首先停止其循环)
    for actor_proc in g_actor_processes:
        if actor_proc and actor_proc.is_alive():
            log_func(f"Waiting for Actor process ({actor_proc.name}, PID: {actor_proc.pid}) to join...")
            actor_proc.join(timeout=15) # 每个 actor 等待15秒
            if actor_proc.is_alive():
                log_func(f"Actor process ({actor_proc.name}) did not exit gracefully. Terminating...")
                actor_proc.terminate() # 强制终止
                actor_proc.join(timeout=5) # 等待强制终止完成
            if actor_proc.is_alive(): # 再次检查
                 log_func(f"Actor process ({actor_proc.name}) could not be terminated.")
            else:
                 log_func(f"Actor process ({actor_proc.name}) finished.")
    
    # 2. 等待 Learner 进程退出 (它可能需要向 InferenceServer 发送最后更新)
    if g_learner_process and g_learner_process.is_alive():
        log_func(f"Waiting for Learner process ({g_learner_process.name}, PID: {g_learner_process.pid}) to join...")
        g_learner_process.join(timeout=30) # 给 Learner 30秒时间
        if g_learner_process.is_alive():
            log_func(f"Learner process ({g_learner_process.name}) did not exit gracefully. Terminating...")
            g_learner_process.terminate()
            g_learner_process.join(timeout=5)
        if g_learner_process.is_alive():
            log_func(f"Learner process ({g_learner_process.name}) could not be terminated.")
        else:
            log_func(f"Learner process ({g_learner_process.name}) finished.")

    # 3. 通知 InferenceServer 关闭并等待其退出
    if g_inference_server_process and g_inference_server_process.is_alive():
        log_func("Sending SHUTDOWN command to InferenceServer...")
        if g_learner_to_server_cmd_q: # 确保命令队列存在
            try:
                # Learner 通常会发送这个命令，但作为备用，主进程也可以发送
                g_learner_to_server_cmd_q.put(("SHUTDOWN", None), timeout=5) # 带超时的 put
            except Exception as e_cmd_q_put: # 例如 queue.Full
                log_func(f"Error sending SHUTDOWN to InferenceServer via queue: {e_cmd_q_put}")
        
        log_func(f"Waiting for InferenceServer process ({g_inference_server_process.name}, PID: {g_inference_server_process.pid}) to join...")
        g_inference_server_process.join(timeout=20) # 给服务器20秒时间处理关闭
        if g_inference_server_process.is_alive():
            log_func(f"InferenceServer process ({g_inference_server_process.name}) did not exit gracefully. Terminating...")
            g_inference_server_process.terminate()
            g_inference_server_process.join(timeout=5)
        if g_inference_server_process.is_alive():
            log_func(f"InferenceServer process ({g_inference_server_process.name}) could not be terminated.")
        else:
            log_func(f"InferenceServer process ({g_inference_server_process.name}) finished.")
    
    # 4. 清理主进程的资源 (例如 TensorBoard writer)
    if g_main_writer:
        log_func("Main process closing its TensorBoard writer.")
        try: g_main_writer.close()
        except Exception as e_writer_close: log_func(f"Error closing main TensorBoard writer: {e_writer_close}")
    
    log_func("Main process cleanup finished.")
    if signal_received: # 如果是由信号触发的，则以相应方式退出
        sys.exit(128 + signal_received) # 常见的Unix退出码约定

# --- Configuration ---
# It's often better to load config from a file (e.g., YAML, JSON) or use argparse
CONFIG = {
    # 实验元数据
    'experiment_meta': {
        'experiment_name': "Using_Inference_Server", # Use underscores or avoid special chars for dir names
        'log_base_dir': 'logs', # Base directory for logs and TensorBoard
        'checkpoint_base_dir': 'log/model', # Base directory for checkpoints
    },
    
    # Model & Agent 设置
    'model_agent_config' : {
        'in_channels': 187,
        'out_channels': 235,
        'critic_extra_in_channels': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'actor_class': ResNet34Actor, # 指定 Actor 模型类
        'critic_class': ResNet34CentralizedCritic, # 指定 Critic 模
        'agent_clz': FeatureAgent, # 指定环境内部使用的 Agent 类
        'initial_actor_eval_path': "initial_models/actor_from_sl.pth",
        'initial_critic_eval_path': "initial_models/centralized_critic_initialized.pth",
    },

    # Replay Buffer 设置
    'replay_buffer_size': 50000,
    'replay_buffer_episode_capacity': 400, # Renamed for clarity

    # Benchmark 策略设置
    'benchmark_policies': {
        'initial_il_policy': "initial_models/actor_from_sl.pth",
    },
    'server_hosted_benchmark_names': ["initial_il_policy"],

    # 关于 inference_server 的相关设置
    'inference_server_config': {
        'inference_batch_size': 128,
        'inference_max_wait_ms': 5, # 单位 ms
        'max_history_actors': 10, # 历史 Actor 的最大数量
        'history_actors_update_every': 100, # 每 100 次更新模型保存一次历史 Actor 模型
    },

    # 分布式训练配置
    'num_actors': 8,

    # 关于 Actor 的相关设置
    'actor_config': {
        'log_interval': 10, # Actor 日志间隔
        'num_env_steps': 10000000, # Total number of environment steps to run across all actors
        'num_envs_per_actor': 8, # 每个 Actor 运行的环境数量
        # 对手的相关涉资

        # 多样化 opponent
        'p_opponent_historical' : 0.05,
        'p_opponent_benchmark': 1,
        'opponent_model_change_interval': 500, # 每多少个 episode 替换一次对手

        # 收集数据处理
        'filter_single_action_steps': False, # [Deprecated] 是否过滤掉只有单个可能 action 的时间步
        'use_normalized_reward': True,
        'draw_reward': -0.5,

        'inference_timeout_seconds': 5, 
    },

    # 关于 Learner 的相关设置
    'learner_config': {
        'log_interval': 100, 
        'model_push_interval': 10,
        'ckpt_save_interval_seconds': 600, # 保存检查点的间隔时间
        'min_sample_to_start_learner': 20, # 开始训练需要的 buffer 样本数
        'training_components_log_freq': 1000,  # 训练组件状态日志频率
    },

    # PPO 基本设置
    'ppo_config': {
        'gamma': 0.98,      # Discount factor for GAE/TD Target
        'lambda': 0.97,     # Lambda for GAE
        'clip': 0.2,        # PPO clip epsilon
        'grad_clip_norm': 0.3,
        'value_coeff': 0.5, # Coefficient for value loss (common to scale down)
        'entropy_coeff': -1e-3, # Coefficient for entropy bonus 正常是正数，但这里是负数表示惩罚
        'batch_size': 1024, # Increased batch size
        'epochs_per_batch': 5, # Renamed 'epochs' for clarity (PPO inner loops)
        'normalize_adv': True,


        # Learner Hyperparameters        'lr_actor': 3e-5,  # Actor overall learning rate
        'lr_critic_feature_extractor': 3e-4,  # Critic feature_extractor_obs learning rate  
        'lr_critic_head': 3e-4,  # Critic head (includes feature_extractor_extra + critic_head_mlp) learning rate
        
        # Learning rate scheduling
        'use_lr_scheduler': True,  # Enable learning rate scheduling
        "warmup_iterations": 1000,
        'total_iterations_for_lr_decay': 500000,
        'min_lr_for_scheduled_components': 1e-6,
        'initial_lr_warmup_actor': 3e-6,
        'initial_lr_warmup_critic_fe': 3e-6,
        'initial_lr_warmup_critic_head': 3e-5,
        # Staged training configuration
        'stage1_iterations': 1000,  # Only train critic_head group (fe_extra + head_mlp)
        'stage2_iterations': 2000,  # Unfreeze critic_fe_obs, keep actor frozen
        # Stage 3 starts at stage2_iterations: joint training of all components
    },

    # 性能 profiling 相关设置
    'profiling_config': {
        'enable_profiling': False,
        'enable_profiling_actor': False
    }
}


def main():
    global g_learner_process, g_actor_processes, g_inference_server_process
    global g_main_logger, g_main_writer, shutdown_event, g_learner_to_server_cmd_q

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [ROOT/%(levelname)s] %(message)s', # Differentiate root logs
        handlers=[logging.StreamHandler()] # Root logs to console
    )
    
    run_name = CONFIG['experiment_meta']['experiment_name']
    log_base_dir = CONFIG['experiment_meta']['log_base_dir']  #logs

    try:
        # 使用新的日志设置函数
        g_main_logger, g_main_writer, main_log_paths = setup_process_logging_and_tensorboard(
            log_base_dir, CONFIG, process_name='main_train', log_type='main'
        )
        
        # 保存实验配置
        save_experiment_config(CONFIG, main_log_paths['config_save_path'])
        
    except Exception as e_log_setup:
        logging.error(f"Main process logger/writer setup failed: {e_log_setup}", exc_info=True)
        g_main_logger = logging.getLogger("main_fallback_logger") # 使用一个备用logger，以防万一
        g_main_writer = None # Writer 可能无法创建
        main_log_paths = {'checkpoint_dir': 'log/model'}  # 提供默认路径

    # 设置信号处理器
    original_sigint_handler = signal.getsignal(signal.SIGINT) # 保存原始的 SIGINT 处理器
    original_sigterm_handler = signal.getsignal(signal.SIGTERM) # 保存原始的 SIGTERM 处理器
    try:
        signal.signal(signal.SIGINT, cleanup_all_processes)  # 捕获 Ctrl+C
        signal.signal(signal.SIGTERM, cleanup_all_processes) # 捕获 kill 命令
        g_main_logger.info("Main process signal handlers set. Press Ctrl+C to attempt graceful shutdown.")
    except Exception as e_signal_setup: # 例如在非主线程中设置 (虽然这里是主线程)
        g_main_logger.error(f"Failed to set signal handlers: {e_signal_setup}", exc_info=True)

    # --- 核心训练流程 ---
    try:
        g_main_logger.info("="*60)
        g_main_logger.info(f"Starting Experiment: {CONFIG['experiment_meta']['experiment_name']} (Main Process PID: {os.getpid()})")
        g_main_logger.info(f"Log paths: {main_log_paths}")
        g_main_logger.info("="*60)

        # 1. 创建 InferenceServer 通信队列
        g_learner_to_server_cmd_q = mp.Queue()   # Learner -> Server 的命令队列
        actors_to_server_req_q = mp.Queue()      # Actors -> Server 的推理请求队列
        
        # Server -> Actors 的响应队列 (每个 Actor 一个)
        server_to_actors_resp_qs = {}
        for i in range(CONFIG['num_actors']):
            actor_id_key = f'Actor-{CONFIG.get("actor_id_base", 0) + i}' # 与 Actor 配置中的 name 保持一致
            server_to_actors_resp_qs[actor_id_key] = mp.Queue()
        g_main_logger.info("Communication queues for InferenceServer created.")        # 2. 准备 InferenceServer 配置
        
        benchmark_models_info_for_server = {}
        # 从 benchmark_policies 配置中获取基准模型信息
        for policy_name, policy_path in CONFIG.get('benchmark_policies', {}).items():
            if policy_path and os.path.isfile(policy_path):
                benchmark_models_info_for_server[policy_name] = {"path": policy_path}
                g_main_logger.info(f"Added benchmark model '{policy_name}' from path: {policy_path}")
            else:
                g_main_logger.warning(f"Benchmark model path not found: {policy_path} for policy '{policy_name}'")
        
        server_config = {
            'name': 'InferenceServer', # 服务器进程的名称
            'server_id': f"inference_server_{os.getpid()}", # 给服务器一个唯一ID，用于日志

            'benchmark_models_info': benchmark_models_info_for_server, # 要托管的基准模型
        }
        server_config.update(CONFIG['experiment_meta']) # 添加实验元数据
        server_config.update(CONFIG['model_agent_config']) # 添加模型和代理配置
        server_config.update(CONFIG['inference_server_config']) # 添加 InferenceServer 特定配置

        g_main_logger.info("InferenceServer configuration prepared.")

        # 3. 实例化并启动 InferenceServer
        g_inference_server_process = InferenceServer(
            server_config,
            g_learner_to_server_cmd_q,
            actors_to_server_req_q,
            server_to_actors_resp_qs
        )
        g_main_logger.info("Starting InferenceServer process...")
        g_inference_server_process.start()
        time.sleep(3) # 给服务器一点时间初始化 (可选，但有时有帮助)
        if not g_inference_server_process.is_alive():
            g_main_logger.critical("InferenceServer failed to start! Check its logs. Exiting.")
            cleanup_all_processes() # 尝试清理
            return # 主程序退出
        g_main_logger.info(f"InferenceServer process (PID: {g_inference_server_process.pid}) started.")


        # 4. 初始化 Replay Buffer
        g_main_logger.info("Initializing Replay Buffer...")
        replay_buffer = ReplayBuffer(
            CONFIG['replay_buffer_size'], 
            CONFIG['replay_buffer_episode_capacity'],
        )

        g_main_logger.info("Replay Buffer initialized.")

        # 5. 准备并启动 Learner
        checkpoint_dir = main_log_paths['checkpoint_dir']  # 使用新的路径
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        learner_config = CONFIG['learner_config'].copy() # 为 Learner 创建配置副本
        learner_config.update(CONFIG['experiment_meta']) # 添加实验元数据
        learner_config.update(CONFIG['model_agent_config']) # 添加模型和代理配置
        learner_config.update(CONFIG['ppo_config']) # 添加 PPO 配置

        learner_config['ckpt_save_path'] = checkpoint_dir # 传递检查点保存路径
        learner_config['shutdown_event'] = shutdown_event # 传递关闭事件
        learner_config['inference_server_cmd_queue'] = g_learner_to_server_cmd_q # 传递命令队列
        learner_config['log_base_dir'] = log_base_dir


        g_main_logger.info("Initializing Learner...")
        g_learner_process = Learner(learner_config, replay_buffer)
        g_main_logger.info("Learner initialized.")
        g_main_logger.info("Starting Learner process...")
        g_learner_process.start()
        time.sleep(1) # 给 Learner 一点时间启动
        if not g_learner_process.is_alive():
            g_main_logger.critical("Learner failed to start! Check its logs. Exiting.")
            cleanup_all_processes()
            return
        g_main_logger.info(f"Learner process (PID: {g_learner_process.pid}) started.")


        # 6. 准备并启动 Actors
        g_main_logger.info(f"Initializing {CONFIG['num_actors']} Actors...")
        g_actor_processes.clear() # 清空全局列表以防重用
        for i in range(CONFIG['num_actors']):

            actor_config = CONFIG['actor_config'].copy() # 获取 Actor 特定配置
            actor_config.update(CONFIG['experiment_meta']) # 添加实验元数据
            actor_config.update(CONFIG['model_agent_config']) # 添加模型和代理配置
            actor_config.update(CONFIG['ppo_config']) # 添加 PPO 配置
            actor_config.update(CONFIG['profiling_config']) # 添加 profiling 配置

            actor_config['server_hosted_benchmark_names'] = CONFIG['server_hosted_benchmark_names'] # 添加服务器托管的基准模型名称

            actor_id_val = i # 计算唯一的 Actor ID
            actor_name_key = f'Actor-{actor_id_val}' # 创建 Actor 名称
            
            actor_config['name'] = actor_name_key
            actor_config['actor_id'] = actor_id_val # 将 actor_id 传入配置，供 Actor 内部使用
            actor_config['shutdown_event'] = shutdown_event # 传递关闭事件
            actor_config['inference_server_req_queue'] = actors_to_server_req_q # 推理请求队列
            actor_config['inference_server_resp_queue'] = server_to_actors_resp_qs[actor_name_key] # 专属的响应队列
            actor_config['log_base_dir'] = log_base_dir # 日志目录
                        # 简化：只为Actor-0在主TensorBoard中记录指标，避免多进程写入冲突
            # 其他Actor只写入自己的detailed TensorBoard
            if actor_id_val == 0:  # 只有Actor-0写入主TensorBoard
                main_tensorboard_dir = os.path.join(main_log_paths['main_experiment_dir'], 'tensorboard', 'main_train')
                actor_main_tb_path = os.path.join(main_tensorboard_dir, 'Actor-0')
                actor_config['actor_main_tensorboard_path'] = actor_main_tb_path
            else:
                # 其他Actor不写入主TensorBoard
                actor_config['actor_main_tensorboard_path'] = None

            actor = Actor(actor_config, replay_buffer)
            g_actor_processes.append(actor)
        g_main_logger.info(f"{CONFIG['num_actors']} Actors initialized.")
        g_main_logger.info("Starting Actor processes...")
        for actor in g_actor_processes:
            actor.start()
        g_main_logger.info("All Actor processes started.")

        # --- 7. 等待进程结束 (主训练循环对于 train.py 来说就是等待) ---
        g_main_logger.info("Main process is now waiting for Actor processes to complete their configured episodes (or until a shutdown signal is received)...")
        
        # 记录训练开始的重要信息
        total_env_steps = CONFIG['actor_config']['num_env_steps'] 
        actors_count = CONFIG['num_actors']
        envs_per_actor = CONFIG['actor_config']['num_envs_per_actor']
        total_parallel_envs = actors_count * envs_per_actor
        
        g_main_logger.info(f"🚀 TRAINING_START | Target: {total_env_steps:,} steps | "
                          f"Parallel Envs: {total_parallel_envs} ({actors_count} actors × {envs_per_actor} envs) | "
                          f"Expected Duration: ~{total_env_steps / (total_parallel_envs * 60):.1f} hours")
        
        # 监控变量
        training_start_time = time.time()  # 记录训练开始时间
        last_status_time = time.time()
        status_interval = 60  # 每60秒输出一次状态
        
        for actor in g_actor_processes:
            while actor.is_alive(): # 只要 actor 还在运行
                if shutdown_event.is_set(): # 检查是否收到了关闭信号
                    g_main_logger.info(f"Main process detected shutdown_event, no longer actively waiting for Actor {actor.name}.")
                    break # 跳出对此 actor 的等待
                
                # 定期输出训练状态
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
                    try:
                        # 获取ReplayBuffer状态
                        buffer_size = replay_buffer.size()
                        buffer_episodes = replay_buffer.queue.qsize() if hasattr(replay_buffer, 'queue') and hasattr(replay_buffer.queue, 'qsize') else 'N/A'
                        
                        # 检查进程状态
                        learner_status = "RUNNING" if g_learner_process and g_learner_process.is_alive() else "STOPPED"
                        server_status = "RUNNING" if g_inference_server_process and g_inference_server_process.is_alive() else "STOPPED"
                        alive_actors = sum(1 for a in g_actor_processes if a.is_alive())
                        
                        elapsed_minutes = (current_time - training_start_time) / 60
                        estimated_progress = min(buffer_size / total_env_steps * 100, 100) if total_env_steps > 0 else 0
                        
                        g_main_logger.info(f"📊 TRAINING_STATUS | ReplayBuffer: {buffer_size:,} steps ({buffer_episodes} episodes) | "
                                         f"Processes: Learner={learner_status}, Server={server_status}, Actors={alive_actors}/{len(g_actor_processes)} | "
                                         f"Runtime: {elapsed_minutes:.1f}min | Progress: {estimated_progress:.1f}%")
                        
                        # 记录主要状态到TensorBoard
                        if g_main_writer:
                            try:
                                g_main_writer.add_scalar('System/ReplayBuffer_Size', buffer_size, int(elapsed_minutes * 60))
                                g_main_writer.add_scalar('System/Training_Progress_Percent', estimated_progress, int(elapsed_minutes * 60))
                                g_main_writer.add_scalar('System/Active_Actors', alive_actors, int(elapsed_minutes * 60))
                                g_main_writer.flush()
                            except Exception as e_main_tb:
                                g_main_logger.warning(f"Failed to write system metrics to main TensorBoard: {e_main_tb}")
                        
                        last_status_time = current_time
                    except Exception as e_status:
                        g_main_logger.warning(f"Error collecting training status: {e_status}")
                
                actor.join(timeout=1.0) # 带超时的 join，允许主进程周期性地检查 shutdown_event
            if not shutdown_event.is_set() and not actor.is_alive(): # 如果 actor 正常结束
                 g_main_logger.info(f"✅ ACTOR_COMPLETED | {actor.name} (PID: {actor.pid if actor.pid else 'N/A'}) has finished its episodes.")
        
        # 训练完成统计
        total_training_time = (time.time() - training_start_time) / 60
        final_buffer_size = replay_buffer.size()
        
        if not shutdown_event.is_set(): # 如果不是因为外部信号中断的 (即所有 Actors 都正常完成了)
            g_main_logger.info(f"🎯 TRAINING_COMPLETED | All Actor processes have completed their tasks. | "
                              f"Total Runtime: {total_training_time:.1f}min | "
                              f"Final Buffer Size: {final_buffer_size:,} steps")
            
            # 记录训练完成指标到TensorBoard
            if g_main_writer:
                try:
                    g_main_writer.add_scalar('Training/Total_Runtime_Minutes', total_training_time, final_buffer_size)
                    g_main_writer.add_scalar('Training/Final_Buffer_Size', final_buffer_size, final_buffer_size)
                    g_main_writer.add_scalar('Training/Completion_Status', 1.0, final_buffer_size)  # 1.0 表示正常完成
                    g_main_writer.flush()
                except Exception as e_final_tb:
                    g_main_logger.warning(f"Failed to write completion metrics to main TensorBoard: {e_final_tb}")
                    
            g_main_logger.info("Signaling Learner and InferenceServer to shut down gracefully...")
            shutdown_event.set() # 通知 Learner 优雅关闭
            if g_learner_to_server_cmd_q: # 通知 InferenceServer 优雅关闭
                try:
                    g_learner_to_server_cmd_q.put(("SHUTDOWN", None), timeout=5)
                except Exception as e_shutdown_cmd:
                     g_main_logger.error(f"Error sending SHUTDOWN command to InferenceServer: {e_shutdown_cmd}")
        
        # 最终的等待和清理由 cleanup_all_processes 函数处理

    except KeyboardInterrupt:
        g_main_logger.warning("Main process caught KeyboardInterrupt (Ctrl+C). Cleanup will be called by signal handler or finally block.")
        # 信号处理器 cleanup_all_processes 应该已经被触发了
        # 如果由于某种原因没有（例如，信号处理器设置失败），finally 块会作为后备
    except Exception as e:
        g_main_logger.critical(f"Main training orchestration loop encountered an unhandled exception: {e}", exc_info=True)
        # 在发生其他严重错误时，也尝试清理
    finally:
        g_main_logger.info("Main process entering 'finally' block for cleanup.")
        # 确保 cleanup_all_processes 至少被调用一次，以处理所有子进程和资源
        if not (hasattr(cleanup_all_processes, 'is_shutting_down') and cleanup_all_processes.is_shutting_down):
            cleanup_all_processes() # 如果信号处理器未运行或提前失败，这里会调用
        
        # 恢复原始信号处理器 (通常在程序真正退出前执行，以防影响后续可能的Python代码)
        try:
            if 'original_sigint_handler' in locals() and original_sigint_handler is not None: # 检查变量是否已定义
                 signal.signal(signal.SIGINT, original_sigint_handler)
            if 'original_sigterm_handler' in locals() and original_sigterm_handler is not None:
                 signal.signal(signal.SIGTERM, original_sigterm_handler)
        except Exception as e_restore_final:
            # 使用 log_func 以防 g_main_logger 此时也可能出问题
            log_func_final = g_main_logger.warning if g_main_logger else print
            log_func_final(f"Finally block: Could not restore original signal handlers: {e_restore_final}")

        log_func_final = g_main_logger.info if g_main_logger else print
        log_func_final("Training script main function finished.")

if __name__ == '__main__':
    try:
        # 尝试设置多进程启动方法为 'spawn'，这在某些系统（如macOS, Windows）和使用CUDA时更稳定
        # 需要在任何多进程对象创建之前调用
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method != 'spawn': # 只有当不是 'spawn' 时才尝试设置
            mp.set_start_method('spawn', force=True) 
            print(f"多进程启动方法已设置为 'spawn'。之前为: {current_start_method}")
        else:
            print(f"多进程启动方法已经是 'spawn'。")
    except RuntimeError as e_sm_runtime:
        # 例如，如果上下文已经设置，或者在某些环境中不允许更改
        print(f"信息: 多进程启动方法未能设置为 'spawn' (例如，已启动或平台不支持更改): {e_sm_runtime}")
    except Exception as e_sm_main:
        print(f"设置多进程启动方法时发生错误: {e_sm_main}")
        
    main()
