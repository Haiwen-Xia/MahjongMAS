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
from actor import Actor # Actor 类 
from learner import Learner # Learner 类 
from inference_server import InferenceServer # 导入新的 InferenceServer
from model import ResNet34AC # 导入您的模型类
from utils import setup_process_logging_and_tensorboard # 日志和 TensorBoard 设置工具

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
    # Experiment Meta
    'experiment_name': "Using_Inference_Server", # Use underscores or avoid special chars for dir names
    'log_base_dir': '/home/dataset-assist-0/data/Mahjong/RL/log', # Base directory for logs and TensorBoard
    'checkpoint_base_dir': '/home/dataset-assist-0/data/Mahjong/RL/model', # Base directory for checkpoints

    # Data & Replay Buffer
    'replay_buffer_size': 50000,
    'replay_buffer_episode_capacity': 400, # Renamed for clarity
    'min_sample_to_start_learner': 20000, # Increased wait time based on batch size? Renamed.

    # Model & Training
    'supervised_model_path': '/home/dataset-assist-0/data/Mahjong/RL/supervised_model/best_model.pkl',
    'in_channels': 187,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Distributed Setup
    # 'model_pool_size': 120,
    # 'model_pool_name': 'model-pool-v2', # Example name
    'num_actors': 8,
    'episodes_per_actor': 25000, # How many episodes each actor runs before exiting

    # Learner Hyperparameters
    'batch_size': 1024, # Increased batch size
    'epochs_per_batch': 2, # Renamed 'epochs' for clarity (PPO inner loops)

    ## Learning Rate Scheduler
    'lr_critic_head': 3e-5,
    'lr_critic_feature_extractor': 2e-5,
    'lr_actor_head_finetune': 2e-5,
    'lr_actor_feature_extractor_finetune': 2e-5,
    
    'unfreeze_actor_head_after_iters': 500,
    'unfreeze_actor_feature_extractor_after_iters': 500,

    'use_lr_scheduler': True,
    "warmup_iterations": 500,
    'total_iterations_for_lr_decay': 500000,
    'initial_lr_warmup_critic': 1e-6,
    'min_lr_critic_schedule': 1e-6, # Minimum learning rate for scheduler

    # PPO 基本设置
    'gamma': 0.98,      # Discount factor for GAE/TD Target
    'lambda': 0.97,     # Lambda for GAE
    'clip': 0.2,        # PPO clip epsilon
    'grad_clip_norm': 0.3,
    'value_coeff': 0.8, # Coefficient for value loss (common to scale down)
    'entropy_coeff': -1e-4, # Coefficient for entropy bonus


    'ckpt_save_interval_seconds': 600, # Save checkpoint every N seconds (e.g., 10 minutes)

    # 麻将 RL 的特殊设置
    'filter_single_action_steps': False, # 是否过滤掉只有单个可能 action 的时间步
    'use_normalized_reward': True,

    # 多样化 opponent
    'p_opponent_historical' : 0.4,
    "benchmark_policies": {
        "initial_il_policy": "/home/dataset-assist-0/data/Mahjong/RL/model/Separate_Feature_Extractor/model_iter_249.pt",
    },
    "initial_model_eval_path": "/home/dataset-assist-0/data/Mahjong/RL/model/Separate_Feature_Extractor/model_iter_249.pt",
    "prob_opponent_is_benchmark": 0.15,

    "server_hosted_benchmark_names": ["initial_il_policy"],

    'opponent_sampling_k' : 115, 
    'opponent_model_change_interval': 1, # 每多少个 episode 替换一次对手
    'actor_model_change_interval': 1,


    # Added for potential use in Learner/Actor logging/config
    'log_interval_learner': 100, # Log Learner stats every N iterations
    'model_push_interval': 10,   # Push model every N Learner iterations
    'enable_profiling': True,
    'enable_profiling_actor': True,

    # 关于 inference_server 的相关设置
    'use_centralize_critic': False,
    "inference_batch_size": 32,
    "inference_max_wait_ms": 10, # 单位 ms
}


def main():
    global g_learner_process, g_actor_processes, g_inference_server_process
    global g_main_logger, g_main_writer, shutdown_event, g_learner_to_server_cmd_q

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [ROOT/%(levelname)s] %(message)s', # Differentiate root logs
        handlers=[logging.StreamHandler()] # Root logs to console
    )
    
    run_name = CONFIG['experiment_name']
    log_base_dir = CONFIG['log_base_dir']

    try:
        g_main_logger, g_main_writer = setup_process_logging_and_tensorboard(
            log_base_dir, run_name, process_type='main_train', process_id=os.getpid()
        )
    except Exception as e_log_setup:
        logging.error(f"Main process logger/writer setup failed: {e_log_setup}", exc_info=True)
        g_main_logger = logging.getLogger("main_fallback_logger") # 使用一个备用logger，以防万一
        g_main_writer = None # Writer 可能无法创建

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
        g_main_logger.info(f"Starting Experiment: {run_name} (Main Process PID: {os.getpid()})")
        g_main_logger.info(f"Full Configuration:\n{json.dumps(CONFIG, indent=2, default=str, ensure_ascii=False)}") # default=str 处理无法序列化的对象
        g_main_logger.info("="*60)

        # 1. 创建 InferenceServer 通信队列
        g_learner_to_server_cmd_q = mp.Queue()   # Learner -> Server 的命令队列
        actors_to_server_req_q = mp.Queue()      # Actors -> Server 的推理请求队列
        
        # Server -> Actors 的响应队列 (每个 Actor 一个)
        server_to_actors_resp_qs = {}
        for i in range(CONFIG['num_actors']):
            actor_id_key = f'Actor-{CONFIG.get("actor_id_base", 0) + i}' # 与 Actor 配置中的 name 保持一致
            server_to_actors_resp_qs[actor_id_key] = mp.Queue()
        g_main_logger.info("Communication queues for InferenceServer created.")

        # 2. 准备 InferenceServer 配置
        benchmark_models_info_for_server = {}
        # 示例：从主配置中获取一个基准模型信息 (如果存在)
        # if CONFIG.get('supervised_model_path') and os.path.isfile(CONFIG['supervised_model_path']):
        benchmark_models_info_for_server["initial_il_policy"] = {"path": CONFIG['initial_model_eval_path']}
        # 您可以在这里添加更多基准模型到 benchmark_models_info_for_server
        
        server_config = {
            'name': 'InferenceServer-Main', # 服务器进程的名称
            'evaluator_id': f"inference_server_{os.getpid()}", # 给服务器一个唯一ID，用于日志
            'log_base_dir': CONFIG['log_base_dir'],
            'experiment_name': CONFIG['experiment_name'],
            'in_channels': CONFIG['in_channels'],
            'device': CONFIG['device'], # InferenceServer 将模型加载到哪个设备
            'initial_model_eval_path': CONFIG.get('initial_model_eval_path'), # 初始训练模型的路径
            'benchmark_models_info': benchmark_models_info_for_server, # 要托管的基准模型
            'model_definition_class': ResNet34AC # 传递实际的模型类
        }
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
        checkpoint_dir = os.path.join(CONFIG['checkpoint_base_dir'], run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        learner_config = CONFIG.copy() # 为 Learner 创建配置副本
        learner_config['name'] = 'Learner-Main' # 给 Learner 一个明确的名称
        learner_config['ckpt_save_path'] = checkpoint_dir # 传递检查点保存路径
        learner_config['shutdown_event'] = shutdown_event # 传递关闭事件
        learner_config['inference_server_cmd_queue'] = g_learner_to_server_cmd_q # 传递命令队列
        # Learner 不再需要旧的 ModelPool 配置 (如果完全依赖 InferenceServer)
        learner_config.pop('model_pool_name', None) 
        learner_config.pop('model_pool_size', None)
        # Learner 需要知道模型同步间隔
        learner_config['model_sync_interval'] = CONFIG.get('model_sync_interval_learner', 10)


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
            actor_config = CONFIG.copy() # 为每个 Actor 创建配置副本
            actor_id_val = CONFIG.get('actor_id_base', 0) + i # 计算唯一的 Actor ID
            actor_name_key = f'Actor-{actor_id_val}' # 创建 Actor 名称
            
            actor_config['name'] = actor_name_key
            actor_config['actor_id'] = actor_id_val # 将 actor_id 传入配置，供 Actor 内部使用
            actor_config['shutdown_event'] = shutdown_event # 传递关闭事件
            actor_config['inference_server_req_queue'] = actors_to_server_req_q # 推理请求队列
            actor_config['inference_server_resp_queue'] = server_to_actors_resp_qs[actor_name_key] # 专属的响应队列
            # Actors 也不再需要旧的 ModelPool 配置 (如果完全依赖 InferenceServer)
            actor_config.pop('model_pool_name', None)

            actor = Actor(actor_config, replay_buffer)
            g_actor_processes.append(actor)
        g_main_logger.info(f"{CONFIG['num_actors']} Actors initialized.")
        g_main_logger.info("Starting Actor processes...")
        for actor in g_actor_processes:
            actor.start()
        g_main_logger.info("All Actor processes started.")

        # --- 7. 等待进程结束 (主训练循环对于 train.py 来说就是等待) ---
        g_main_logger.info("Main process is now waiting for Actor processes to complete their configured episodes (or until a shutdown signal is received)...")
        for actor in g_actor_processes:
            while actor.is_alive(): # 只要 actor 还在运行
                if shutdown_event.is_set(): # 检查是否收到了关闭信号
                    g_main_logger.info(f"Main process detected shutdown_event, no longer actively waiting for Actor {actor.name}.")
                    break # 跳出对此 actor 的等待
                actor.join(timeout=1.0) # 带超时的 join，允许主进程周期性地检查 shutdown_event
            if not shutdown_event.is_set() and not actor.is_alive(): # 如果 actor 正常结束
                 g_main_logger.info(f"Actor {actor.name} (PID: {actor.pid if actor.pid else 'N/A'}) has finished its episodes.")
        
        if not shutdown_event.is_set(): # 如果不是因为外部信号中断的 (即所有 Actors 都正常完成了)
            g_main_logger.info("All Actor processes have completed their tasks.")
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
