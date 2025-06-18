import torch
import torch.multiprocessing as mp
import time
import os
import logging
import numpy as np
import sys
from queue import Empty as QueueEmpty
import random

# --- 模块导入 ---
# 确保这些模块的路径在 PYTHONPATH 中，或者此脚本与它们在同一目录下
try:
    from inference_server.inference_server import InferenceServer
    from models.actor import ResNet34Actor
    # Critic 现在是中心化的，它需要不同的输入，但为了测试，我们可能需要一个简化版本
    # 或者确保测试时 Actor 发送了 critic 需要的额外信息
    from models.critic import ResNet34CentralizedCritic 
    from utils import setup_process_logging_and_tensorboard
except ImportError as e:
    print(f"导入模块失败: {e}。请确保所有相关文件（inference_server.py, models/, utils.py）都在可访问的路径中。")
    # 为了让脚本能运行，这里使用占位符类
    class ResNet34Actor(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
        def forward(self, x): return self.fc(torch.zeros(x['obs']['observation'].shape[0], 1)), torch.zeros(x['obs']['observation'].shape[0], 1)
    class ResNet34CentralizedCritic(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
        def forward(self, x, y): return self.fc(torch.zeros(x['obs']['observation'].shape[0], 1))
    def setup_process_logging_and_tensorboard(*args, **kwargs): return logging.getLogger("test"), None
    print("警告：使用占位符模型/工具类进行测试。")


# --- 辅助函数 ---

def create_dummy_model_file(path: str, model_class, config: dict):
    """创建一个虚拟的模型权重文件用于测试。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 根据模型类需要不同的初始化参数
        if model_class == ResNet34Actor:
            model = model_class(config['in_channels'], config['out_channels'])
        elif model_class == ResNet34CentralizedCritic:
            model = model_class(config['in_channels'], config.get('critic_extra_in_channels', 16))
        else:
            model = model_class(config['in_channels']) # 通用情况

        torch.save(model.state_dict(), path)
        print(f"[Helper] Dummy model file created at: {path}")
        return True
    except Exception as e:
        print(f"[Helper] Error creating dummy model file at {path}: {e}")
        return False

# --- 模拟进程 ---

def mock_actor_process(
    actor_id: str,
    config: dict,
    req_q: mp.Queue,
    resp_q: mp.Queue,
    shutdown_event: mp.Event
):
    """
    模拟单个 Actor 的行为。
    它会发送推理请求并等待响应。
    """
    actor_name = f"MockActor-{actor_id}"
    print(f"[{actor_name}] Process started (PID: {os.getpid()}).")
    
    # 从配置中获取可用基准模型的名称
    benchmark_keys = list(config.get('benchmark_models_info', {}).keys())
    
    request_id_counter = 0
    num_requests_to_send = 10 # 每个 Actor 发送10个请求

    for i in range(num_requests_to_send):
        if shutdown_event.is_set():
            print(f"[{actor_name}] Shutdown signal received. Exiting.")
            break
        
        request_id_counter += 1
        
        # 决定请求哪个模型
        if i % 3 == 0 and benchmark_keys: # 每3次请求一次基准模型
            model_key = random.choice(benchmark_keys)
        else: # 其他时候请求最新模型
            model_key = "latest_eval"
            
        # 创建一个虚拟的观测数据
        obs_shape = (config['in_channels'], 4, 9) # 假设形状
        mask_shape = (config['out_channels'],)
        observation_data = {
            'obs': {
                'observation': np.random.rand(*obs_shape).astype(np.float32),
                'action_mask': np.random.randint(0, 2, size=mask_shape).astype(np.int8)
            },
            # 对于中心化 Critic，Actor 也需要提供全局状态
            'centralized_extra_info': np.random.rand(config.get('critic_extra_in_channels', 16), 4, 9).astype(np.float32)
        }
        
        # 发送请求
        payload = (actor_id, request_id_counter, model_key, observation_data)
        print(f"[{actor_name}] Sending request #{request_id_counter} for model '{model_key}'...")
        req_q.put(payload)
        
        # 等待响应
        try:
            resp_id, action, value, log_prob = resp_q.get(timeout=5)
            if resp_id == request_id_counter:
                print(f"[{actor_name}] SUCCESS: Received response for request #{resp_id}. Action: {action}, Value: {value:.3f}")
            else:
                print(f"[{actor_name}] ERROR: Mismatched request ID! Expected {request_id_counter}, got {resp_id}.")
        except QueueEmpty:
            print(f"[{actor_name}] ERROR: Timeout waiting for response for request #{request_id_counter}.")
        except Exception as e:
            print(f"[{actor_name}] ERROR: Exception while receiving response: {e}")
            
        time.sleep(random.uniform(0.1, 0.5)) # 模拟与环境交互的时间

    print(f"[{actor_name}] Finished sending all requests. Process exiting.")

def mock_learner_logic(config, cmd_q):
    """
    模拟 Learner 的行为，在主进程中运行。
    它会定期向 InferenceServer 发送模型更新命令。
    """
    print("[MockLearner] Logic started in main process.")
    
    # 模拟两次模型更新
    for update_step in range(2):
        print(f"[MockLearner] Simulating training step {update_step + 1}...")
        time.sleep(5) # 模拟训练耗时

        # 创建新的虚拟权重
        print(f"[MockLearner] Creating new dummy state_dicts for update...")
        actor_model = config['actor_class'](config['in_channels'], config['out_channels'])
        critic_model = config['critic_class'](config['in_channels'], config['critic_extra_in_channels'])
        
        new_actor_state_dict = actor_model.state_dict()
        new_critic_state_dict = critic_model.state_dict()
        
        # 发送更新命令
        print("[MockLearner] Sending 'UPDATE_ACTOR_MODEL' and 'UPDATE_CRITIC_MODEL' commands...")
        cmd_q.put(("UPDATE_ACTOR_MODEL", new_actor_state_dict))
        cmd_q.put(("UPDATE_CRITIC_MODEL", new_critic_state_dict))
    
    # 发送关闭命令
    time.sleep(5)
    print("[MockLearner] Sending 'SHUTDOWN' command...")
    cmd_q.put(("SHUTDOWN", None))
    print("[MockLearner] Logic finished.")


def main():
    """主测试函数"""
    global g_learner_to_server_cmd_q, g_inference_server_process, g_actor_processes, g_main_logger, g_main_writer
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # 可能已设置

    # --- 1. 定义测试配置 ---
    # 创建一个临时目录来存放虚拟模型文件
    dummy_model_dir = './tmp_test_models'
    os.makedirs(dummy_model_dir, exist_ok=True)
    initial_model_path = os.path.join(dummy_model_dir, 'initial_model.pth')
    benchmark_model_path = os.path.join(dummy_model_dir, 'benchmark_model.pth')

    test_config = {
        'experiment_name': "inference_server_test",
        'log_base_dir': './test_logs',
        'device': 'cpu', # 测试时使用 CPU 更方便，避免占用 GPU
        'in_channels': 187,
        'out_channels': 235,
        'critic_extra_in_channels': 16, # 示例值
        'actor_class': ResNet34Actor,
        'critic_class': ResNet34CentralizedCritic,
        'initial_model_path': initial_model_path, # Server将用这个初始化 actor 和 critic
        'benchmark_models_info': {
            "il_benchmark": {"path": benchmark_model_path}
        },
        'num_actors': 2 # 用2个模拟 Actor 进行测试
    }
    
    # 创建虚拟模型文件
    create_dummy_model_file(initial_model_path, ResNet34Actor, test_config)
    create_dummy_model_file(benchmark_model_path, ResNet34Actor, test_config)


    # --- 2. 创建通信和同步原语 ---
    shutdown_event = mp.Event()
    g_learner_to_server_cmd_q = mp.Queue()
    actors_to_server_req_q = mp.Queue()
    actor_resp_qs = {f'Actor-{i}': mp.Queue() for i in range(test_config['num_actors'])}


    # --- 3. 准备并启动所有进程 ---
    processes = []
    try:
        # a. 启动 InferenceServer
        server_config = {
            **test_config, # 继承主配置
            'name': 'TestInferenceServer',
            'server_id': 'test_server_01',
        }
        g_inference_server_process = InferenceServer(
            server_config,
            g_learner_to_server_cmd_q,
            actors_to_server_req_q,
            actor_resp_qs
        )
        g_inference_server_process.start()
        processes.append(g_inference_server_process)
        print(f"[Main] Test Inference Server started (PID: {g_inference_server_process.pid}).")
        time.sleep(3) # 等待服务器完全初始化

        if not g_inference_server_process.is_alive():
             raise RuntimeError("InferenceServer failed to start.")

        # b. 启动模拟 Actors
        for i in range(test_config['num_actors']):
            actor_id = f'Actor-{i}'
            actor_proc = mp.Process(
                target=mock_actor_process,
                args=(actor_id, test_config, actors_to_server_req_q, actor_resp_qs[actor_id], shutdown_event)
            )
            actor_proc.start()
            processes.append(actor_proc)
            print(f"[Main] Mock Actor process '{actor_id}' started (PID: {actor_proc.pid}).")
        
        # --- 4. 运行模拟 Learner 逻辑 ---
        mock_learner_logic(test_config, g_learner_to_server_cmd_q)

        # --- 5. 等待所有进程结束 ---
        print("[Main] Waiting for all processes to finish...")
        for p in processes:
            p.join(timeout=30) # 等待30秒

    except Exception as e:
        print(f"[Main] An unexpected error occurred in the main test script: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 6. 清理和关闭 ---
        print("\n" + "="*50)
        print("[Main] Test finished or interrupted. Initiating final cleanup...")
        shutdown_event.set()
        
        for p in reversed(processes): # 先关闭客户端，再关闭服务器
            if p.is_alive():
                print(f"  Waiting for process {p.name} (PID: {p.pid}) to join...")
                p.join(timeout=5)
                if p.is_alive():
                    print(f"  Process {p.name} did not join gracefully. Terminating...")
                    p.terminate()
        
        # 清理虚拟文件
        if os.path.exists(dummy_model_dir):
            import shutil
            shutil.rmtree(dummy_model_dir)
            print(f"[Main] Cleaned up dummy model directory: {dummy_model_dir}")

        print("[Main] All processes shut down. Test complete.")


if __name__ == '__main__':
    main()

