
import torch
import torch.multiprocessing as mp
import time
import os
import logging
import numpy as np
import sys
from queue import Empty as QueueEmpty

# 假设这些模块可被此文件访问
try:
    from learner.learner import Learner
    from models.actor import ResNet34Actor
    from models.critic import ResNet34CentralizedCritic
except ImportError as e:
    print(f"导入模块失败: {e}。为演示将使用占位符。")
    class Process: 
        def __init__(*args, **kwargs): pass

        def start(self): self.run()

        def join(self, *args, **kwargs): pass
    class Learner(Process): 
        def run(self): print("Running Mock Learner")

# --- 模拟和伪造的组件 ---

class MockPPOAlgorithm:
    """一个伪造的 PPO 算法类，用于测试 Learner 的调用逻辑。"""
    def __init__(self, config, actor_model, critic_model):
        print("[MockPPOAlgorithm] Initialized.")
        self.config = config
        self.actor_model = actor_model
        self.critic_model = critic_model
        # 伪造优化器以便 Learner 可以访问 (例如保存检查点)
        self.optimizer_actor = torch.optim.Adam(self.actor_model.parameters(), lr=1e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic_model.parameters(), lr=1e-4)

    def update_freezing_status(self, iterations):
        print(f"[MockPPOAlgorithm] update_freezing_status called at iter {iterations}.")

    def schedule_learning_rate(self, iterations):
        print(f"[MockPPOAlgorithm] schedule_learning_rate called at iter {iterations}.")
        
    def train_on_batch(self, batch):
        print(f"[MockPPOAlgorithm] train_on_batch called with a batch of size {len(batch['action'])}.")
        # 返回伪造的训练指标
        return {'policy_loss': 0.1, 'critic_loss': 0.2, 'entropy_loss': 0.01}

class MockReplayBuffer:
    """一个伪造的回放缓冲区，用于给 Learner 提供数据。"""
    def __init__(self, config):
        self.config = config
        self.batch_size = config.get('batch_size', 128)
        self.in_channels = config.get('in_channels', 187)
        self.out_channels = config.get('out_channels', 235)
        self.critic_extra_channels = config.get('critic_extra_in_channels', 16)
        print("[MockReplayBuffer] Initialized.")

    def size(self):
        # 总是返回一个足够大的值，让 Learner 开始训练
        return self.config.get('min_sample_to_start_learner', 10000) + 1

    def sample(self, batch_size):
        print(f"[MockReplayBuffer] sample() called for batch_size {batch_size}.")
        # 生成符合结构的、随机的 NumPy 数据
        return {
            'state': {
                'observation': np.random.rand(batch_size, self.in_channels, 4, 9).astype(np.float32),
                'action_mask': np.ones((batch_size, self.out_channels), dtype=np.int8)
            },
            'centralized_extra_info': np.random.rand(batch_size, self.critic_extra_channels, 4, 9).astype(np.float32),
            'action': np.random.randint(0, self.out_channels, size=(batch_size,)),
            'adv': np.random.randn(batch_size),
            'target': np.random.randn(batch_size),
            'log_prob': -np.random.rand(batch_size)
        }

def mock_inference_server_listener(cmd_q, shutdown_event):
    """一个简单的进程，用于监听 Learner 发送的命令并打印。"""
    print(f"[MockInferenceServer] Listener started (PID: {os.getpid()}).")
    while not shutdown_event.is_set():
        try:
            cmd_type, cmd_data = cmd_q.get(timeout=0.1)
            print(f"[MockInferenceServer] SUCCESS: Received command '{cmd_type}'.")
            if cmd_type == "UPDATE_ACTOR_MODEL":
                print(f"  - Received actor state_dict with {len(cmd_data)} tensors.")
            elif cmd_type == "UPDATE_CRITIC_MODEL":
                print(f"  - Received critic state_dict with {len(cmd_data)} tensors.")
        except QueueEmpty:
            continue
    print(f"[MockInferenceServer] Shutdown signal received. Exiting.")


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    # --- 1. 定义测试配置 ---
    test_config = {
        'experiment_name': "learner_refactor_test",
        'log_base_dir': './test_logs',
        'device': 'cpu',
        'in_channels': 187,
        'out_channels': 235,
        'critic_extra_in_channels': 16,
        'actor_class': ResNet34Actor,
        'critic_class': ResNet34CentralizedCritic,
        'supervised_model_path': None, # 测试时不从SL加载
        'batch_size': 128,
        'epochs_per_batch': 1, # 测试时只跑一个epoch
        'model_push_interval': 5, # 每5次迭代发送一次更新
        'min_sample_to_start_learner': 1000, # 快速开始
        'total_actor_steps': 100 # 让 Learner 运行大约 100/128 * 5 = 4次更新
    }
    
    # 将 PPOAlgorithm 替换为我们的 Mock 版本，以便测试
    # 在真实的 train.py 中，这里会是 PPOAlgorithm
    # 这里我们注入到 config 中，让 Learner 使用 Mock
    # 或者，更简单的方法是在 Learner 内部的 PPOAlgorithm = ... 处修改
    # 假设 Learner 内部会 from algos.ppo import PPOAlgorithm
    # 我们需要确保测试时能替换它，或者让 Learner 接受一个 algo_class 参数

    # --- 2. 创建通信和同步原语 ---
    shutdown_event = mp.Event()
    inference_server_cmd_q = mp.Queue()
    
    # --- 3. 准备依赖项 ---
    mock_buffer = MockReplayBuffer(test_config)
    
    # Learner 的配置
    learner_config = test_config.copy()
    learner_config['name'] = 'TestLearner-0'
    learner_config['shutdown_event'] = shutdown_event
    learner_config['inference_server_cmd_queue'] = inference_server_cmd_q
    
    # --- 4. 启动所有进程 ---
    processes = []
    try:
        mock_server = mp.Process(target=mock_inference_server_listener, args=(inference_server_cmd_q, shutdown_event))
        mock_server.start()
        processes.append(mock_server)
        print(f"[Main] Mock Inference Server Listener started (PID: {mock_server.pid}).")
        
        # 将 PPOAlgorithm 替换为 Mock
        # 这是一种 Monkey Patching，仅用于测试
        import learner as learner_module
        learner_module.PPOAlgorithm = MockPPOAlgorithm 

        learner_process = Learner(learner_config, mock_buffer)
        processes.append(learner_process)
        print(f"[Main] Starting Learner process...")
        learner_process.start()
        
        # --- 5. 运行测试 ---
        print("\n" + "="*50)
        print("[Main] Test running... Learner should start training and send updates.")
        print("Test will run for a few seconds...")
        print("="*50 + "\n")
        
        # 让 Learner 运行一段时间，例如15秒
        learner_process.join(timeout=15)
        if learner_process.is_alive():
            print("[Main] Learner is still running after 15s, which is expected. Test seems OK.")
        else:
            print(f"[Main] WARNING: Learner process terminated unexpectedly with exit code {learner_process.exitcode}.")

    finally:
        # --- 6. 清理 ---
        print("\n[Main] Test duration finished. Initiating shutdown...")
        shutdown_event.set()
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        print("[Main] Test complete.")

if __name__ == '__main__':
    # 为了让这个测试脚本能找到其他模块，将项目根目录添加到路径中
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    main()
