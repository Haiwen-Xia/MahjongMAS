import torch
import torch.multiprocessing as mp
import time
import os
import logging
import json
import numpy as np
import random
import sys
from queue import Empty as QueueEmpty

# 假设这些模块与测试脚本在同一路径或已在 PYTHONPATH 中
# try:
from actor.actor import Actor # 导入我们想要测试的 Actor 类
from replay_buffer import ReplayBuffer # 我们将创建一个 MockReplayBuffer
from env.env_wrapper import SubprocVecEnv, CloudpickleWrapper, worker
from env.env import MahjongGBEnv # Actor 内部需要
from agent.feature import FeatureAgent # Actor 内部需要

def mock_inference_server_process(
    config: dict,
    request_queue: mp.Queue,
    response_queues: dict,
    shutdown_event: mp.Event
):
    """
    一个运行在独立进程中的“伪”推理服务器。
    它模拟 InferenceServer 的行为，接收请求并返回随机生成的有效动作。
    """
    server_name = config.get('name', 'MockInferenceServer')
    print(f"[{server_name}] Process started (PID: {os.getpid()}).")
    
    while not shutdown_event.is_set():
        try:
            # 非阻塞地从请求队列获取一个请求
            # 请求格式: (actor_id, request_id, model_key, observation_data_cpu)
            request_payload = request_queue.get(timeout=0.01) # 短暂超时以响应关闭事件
            
            actor_id, request_id, model_key, obs_data = request_payload
            print(f"[{server_name}] Received request ID {request_id} from {actor_id} for model '{model_key}'.")

            # --- 模拟推理 ---
            action_mask = obs_data.get('obs', {}).get('action_mask', [])
            valid_action_indices = np.where(action_mask == 1)[0]
            
            if len(valid_action_indices) > 0:
                action = np.random.choice(valid_action_indices) # 从有效动作中随机选择一个
            else:
                action = 0 # 如果没有有效动作，则默认为0 (Pass)
            
            value = random.uniform(-1, 1) # 生成一个伪造的价值
            log_prob = -np.log(len(valid_action_indices)) if len(valid_action_indices) > 0 else -10.0 # 生成一个伪造的log_prob

            # --- 准备并发送响应 ---
            # 响应格式: (request_id, action, value, log_prob)
            response_payload = (request_id, action, value, log_prob)
            
            if actor_id in response_queues:
                response_queues[actor_id].put(response_payload)
            else:
                print(f"[{server_name}] Error: No response queue found for actor_id '{actor_id}'.")

        except QueueEmpty:
            # 队列为空是正常现象，继续循环
            continue
        except Exception as e:
            print(f"[{server_name}] An error occurred: {e}")
            time.sleep(0.1)

    print(f"[{server_name}] Shutdown signal received. Exiting.")


class MockReplayBuffer:
    """
    一个“伪”回放缓冲区，用于测试 Actor。
    它的 push 方法会将接收到的数据放入一个输出队列，供主测试进程检查。
    """
    def __init__(self, output_queue: mp.Queue):
        self.output_queue = output_queue
        print("[MockReplayBuffer] Initialized.")

    def push(self, data: dict):
        """将 Actor 处理好的数据放入输出队列。"""
        print(f"[MockReplayBuffer] Received data to push. Num steps: {len(data.get('action', []))}. Pushing to output queue...")
        try:
            # 只放入关键数据以供检查
            simplified_data = {
                'action_shape': data['action'].shape,
                'adv_mean': np.mean(data['adv']),
                'target_mean': np.mean(data['target'])
            }
            self.output_queue.put(simplified_data)
        except Exception as e:
            print(f"[MockReplayBuffer] Error pushing data: {e}")

def main():
    """主测试函数"""
    # 使用 'spawn' 启动方法，这对于 PyTorch 和 CUDA 更安全
    try:
        mp.set_start_method('spawn', force=True)
        print("[Main] Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("[Main] Info: Multiprocessing start method was already set.")

    # --- 1. 定义测试配置 ---
    # 这个配置会传递给 Actor
    test_config = {
        'experiment_name': "actor_test_run",
        'log_base_dir': './test_logs',
        'num_envs_per_actor': 2, # 使用2个并行环境进行测试
        'episodes_per_actor': 3, # 让 Actor 运行3个完整的 episodes
        'in_channels': 187,      # 确保与您的模型定义一致
        'device': 'cpu',         # 测试时使用 CPU 更方便
        'inference_timeout_seconds': 10, # 增加超时以防调试时阻塞
        # 假设 Actor 不需要以下配置，因为它们由 InferenceServer 或 Learner 使用
        # 但为了让 Actor 的 .get() 不出错，提供一些默认值
        'prob_opponent_is_benchmark_server': 0.5,
        'opponent_model_change_interval': 1,
        'filter_single_action_steps': False, # 测试时不进行过滤，以便观察所有步骤
        'use_normalized_reward': True, # 测试奖励变换逻辑
        'gamma': 0.99,
        'lambda_gae': 0.95,
        'server_hosted_benchmark_names': ["benchmark_A", "benchmark_B"], # 告知 Actor 有哪些基准模型可用
        'env_config': {'agent_clz': FeatureAgent} # 指定环境内部使用的 Agent
    }

    # --- 2. 创建通信和同步原语 ---
    shutdown_event = mp.Event() # 用于通知所有进程关闭
    
    # 用于 Actor 和 Mock InferenceServer 之间的通信
    inference_req_q = mp.Queue()
    actor_resp_q_dict = {
        'TestActor-0': mp.Queue() # 为我们的测试 Actor 创建一个响应队列
    }
    
    # 用于 Mock ReplayBuffer 将结果传回主进程
    replay_buffer_output_q = mp.Queue()

    # --- 3. 准备并启动所有进程 ---
    processes = []
    
    try:
        # a. 启动伪 InferenceServer
        mock_server_config = {'name': 'MockInferenceServer'}
        mock_server = mp.Process(
            target=mock_inference_server_process,
            args=(mock_server_config, inference_req_q, actor_resp_q_dict, shutdown_event)
        )
        mock_server.start()
        processes.append(mock_server)
        print(f"[Main] Mock Inference Server started (PID: {mock_server.pid}).")

        # b. 创建伪 ReplayBuffer
        mock_buffer = MockReplayBuffer(replay_buffer_output_q)

        # c. 准备 Actor 配置
        actor_config = test_config.copy()
        actor_config['name'] = 'TestActor-0'
        actor_config['actor_id'] = 'TestActor-0'
        actor_config['shutdown_event'] = shutdown_event
        actor_config['inference_server_req_queue'] = inference_req_q
        actor_config['inference_server_resp_queue'] = actor_resp_q_dict['TestActor-0']
        
        # d. 启动被测试的 Actor
        actor_process = Actor(actor_config, mock_buffer)
        actor_process.start()
        processes.append(actor_process)
        print(f"[Main] Actor process '{actor_process.name}' started (PID: {actor_process.pid}).")

        # --- 4. 监控和测试 ---
        print("\n" + "="*50)
        print("[Main] Now monitoring... Waiting for Actor to push data to ReplayBuffer.")
        print("The test will run for a maximum of 60 seconds or until 6 episodes are pushed.")
        print("="*50 + "\n")
        
        start_time = time.time()
        pushed_data_count = 0
        # Actor 运行3个 episodes, 每个 episode 有4个 agent, 但我们只收集 main agent 的数据
        # (根据您之前的 Actor GAE 逻辑)。
        # 如果 Actor 的 `main_agent_seat_idx` 是随机的，平均会有 3/4 的数据被推送。
        # 我们期望至少收到几次推送。
        expected_pushes = 3 

        while time.time() - start_time < 60 and pushed_data_count < expected_pushes:
            try:
                # 从伪 ReplayBuffer 的输出队列中获取数据
                pushed_data = replay_buffer_output_q.get(timeout=1.0)
                pushed_data_count += 1
                print("\n" + "✓" * 20)
                print(f"[Main] SUCCESS: Received data chunk #{pushed_data_count} from Actor via ReplayBuffer!")
                print(f"  Data content (summary): {pushed_data}")
                print("✓" * 20 + "\n")
            except QueueEmpty:
                print(f"[Main] Waiting for data... (Actor alive: {actor_process.is_alive()}, Server alive: {mock_server.is_alive()})")
                if not actor_process.is_alive():
                    print("[Main] ERROR: Actor process terminated unexpectedly.")
                    break
        
        if pushed_data_count > 0:
            print(f"\n[Main] TEST PASSED: Successfully received {pushed_data_count} data chunks from the Actor.")
        else:
            print(f"\n[Main] TEST FAILED: Did not receive any data from the Actor within the time limit.")

    except Exception as e:
        print(f"[Main] An unexpected error occurred in the main test script: {e}")
    finally:
        # --- 5. 清理和关闭 ---
        print("\n" + "="*50)
        print("[Main] Test finished or interrupted. Initiating shutdown...")
        shutdown_event.set() # 通知所有子进程关闭
        
        for p in processes:
            if p.is_alive():
                print(f"  Waiting for process {p.name} (PID: {p.pid}) to join...")
                p.join(timeout=5)
                if p.is_alive():
                    print(f"  Process {p.name} did not join gracefully. Terminating...")
                    p.terminate()
        
        print("[Main] All processes shut down. Test complete.")


if __name__ == '__main__':
    main()