import torch
# 导入多进程模块，Actor 将作为一个独立的进程运行
from multiprocessing import Process
import numpy as np
import time # 引入 time 模块用于可能的延迟或计时
# import logging # 日志由 setup_process_logging_and_tensorboard 处理
import os # 引入 os 模块用于路径操作
from queue import Empty as QueueEmpty, Full as QueueFull # 用于队列操作

# 导入自定义模块
from replay_buffer import ReplayBuffer      # 用于存储和提供训练数据的经验回放缓冲区
# from model_pool_extended import ModelPoolClient # 不再直接使用 ModelPoolClient 获取最新模型
from models.actor import ResNet34Actor

# from torch.utils.tensorboard import SummaryWriter # 由 setup_process_logging_and_tensorboard 处理
from utils import setup_process_logging_and_tensorboard # 用于设置日志和 TensorBoard 的工具函数
import random


from env.env_wrapper import SubprocVecEnv # 导入您新创建的类
from env.env import MahjongGBEnv
from agent.feature_timeseries import FeatureAgentTimeSeries
from agent.feature import FeatureAgent # 导入 FeatureAgent 以便在 env_config 中使用

import cProfile, pstats # 用于记录效率信息 (如果启用)

# 多线程 Actor 相关
import threading
from queue import Queue as ThreadQueue # Python标准库的线程安全队列，用于Actor内部线程间通信
from collections import defaultdict # 用于方便地管理待处理的动作

DEFAULT_AGENT_NAMES_LIST = ['player_1', 'player_2', 'player_3', 'player_4']

# Actor 类，继承自 Process，每个 Actor 负责在一个或多个环境中与自身或其他策略进行交互，收集经验数据
class Actor(Process):

    # 初始化函数
    def __init__(self, config, replay_buffer: ReplayBuffer):
        """
        初始化 Actor 进程。

        Args:
            config (dict): 包含 Actor 配置参数的字典 (例如 model_pool 名称, gamma, lambda, episodes_per_actor 等)。
            replay_buffer (ReplayBuffer): Actor 将收集到的数据推送到的共享经验回放缓冲区。
        """
        super(Actor, self).__init__() # 调用父类 Process 的初始化方法
        self.replay_buffer = replay_buffer # 存储传入的 replay_buffer 实例
        self.config = config               # 存储配置字典
        self.name = config.get('name', 'Actor-?') # Fallback name if not provided

        # 在 __init__ 中配置 TensorBoard writer 可能不适合多进程
        # 最好在 run() 方法中初始化
        self.logger = None
        self.writer = None
        self.detailed_writer = None  # 新增：详细日志的writer

        self.inference_req_queue = config.get('inference_server_req_queue')
        self.inference_resp_queue = config.get('inference_server_resp_queue') 
        # self.shutdown_event = config.get('shutdown_event')
        self.request_counter = 0 

        # 这个队列用于从响应处理线程向主逻辑线程传递已获取并匹配的推理结果
        # maxsize=1 意味着如果主线程还未处理上一个结果，响应线程在put时会阻塞，
        # 这对于单环境顺序请求-响应模式是合适的，防止结果堆积。
        self.pending_requests = {} # 这个字典本身没问题，但访问它的锁需要在run()中创建
        self.pending_requests_lock = None # 将在 run() 中创建 threading.Lock
        self.internal_result_queue = None # 将在 run() 中创建 ThreadQueue
        self.response_worker_thread = None # 将在 run() 中创建 threading.Thread

        if self.inference_req_queue is None or self.inference_resp_queue is None:
            print(f"严重错误 for {self.name}: Inference server 队列未在配置中提供。")
        if self.inference_req_queue is None or self.inference_resp_queue is None:
            # 致命错误，如果队列未正确传递
            print(f"CRITICAL ERROR for {self.name}: Inference server queues not provided in config.")
            # 在实际应用中，这里应该导致进程无法启动或抛出异常

        
        # Actor 需要知道可以在服务器上请求哪些基准模型的键名
        self.server_hosted_benchmark_names = self.config.get('server_hosted_benchmark_names', [])
        if not self.server_hosted_benchmark_names:
            # logger 可能还未初始化
            print(f"[{self.name}] Info: No server-hosted benchmark model names provided in config.")
        else:
            print(f"[{self.name}] Will be able to request server-hosted benchmarks: {self.server_hosted_benchmark_names}")

        # 修改统计变量：删除win_count，保留draw_count用于计算draw_rate
        self.draw_count = 0  # 平局次数
        self.total_episodes = 0  # 总episode数
        self.win_rate_window_size = config.get('win_rate_window_size', 100)  # 运行平均窗口大小
        self.recent_results = []  # 最近的胜负结果列表，用于计算running mean
        self.current_win_rate = 0.0  # 当前胜率
        self.current_draw_rate = 0.0  # 当前平局率

    def _prepare_data_for_buffer(self, agent_name: str, agent_data: dict):
        """
        为单个 agent 的一局完整数据计算 GAE 和 TD-Target，并打包成适合 push 到 Replay Buffer 的格式。

        Args:
            agent_name (str): 智能体的名称。
            agent_data (dict): 包含该智能体在一局游戏中收集到的所有数据的字典 
                            (例如 'state', 'action', 'reward', 'value', 'log_prob')。

        Returns:
            dict or None: 如果数据有效，返回一个包含 'state', 'action', 'adv', 'target', 'log_prob' 的字典。
                        如果数据无效或处理失败，返回 None。
        """
        T = len(agent_data.get('action', []))
        if T == 0:
            if self.logger:
                self.logger.debug(f"No actions recorded for Env {agent_name}. Skipping GAE/push.")
            return None

        # 1. 验证数据一致性
        # 确保所有用于计算的列表都具有相同的长度 T
        required_keys = ['observation', 'action_mask', 'global_obs'] # state 的子键
        for key in required_keys:
            if len(agent_data.get('state', {}).get(key, [])) != T:
                if self.logger:
                    self.logger.error(f"Data Mismatch for {agent_name}: actions len({T}) != state.{key} len({len(agent_data.get('state', {}).get(key, []))}).")
                return None
        
        required_keys = ['reward', 'value', 'log_prob'] # 顶层键
        for key in required_keys:
            if len(agent_data.get(key, [])) != T:
                if self.logger:
                    self.logger.error(f"Data Mismatch for {agent_name}: actions len({T}) != {key} len({len(agent_data.get(key, []))}).")
                return None        # 2. 将数据转换为 NumPy 数组
        try:
            obs_np = np.stack(agent_data['state']['observation'])
            mask_np = np.stack(agent_data['state']['action_mask'])
            global_obs_np = np.stack(agent_data['state']['global_obs']) 
            actions_np = np.array(agent_data['action'], dtype=np.int64)
            rewards_np = np.array(agent_data['reward'], dtype=np.float32)
            values_np = np.array(agent_data['value'], dtype=np.float32)
            log_probs_np = np.array(agent_data['log_prob'], dtype=np.float32)
            
        except ValueError as e:
            if self.logger:
                self.logger.error(f"Error stacking data for {agent_name}: {e}. Shapes might be inconsistent.", exc_info=True)
            return None
        
        # 3. 计算 GAE 和 TD-Target
        gamma = self.config.get('gamma', 0.99)
        lambda_gae = self.config.get('lambda', 0.95) # 确保配置中使用 lambda
        
        advantages = np.zeros_like(rewards_np)
        last_gae_lam = 0.0
        
        # 注意：这里的 values_np 应该是 V(s_0), V(s_1), ..., V(s_{T-1})
        # 我们需要 V(s_T) 来计算最后一个 delta。如果 s_T 是终止状态，V(s_T) = 0。
        # 我们通过 next_non_terminal 标志来处理
        for t in reversed(range(T)):
            # 如果是最后一个时间步 (t = T-1)，那么下一个状态是终止状态
            next_non_terminal = 0.0 if t == T - 1 else 1.0
            # 下一个状态的价值。如果是终止状态，则为0。否则，是 t+1 步的价值。
            next_value = 0.0 if t == T - 1 else values_np[t+1] # 注意：这要求 values_np 至少有 T 个元素，且 V(s_t) 和 a_t 对应
                                                            # 但 GAE 需要 V(s_{t+1})。如果 value 列表是在执行动作 a_t 后得到的 V(s_t)，那么这里是正确的。
                                                            # 如果您的 value 列表长度为 T+1 (包含了V(s_T))，这里的逻辑需要调整。
                                                            # 鉴于您之前的代码，我们假设 len(value) == T，且 values_np[t+1] 是有效的（除了 t=T-1）
            
            delta = rewards_np[t] + gamma * next_value * next_non_terminal - values_np[t]
            advantages[t] = last_gae_lam = delta + gamma * lambda_gae * next_non_terminal * last_gae_lam
        
        td_target = advantages + values_np        # 4. 打包最终数据
        data_to_push = {
            'state': {'observation': obs_np, 'action_mask': mask_np, 'global_obs': global_obs_np},
            'action': actions_np, 
            'adv': advantages, 
            'target': td_target, 
            'log_prob': log_probs_np
        }

        return data_to_push

    def _insert_into_buffer(self, data_to_push: dict):
        """将准备好的数据推送到 Replay Buffer。"""
        if not data_to_push:
            return
        
        try:
            self.replay_buffer.push(data_to_push)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error pushing data to replay buffer: {e}", exc_info=True)

    def _init_episode_data_single_buffer(self, env_idx: int) -> dict:
        agent_names = self.config.get('agent_names', DEFAULT_AGENT_NAMES_LIST)
        buffer = {}
        for name in agent_names:
            buffer[name] = {
                'state': {'observation': [], 'action_mask': [], 'global_obs': []},
                'action': [], 'value': [], 'log_prob': [], 'reward': []
            }
        return buffer
        

    def _init_episode_data_buffers(self, num_envs: int) -> list:
        """
        为所有并行环境初始化临时数据缓冲区。
        每个环境一个，用于存储当前 episode 的轨迹。
        """
        if self.logger:
            self.logger.info(f"Initializing temporary data buffers for {num_envs} parallel environments.")
        # 每个缓冲区将存储一个 episode 的 (s, a, v, lp, r) 等信息
        # 结构: [{'state': {'obs':[], 'mask':[]}, 'action': [], ...}, {...}, ...]
        # 我们为每个并行环境创建一个这样的字典
        episode_buffers = []
        for i in range(num_envs):
            episode_buffers.append(self._init_episode_data_single_buffer(i)) # 初始化第一个环境的缓冲区

        return episode_buffers

    def _store_step_data(self, episode_buffers: list, step_data: tuple, episode_raw_rewards: list):
        """
        将一步的数据存储到所有并行环境各自的临时缓冲区中。

        Args:
            episode_buffers (list): 所有并行环境的临时数据缓冲区列表。
            step_data (tuple): 包含从 InferenceServer 和 env.step() 获取的、
                            经过批处理的数据。例如 (obs_batch, actions_batch, ...)。
            episode_raw_rewards (list): 每个环境的原始奖励累积字典列表。
        """
        # 从 step_data 解包出批处理数据
        obs_batch, actions_batch, values_batch, log_probs_batch, rewards_batch, global_obs_batch = step_data
        num_envs = len(obs_batch) # 假设批次维度是0

        for i in range(num_envs): # 遍历每个环境
            env_buffer = episode_buffers[i]
            # 获取第 i 个环境的数据
            obs_dict_env = obs_batch[i]
            actions_dict_env = actions_batch[i]
            values_dict_env = values_batch[i]
            log_prob_dict_env = log_probs_batch[i]
            rewards_dict_env = rewards_batch[i]
            global_obs_env = global_obs_batch[i] 

            # 遍历在该步骤中实际行动了的 agent
            for agent_name, action_taken in actions_dict_env.items():
                agent_buffer = env_buffer[agent_name]
                # 获取该 agent 的观测数据
                observation_data = obs_dict_env[agent_name]
                action_taken = actions_dict_env[agent_name]
                value = values_dict_env[agent_name]
                log_prob = log_prob_dict_env[agent_name]
                reward = rewards_dict_env.get(agent_name, 0.0) # 如果没有奖励，默认为0.0

                agent_buffer['state']['observation'].append(observation_data['observation'])
                agent_buffer['state']['action_mask'].append(np.array(observation_data['action_mask']))
                agent_buffer['state']['global_obs'].append(global_obs_env)
                agent_buffer['action'].append(action_taken)
                agent_buffer['value'].append(value)
                agent_buffer['log_prob'].append(log_prob)
                agent_buffer['reward'].append(reward)
                
                # 累积原始奖励用于episode结束时的统计
            
            for agent_name, reward in rewards_dict_env.items():
                episode_raw_rewards[i][agent_name] += reward
                # self.logger.info(f"Agent: {agent_name}, Action: {action_taken}, Reward: {reward:.2f}")

    def _handle_done_envs(self, episode_buffers: list, dones: np.ndarray, env_policy_keys: dict, 
                          episode_start_times: list, episode_step_counts: list, episode_raw_rewards: list,
                          episode_nums: list, total_actor_steps: int):
        """
        检查哪些环境已完成，处理它们的 episode 数据（GAE 计算、奖励变换），
        然后将数据推送到 Replay Buffer，并重置对应的临时缓冲区。

        Args:
            episode_buffers (list): 所有并行环境的临时数据缓冲区列表。
            dones (np.ndarray): 来自 vec_env.step() 的完成状态数组，形状为 (num_envs,)。
            env_policy_keys (dict): 每个环境中每个agent使用的策略名称。
            episode_start_times (list): 每个环境episode开始时间列表。
            episode_step_counts (list): 每个环境episode步数计数列表。
            episode_raw_rewards (list): 每个环境的原始奖励字典列表。
            episode_nums (list): 每个环境的episode编号列表。
            total_actor_steps (int): Actor的总步数。
        """
        # 在麻将中，done 通常是 (num_envs,) 的布尔数组
        for i, done in enumerate(dones):
            if done:
                episode_duration = time.time() - episode_start_times[i]
                episode_step_count = episode_step_counts[i]
                episode_raw_reward_dict = episode_raw_rewards[i]
                episode_num = episode_nums[i]
                # 获取 agent 的奖励用于日志记录
                agent_names = self.config.get('agent_names', DEFAULT_AGENT_NAMES_LIST)

                main_agent_idx = self.main_agent_idxs[i]  # 获取当前环境的主智能体索引
                # print(main_agent_idx)
                main_agent_name = agent_names[main_agent_idx] if main_agent_idx is not None else 'unknown'

                main_agent_reward = episode_raw_reward_dict.get(main_agent_name, 0.0)


                self.total_episodes += 1
                game_result = self._determine_game_result(episode_raw_reward_dict, main_agent_name)
                
                if game_result == 'win':
                    result_value = 1.0
                elif game_result == 'draw':
                    self.draw_count += 1
                    result_value = 0.2
                else:  # loss
                    result_value = 0.0

                # 更新运行平均胜率
                self.recent_results.append(result_value)
                if len(self.recent_results) > self.win_rate_window_size:
                    self.recent_results.pop(0)  # 移除最旧的结果
                
                # 计算当前胜率（包括平局算0.2分）和平局率
                self.current_win_rate = sum(self.recent_results) / len(self.recent_results) if self.recent_results else 0.0
                self.current_draw_rate = self.draw_count / self.total_episodes if self.total_episodes > 0 else 0.0


                sum_raw_rewards_for_latest_policy = 0
                for agent_name, agent_data in episode_buffers[i].items():
                    if env_policy_keys[i].get(agent_name) == 'latest_eval':
                        sum_raw_rewards_for_latest_policy += episode_raw_reward_dict.get(agent_name)

                reward_string = ', '.join([(str(episode_raw_reward_dict.get(agent_name))) for agent_name, agent_data in episode_buffers[i].items()])
                policy_string = ', '.join([f"{agent_name}: {env_policy_keys[i].get(agent_name, 'unknown')}" for agent_name in agent_names])

                if (episode_num + 1) % self.config.get('log_interval', 10) == 0:
                    self.logger.info(f"Environment {i} in Actor {self.name} finished episode {episode_num+1} "
                                f"in {episode_step_count} steps ({episode_duration:.2f}s). "
                                f"Total Raw Reward for Latest Agent: {sum_raw_rewards_for_latest_policy:.2f}. "
                                f"Reward String ({reward_string}), Policies: {policy_string}"
                                f"Result: {game_result}, Win Rate: {self.current_win_rate:.3f}"
                                )
                
                # 1. 获取这个完成的 episode 的数据
                completed_episode_data = episode_buffers[i]
                
                # 2. 应用奖励变换逻辑（如果配置了）
                if self.config.get('use_normalized_reward', False) and len(agent_names) == 4:
                    self.logger.debug(f"Episode {episode_num+1} Env {i}: Applying reward normalization/transformation.")
                    
                    # 获取原始分数
                    raw_scores_ordered = [episode_raw_reward_dict.get(name, 0.0) for name in agent_names]
                    transformed_scores_ordered = [0.0] * 4
                    
                    # 找到最小分数和最小分数的个数
                    min_raw_score = min(raw_scores_ordered)
                    min_raw_score_count = raw_scores_ordered.count(min_raw_score)
                    
                    # 应用变换规则
                    for idx, raw_score in enumerate(raw_scores_ordered):
                        if raw_score > 0:
                            transformed_scores_ordered[idx] = np.sqrt(raw_score / 2.0)
                        elif raw_score < 0:
                            if raw_score == min_raw_score and min_raw_score_count == 1:
                                transformed_scores_ordered[idx] = -2.0
                            else:
                                transformed_scores_ordered[idx] = -1.0
                        else:
                            transformed_scores_ordered[idx] = self.config.get('draw_reward', -0.5)
                    
                    self.logger.debug(f"  Raw scores: {list(zip(agent_names, raw_scores_ordered))}")
                    self.logger.debug(f"  Transformed scores: {list(zip(agent_names, transformed_scores_ordered))}")
                    
                    # 更新episode数据中的最终奖励
                    for idx, agent_name in enumerate(agent_names):
                        if agent_name in completed_episode_data:
                            agent_reward_list = completed_episode_data[agent_name]['reward']
                            if agent_reward_list:  # 如果有记录的步骤，最后一个奖励是最终得分
                                agent_reward_list[-1] = transformed_scores_ordered[idx]
                
                # 3. TensorBoard记录
                if self.writer:
                    try:
                        self.writer.add_scalar(f'Actor_{self.name}/Reward/MainAgentRawEp', main_agent_reward, total_actor_steps)
                        self.writer.add_scalar(f'Actor_{self.name}/Episode/Length', episode_step_count, total_actor_steps)
                        self.writer.add_scalar(f'Actor_{self.name}/Episode/Duration', episode_duration, total_actor_steps)
                        self.writer.add_scalar(f'Actor_{self.name}/WinRate/Current', self.current_win_rate, total_actor_steps)
                        self.writer.add_scalar(f'Actor_{self.name}/WinRate/TotalEpisodes', self.total_episodes, total_actor_steps)
                        self.writer.add_scalar(f'Actor_{self.name}/DrawRate/Current', self.current_draw_rate, total_actor_steps)
                        
                        # 记录每个agent的原始奖励
                        for ag_name, raw_score in episode_raw_reward_dict.items():
                            safe_ag_name = "".join(c if c.isalnum() else '_' for c in str(ag_name))
                            self.writer.add_scalar(f'Actor_{self.name}/RawReward_Detailed/Agent_{safe_ag_name}_EpTotal', 
                                                 raw_score, total_actor_steps)
                        
                        self.writer.flush()
                    except Exception as e:
                        self.logger.warning(f"Actor {self.name}: Failed to write to detailed TensorBoard: {e}", exc_info=True)

                # 4. 主要指标记录到main writer（核心指标）
                if self.main_writer:
                    try:
                        # 只记录最重要的指标到主TensorBoard
                        self.main_writer.add_scalar(f'Training/Episode_Reward_{self.name}', main_agent_reward, total_actor_steps)
                        self.main_writer.add_scalar(f'Training/WinRate_{self.name}', self.current_win_rate, total_actor_steps)
                        self.main_writer.add_scalar(f'Training/Episode_Length_{self.name}', episode_step_count, total_actor_steps)
                        self.main_writer.add_scalar(f'Training/Draw_Rate_{self.name}', self.current_draw_rate, total_actor_steps)
                        
                        self.main_writer.flush()
                    except Exception as e:
                        self.logger.warning(f"Actor {self.name}: Failed to write to main TensorBoard: {e}", exc_info=True)
                
                # 4. 准备数据 (计算 GAE) 并推送到 Replay Buffer
                for agent_name, agent_data in completed_episode_data.items():
                    # 只有使用了 "latest_eval" 策略的 agent 的数据才会被用于训练
                    agent_policy = env_policy_keys[i].get(agent_name, 'latest_eval')
                    if agent_policy != 'latest_eval':
                        self.logger.debug(f"Skipping data push for env {i} agent {agent_name} "
                                        f"(policy: {agent_policy}, not latest_eval)")
                        continue
                    
                    prepared_data = self._prepare_data_for_buffer(f"env{i}_{agent_name}", agent_data)
                    if prepared_data:
                        self._insert_into_buffer(prepared_data)
                        self.logger.debug(f"Successfully pushed {len(prepared_data['action'])} timesteps "
                                        f"from env {i} agent {agent_name} to replay buffer.")

                # 5. 重置这个环境的各种统计数据
                episode_buffers[i] = self._init_episode_data_single_buffer(i)
                episode_start_times[i] = time.time()
                episode_step_counts[i] = 0
                episode_raw_rewards[i] = {name: 0.0 for name in agent_names}
                episode_nums[i] += 1

                self._prepare_opponents(episode_num, env_policy_keys, i) # 更新对手策略


    def _response_worker_loop(self):
        """
        运行在独立线程中的循环，负责从InferenceServer的响应队列接收数据，
        并将其放入Actor内部的线程安全队列中，供主逻辑线程处理。
        """
        if self.logger: # logger 可能在线程启动时还未完全初始化完毕，做个检查
            self.logger.info(f"{self.name}: 响应处理工作线程已启动。")
        else:
            print(f"{self.name}: 响应处理工作线程已启动 (logger未初始化)。")

        while True:
            # 检查关闭事件
            # if self.shutdown_event and self.shutdown_event.is_set():
            #     if self.logger: self.logger.info(f"{self.name}: 响应处理工作线程检测到关闭信号，正在退出。")
            #     break
            
            try:
                # 从与 InferenceServer 连接的 multiprocessing.Queue 获取响应
                # InferenceServer 发送的响应格式应为: (original_request_id, action, value, log_prob)
                # original_request_id 是 Actor 发送请求时生成的那个 request_counter 值
                response_payload = self.inference_resp_queue.get(timeout=0.1) # 使用短超时以便能周期性检查 shutdown_event

                # 将获取到的响应放入 Actor 内部的 ThreadQueue，供主线程消费
                # put 操作会阻塞，直到主线程从 internal_result_queue 中 get (因为 maxsize=1)
                # 这确保了响应是按顺序（或至少是单个未处理）传递给主线程的
                self.internal_result_queue.put(response_payload) 

            except QueueEmpty: # get(timeout=0.1) 超时，是正常现象，继续循环检查 shutdown_event
                continue
            except Exception as e:
                # 记录工作线程中的其他异常
                if self.logger:
                    self.logger.error(f"{self.name}: 响应处理工作线程发生错误: {e}", exc_info=True)
                else:
                    print(f"{self.name}: 响应处理工作线程发生错误: {e}")
                # 发生严重错误时，可以考虑也通知主线程或尝试优雅退出
                time.sleep(0.5) # 避免在持续错误时CPU空转

        if self.logger: self.logger.info(f"{self.name}: 响应处理工作线程已结束。")


    def _get_action_from_inference_server(self, model_key_to_request: str, observation_dict_for_model: dict) -> tuple:
        """
        向 InferenceServer 发送推理请求，并从内部工作线程获取匹配的响应。
        """
        if self.inference_req_queue is None:
            if self.logger: self.logger.error("请求队列未配置。无法获取动作。")
            else: print("请求队列未配置。无法获取动作。")
            return 0, 0.0, 0.0 # 返回默认/安全值

        self.request_counter += 1
        current_internal_request_id = self.request_counter # Actor内部生成的、期望匹配的请求ID

        # payload 发送给 InferenceServer，包含 name (例如 "Actor-0") 和这个内部请求ID
        payload = (self.name, current_internal_request_id, model_key_to_request, observation_dict_for_model)
        
        log_msg_prefix = f"{self.name} ReqID {current_internal_request_id} for model '{model_key_to_request}'"

        try:
            if self.logger: self.logger.debug(f"{log_msg_prefix}: 正在发送推理请求...")
            self.inference_req_queue.put(payload, timeout=self.config.get('queue_put_timeout_seconds', 2.0))
            
            if self.logger: self.logger.debug(f"{log_msg_prefix}: 请求已发送。正在等待内部工作线程的响应...")
            
            # 从内部的 ThreadQueue 获取响应，这个响应是由 _response_worker_loop 放入的
            # 需要设置一个合理的超时时间
            timeout_duration = self.config.get('inference_timeout_seconds', 5.0)
            
            # 循环获取，直到拿到与 current_internal_request_id 匹配的响应，或者超时
            # 这是为了处理一种可能性：如果之前的 get 超时了，但响应稍后到达并被worker线程放入队列，
            # 而此时我们正在为新的请求等待，get() 可能会拿到旧的响应。
            # (虽然 internal_result_queue 的 maxsize=1 会缓解这个问题，但多一层校验更稳妥)
            start_wait_time = time.time()
            while True:
                if time.time() - start_wait_time > timeout_duration:
                    # 整体等待超时
                    raise QueueEmpty # 重新抛出 QueueEmpty 以便被外部的 except 块捕获

                try:
                    # 从内部队列获取，短超时以便能快速检查外部的整体超时
                    # 响应格式应为 (server_returned_request_id, action, value, log_prob)
                    response_from_worker = self.internal_result_queue.get(timeout=0.05) # 短超时
                    
                    server_returned_request_id, action, value, log_prob = response_from_worker
                    
                    if server_returned_request_id == current_internal_request_id:
                        # 成功匹配到当前请求的响应
                        if self.logger: self.logger.debug(f"{log_msg_prefix}: 已收到匹配的响应。动作: {action}, 价值: {value:.4f}, LogProb: {log_prob:.4f}")
                        return action, value, log_prob
                    else:
                        # 收到的响应ID与当前期望的ID不匹配，可能是过期的响应
                        if self.logger:
                            self.logger.warning(f"{log_msg_prefix}: 从工作线程收到不匹配的响应ID。期望 {current_internal_request_id}，但收到 {server_returned_request_id}。正在丢弃并继续等待...")
                        # 这个过期的响应已经被从 internal_result_queue 中取出了，继续循环等待正确的
                
                except QueueEmpty: # internal_result_queue.get(timeout=0.05) 超时
                    # 只是内部get超时，继续外层while循环，直到整体超时或工作线程停止
                    if self.response_worker_thread and not self.response_worker_thread.is_alive():
                        if self.logger: self.logger.error(f"{log_msg_prefix}: 响应处理工作线程已停止。无法获取推理结果。")
                        # 可以选择抛出特定异常或返回错误
                        return 0, 0.0, 0.0 # 或者 raise RuntimeError("Response worker died")
                    # 否则，继续等待 (外部的 while True + timeout_duration 会处理最终超时)
                    pass 
                except ValueError as ve_unpack: # 解包 response_from_worker 可能出错
                    if self.logger: self.logger.error(f"{log_msg_prefix}: 从工作线程获取的响应解包失败: {ve_unpack}. 响应: {response_from_worker if 'response_from_worker' in locals() else 'N/A'}", exc_info=True)
                    # 这种情况通常意味着响应格式错误，可能需要返回错误或重试
                    # 为了简单起见，这里也当作一种超时或错误处理
                    raise QueueEmpty # 触发外部的超时/错误处理


        except QueueFull: # put 到 inference_req_queue 超时
            if self.logger: self.logger.error(f"{log_msg_prefix}: 发送请求到InferenceServer时超时或队列已满。")
            return 0, 0.0, 0.0
        except QueueEmpty: # 从 internal_result_queue 获取响应最终超时 (由 raise QueueEmpty 触发)
            if self.logger: self.logger.error(f"{log_msg_prefix}: 等待推理响应超时。")
            return 0, 0.0, 0.0
        except ValueError as ve: # 例如，如果 response_payload 解包失败 (虽然内部循环也尝试处理)
            err_msg = f"{log_msg_prefix}: 解包推理响应时发生ValueError: {ve}."
            if self.logger: self.logger.error(err_msg, exc_info=True)
            else: print(err_msg)
            return 0, 0.0, 0.0
        except Exception as e:
            err_msg = f"{log_msg_prefix}: 推理请求/响应过程中发生未知错误: {e}"
            if self.logger: self.logger.error(err_msg, exc_info=True)
            else: print(err_msg)
            return 0, 0.0, 0.0

    def _prepare_opponents(self, episode_num: int, env_policy_keys: dict, env_to_update: int):

        if (episode_num + 1) % self.config.get('opponent_model_change_interval', 10) == 0:

            env_policy_keys[env_to_update] = {}

            current_agent_seat_idx = np.random.randint(0, 4) # 随机选择一个座位作为主智能体

            self.main_agent_idxs[env_to_update] = current_agent_seat_idx

            p = np.random.random((4,))


            p_historical = self.config.get('p_opponent_historical', 0.15) # 从配置中获取概率
            p_benchmark = self.config.get('p_opponent_benchmark', 0.4) # 从配置中获取概率

            for seat_ix in range(4):
                agent_name_for_seat = DEFAULT_AGENT_NAMES_LIST[seat_ix]
                if seat_ix == current_agent_seat_idx:
                    # 主智能体使用最新评估模型
                    env_policy_keys[env_to_update][agent_name_for_seat] = "latest_eval"
                else:
                    # 其他座位的智能体根据概率选择模型
                    if self.server_hosted_benchmark_names and p[seat_ix] < p_benchmark:
                        env_policy_keys[env_to_update][agent_name_for_seat] = random.choice(self.server_hosted_benchmark_names)
                    elif p[seat_ix] < p_historical + p_benchmark:
                        env_policy_keys[env_to_update][agent_name_for_seat] = "random_historical" # 使用历史模型
                    else:
                        env_policy_keys[env_to_update][agent_name_for_seat] = "latest_eval"


    # 进程启动时执行的主函数
    def run(self):
        """
        Actor 进程的主要执行逻辑。包括：
        1. 初始化日志和 TensorBoard。
        2. 初始化环境和模型。
        3. 从 Model Pool 获取最新的模型。
        4. 在环境中运行多个 episode 进行自对弈 (self-play)。
        5. 收集状态、动作、奖励等数据。
        6. 计算 GAE (Generalized Advantage Estimation) 和 TD Target。
        7. 将处理后的数据推送到 Replay Buffer。
        8. 定期 更新本地模型。
        """
        # --- 初始化日志和 TensorBoard ---
        # 创建主要日志和详细日志的writer
        self.logger, self.writer, actor_log_paths = setup_process_logging_and_tensorboard(
            self.config['log_base_dir'], self.config, self.name, log_type='detailed'
        )
        
        # 同时创建主要日志的writer，用于记录核心指标
        try:
            self.main_logger, self.main_writer, main_log_paths = setup_process_logging_and_tensorboard(
                self.config['log_base_dir'], self.config, self.name, log_type='main'
            )
        except Exception as e:
            self.logger.warning(f"Failed to setup main TensorBoard writer: {e}")
            self.main_writer = None

        # --- 新增：在 run() 方法内部初始化线程相关对象 ---
        self.pending_requests_lock = threading.Lock()
        self.internal_result_queue = ThreadQueue(maxsize=1) # 或者其他合适的 maxsize

        # 设置 PyTorch 在该进程中使用的线程数为 1，避免多进程场景下线程过多导致竞争和性能下降
        try:
            torch.set_num_threads(1)
            self.logger.info(f"PyTorch num_threads set to 1.")
        except Exception as e:
            self.logger.warning(f"Failed to set torch num_threads: {e}")

        # if self.shutdown_event is None:
        #     self.logger.critical(f"Actor {self.name}: shutdown_event 未提供! 响应工作线程可能无法优雅关闭。")
            # 这是一个严重问题，应该在 __init__ 中就阻止启动，或者在这里抛出异常
            # raise RuntimeError("shutdown_event is required for response worker thread.")

        self.response_worker_thread = threading.Thread(target=self._response_worker_loop, daemon=True)
        self.response_worker_thread.name = f"{self.name}-RespWorker" # 给线程一个名字
        self.response_worker_thread.start()
        self.logger.info(f"Actor {self.name}: 响应处理工作线程已启动。")

        self.logger.info(f"Actor {self.name} will request inferences from InferenceServer.")
        
        def make_env_func(config):
            def _init():
                from env.env import MahjongGBEnv
                from agent.feature_timeseries import FeatureAgentTimeSeries
                from agent.feature import FeatureAgent

                # (如果需要，在这里解析字符串为类对象)
                if isinstance(config.get('agent_clz'), str):
                    if config['agent_clz'] == 'FeatureAgentTimeSeries':
                        config['agent_clz'] = FeatureAgentTimeSeries
                    elif config['agent_clz'] == 'FeatureAgent':
                        config['agent_clz'] = FeatureAgent
                return MahjongGBEnv(config=config)
            return _init

        # 2. 从配置中读取并行环境的数量
        num_envs_per_actor = self.config.get('num_envs_per_actor', 8) # 例如，每个 Actor 进程管理8个并行环境
        self.logger.info(f"Actor {self.name} will manage {num_envs_per_actor} parallel environments.")

        # 3. 创建函数列表并实例化 VecEnv
        env_functions = [make_env_func(self.config) for _ in range(num_envs_per_actor)]
        env = SubprocVecEnv(env_functions)
        self.logger.info(f"SubprocVecEnv created for Actor {self.name}.")

        # --- 主运行循环 (按步数迭代) ---
        self.logger.info(f"Actor {self.name}: Starting main rollout loop...")        # 1. 为向量化环境创建数据缓冲区并获取初始观测
        episode_data_buffers = self._init_episode_data_buffers(num_envs_per_actor)
        obs_batch, global_obs_batch = env.reset() # 现在返回观测字典列表和全局观测列表
        
        # 2. 初始化episode跟踪变量
        agent_names = self.config.get('agent_names', DEFAULT_AGENT_NAMES_LIST)
        episode_start_times = [time.time() for _ in range(num_envs_per_actor)]
        episode_step_counts = [0 for _ in range(num_envs_per_actor)]
        episode_raw_rewards = [{name: 0.0 for name in agent_names} for _ in range(num_envs_per_actor)]
        episode_nums = [0 for _ in range(num_envs_per_actor)]
        
        # 3. Profiler支持（仅对第一个episode启用）
        profiler = None
        profile_first_episode = self.config.get('profile_first_episode', False)
        if profile_first_episode:
            profiler = cProfile.Profile()
            profiler.enable()
            self.logger.info(f"Actor {self.name}: Profiling enabled for first episode.")

        num_env_steps = self.config.get('num_env_steps', 1000000)
        total_steps_to_run = num_env_steps // num_envs_per_actor
        total_actor_steps = 0 # 用于跟踪 Actor 的总步数

        # --- 对手选择参数 ---        # policies 字典现在存储每个并行环境的策略分配
        # 结构: {env_idx: {'player_0': 'model_key', 'player_1': 'model_key', ...}}
        env_policy_keys = [{} for _ in range(num_envs_per_actor)]
        self.main_agent_idxs = [None] * num_envs_per_actor # 用于跟踪每个环境的主智能体索引
        

        for i in range(num_envs_per_actor):
            self._prepare_opponents(-1, env_policy_keys, i)
        # 初始化每个环境的策略分配
        # for i in range(num_envs_per_actor):
        #     for seat_idx in range(4):
        #         agent_name_for_seat = agent_names[seat_idx] if seat_idx < len(agent_names) else f"player_{seat_idx+1}"
        #         env_policy_keys[i][agent_name_for_seat] = "latest_eval"  # 默认使用最新评估模型

        for step in range(total_steps_to_run):
            # if self.shutdown_event and self.shutdown_event.is_set():
            #     self.logger.info(f"Actor {self.name} received shutdown signal. Stopping rollout loop at step {step}.")
            #     break

            total_actor_steps += num_envs_per_actor # 每一步都增加了 num_envs 个时间步

            flat_inference_requests = []
            for i in range(num_envs_per_actor):
                obs_dict_for_env = obs_batch[i]
                global_obs_for_env = global_obs_batch[i] if global_obs_batch[i] is not None else None
                
                if not isinstance(obs_dict_for_env, dict) or not obs_dict_for_env:
                    continue # 跳过无效或已结束的环境的观测
                
                for agent_name, observation_data in obs_dict_for_env.items():
                    model_key = env_policy_keys[i].get(agent_name, "latest_eval") # 获取策略，默认为 latest_eval
                    
                    # 构造全局状态 (centralized_extra_info) - 现在传递 global_obs
                    centralized_extra_info = self._create_centralized_info(obs_dict_for_env, agent_name, i, global_obs_for_env)
                    
                    # 将需要发送给服务器的数据打包，包含inference_server期望的所有字段
                    obs_to_send = {
                        'obs': {
                            'observation': observation_data['observation'],
                            'action_mask': np.array(observation_data['action_mask'])
                        },
                        'centralized_extra_info': centralized_extra_info
                    }
                    # 记录请求来源，以便后续重组结果
                    flat_inference_requests.append({
                        "env_idx": i,
                        "agent_name": agent_name,
                        "model_key": model_key,
                        "obs_data": obs_to_send
                    })

            if not flat_inference_requests: # 如果没有需要行动的 agent
                time.sleep(0.01) # 短暂等待
                continue

            # (C) 批量获取推理结果 - 利用InferenceServer的批处理能力
            flat_results = self._get_batch_actions_from_inference_server(flat_inference_requests)

            # (D) 重组动作用于 env.step()，并收集推理结果
            actions_batch_for_env = [{} for _ in range(num_envs_per_actor)]
            values_batch = [{} for _ in range(num_envs_per_actor)]
            log_probs_batch = [{} for _ in range(num_envs_per_actor)]

            for i, request in enumerate(flat_inference_requests):
                env_idx = request['env_idx']
                agent_name = request['agent_name']
                action, value, log_prob = flat_results[i]
                if action is None: # 推理失败
                    self.logger.error(f"Inference failed for env {env_idx}, agent {agent_name}. Terminating actor.")
                    # self.shutdown_event.set(); break
                
                actions_batch_for_env[env_idx][agent_name] = action
                values_batch[env_idx][agent_name] = value
                log_probs_batch[env_idx][agent_name] = log_prob
            # if self.shutdown_event.is_set(): break            # (E) 与所有环境交互
            next_obs_batch, rewards_batch, dones_batch, global_obs_batch = env.step(actions_batch_for_env)
              # (F) 更新步数计数
            for i in range(num_envs_per_actor):
                episode_step_counts[i] += 1
            
            # (G) 将这一步的数据存储到缓冲区
            # step_data_tuple 的结构现在需要与 _store_step_data 的输入匹配
            step_data_tuple = (obs_batch, actions_batch_for_env, values_batch, log_probs_batch, rewards_batch, global_obs_batch)
            self._store_step_data(episode_data_buffers, step_data_tuple, episode_raw_rewards)

            # (H) 检查并处理已完成的 episodes
            if np.any(dones_batch):
                # 关闭第一个episode的profiler（如果启用了）
                if profiler and any(episode_nums[i] == 0 for i in range(num_envs_per_actor) if dones_batch[i]):
                    profiler.disable()
                    self._save_profile_results(profiler)
                    profiler = None
                    
                self._handle_done_envs(episode_data_buffers, dones_batch, env_policy_keys,
                                     episode_start_times, episode_step_counts, episode_raw_rewards,
                                     episode_nums, total_actor_steps)            # (H) 更新观测，为下一步做准备
            obs_batch = next_obs_batch
            # global_obs_batch 已经在 env.step() 调用时更新了
        
        # --- 循环结束后 ---
        self.logger.info(f"Actor {self.name} finished its rollout loop.")
        
        # 保存任何剩余的profile结果
        if profiler:
            profiler.disable()
            self._save_profile_results(profiler)
            
        env.close()

        # 清理线程
        if self.response_worker_thread and self.response_worker_thread.is_alive():
            self.logger.info(f"Actor {self.name}: Waiting for response worker thread to join...")
            self.response_worker_thread.join(timeout=5.0) # 给工作线程一些时间退出
            if self.response_worker_thread.is_alive():
                self.logger.warning(f"Actor {self.name}: Response worker thread did not join in time.")

        # 关闭TensorBoard writer
        if self.writer:
            self.writer.close()

        self.logger.info(f"Actor {self.name}: Run completed successfully.")

    def _save_profile_results(self, profiler):
        """保存profiler结果到文件"""
        try:
            ps = pstats.Stats(profiler).sort_stats('cumulative')
            profile_dir = os.path.join(self.config['log_base_dir'], "file_logs", 
                                     self.config['experiment_name'], "actor_profiles")
            os.makedirs(profile_dir, exist_ok=True)
            profile_log_path = os.path.join(profile_dir, f"{self.name}_first_ep_profile.prof")
            ps.dump_stats(profile_log_path)
            self.logger.info(f"Actor {self.name}: Profile results for first episode saved to {profile_log_path}")
        except Exception as e:
            self.logger.warning(f"Actor {self.name}: Failed to save profile results: {e}", exc_info=True)

    def _create_centralized_info(self, obs_dict_for_env: dict, current_agent_name: str, env_idx: int, global_obs: np.ndarray = None):
        """
        创建用于中心化critic的全局状态信息。
        
        Args:
            obs_dict_for_env: 当前环境中所有agent的观测字典
            current_agent_name: 当前请求推理的agent名称
            env_idx: 环境索引
            global_obs: 来自环境的全局观测 (16, 4, 9)
            
        Returns:
            np.ndarray: 中心化状态信息，格式为适合ExtraInfoFeatureExtractor的4D张量
        """        
        centralized_base = global_obs.copy()  # (16, 4, 9)
        return centralized_base.astype(np.float32)

    def _get_batch_actions_from_inference_server(self, flat_inference_requests: list):
        """
        批量向InferenceServer发送推理请求并等待响应。
        利用InferenceServer的批处理能力来提高效率。
        
        Args:
            flat_inference_requests: 推理请求列表
            
        Returns:
            list: 与请求对应的推理结果列表 [(action, value, log_prob), ...]
        """
        if not flat_inference_requests:
            return []
        
        # 发送所有请求到InferenceServer
        request_ids = []
        for i, req in enumerate(flat_inference_requests):
            self.request_counter += 1
            current_request_id = self.request_counter
            request_ids.append(current_request_id)
            
            # InferenceServer期望的格式: (actor_id, request_id, model_key, obs_data)
            payload = (self.name, current_request_id, req['model_key'], req['obs_data'])
            
            try:
                self.inference_req_queue.put(payload, timeout=self.config.get('queue_put_timeout_seconds', 2.0))
            except QueueFull:
                self.logger.error(f"Failed to send request {current_request_id} to InferenceServer: queue full")
                # 对于发送失败的请求，返回默认值
                return [(0, 0.0, 0.0) for _ in flat_inference_requests]
        
        # 等待所有响应
        results = []
        timeout_duration = self.config.get('inference_timeout_seconds', 10.0)  # 批处理可能需要更长时间
        start_time = time.time()
        
        received_responses = {}  # request_id -> (action, value, log_prob)
        
        while len(received_responses) < len(request_ids):
            if time.time() - start_time > timeout_duration:
                self.logger.error(f"Batch inference timeout: received {len(received_responses)}/{len(request_ids)} responses")
                break
            
            try:
                # 从内部队列获取响应
                response_payload = self.internal_result_queue.get(timeout=0.1)
                request_id, action, value, log_prob = response_payload
                
                if request_id in request_ids:
                    received_responses[request_id] = (action, value, log_prob)
                else:
                    self.logger.warning(f"Received unexpected response for request_id {request_id}")
                    
            except QueueEmpty:
                continue
            except Exception as e:
                self.logger.error(f"Error receiving batch response: {e}", exc_info=True)
                break
        
        # 按原始请求顺序组织结果
        for request_id in request_ids:
            if request_id in received_responses:
                results.append(received_responses[request_id])
            else:
                # 对于未收到响应的请求，使用默认值
                self.logger.warning(f"No response received for request_id {request_id}, using default values")
                results.append((0, 0.0, 0.0))
        
        return results

    def _determine_game_result(self, episode_raw_reward_dict: dict, main_agent_name: str) -> str:
        """
        根据原始奖励字典判断一局游戏的结果。

        Args:
            episode_raw_reward_dict (dict): 包含每个agent原始奖励的字典。
            main_agent_name (str): 主要agent的名称，用于结果判断。

        Returns:
            str: 游戏结果，可能的值有 'win', 'loss', 'draw'。
        """
        main_agent_reward = episode_raw_reward_dict.get(main_agent_name, 0.0)
        all_rewards = list(episode_raw_reward_dict.values())
        
        # 获取所有玩家的奖励并排序
        sorted_rewards = sorted(all_rewards, reverse=True)
        
        # 判断平局：主要agent的得分与其他玩家相同的情况
        if main_agent_reward == sorted_rewards[0] and sorted_rewards.count(sorted_rewards[0]) > 1:
            # 如果主要agent获得最高分但有多人并列第一，算作平局
            return 'draw'
        elif main_agent_reward == 0 and all(reward <= 0 for reward in all_rewards):
            # 如果所有人得分都不为正，且主要agent得分为0，算作平局
            return 'draw'
        elif main_agent_reward > 0:
            # 主要agent获得正分，算作胜利
            return 'win'
        else:
            # 主要agent得分为负或0（且不满足平局条件），算作失败
            return 'loss'