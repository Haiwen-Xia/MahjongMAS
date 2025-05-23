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
from env import MahjongGBEnv                # 麻将游戏环境
from feature import FeatureAgent            # 用于处理麻将特征的 Agent
from model import ResNet34AC                # Actor 使用的神经网络模型定义
# from torch.utils.tensorboard import SummaryWriter # 由 setup_process_logging_and_tensorboard 处理
from utils import setup_process_logging_and_tensorboard # 用于设置日志和 TensorBoard 的工具函数
import random

import cProfile, pstats # 用于记录效率信息 (如果启用)

# 多线程 Actor 相关
import threading
from queue import Queue as ThreadQueue # Python标准库的线程安全队列，用于Actor内部线程间通信
from collections import defaultdict # 用于方便地管理待处理的动作

DEFAULT_AGENT_NAMES_LIST = ['player_1', 'player_2', 'player_3', 'player_4']

# Actor 类，继承自 Process，每个 Actor 负责在一个或多个环境中与自身或其他策略进行交互，收集经验数据
class Actor(Process):

    # 初始化函数
    def __init__(self, config, replay_buffer):
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
        self.server_hosted_benchmark_names = self.config.get("server_hosted_benchmark_names", [])
        if not self.server_hosted_benchmark_names:
            # logger 可能还未初始化
            print(f"[{self.name}] Info: No server-hosted benchmark model names provided in config.")
        else:
            print(f"[{self.name}] Will be able to request server-hosted benchmarks: {self.server_hosted_benchmark_names}")


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
            self.inference_req_queue.put(payload, timeout=self.config.get("queue_put_timeout_seconds", 2.0))
            
            if self.logger: self.logger.debug(f"{log_msg_prefix}: 请求已发送。正在等待内部工作线程的响应...")
            
            # 从内部的 ThreadQueue 获取响应，这个响应是由 _response_worker_loop 放入的
            # 需要设置一个合理的超时时间
            timeout_duration = self.config.get("inference_timeout_seconds", 5.0)
            
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
        8. 定期更新本地模型。
        """
        # --- 初始化日志和 TensorBoard ---
        self.logger, self.writer = setup_process_logging_and_tensorboard(
            self.config['log_base_dir'], self.config['experiment_name'], self.name
        )

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

        # --- 初始化环境 ---
        env = None
        try:
            env_config = self.config.get('env_config', {})
            if 'agent_clz' not in env_config:
                env_config['agent_clz'] = FeatureAgent
            self.logger.info(f"Actor {self.name} using env_config: {env_config}")
            env = MahjongGBEnv(config=env_config)
            self.logger.info(f"Mahjong environment created for Actor {self.name} with agent class: {env_config['agent_clz'].__name__}.")
        except Exception as e:
            self.logger.error(f"Failed to create Mahjong environment for Actor {self.name}: {e}. Exiting.", exc_info=True)
            if self.writer: self.writer.close()
            # 确保工作线程也被通知关闭 (如果已启动)
            # if self.shutdown_event: self.shutdown_event.set()
            if self.response_worker_thread and self.response_worker_thread.is_alive():
                self.response_worker_thread.join(timeout=1.0)
            return

        # --- 对手选择参数 ---
        # prob_opponent_is_benchmark: 从服务器请求基准模型的概率
        prob_opponent_is_benchmark_server = self.config.get('prob_opponent_is_benchmark', 0.15)
        # prob_opponent_is_historical_via_server: 从服务器请求历史模型的概率 (当前简化为也请求 'latest_eval' 或其他 benchmark)
        # p_opponent_historical_via_server = self.config.get('p_opponent_historical_via_server', 0.2) 
        
        opponent_model_change_interval = self.config.get('opponent_model_change_interval', 1)
        
        # policies 字典将存储当前 episode 中每个 agent_name -> 要发送给 InferenceServer 的模型键名
        current_episode_policy_keys = {} 
        current_episode_policy_sources_log = {} # 仅用于日志，记录来源

        total_actor_steps = 0
        episodes_per_actor_run = self.config.get('episodes_per_actor', 100000)

        self.logger.info(f"Actor {self.name}: Preparing to start episode loop for {episodes_per_actor_run} episodes.") # <--- 新增日志点 (Alpha)

        # --- 主循环 ---
        for episode_num in range(episodes_per_actor_run):
            self.logger.info(f"Actor {self.name}: START of episode {episode_num + 1}.") # <--- 新增日志点 (Beta)

            # if self.shutdown_event and self.shutdown_event.is_set():
            #     self.logger.info(f"Actor {self.name} received shutdown signal. Stopping.")
            #     break
            
            profiler = None # Profiler 初始化
            if self.config.get('enable_profiling_actor', False) and episode_num == 0 : 
                profiler = cProfile.Profile(); profiler.enable()
                self.logger.info(f"Actor {self.name}: Profiling enabled for the first episode.")

            episode_start_time = time.time()

            self.logger.info(f"Arranging Policy for Actor {self.name} at Episode {episode_num + 1}.")
            # --- 1. 定期为本局游戏设置所有席位的策略 (模型键名) ---
            if episode_num % opponent_model_change_interval == 0:
                self.logger.info(f"Actor {self.name}, Episode {episode_num + 1}: Setting policy keys for all seats.")
                current_episode_policy_keys.clear()
                current_episode_policy_sources_log.clear()

                # 随机分配一个位置给 main agent
                main_agent_seat_idx = np.random.randint(0, 4)

                for seat_idx in range(4):
                    agent_name_for_seat = env.agent_names[seat_idx]
                    
                    if seat_idx == main_agent_seat_idx:
                        current_episode_policy_keys[agent_name_for_seat] = "latest_eval"
                        current_episode_policy_sources_log[agent_name_for_seat] = "server_latest_eval_main"
                    else: # 对手席位
                        use_benchmark = self.server_hosted_benchmark_names and \
                                        np.random.rand() < prob_opponent_is_benchmark_server
                        
                        if use_benchmark:
                            chosen_benchmark_key = random.choice(self.server_hosted_benchmark_names)
                            current_episode_policy_keys[agent_name_for_seat] = chosen_benchmark_key
                            current_episode_policy_sources_log[agent_name_for_seat] = f"server_benchmark_{chosen_benchmark_key}"
                        else:
                            # 对于非基准对手，也使用 "latest_eval" (或未来扩展为从服务器采样历史)
                            current_episode_policy_keys[agent_name_for_seat] = "latest_eval"
                            current_episode_policy_sources_log[agent_name_for_seat] = "server_latest_eval_opponent"
                
                log_parts = [f"P{s}={current_episode_policy_sources_log.get(env.agent_names[s],'ERR')}" for s in range(4)]
                self.logger.info(f"  Episode Policy Setup: {', '.join(log_parts)}")


            # --- 运行一个 episode 并收集数据 ---
            # (日志在循环开始时打印更清晰)
            # self.logger.info(f"Actor {self.name}, Ep {episode_num+1}/{episodes_per_actor_run}.")

            self.logger.info(f"Actor {self.name}: Starting actual gameplay for Ep {episode_num+1}/{episodes_per_actor_run}.") # <--- 新增日志点 (Gamma)

            try: 
                obs = env.reset() # obs 是一个字典: {agent_name: state_dict_from_feature_agent}
            except Exception as e_reset:
                self.logger.error(f"Failed to reset env for ep {episode_num+1}: {e_reset}. Skipping.", exc_info=True)
                if profiler: profiler.disable(); # ... (save profile logic)
                continue 

            episode_data = {name_init: {
                'state' : {'observation': [], 'action_mask': []},
                'action' : [], 'reward' : [], 'value' : [], 'log_prob' : []
            } for name_init in env.agent_names}
            episode_raw_rewards = {name_init: 0.0 for name_init in env.agent_names}

            done = False 
            step_count = 0 
            record_data_for_agent_this_step = {} 

            while not done:
                # if self.shutdown_event and self.shutdown_event.is_set():
                #     self.logger.info(f"Actor {self.name} received shutdown mid-ep {episode_num+1}. Ending early.")
                #     done = True; break

                step_count += 1
                total_actor_steps += 1 
                actions_for_env_step = {} 
                record_data_for_agent_this_step.clear()

                current_agents_to_act = list(obs.keys())
                if not current_agents_to_act and not done:
                    self.logger.warning(f"Ep {episode_num+1}, Step {step_count}: No agents in 'obs'. Ending ep.")
                    done = True; break

                for agent_name_env in current_agents_to_act:
                    agent_specific_episode_data = episode_data.get(agent_name_env)
                    if agent_specific_episode_data is None: 
                        self.logger.warning(f"Agent '{agent_name_env}' in obs but not in episode_data. Skipping.")
                        actions_for_env_step[agent_name_env] = 0 # Default pass
                        continue
                        
                    current_state_raw = obs[agent_name_env] 
                    action_mask_numpy = np.array(current_state_raw['action_mask'])
                    num_valid_actions = action_mask_numpy.sum()
                    should_filter_step = self.config.get('filter_single_action_steps', True) and num_valid_actions <= 1
                    record_data_for_agent_this_step[agent_name_env] = not should_filter_step

                    if record_data_for_agent_this_step[agent_name_env]:
                        agent_specific_episode_data['state']['observation'].append(current_state_raw['observation'])
                        agent_specific_episode_data['state']['action_mask'].append(action_mask_numpy)
                    
                    action_item_for_env = 0 
                    log_prob_item_result = 0.0
                    value_item_result = 0.0
                    
                    model_key_for_current_agent = current_episode_policy_keys.get(agent_name_env)
                    if model_key_for_current_agent is None:
                        self.logger.error(f"No model_key found for {agent_name_env} in current_episode_policy_keys! Using 'latest_eval'.")
                        model_key_for_current_agent = "latest_eval" # Fallback

                    # 准备发送给 InferenceServer 的观测数据 (已经是 NumPy 数组)
                    obs_to_send_to_server = {
                        'obs': {
                            'observation': current_state_raw['observation'], 
                            'action_mask': action_mask_numpy 
                        }
                    }
                    # 从 InferenceServer 获取动作、价值、log_prob
                    action_item_for_env, value_item_result, log_prob_item_result = \
                        self._get_action_from_inference_server(model_key_for_current_agent, obs_to_send_to_server)
                    
                    if action_item_for_env is None: # 推理失败或超时
                        self.logger.error(f"Failed to get action from InferenceServer for {agent_name_env} (model_key: {model_key_for_current_agent}). Ending episode.")
                        done = True; break 
                            
                    actions_for_env_step[agent_name_env] = action_item_for_env

                    if record_data_for_agent_this_step[agent_name_env]:
                        agent_specific_episode_data['action'].append(action_item_for_env)
                        agent_specific_episode_data['value'].append(value_item_result)
                        agent_specific_episode_data['log_prob'].append(log_prob_item_result)

                if done: break 

                # if not actions_for_env_step:
                #     if not done: self.logger.warning(f"No actions for env.step at step {step_count}. Ending episode.")
                #     done = True; break
                
                try:
                    next_obs, rewards, done_info = env.step(actions_for_env_step) 
                except Exception as e_env_step:
                    self.logger.error(f"Error during env.step at step {step_count}: {e_env_step}. Ending episode.", exc_info=True)
                    done = True 
                
                if not done: 
                    for r_agent_name, r_val in rewards.items():
                        if r_agent_name in episode_raw_rewards:
                            episode_raw_rewards[r_agent_name] += r_val

                        # if r_agent_name in episode_data and record_data_for_agent_this_step.get(r_agent_name, False):
                            episode_data[r_agent_name]['reward'].append(r_val)
                            
                    obs = next_obs
                    # (done_info 处理逻辑与之前类似)
                    if isinstance(done_info, dict):
                        if done_info.get("__all__", False): done = True
                        else:
                            # active_next = {k for k, v_done in done_info.items() if not v_done and k in obs}
                            obs = {k: v for k, v in next_obs.items() if k in active_next}
                            if not obs and not done_info.get("__all__", False): done = True
                    elif isinstance(done_info, bool): done = done_info
                    else: self.logger.error(f"Unknown done_info type: {type(done_info)}"); done = True
            
            # --- Episode 结束 ---
            # (日志记录, TensorBoard, GAE 计算, 数据推送逻辑与之前版本类似)
            # ... (确保使用正确的变量名和日志格式) ...
            # --- START: Episode End (Copied & adapted from previous Actor for completeness) ---
            episode_duration = time.time() - episode_start_time
            main_agent_for_log_name = env.agent_names[self.config.get('actor_main_seat_idx',0)]
            main_model_reward_this_episode = episode_raw_rewards.get(main_agent_for_log_name, 0.0)

            # 应用奖励变换（如果配置了）
            if self.config.get('use_normalized_reward', False) and len(env.agent_names) == 4:
                # ... (将之前提供的奖励变换逻辑粘贴到这里，作用于 episode_raw_rewards) ...
                # ... (然后用变换后的分数更新 episode_data[agent_name]['reward'] 的最后一个元素) ...
                # (这部分奖励变换逻辑需要从上一个回答中复制过来)
                self.logger.info(f"Episode {episode_num+1}: Applying reward normalization/transformation to final scores.")
                raw_scores_ordered_list = [episode_raw_rewards.get(name, 0.0) for name in env.agent_names]
                transformed_scores_ordered_list = [0.0] * 4
                min_raw_score = min(raw_scores_ordered_list)
                min_raw_score_count = raw_scores_ordered_list.count(min_raw_score)

                for idx, raw_score in enumerate(raw_scores_ordered_list):
                    if raw_score > 0: transformed_scores_ordered_list[idx] = np.sqrt(raw_score / 2.0)
                    elif raw_score < 0:
                        if raw_score == min_raw_score and min_raw_score_count == 1: transformed_scores_ordered_list[idx] = -2.0
                        else: transformed_scores_ordered_list[idx] = -1.0
                    else: transformed_scores_ordered_list[idx] = 0.0
                
                self.logger.debug(f"  Raw scores: {list(zip(env.agent_names, raw_scores_ordered_list))}")
                self.logger.debug(f"  Transformed scores: {list(zip(env.agent_names, transformed_scores_ordered_list))}")

                for idx, agent_name_key_reward in enumerate(env.agent_names):
                    if agent_name_key_reward in episode_data:
                        agent_reward_list_ref = episode_data[agent_name_key_reward]['reward']
                        if agent_reward_list_ref: # 如果有记录的步骤，最后一个奖励是最终得分
                            agent_reward_list_ref[-1] = transformed_scores_ordered_list[idx]

            self.logger.info(f"Actor {self.name}: Ep {episode_num+1} finished in {step_count} steps ({episode_duration:.2f}s). "
                             f"Main Agent ({main_agent_for_log_name}) Final Raw Reward: {main_model_reward_this_episode:.2f}.")
            
            if self.writer:
                try:
                    self.writer.add_scalar(f'Actor_{self.name}/Reward/MainAgentRawEp', main_model_reward_this_episode, total_actor_steps)
                    self.writer.add_scalar(f'Actor_{self.name}/Episode/Length', step_count, total_actor_steps)
                    # 可以记录变换后的奖励总和或每个 agent 的变换后奖励
                    for ag_name_tb, raw_score_tb in episode_raw_rewards.items(): # Log raw scores per agent
                        sane_ag_name_tb = "".join(c if c.isalnum() else '_' for c in str(ag_name_tb))
                        self.writer.add_scalar(f'Actor_{self.name}/RawReward_Detailed/Agent_{sane_ag_name_tb}_EpTotal', raw_score_tb, total_actor_steps)
                    self.writer.flush()
                except Exception as e_tb_actor:
                    self.logger.warning(f"Actor {self.name}: Failed to write to TensorBoard: {e_tb_actor}", exc_info=True)

            # GAE 和数据推送
            for agent_name_proc, agent_data_proc in episode_data.items():
                # 仅收集那些最新 policy 视角下的数据
                if current_episode_policy_keys[agent_name_proc] != 'latest_eval':
                    continue

                T = len(agent_data_proc.get('action', [])) 
                if T == 0: continue 
                # (确保所有列表长度都为 T 的检查)
                # ...
                try:
                    if len(agent_data_proc['reward']) > T: agent_data_proc['reward'].pop(0) # 示例对齐
                    if len(agent_data_proc['value']) == T+1:
                        values = np.array(agent_data_proc['value'][:T], dtype=np.float32)
                        next_values = np.array(agent_data_proc['value'][1:], dtype=np.float32)
                    elif len(agent_data_proc['value']) == T:
                        # 依照惯例, 最后第 T 时间步的 value 为 0
                        values = np.array(agent_data_proc['value'], dtype=np.float32)
                        next_values = np.zeros_like(values)
                        if T > 1: next_values[:-1] = values[1:]
                        # self.logger.warning(f"Value list length ({len(agent_data['value'])}) equals action length ({T}). Assuming V(s_T)=0 for {agent_name}.")
                    else:
                         self.logger.error(f"Value list length mismatch for {agent_name}. Skipping GAE.")
                         continue
                    # (GAE 计算和 data_to_push 构建逻辑，与您之前 Actor 代码中一致)
                    # ... 使用 agent_data_proc 中的数据 ...
                    obs_np_gae = np.stack(agent_data_proc['state']['observation']) # 使用局部变量避免覆盖外部 obs
                    mask_np_gae = np.stack(agent_data_proc['state']['action_mask'])
                    actions_np_gae = np.array(agent_data_proc['action'], dtype=np.int64)
                    rewards_np_gae = np.array(agent_data_proc['reward'], dtype=np.float32)
                    values_np_gae = np.array(agent_data_proc['value'], dtype=np.float32)
                    log_probs_np_gae = np.array(agent_data_proc['log_prob'], dtype=np.float32)

                    gamma = self.config.get('gamma', 0.99)
                    lambda_gae = self.config.get('lambda', 0.95)
                    
                    td_target = rewards_np_gae + gamma * next_values
                    td_delta = td_target - values
                    advantages = np.zeros_like(rewards_np_gae)
                    
                    adv = 0.0
                    for t in reversed(range(T)):
                        adv = td_delta[t] + gamma * lambda_gae * adv
                        advantages[t] = adv

                    data_to_push = {
                        'state': {'observation': obs_np_gae, 'action_mask': mask_np_gae },
                        'action': actions_np_gae, 'adv': advantages, 
                        'target': td_target, 'log_prob': log_probs_np_gae
                    }
                    self.replay_buffer.push(data_to_push)
                except Exception as e_gae_push:
                    self.logger.error(f"Error during GAE/push for {agent_name_proc}: {e_gae_push}. Skipping.", exc_info=True)
            # --- END: Episode End (Copied & adapted) ---

            if profiler: 
                profiler.disable()
                # ... (保存 profile 结果的代码，与您之前版本类似) ...
                ps = pstats.Stats(profiler).sort_stats('cumulative')
                profile_dir = os.path.join(self.config['log_base_dir'], "file_logs", self.config['experiment_name'], "actor_profiles")
                os.makedirs(profile_dir, exist_ok=True)
                profile_log_path = os.path.join(profile_dir, f"{self.name}_ep0_profile.prof")
                ps.dump_stats(profile_log_path)
                self.logger.info(f"Actor {self.name}: Profile results for first episode saved to {profile_log_path}")


        # --- Actor 结束 ---
        self.logger.info(f"Actor {self.name} finished {episodes_per_actor_run} episodes. Exiting.")
        if self.writer: 
            self.writer.close()

        # if self.shutdown_event and not self.shutdown_event.is_set(): # 如果循环正常结束，但主程序未发关闭信号
        #     self.logger.info(f"Actor {self.name}: Episode loop finished, now setting shutdown_event for worker.")
        #     self.shutdown_event.set() # 主动设置关闭事件，让工作线程退出

        if self.response_worker_thread and self.response_worker_thread.is_alive():
            self.logger.info(f"Actor {self.name}): Waiting for response worker thread to join...")
            self.response_worker_thread.join(timeout=5.0) # 给工作线程一些时间退出
            if self.response_worker_thread.is_alive():
                self.logger.warning(f"Actor {self.name}: Response worker thread did not join in time.")