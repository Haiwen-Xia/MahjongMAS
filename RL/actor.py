import torch
# 导入多进程模块，Actor 将作为一个独立的进程运行
from multiprocessing import Process
import numpy as np
import time # 引入 time 模块用于可能的延迟或计时
import logging # 引入日志模块
import os # 引入 os 模块用于路径操作

# 导入自定义模块
from replay_buffer import ReplayBuffer       # 用于存储和提供训练数据的经验回放缓冲区
from model_pool_extended import ModelPoolClient        # 用于从中央模型池获取最新模型参数的客户端
from env import MahjongGBEnv                 # 麻将游戏环境
from feature import FeatureAgent             # 用于处理麻将特征的 Agent
from model import ResNet34AC                 # Actor 使用的神经网络模型定义 (假设包含策略和价值输出)
from torch.utils.tensorboard import SummaryWriter # 引入 TensorBoard
from utils import setup_process_logging_and_tensorboard # 用于设置日志和 TensorBoard 的工具函数

import cProfile, pstats # 用于记录效率信息

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
        # 设置 Actor 的名字，便于日志区分，如果未提供则使用 PID (Process ID)
        # 注意: 直接访问 self.pid 在 __init__ 中可能还不可用，因为它在 start() 后才分配
        # 但可以在 run() 方法中使用 self.pid，或者依赖传入的 config['name']
        self.name = config.get('name', 'Actor-?') # Fallback name if not provided

        # 在 __init__ 中配置 TensorBoard writer 可能不适合多进程
        # 最好在 run() 方法中初始化
        self.logger = None
        self.writer = None

        # 添加 benchmark 模型
        self.benchmark_models = {}
        self.benchmark_policy_paths = self.config.get("benchmark_policies", {})
        for name, path in self.benchmark_policy_paths.items():
            if os.path.isfile(path):
                try:
                    model_instance = ResNet34AC(self.config['in_channels'])
                    # self._load_state_dict_to_model is a helper you have
                    state_dict = torch.load(path)
                    self._load_state_dict_to_model(model_instance, state_dict)
                    self.benchmark_models[name] = model_instance
                    # self.logger.info(f"Actor {self.name}: Loaded benchmark policy '{name}' from {path}")
                except Exception as e:
                    # self.logger.error(f"Actor {self.name}: Failed to load benchmark policy '{name}' from {path}: {e}", exc_info=True)
                    pass
            else:
                # self.logger.warning(f"Actor {self.name}: Benchmark policy path not found for '{name}': {path}")
                pass




    def _load_state_dict_to_model(self, model_instance, state_dict_to_load):
        """Helper to load state_dict, handling potential nested dicts from checkpoints."""
        if isinstance(state_dict_to_load, dict) and 'model_state_dict' in state_dict_to_load:
            model_instance.load_state_dict(state_dict_to_load['model_state_dict'])
        else:
            model_instance.load_state_dict(state_dict_to_load)
        model_instance.eval() # Set to eval mode after loading
        
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

        # 设置 PyTorch 在该进程中使用的线程数为 1，避免多进程场景下线程过多导致竞争和性能下降
        try:
            torch.set_num_threads(1)
            self.logger.info(f"PyTorch num_threads set to 1.")
        except Exception as e:
            self.logger.warning(f"Failed to set torch num_threads: {e}")

        # 连接到模型池 (Model Pool)
        try: # 添加异常处理，确保连接失败时能看到错误
            model_pool = ModelPoolClient(self.config['model_pool_name'])
            self.logger.info(f"Connected to Model Pool '{self.config['model_pool_name']}'.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Model Pool - {e}. Actor exiting.")
            if self.writer: self.writer.close()
            return # 连接失败则退出

        model_latest = None
        try:
            model_latest = ResNet34AC(self.config['in_channels'])
        except Exception as e:
            self.logger.error(f"Failed to create main model instance: {e}. Actor exiting.", exc_info=True)
            if self.writer: self.writer.close()
            return

        # 加载初始主模型参数
        latest_model_version_id = "N/A"
        try:
            latest_version_meta = model_pool.get_latest_model_metadata()
            if latest_version_meta:
                latest_model_version_id = latest_version_meta.get('id', 'N/A')
                self.logger.info(f"Got latest model metadata from pool: ID {latest_model_version_id}")
                state_dict = model_pool.load_model_parameters(latest_version_meta)
                if state_dict:
                    self._load_state_dict_to_model(model_latest, state_dict)
                    self.logger.info(f"Loaded initial parameters for model_latest, version {latest_model_version_id}")
                else:
                    self.logger.error(f"Failed to load parameters for latest model ID {latest_model_version_id}. Actor exiting.")
                    return
            else:
                self.logger.error("Failed to get any model metadata from pool at startup. Actor exiting.")
                return
        except Exception as e:
            self.logger.error(f"Failed to load initial model_latest: {e}. Actor exiting.", exc_info=True)
            if self.writer: self.writer.close()
            return

        env = None
        try:
            env_config = self.config.get('env_config', {})
            if 'agent_clz' not in env_config: # Default if not specified
                 # Assuming FeatureAgent is the one compatible with MahjongGBEnv's internal processing
                env_config['agent_clz'] = FeatureAgent 
            self.logger.info(f"Actor using env_config: {env_config}")
            env = MahjongGBEnv(config=env_config)
            self.logger.info(f"Mahjong environment created with agent class: {env_config['agent_clz'].__name__}.")
        except Exception as e:
            self.logger.error(f"Failed to create Mahjong environment: {e}. Actor exiting.", exc_info=True)
            if self.writer: self.writer.close()
            return

        # --- 对手模型管理 ---
        # policies: 字典，存储当前 episode 中每个 agent_name -> model_instance 的映射
        policies = {} 
        # opponent_model_instances: 字典，存储为对手席位专门创建（或复用）的模型实例
        # 键是 agent_name (例如 "player_1"), 值是 ResNet34AC 实例
        opponent_model_instances = {} 
        # opponent_model_ids: 字典，存储对手席位当前使用的模型ID (如果是从pool加载的)
        opponent_model_ids = {} 

        # 从配置获取对手选择参数
        p_opponent_historical = self.config.get('p_opponent_historical', 0.2) # 例如 20% 概率使用历史模型
        opponent_sampling_k = self.config.get('opponent_sampling_k', 8)
        opponent_model_change_interval = self.config.get('opponent_model_change_interval', 1) # 每1个episode就可能换
        actor_update_model_latest_interval = self.config.get('actor_model_change_interval', 1)

        total_actor_steps = 0
        episodes_per_actor_run = self.config.get('episodes_per_actor', 1000) # 总共运行多少局

        for episode in range(episodes_per_actor_run):
            if self.config.get('enable_profiling', False) and episode == 0: # 只 profile 第一个 episode
                pr = cProfile.Profile()
                pr.enable()

            episode_start_time = time.time()

            # --- 1. 定期更新主模型 (model_latest) ---
            if episode % actor_update_model_latest_interval == 0:
                try:
                    current_pool_latest_meta = model_pool.get_latest_model_metadata()
                    if current_pool_latest_meta:
                        current_pool_latest_id = current_pool_latest_meta.get('id', 'N/A')
                        is_newer = False
                        # 版本比较逻辑 (确保 ID 类型一致或可比较)
                        if isinstance(current_pool_latest_id, type(latest_model_version_id)):
                            if isinstance(current_pool_latest_id, (int, float)):
                                is_newer = current_pool_latest_id > latest_model_version_id
                            elif isinstance(current_pool_latest_id, str):
                                # 假设版本ID越高越新；对于字符串ID，可能需要更复杂的比较逻辑
                                is_newer = current_pool_latest_id > latest_model_version_id 
                        
                        if is_newer and current_pool_latest_id != latest_model_version_id: # 避免不必要的加载
                            # self.logger.info(f"Actor {self.name}: Found newer main model ID {current_pool_latest_id} (current: {latest_model_version_id}). Updating model_latest.")
                            new_state_dict = model_pool.load_model_parameters(current_pool_latest_meta)
                            if new_state_dict:
                                self._load_state_dict_to_model(model_latest, new_state_dict)
                                latest_model_version_id = current_pool_latest_id
                                # self.logger.info(f"Actor {self.name}: Updated model_latest to version {latest_model_version_id}.")
                            else:
                                self.logger.warning(f"Actor {self.name}: Failed to load parameters for new main model ID {current_pool_latest_id}. Continuing with old version.")
                        elif current_pool_latest_id != latest_model_version_id and latest_model_version_id != "N/A":
                             self.logger.debug(f"Actor {self.name}: Pool latest ID {current_pool_latest_id} is not considered newer than local {latest_model_version_id}.")
                    else:
                        self.logger.warning(f"Actor {self.name}: Could not get latest model metadata from pool for updating model_latest.")
                except Exception as e:
                    self.logger.warning(f"Actor {self.name}: Error checking/updating model_latest: {e}", exc_info=True)
            
            
            # --- 2. 定期决定并设置本轮对局中所有席位的策略 ---
            if episode % opponent_model_change_interval == 0:
                # self.logger.info(f"Actor {self.name}, Episode {episode + 1}: Re-evaluating/setting policies for all seats.")
                policies.clear()
                opponent_model_ids.clear() # 清除上一轮的对手ID记录

                main_agent_seat_idx_in_env = np.random.randint(0, 4)
                main_agent_name = env.agent_names[main_agent_seat_idx_in_env]
                for seat_idx in range(4): # 遍历所有0, 1, 2, 3号席位
                    current_env_agent_name = env.agent_names[seat_idx] # 获取该席位对应的 player_name

                    if seat_idx == main_agent_seat_idx_in_env:
                        policies[current_env_agent_name] = model_latest
                        opponent_model_ids[current_env_agent_name] = latest_model_version_id
                        # self.logger.info(f"  Seat {seat_idx} ({current_env_agent_name}): Using main learning model (ID: {latest_model_version_id})")
                    else: # 这是对手席位

                        use_benchmark_opponent = np.random.rand() < self.config.get('prob_opponent_is_benchmark', 0.0)

                        if use_benchmark_opponent and self.benchmark_models:
                            # 选择一个基准模型 (可以简单随机选，或按配置的概率选)
                            # Example: choose 'initial_il_policy' with higher probability
                            chosen_benchmark_name = None
                            if 'initial_il_policy' in self.benchmark_models and np.random.rand() < self.config.get('prob_benchmark_is_initial_il', 1.0):
                                chosen_benchmark_name = 'initial_il_policy'
                            else: # Fallback to random benchmark or other logic
                                available_benchmarks = list(self.benchmark_models.keys())
                                if available_benchmarks:
                                    chosen_benchmark_name = np.random.choice(available_benchmarks)

                            if chosen_benchmark_name:
                                policies[current_env_agent_name] = self.benchmark_models[chosen_benchmark_name]
                                opponent_model_ids[current_env_agent_name] = f"benchmark_{chosen_benchmark_name}"
                                # self.logger.info(f"  Seat {seat_idx} ({current_env_agent_name}): Using BENCHMARK model '{chosen_benchmark_name}'")
                            else: # Should not happen if self.benchmark_models is not empty
                                use_benchmark_opponent = False # Fallback to pool/latest

                        elif np.random.rand() < p_opponent_historical:
                            # self.logger.debug(f"  Seat {seat_idx} ({current_env_agent_name}): Attempting to sample historical opponent (k={opponent_sampling_k}).")
                            exclude_ids_list = [latest_model_version_id] if latest_model_version_id != "N/A" else None
                            
                            sampled_meta = model_pool.sample_model_metadata(
                                strategy='latest_k', 
                                k=opponent_sampling_k
                                # exclude_ids=exclude_ids_list,
                                # require_distinct_from_latest=True
                            )
                            
                            historical_model_loaded_successfully = False
                            if sampled_meta:
                                sampled_id = sampled_meta.get('id', 'N/A')
                                params = model_pool.load_model_parameters(sampled_meta)
                                if params:
                                    if current_env_agent_name not in opponent_model_instances:
                                        opponent_model_instances[current_env_agent_name] = ResNet34AC(self.config['in_channels'])
                                        # self.logger.info(f"    Created new model instance for opponent {current_env_agent_name}")
                                    
                                    self._load_state_dict_to_model(opponent_model_instances[current_env_agent_name], params)
                                    policies[current_env_agent_name] = opponent_model_instances[current_env_agent_name]
                                    opponent_model_ids[current_env_agent_name] = sampled_id
                                    # self.logger.info(f"  Seat {seat_idx} ({current_env_agent_name}): Using sampled historical model (ID: {sampled_id})")
                                    historical_model_loaded_successfully = True
                                else:
                                    self.logger.warning(f"    Failed to load params for sampled ID {sampled_id}.")
                            else:
                                self.logger.warning(f"    Failed to sample historical model from pool.")
                            
                            if not historical_model_loaded_successfully:
                                self.logger.warning(f"    Seat {seat_idx} ({current_env_agent_name}) will use model_latest as fallback for historical.")
                                policies[current_env_agent_name] = model_latest 
                                opponent_model_ids[current_env_agent_name] = latest_model_version_id
                        else:
                            # self.logger.info(f"  Seat {seat_idx} ({current_env_agent_name}): Using main learning model (ID: {latest_model_version_id}) as opponent.")
                            policies[current_env_agent_name] = model_latest
                            opponent_model_ids[current_env_agent_name] = latest_model_version_id
            
            # --- 运行一个 episode 并收集数据 ---
            # self.logger.info(f"Actor {self.name}, Ep {episode+1}/{episodes_per_actor_run}. P1={latest_model_version_id}, "
            #                  f"P2={opponent_model_ids.get(env.agent_names[1], 'N/A')}, "
            #                  f"P3={opponent_model_ids.get(env.agent_names[2], 'N/A')}, "
            #                  f"P4={opponent_model_ids.get(env.agent_names[3], 'N/A')}")

            try: 
                obs = env.reset() 
            except Exception as e:
                self.logger.error(f"Failed to reset environment for episode {episode+1}: {e}. Skipping episode.", exc_info=True)
                continue 

            episode_data = {agent_name: {
                'state' : {'observation': [], 'action_mask': []},
                'action' : [], 'reward' : [], 'value' : [], 'log_prob' : []
            } for agent_name in env.agent_names}
            episode_raw_rewards = {agent_name: 0.0 for agent_name in env.agent_names}

            done = False 
            step_count = 0 
            record_data_for_agent_this_step = {}

            # 在一个 episode 内部循环，直到结束
            while not done:
                step_count += 1
                total_actor_steps += 1 # 增加总步数计数器
                actions = {}
                values = {}

                record_step_for_agent = {} 

                # 为当前 obs 字典中的每个 agent 获取动作
                current_agents = list(obs.keys()) # 获取当前需要行动的 agent
                for agent_name in current_agents:
                    if agent_name not in episode_data: # 防御性检查
                        self.logger.warning(f"Agent '{agent_name}' appeared mid-episode unexpectedly. Skipping.")
                        continue
                    
                    agent_data = episode_data[agent_name] # 获取该 agent 的数据存储区
                    state = obs[agent_name]             # 获取该 agent 当前的观察状态

                    # 检查 action_mask
                    action_mask_numpy = np.array(state['action_mask'])
                    num_valid_actions = action_mask_numpy.sum()

                    # 根据配置决定是否过滤此步骤的数据
                    should_filter_this_agent_step = self.config.get('filter_single_action_steps', True) and \
                                                    num_valid_actions <= 1
                    
                    record_step_for_agent[agent_name] = not should_filter_this_agent_step

                    if record_step_for_agent[agent_name]:
                        # 如果记录，则存储状态信息
                        agent_data['state']['observation'].append(state['observation'])
                        agent_data['state']['action_mask'].append(action_mask_numpy) # 存储 numpy 格式的 mask
                        self.logger.debug(f"Step {step_count} for agent {agent_name}: Recording data (num_valid_actions: {num_valid_actions}).")
                    else:
                        self.logger.debug(f"Step {step_count} for agent {agent_name}: Filtering data (num_valid_actions: {num_valid_actions}).")

                    # 存储原始观察状态和动作掩码 (numpy 格式)
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])

                    # 准备模型输入
                    try:
                        state_obs_tensor = torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0)
                        state_mask_tensor = torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
                        model_input = {'obs': {'observation': state_obs_tensor, 'action_mask': state_mask_tensor}}
                    except Exception as e:
                         logger.error(f"Error converting state to tensor for {agent_name} at step {step_count}: {e}. Skipping episode.")
                         # 可能需要更精细的处理，比如仅跳过这个 agent 或结束 episode
                         done = True # 强制结束 episode
                         break # 跳出 for agent_name 循环

                    # 模型推理获取动作
                    try:
                        policies[agent_name].eval() # 设置为评估模式
                        with torch.no_grad():
                            logits, value = policies[agent_name](model_input) # 获取动作 logits 和状态价值 V(s)
                            action_dist = torch.distributions.Categorical(logits=logits)

                            action_tensor = action_dist.sample() # Sample action as a tensor
                            action_item = action_tensor.item()   # Get Python number for env
                            log_prob_item = action_dist.log_prob(action_tensor).item()

                            value = value.item() # 获取价值
                    except Exception as e:
                         self.logger.error(f"Error during model inference for {agent_name} at step {step_count}: {e}. Skipping episode.")
                         done = True
                         break

                    # 存储选择的动作、价值（和 log_prob）
                    actions[agent_name] = action_item
                    values[agent_name] = value

                    if record_step_for_agent[agent_name]:
                        # 如果记录，则存储动作、价值和log_prob
                        agent_data['action'].append(actions[agent_name])
                        agent_data['value'].append(values[agent_name])
                        agent_data['log_prob'].append(log_prob_item)

                if done: break # 如果在动作选择中出错，跳出 while 循环

                # --- 与环境交互 ---
                try:
                    next_obs, rewards, done_info = env.step(actions) # done_info 可能包含 __all__
                except Exception as e:
                    self.logger.error(f"Error during env.step at step {step_count}: {e}. Ending episode.")
                    # 记录已收集的数据可能仍然有用
                    done = True # 强制标记结束
                    # break # 跳出 while 循环，让后续处理逻辑执行
                
                # 处理奖励和 done 状态
                if not done: # 仅在未结束时处理奖励和状态更新
                     # 存储这一步获得的奖励并累加总奖励
                     for agent_name in rewards:
                         if agent_name in episode_data:
                             episode_data[agent_name]['reward'].append(rewards[agent_name])
                             episode_raw_rewards[agent_name] += rewards[agent_name]
                         
                     # 更新观察状态
                     obs = next_obs
                     
                     # 处理 done 信号 (可能是字典)
                     if isinstance(done_info, dict):
                         if done_info.get("__all__", False):
                             done = True
                         else:
                             # 多智能体环境中，如果不是所有人都结束，游戏继续
                             # 但 obs 字典可能只包含未结束的智能体
                             # 更新 obs 只包含需要下一步动作的 agent
                             obs = {k: v for k, v in next_obs.items() if not done_info.get(k, False)}
                             if not obs: # 如果所有 agent 都结束了，虽然 __all__ 不是 True
                                  self.logger.warning("Episode ended because all agents were done individually, but __all__ was not True.")
                                  done = True
                     elif isinstance(done_info, bool): # 如果 done 是布尔值
                          done = done_info
                     else:
                          self.logger.error(f"Unexpected type for 'done' signal from env.step: {type(done_info)}. Ending episode.")
                          done = True

            # --- Episode 结束 ---
            episode_duration = time.time() - episode_start_time
            episode_transformed_rewards = {agent_name: 0.0 for agent_name in env.agent_names}

            if self.config['use_normalized_reward']:
                self.logger.info(f"Episode {episode+1}: Applying reward normalization/transformation.")
                
                # 1. 从 episode_raw_rewards 获取所有玩家的原始最终得分
                #    env.agent_names 提供了标准的玩家顺序，例如 ['player_0', 'player_1', 'player_2', 'player_3']
                #    我们需要确保得到所有4个玩家的得分，即使某个玩家可能因为某种原因没有分数记录 (默认为0)
                
                # 检查 env.agent_names 是否包含4个玩家
                if len(env.agent_names) != 4:
                    self.logger.error(f"Reward normalization expects 4 players, but found {len(env.agent_names)}. Skipping normalization.")
                else:
                    raw_scores_ordered_list = [episode_raw_rewards.get(name, 0.0) for name in env.agent_names]
                    self.logger.debug(f"  Raw scores for normalization: {list(zip(env.agent_names, raw_scores_ordered_list))}")

                    transformed_scores_ordered_list = [0.0] * 4
                    
                    min_raw_score = min(raw_scores_ordered_list)
                    min_raw_score_count = raw_scores_ordered_list.count(min_raw_score)

                    for idx, raw_score in enumerate(raw_scores_ordered_list):
                        if raw_score > 0:
                            transformed_scores_ordered_list[idx] = np.sqrt(raw_score / 2.0)
                        elif raw_score < 0:
                            # 检查是否是唯一的严格最小负分 (点炮者)
                            if raw_score == min_raw_score and min_raw_score_count == 1:
                                transformed_scores_ordered_list[idx] = -2.0
                            else: # 其他负分玩家
                                transformed_scores_ordered_list[idx] = -1.0
                        else: # raw_score == 0
                            transformed_scores_ordered_list[idx] = 0.0
                    
                    self.logger.debug(f"  Transformed scores: {list(zip(env.agent_names, transformed_scores_ordered_list))}")


                    for idx, agent_name_key in enumerate(env.agent_names):
                        episode_transformed_rewards[agent_name_key] = transformed_scores_ordered_list[idx]

                    # 2. 更新 episode_data 中每个玩家的奖励序列
                    #    假设：只有奖励序列的最后一个元素代表该局的最终得分，之前都是0。
                    #    如果您的 env.step() 会在中间步骤返回非零奖励，这里的逻辑需要调整。
                    for idx, agent_name_key in enumerate(env.agent_names):
                        if agent_name_key in episode_data:
                            agent_reward_list = episode_data[agent_name_key]['reward']
                            if agent_reward_list: # 确保奖励列表不为空 (即该玩家有记录的步骤)
                                # 替换最后一个元素为变换后的得分
                                # 这假设了原始的最终得分已经被正确地作为最后一个元素添加到了这个列表
                                original_terminal_reward = agent_reward_list[-1] # 用于日志
                                agent_reward_list[-1] = transformed_scores_ordered_list[idx]
                                self.logger.debug(f"  Updated terminal reward for {agent_name_key}: {original_terminal_reward} -> {transformed_scores_ordered_list[idx]}")
                            elif len(agent_data_proc.get('action', [])) > 0 : # 有动作但奖励列表为空，这不应该发生
                                 self.logger.warning(f"  Agent {agent_name_key} has actions but reward list is empty. Cannot apply normalized terminal reward.")
                        else:
                            # 如果某个 agent_name 不在 episode_data 中 (例如，因为过滤或从未行动)
                            # 则无需操作。GAE 计算只会处理 episode_data 中存在的 agent。
                            pass 
            else:
                episode_transformed_rewards = episode_raw_rewards

            lastest_model_reward = episode_transformed_rewards[main_agent_name]
            self.logger.info(f"Episode {episode+1} finished in {step_count} steps ({episode_duration:.2f}s). "
                        f"Model Version {latest_model_version_id}. Latest Model Reward: {lastest_model_reward:.2f}.")
            
            
            # --- 记录到 TensorBoard ---
            if self.writer:
                try:
                     # 使用 actor 内部的总步数或 episode 数作为 x 轴
                     # 如果能获取全局步数会更好，这里用 episode
                     self.writer.add_scalar('Reward/LastestModel', lastest_model_reward, episode + 1)
                     self.writer.add_scalar('Episode/Length', step_count, episode + 1)
                     # 可以记录每个 agent 的奖励
                     for agent_id, total_reward in episode_transformed_rewards.items():
                          self.writer.add_scalar(f'Reward/Agent_{agent_id}_EpisodeTotal', total_reward, episode + 1)
                     self.writer.flush() # 确保写入
                except Exception as e:
                     self.logger.warning(f"Failed to write to TensorBoard: {e}")

            # --- 对收集到的 episode 数据进行后处理，计算 GAE 和 TD-Target ---
            # (这部分逻辑与之前的版本类似，假设已根据需要调整和确认对齐方式)
            for agent_name, agent_data in episode_data.items():
                if agent_name != main_agent_name and opponent_model_ids[agent_name] != latest_model_version_id:
                    # 如果不是我们关心的学习智能体，则直接跳过后续所有处理
                    # self.logger.debug(f"Skipping data processing for non-learning agent: {agent_name}") # 可选的调试日志
                    continue

                
                T = len(agent_data['action'])
                if T == 0: continue # 跳过没有动作的 agent/episode

                # --- 数据对齐检查与准备 (同前一版本，需要仔细核对) ---
                # ... (此处省略详细的数据对齐和准备代码，假设它存在且逻辑正确) ...
                # 关键是准备好长度为 T 的 obs, mask, actions, rewards
                # 以及长度为 T 的 values (V(s_0..T-1)) 和 next_values (V(s_1..T))
                # 这里仅作示意，需要替换为实际的对齐和转换代码
                try:
                    if len(agent_data['reward']) > T: agent_data['reward'].pop(0) # 示例对齐
                    if len(agent_data['value']) == T+1:
                        values = np.array(agent_data['value'][:T], dtype=np.float32)
                        next_values = np.array(agent_data['value'][1:], dtype=np.float32)
                    elif len(agent_data['value']) == T:
                        # 依照惯例, 最后第 T 时间步的 value 为 0
                        values = np.array(agent_data['value'], dtype=np.float32)
                        next_values = np.zeros_like(values)
                        if T > 1: next_values[:-1] = values[1:]
                        # self.logger.warning(f"Value list length ({len(agent_data['value'])}) equals action length ({T}). Assuming V(s_T)=0 for {agent_name}.")
                    else:
                         self.logger.error(f"Value list length mismatch for {agent_name}. Skipping GAE.")
                         continue

                    obs = np.stack(agent_data['state']['observation'][:T])
                    mask = np.stack(agent_data['state']['action_mask'][:T])
                    actions = np.array(agent_data['action'], dtype=np.int64)
                    rewards = np.array(agent_data['reward'][:T], dtype=np.float32)
                    log_probs_np = np.array(agent_data['log_prob'], dtype=np.float32) # PPO 需要

                    # --- GAE 计算 (同前一版本) ---
                    gamma = self.config.get('gamma', 0.99)
                    lambda_gae = self.config.get('lambda', 0.95)
                    td_target = rewards + gamma * next_values
                    td_delta = td_target - values
                    advantages = np.zeros_like(rewards)
                    
                    adv = 0.0
                    for t in reversed(range(T)):
                        adv = td_delta[t] + gamma * lambda_gae * adv
                        advantages[t] = adv

                except Exception as e:
                     self.logger.error(f"Error during post-processing for {agent_name}: {e}. Skipping data push.")
                     continue # 跳过这个 agent 的数据推送

                # --- 推送数据到 Replay Buffer ---
                try:
                    # 确保所有数据的第一个维度都是 T (时间步数量)
                    data_to_push = {
                        'state': {
                            'observation': obs,  # (T, C, H, W) or (T, FeatureDim)
                            'action_mask': mask # (T, ActionDim)
                        },
                        'action': actions,       # (T,)
                        'adv': advantages,     # (T,) GAE 优势值
                        'target': td_target,    # (T,) TD 目标价值 (或称为 Return)
                        'log_prob': log_probs_np # PPO 需要
                    }
                    self.replay_buffer.push(data_to_push)
                    # logger.debug(f"Pushed {T} steps of data for {agent_name} to replay buffer.") # Debug level log
                except Exception as e:
                     self.logger.error(f"Error pushing data to replay buffer for {agent_name}: {e}")

            if self.config.get('enable_profiling', False) and episode == 0:
                pr.disable()
                ps = pstats.Stats(pr).sort_stats('cumulative')
                profile_log_path = os.path.join(self.config['log_base_dir'], 'file_logs', self.config['experiment_name'], f"{self.name}_profile.log")
                with open(profile_log_path, 'w') as f_profile:
                    ps_profile = pstats.Stats(pr, stream=f_profile).sort_stats('cumulative')
                    ps_profile.print_stats(30) # 打印耗时最多的30个函数
                self.logger.info(f"Profile results saved to {profile_log_path}")

        # --- Actor 结束 ---
        self.logger.info(f"Finished {self.config['episodes_per_actor']} episodes. Exiting.")
        if self.writer: # 关闭 TensorBoard writer
            self.writer.close()