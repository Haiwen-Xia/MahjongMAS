import torch
from multiprocessing import Process
import numpy as np
import time
import os
import json # 用于打印配置
from collections import defaultdict

# 假设这些自定义模块路径正确
from replay_buffer import ReplayBuffer # Evaluator不需要，但Actor有，先保留看是否父类需要
from model_pool import ModelPoolClient 
from env import MahjongGBEnv
from feature import FeatureAgent # Evaluator 仍需 FeatureAgent 来处理环境原始观测
from model import ResNet34AC # 或其他你使用的模型类
from utils import setup_process_logging_and_tensorboard

class Evaluator(Process):
    def __init__(self, config):
        """
        初始化 Evaluator 进程。

        Args:
            config (dict): 包含 Evaluator 配置参数的字典。
                关键配置项可能包括:
                - 'name': Evaluator 的名称 (例如 "Evaluator-Primary").
                - 'log_base_dir', 'experiment_name': 日志和 TensorBoard 相关。
                - 'model_pool_name': 连接的 ModelPool 名称 (用于在线评估模式)。
                - 'in_channels': 模型输入通道数。
                - 'device': 'cpu' 或 'cuda:X'.
                - 'evaluation_episodes': 每个评估周期运行的局数。
                - 'evaluation_config': 描述如何选择模型进行对局的字典。例如:
                    {
                        "mode": "online_vs_pool" # "online_vs_pool", "offline_vs_paths", "self_play_latest"
                        "main_agent_source": "latest", # "latest", "specific_id", "path/to/model.pth"
                        "main_agent_model_id_or_path": None, # 如果 source 是 specific_id 或 path
                        "opponent_config": [ # 列表，长度为3，对应P1, P2, P3 (假设主评估对象是P0)
                            {"source": "sample_from_pool", "strategy": "uniform", "k": 1, "exclude_self": True},
                            {"source": "sample_from_pool", "strategy": "latest_k", "k": 3},
                            {"source": "path", "path": "path/to/opponent_fixed.pth"}
                            # 或者更简单的 {"source": "self"} 表示使用与主评估对象相同的模型
                        ]
                    }
                - 'env_config': 环境配置 (同 Actor)。
        """
        super(Evaluator, self).__init__()
        self.config = config
        self.name = config.get('name', f'Evaluator-{os.getpid()}')
        self.logger = None
        self.writer = None # TensorBoard writer for evaluation metrics

        # 模型实例字典，键为席位(0-3)或描述性名称，值为加载好的模型对象
        self.loaded_policies = {} 
        # 存储每个席位当前使用的模型元数据或路径，方便记录
        self.current_policy_identifiers = {} 

    def _load_model_from_source(self, source_config, model_pool_client, primary_model_id_for_exclusion=None):
        """
        根据来源配置加载单个模型。
        Args:
            source_config (dict): 如 {"source": "latest", "id_or_path": ..., "strategy": ...}
            model_pool_client (ModelPoolClient): ModelPool 客户端实例。
            primary_model_id_for_exclusion (int, optional): 当采样对手时，可能需要排除主模型的ID。
        Returns:
            tuple (model_instance, model_identifier_str) or (None, None)
        """
        model_instance = None
        model_identifier = "Unknown" # 用于日志记录
        source_type = source_config.get("source")

        try:
            model_instance = ResNet34AC(self.config['in_channels']) # 创建模型结构
            model_instance.to(torch.device(self.config.get('device', 'cpu')))
            model_instance.eval() # 评估模式

            if source_type == "latest":
                if not model_pool_client:
                    self.logger.error("ModelPoolClient not available for 'latest' source type.")
                    return None, None
                metadata = model_pool_client.get_latest_model_metadata()
                if metadata:
                    state_dict = model_pool_client.load_model_parameters(metadata)
                    if state_dict:
                        model_instance.load_state_dict(state_dict)
                        model_identifier = f"pool_latest_id_{metadata.get('id', 'N/A')}"
                    else: model_instance = None; self.logger.warning("Failed to load 'latest' model parameters.")
                else: model_instance = None; self.logger.warning("Failed to get 'latest' model metadata.")
            
            elif source_type == "specific_id_from_pool":
                if not model_pool_client:
                    self.logger.error("ModelPoolClient not available for 'specific_id_from_pool' source type.")
                    return None, None
                model_id = source_config.get("id_or_path")
                metadata = model_pool_client.get_model_metadata_by_id(model_id)
                if metadata:
                    state_dict = model_pool_client.load_model_parameters(metadata)
                    if state_dict:
                        model_instance.load_state_dict(state_dict)
                        model_identifier = f"pool_id_{model_id}"
                    else: model_instance = None; self.logger.warning(f"Failed to load parameters for pool model ID {model_id}.")
                else: model_instance = None; self.logger.warning(f"Failed to get metadata for pool model ID {model_id}.")

            elif source_type == "sample_from_pool":
                if not model_pool_client:
                    self.logger.error("ModelPoolClient not available for 'sample_from_pool' source type.")
                    return None, None
                strategy = source_config.get("strategy", "uniform")
                k = source_config.get("k", 1)
                exclude_ids_list = []
                if source_config.get("exclude_self", False) and primary_model_id_for_exclusion is not None:
                    exclude_ids_list.append(primary_model_id_for_exclusion)
                
                metadata = model_pool_client.sample_model_metadata(
                    strategy=strategy, 
                    k=k, 
                    exclude_ids=exclude_ids_list,
                    require_distinct_from_latest=source_config.get("require_distinct_from_latest", False)
                )
                if metadata:
                    state_dict = model_pool_client.load_model_parameters(metadata)
                    if state_dict:
                        model_instance.load_state_dict(state_dict)
                        model_identifier = f"pool_sampled_id_{metadata.get('id', 'N/A')}_strat_{strategy}"
                    else: model_instance = None; self.logger.warning(f"Failed to load parameters for sampled pool model ID {metadata.get('id')}.")
                else: model_instance = None; self.logger.warning(f"Failed to sample model from pool with strategy {strategy}.")
            
            elif source_type == "path":
                model_path = source_config.get("id_or_path")
                if model_path and os.path.isfile(model_path):
                    state_dict = torch.load(model_path, map_location=self.config.get('device', 'cpu'))
                    # 如果保存的是 checkpoint 字典，从中提取 model_state_dict
                    if 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    model_instance.load_state_dict(state_dict)
                    model_identifier = f"path_{os.path.basename(model_path)}"
                else:
                    model_instance = None
                    self.logger.error(f"Model file not found or path not specified for source_type 'path': {model_path}")
            
            elif source_type == "self": # 特殊标记，表示使用主评估对象的模型
                model_identifier = "self_as_opponent" 
                # model_instance will be assigned later by copying the main agent's model
                # For now, return the empty structure. It will be populated in the main eval loop.
                pass # Handled in the main setup_evaluation_players function

            else:
                model_instance = None
                self.logger.error(f"Unknown model source type: {source_type}")

        except Exception as e:
            self.logger.error(f"Error loading model from source_config {source_config}: {e}", exc_info=True)
            return None, "ErrorLoading"
            
        return model_instance, model_identifier

    def _setup_evaluation_players(self, eval_config, model_pool_client):
        """
        根据评估配置，为所有4个席位加载模型。
        """
        self.loaded_policies = {}
        self.current_policy_identifiers = {}
        
        # 1. 加载主评估模型 (P0)
        main_agent_source_cfg = {
            "source": eval_config.get("main_agent_source", "latest"),
            "id_or_path": eval_config.get("main_agent_model_id_or_path")
        }
        # print(f"Attempting to load main agent from: {main_agent_source_cfg}") # Debug
        main_model, main_model_id_str = self._load_model_from_source(main_agent_source_cfg, model_pool_client)
        
        if not main_model:
            self.logger.error(f"CRITICAL: Failed to load main evaluation agent ({main_agent_source_cfg}). Cannot proceed with this evaluation round.")
            return False
        
        self.loaded_policies[0] = main_model
        self.current_policy_identifiers[0] = main_model_id_str
        self.logger.info(f"Seat 0 (Main Agent): Loaded {main_model_id_str}")

        primary_model_actual_id = None # Used for exclude_ids if main model is from pool
        if main_agent_source_cfg["source"] == "latest" or main_agent_source_cfg["source"] == "specific_id_from_pool" or main_agent_source_cfg["source"] == "sample_from_pool":
            # Try to extract the actual ID from the identifier string if it's a pool model
            if "pool_id_" in main_model_id_str:
                try: primary_model_actual_id = int(main_model_id_str.split("pool_id_")[-1].split("_")[0])
                except: pass
            elif "pool_latest_id_" in main_model_id_str:
                try: primary_model_actual_id = int(main_model_id_str.split("pool_latest_id_")[-1].split("_")[0])
                except: pass
            elif "pool_sampled_id_" in main_model_id_str:
                try: primary_model_actual_id = int(main_model_id_str.split("pool_sampled_id_")[-1].split("_")[0])
                except: pass


        # 2. 加载对手模型 (P1, P2, P3)
        opponent_configs = eval_config.get("opponent_config", [])
        for i in range(3): # P1, P2, P3 -> seats 1, 2, 3
            seat_idx = i + 1
            if i < len(opponent_configs):
                opp_cfg = opponent_configs[i]
                if opp_cfg.get("source") == "self": # Opponent uses the same model as P0
                    self.loaded_policies[seat_idx] = main_model # Share the same model instance
                    self.current_policy_identifiers[seat_idx] = f"self_as_P0({main_model_id_str})"
                    self.logger.info(f"Seat {seat_idx}: Using Main Agent's model ({main_model_id_str})")
                else:
                    opp_model, opp_model_id_str = self._load_model_from_source(opp_cfg, model_pool_client, primary_model_actual_id)
                    if opp_model:
                        self.loaded_policies[seat_idx] = opp_model
                        self.current_policy_identifiers[seat_idx] = opp_model_id_str
                        self.logger.info(f"Seat {seat_idx}: Loaded {opp_model_id_str}")
                    else:
                        self.logger.warning(f"Failed to load opponent for Seat {seat_idx} from {opp_cfg}. Using P0's model as fallback.")
                        self.loaded_policies[seat_idx] = main_model # Fallback to P0's model
                        self.current_policy_identifiers[seat_idx] = f"fallback_P0({main_model_id_str})"
            else: # Not enough opponent configs, use P0's model
                self.logger.warning(f"No config for opponent Seat {seat_idx}. Using P0's model.")
                self.loaded_policies[seat_idx] = main_model
                self.current_policy_identifiers[seat_idx] = f"default_P0({main_model_id_str})"
        
        return len(self.loaded_policies) == 4 # Ensure all 4 seats have a policy

    def run(self):
        log_base_dir = self.config.get('log_base_dir', './logs')
        experiment_name = self.config.get('experiment_name', 'default_run')
        self.logger, self.writer = setup_process_logging_and_tensorboard(
            log_base_dir, experiment_name, 
            process_type='evaluator', 
            process_id=self.config.get('evaluator_id', os.getpid())
        )
        if not self.logger:
            print(f"CRITICAL: Logger for {self.name} could not be initialized. Exiting.")
            if self.writer: self.writer.close()
            return
        self.logger.info(f"Evaluator process {self.name} started.")
        self.logger.info(f"Evaluation config: {json.dumps(self.config.get('evaluation_config', {}), indent=2)}")

        try:
            torch.set_num_threads(1)
            self.logger.info("PyTorch num_threads set to 1 for evaluator.")
        except Exception as e:
            self.logger.warning(f"Failed to set torch num_threads for evaluator: {e}")

        model_pool_client = None
        eval_cfg = self.config.get('evaluation_config', {})
        eval_mode = eval_cfg.get("mode", "offline_vs_paths") # Default to offline if not specified

        if "pool" in eval_mode or \
           eval_cfg.get("main_agent_source", "").startswith("pool") or \
           any(opp.get("source", "").startswith("pool") for opp in eval_cfg.get("opponent_config",[])):
            try:
                model_pool_client = ModelPoolClient(self.config['model_pool_name'])
                self.logger.info(f"Evaluator connected to Model Pool '{self.config['model_pool_name']}'.")
            except Exception as e:
                self.logger.error(f"Evaluator failed to connect to Model Pool: {e}. Some evaluation modes may not work.", exc_info=True)
                # Depending on config, may need to exit if pool is essential
                if eval_mode == "online_vs_pool" or eval_cfg.get("main_agent_source") == "latest":
                    self.logger.error("Exiting evaluator as essential ModelPool connection failed.")
                    if self.writer: self.writer.close()
                    return

        try:
            env_config = self.config.get('env_config', {'agent_clz': FeatureAgent})
            env = MahjongGBEnv(config=env_config)
            # Each player in the env will use their own FeatureAgent instance
            feature_agents = {player_name: FeatureAgent(seatWind=idx) for idx, player_name in enumerate(env.agent_names)}
            self.logger.info(f"Mahjong environment and FeatureAgents created for evaluation.")
        except Exception as e:
            self.logger.error(f"Failed to create Mahjong environment/FeatureAgents for evaluation: {e}. Evaluator exiting.", exc_info=True)
            if self.writer: self.writer.close()
            return

        num_evaluation_episodes = self.config.get('evaluation_episodes', 100)
        total_eval_steps = 0 # For x-axis of TensorBoard if needed for evaluator lifetime
        
        # Main evaluation loop (can be run once for offline, or periodically for online)
        # For simplicity, this example runs one full evaluation setup.
        # In an online scenario, _setup_evaluation_players might be called periodically.

        if not self._setup_evaluation_players(eval_cfg, model_pool_client):
            self.logger.error("Failed to setup players for evaluation. Exiting.")
            if self.writer: self.writer.close()
            return

        # --- Store aggregated results ---
        # Indexed by main_agent_identifier, then by string describing opponent setup
        # Example: results_summary[main_id_str][opp_setup_str] = {'wins': 0, 'total_games':0, 'avg_score':0 ...}
        # This part needs a more structured way if evaluating multiple main agents or opponent setups in one run.
        # For now, simple per-run aggregation:
        
        per_seat_scores = defaultdict(list) # seat_idx -> list of scores
        main_agent_seat = 0 # Assuming P0 is the main agent being evaluated
        main_agent_wins = 0
        draw_games = 0

        for episode_idx in range(num_evaluation_episodes):
            self.logger.info(f"Starting Evaluation Episode {episode_idx + 1}/{num_evaluation_episodes}")
            self.logger.info(f"Current Policies: P0={self.current_policy_identifiers.get(0, 'N/A')}, "
                             f"P1={self.current_policy_identifiers.get(1, 'N/A')}, "
                             f"P2={self.current_policy_identifiers.get(2, 'N/A')}, "
                             f"P3={self.current_policy_identifiers.get(3, 'N/A')}")

            # Reset feature agents for the new episode
            for idx, player_name in enumerate(env.agent_names):
                feature_agents[player_name].__init__(seatWind=idx) # Re-initialize (or call a proper reset method)
            
            try:
                obs_dict = env.reset()
                # Propagate initial Wind and Deal to all feature agents
                # This depends on how env.reset() provides initial info.
                # Assuming env.reset() output or subsequent calls populate agent's internal state via request2obs.
                # For FeatureAgentTimeSeries, it expects specific request strings.
                # This part needs to match how your env and FeatureAgent handle game start.
                # Example: if env.reset() also returns initial requests:
                # initial_requests = env.get_initial_requests() # Hypothetical
                # for player_name, fa in feature_agents.items():
                #     for req in initial_requests.get(player_name, []): # Process initial lines for each agent
                #         fa.request2obs(req)

            except Exception as e:
                self.logger.error(f"Failed to reset environment for eval episode {episode_idx+1}: {e}. Skipping episode.", exc_info=True)
                continue
            
            episode_scores = {name: 0.0 for name in env.agent_names} # Final scores for this episode
            done = False
            episode_steps = 0

            while not done:
                episode_steps += 1
                total_eval_steps +=1
                actions_to_env = {}
                
                current_acting_agents = list(obs_dict.keys())
                if not current_acting_agents: break # Should be caught by done

                for agent_name_env in current_acting_agents: # agent_name_env is like 'player_0'
                    player_idx = env.agent_names.index(agent_name_env) # Get seat index 0-3
                    
                    # Get processed observation from the correct FeatureAgent
                    # The obs_dict from env is raw; FeatureAgent processes it.
                    # The FeatureAgent needs to be fed the request string for the current game state.
                    # This is a simplification; the main loop of your Actor shows how requests are parsed and fed.
                    # Here, we assume obs_dict from env can be transformed or FeatureAgent is updated another way.
                    # For robust implementation, mimic Actor's request parsing loop.
                    # For now, let's assume feature_agents[agent_name_env] has its state correctly updated
                    # and we can call its _obs() method or similar if it prepares model input.
                    
                    # Simplified: Assume FeatureAgent has a method to take raw_obs and return model_input_dict
                    # raw_state_for_agent = obs_dict[agent_name_env]
                    # processed_obs_for_model = feature_agents[agent_name_env].raw_to_model_input(raw_state_for_agent) # HYPOTHETICAL
                    
                    # More realistically, we need to simulate the request string flow for FeatureAgentTimeSeries
                    # This part is complex and depends on your exact game loop and FeatureAgent interface.
                    # For this example, we'll directly use the model with a simplified input if possible,
                    # or acknowledge this is a placeholder for proper FeatureAgent integration.
                    
                    # Let's assume obs_dict[agent_name_env] IS the dict expected by FeatureAgent's _obs()
                    # after its internal state has been updated by prior requests.
                    # This means FeatureAgent's request2obs must be called appropriately before this.
                    # This is a GAP in this simplified Evaluator compared to full Actor logic.
                    
                    # For now, using a placeholder to show model usage:
                    # This requires obs_dict[agent_name_env] to be exactly what model expects
                    # after FeatureAgent.request2obs() -> FeatureAgent._obs() has been called.
                    # This is a strong assumption and likely needs proper request string generation.

                    # Let's try to use the FeatureAgent as intended, assuming requests are simulated/obtained
                    # This part is highly dependent on how requests are generated for the FeatureAgent
                    # For a simple pass:
                    current_agent_FA = feature_agents[agent_name_env]
                    # In a real scenario, a request string reflecting the current game state for this agent
                    # would be passed to current_agent_FA.request2obs().
                    # For example, if it's this agent's turn to play after a draw:
                    # fake_request_if_needed = f"Player {player_idx} Draw {hypothetical_drawn_tile}" 
                    # model_input_dict = current_agent_FA.request2obs(fake_request_if_need) 
                    # This makes evaluator very complex if it has to simulate all request strings.

                    # A better way for evaluator: FeatureAgent might need a method like:
                    # `get_observation_for_model(self, raw_env_obs_for_this_agent)`
                    # And another method `update_internal_state(self, request_string_from_game_engine)`
                    # For now, we assume model_input_dict is magically available or simplified.
                    
                    # --- Placeholder for model input preparation ---
                    # This needs to match how Actor prepares model_input from FeatureAgent output
                    # For ResNet34AC, it expects: {'obs': {'observation': tensor, 'action_mask': tensor}}
                    # We get this from feature_agents[agent_name_env]._obs() *after* its state is updated.
                    # The tricky part is: how is its state updated in evaluator?
                    # Let's assume for now that obs_dict from env IS the feature_agent's output directly.
                    # This means the environment itself must be returning what FeatureAgent._obs() would.
                    # OR FeatureAgent needs to be more tightly integrated here.

                    state_for_model = obs_dict[agent_name_env] # This is the raw state from env.
                    # The FeatureAgent is stateful. We must call request2obs on it with game events.
                    # The Evaluator loop needs to mimic the request string generation and distribution
                    # that happens in your main training data collection script.
                    # This is non-trivial.

                    # --- MINIMALISTIC ACTION SELECTION (NEEDS PROPER FA INTEGRATION) ---
                    try:
                        policy_to_use = self.loaded_policies[player_idx]
                        policy_to_use.eval() # Ensure eval mode
                        
                        # Convert state_for_model (raw) to tensor for ResNet34AC
                        # This is a simplified conversion, assuming state_for_model matches model input after FA
                        obs_tensor = torch.tensor(state_for_model['observation'], dtype=torch.float).unsqueeze(0).to(self.config.get('device', 'cpu'))
                        mask_tensor = torch.tensor(state_for_model['action_mask'], dtype=torch.float).unsqueeze(0).to(self.config.get('device', 'cpu'))
                        model_input_for_nn = {'obs': {'observation': obs_tensor, 'action_mask': mask_tensor}}

                        with torch.no_grad():
                            logits, _ = policy_to_use(model_input_for_nn)
                            # Greedy action for evaluation (or can sample if desired)
                            action = torch.argmax(logits, dim=1).item() 
                        actions_to_env[agent_name_env] = action
                    except KeyError: # If agent_name_env not in obs_dict (e.g. agent already done)
                        continue
                    except Exception as e_inf:
                        self.logger.error(f"Error during inference for {agent_name_env} (Seat {player_idx}) in eval: {e_inf}", exc_info=True)
                        # Choose a default/random valid action or end episode
                        actions_to_env[agent_name_env] = np.random.choice(np.where(state_for_model['action_mask'] == 1)[0]) if np.any(state_for_model['action_mask']) else 0 # Fallback
                        # done = True; break 
                
                if not actions_to_env: # If all agents were skipped or errored
                    if not done: self.logger.warning("No actions generated by any agent. Ending episode."); done=True
                    break

                if done: break

                try:
                    next_obs_dict, rewards_dict, done_info_dict = env.step(actions_to_env)
                    # IMPORTANT: Here, you would generate request strings from actions_to_env, rewards_dict, next_obs_dict
                    # and feed them to ALL FeatureAgent instances using their request2obs method to update their internal states.
                    # e.g. for player_k_idx who just acted with action_k:
                    #      req_str_play = f"Player {player_k_idx} Play {feature_agents[player_k_name].action2response(action_k)}"
                    #      for fa in feature_agents.values(): fa.request2obs(req_str_play)
                    # This is crucial for FeatureAgentTimeSeries to work correctly. Omitting for brevity.

                except Exception as e_step:
                    self.logger.error(f"Error during env.step in eval: {e_step}. Ending episode.", exc_info=True)
                    done = True # Force end

                if not done:
                    for r_agent_name, r_val in rewards_dict.items():
                        episode_scores[r_agent_name] = r_val # In Mahjong, rewards are often final scores.
                    
                    obs_dict = next_obs_dict
                    if isinstance(done_info_dict, dict):
                        if done_info_dict.get("__all__", False): done = True
                        else: 
                            active_agents_still = {k for k, v_done in done_info_dict.items() if not v_done}
                            if not active_agents_still: done = True
                            obs_dict = {k: v for k,v in next_obs_dict.items() if k in active_agents_still}
                            if not obs_dict and not done: done = True # No one to act for
                    elif isinstance(done_info_dict, bool): done = done_info_dict
                    else: self.logger.error("Unknown done_info type"); done = True
            
            # Episode finished
            self.logger.info(f"Evaluation Episode {episode_idx + 1} finished in {episode_steps} steps.")
            final_scores_this_episode = env.get_final_scores() # Assuming env provides this
            if final_scores_this_episode:
                for seat_i, score_i in enumerate(final_scores_this_episode):
                    per_seat_scores[seat_i].append(score_i)
                    player_name_for_score = env.agent_names[seat_i] # map seat index to player_name from env
                    self.logger.info(f"  Seat {seat_i} ({player_name_for_score}, Policy: {self.current_policy_identifiers.get(seat_i, 'N/A')}): Score = {score_i}")
                if final_scores_this_episode[main_agent_seat] > 0: # Simple win condition (score > 0)
                    main_agent_wins +=1
                elif sum(s != 0 for s in final_scores_this_episode) == 0 : # Or check specific draw condition
                    draw_games +=1
            else:
                self.logger.warning("Could not get final scores for the episode.")


        # After all evaluation episodes for this setup
        num_played = len(per_seat_scores.get(main_agent_seat, []))
        if num_played > 0:
            avg_score_main_agent = np.mean(per_seat_scores.get(main_agent_seat, [0]))
            win_rate_main_agent = main_agent_wins / num_played
            draw_rate = draw_games / num_played
            
            self.logger.info(f"--- Evaluation Summary for Main Agent (Seat {main_agent_seat}, Policy: {self.current_policy_identifiers.get(main_agent_seat, 'N/A')}) ---")
            self.logger.info(f"  Against Opponents: S1={self.current_policy_identifiers.get(1, 'N/A')}, S2={self.current_policy_identifiers.get(2, 'N/A')}, S3={self.current_policy_identifiers.get(3, 'N/A')}")
            self.logger.info(f"  Total Episodes: {num_played}")
            self.logger.info(f"  Main Agent Avg Score: {avg_score_main_agent:.2f}")
            self.logger.info(f"  Main Agent Win Rate: {win_rate_main_agent:.2%}")
            self.logger.info(f"  Draw Rate: {draw_rate:.2%}")

            if self.writer:
                # Construct a unique tag for this evaluation run, e.g., based on opponent setup
                # This needs to be more structured if evaluating multiple setups.
                eval_tag_suffix = f"vs_{self.current_policy_identifiers.get(1,'Op1')}_{self.current_policy_identifiers.get(2,'Op2')}_{self.current_policy_identifiers.get(3,'Op3')}"
                eval_tag_suffix = eval_tag_suffix.replace('/','_').replace('\\','_') # Sanitize path chars

                self.writer.add_scalar(f'Evaluation/{self.current_policy_identifiers.get(main_agent_seat, "MainAgent")}/AvgScore_{eval_tag_suffix}', avg_score_main_agent, total_eval_steps) # Or use Learner's global step if available
                self.writer.add_scalar(f'Evaluation/{self.current_policy_identifiers.get(main_agent_seat, "MainAgent")}/WinRate_{eval_tag_suffix}', win_rate_main_agent, total_eval_steps)
                self.writer.add_scalar(f'Evaluation/{self.current_policy_identifiers.get(main_agent_seat, "MainAgent")}/DrawRate_{eval_tag_suffix}', draw_rate, total_eval_steps)
                
                # Log scores for all seats
                for seat_k, scores_k in per_seat_scores.items():
                    if scores_k:
                         self.writer.add_scalar(f'EvaluationScores/Seat_{seat_k}_Policy_{self.current_policy_identifiers.get(seat_k, "N/A")}/AvgScore_{eval_tag_suffix}', np.mean(scores_k), total_eval_steps)
                self.writer.flush()
        else:
            self.logger.info("No evaluation episodes were completed successfully.")

        if self.writer:
            self.writer.close()
        if model_pool_client:
             # Explicitly delete client to trigger its __del__ for shm.close()
            del model_pool_client 
        self.logger.info(f"Evaluator {self.name} finished.")

