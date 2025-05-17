import torch
import numpy as np
import time
import os
import json
from collections import defaultdict
import argparse
import sys
import random # 用于随机选择和打乱模型


from env import MahjongGBEnv
# 根据您的项目，这里可能是 FeatureAgentTimeSeries 或 FeatureAgent
from feature_timeseries import FeatureAgentTimeSeries 
from feature import FeatureAgent
from model import ResNet34AC
from utils import setup_process_logging_and_tensorboard # 用于日志记录

class EvaluatorOffline:
    """
    离线评估器类，用于评估一组固定的麻将模型，并在每局开始前随机分配席位。
    """
    def __init__(self, config: dict):
        """
        初始化离线评估器。

        Args:
            config (dict): 配置字典。预期键值:
                - 'evaluator_id': 此评估运行的唯一ID。
                - 'log_base_dir': 日志的基础目录。
                - 'experiment_name': 此评估实验的名称。
                - 'in_channels': 模型输入通道数。
                - 'device': 'cpu' 或 'cuda:X'。
                - 'evaluation_episodes': 要运行的评估局数。
                - 'model_details_dict': 字典，映射描述性模型名称到其信息 
                                         (例如, {'model_A': {'model_dir': '/path/to/model_A.pth'}, ...})。
                                         所有在此定义的模型都将被加载并参与随机席位分配。
                - 'env_config': MahjongGBEnv 的配置。
                - 'seat_assignment_interval': (可选) 每隔多少局重新随机分配一次席位，默认为1 (每局都重新分配)。
        """
        self.config = config
        self.evaluator_id = config.get('evaluator_id', f"offline_dyn_{os.getpid()}")
        self.name = config.get('name', f'EvaluatorOfflineDynamic-{self.evaluator_id}')
        
        self.logger = None
        self.writer = None 
        
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.in_channels = self.config.get('in_channels', 14) 

        self.model_zoo = {} 
        self.model_stats = defaultdict(lambda: {
            'total_reward': 0.0, 
            'games_played': 0, 
            'wins': 0,
            'cumulative_score_sq': 0.0
        })

        self.current_game_policies = {} # 席位索引 -> 模型实例
        self.current_game_policy_names = {} # 席位索引 -> 模型描述性名称
        self.seat_assignment_interval = config.get('seat_assignment_interval', 1)


    def _setup_logging_and_tensorboard(self):
        """初始化日志记录器和 TensorBoard writer。"""
        log_base_dir = self.config.get('log_base_dir', './eval_offline_logs_dynamic')
        experiment_name = self.config.get('experiment_name', 'dynamic_seat_evaluation_run')
        
        self.logger, self.writer = setup_process_logging_and_tensorboard(
            log_base_dir, experiment_name, 
            process_name='evaluator_offline_dyn'
        )
        if not self.logger:
            print(f"严重错误: 评估器 {self.name} (ID: {self.evaluator_id}) 的日志记录器未能初始化。正在退出。")
            if self.writer: self.writer.close()
            raise RuntimeError("日志记录器初始化失败。") 
        self.logger.info(f"离线评估器 (动态席位) {self.name} (ID: {self.evaluator_id}) 已启动。")
        self.logger.info(f"评估配置: {json.dumps(self.config, indent=2, ensure_ascii=False, default=lambda o: str(o))}")

    def _load_models(self) -> bool:
        """
        根据配置中的 'model_details_dict' 加载所有定义的模型。
        """
        model_details_dict = self.config.get('model_details_dict', {})
        if not model_details_dict:
            self.logger.error("配置中未提供 'model_details_dict' 或为空。无法加载模型。")
            return False
        
        self.logger.info(f"将从 'model_details_dict' 加载以下模型: {list(model_details_dict.keys())}")

        for model_name, model_info in model_details_dict.items():
            model_path = model_info.get('model_dir')
            if not model_path or not os.path.isfile(model_path):
                self.logger.error(f"模型 '{model_name}' 的文件路径未找到或无效: {model_path}")
                return False # 任何一个模型加载失败则整体失败
            
            try:
                model_instance = ResNet34AC(self.in_channels)
                model_instance.to(self.device)
                
                self.logger.debug(f"正在从路径加载模型 '{model_name}': {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                state_dict_to_load = checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict_to_load = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict_to_load = checkpoint['state_dict']

                model_instance.load_state_dict(state_dict_to_load)
                model_instance.eval()
                self.model_zoo[model_name] = model_instance
                self.logger.info(f"成功加载模型 '{model_name}' 从 {model_path}")
            except Exception as e:
                self.logger.error(f"加载模型 '{model_name}' 从 {model_path} 时出错: {e}", exc_info=True)
                return False
        
        if not self.model_zoo:
            self.logger.error("模型库 (model_zoo) 为空，没有模型被成功加载。")
            return False
        return True

    def _randomly_assign_policies_to_seats(self) -> bool:
        """
        从 self.model_zoo 中随机选择4个模型（可重复）并随机分配到4个席位。
        更新 self.current_game_policies 和 self.current_game_policy_names。
        """
        available_model_names = list(self.model_zoo.keys())
        if not available_model_names:
            self.logger.error("模型库 (model_zoo) 中没有可用的模型进行席位分配。")
            return False

        num_unique_models = len(available_model_names)
        
        selected_model_names_for_game = []
        if num_unique_models == 0: # Should have been caught by above
            self.logger.error("没有模型可供选择！")
            return False
        elif num_unique_models < 4:
            self.logger.warning(f"只有 {num_unique_models} 个独特模型可用，将进行有放回抽样以填满4个席位。")
            selected_model_names_for_game = random.choices(available_model_names, k=4)
        else: # 有4个或更多独特模型
            selected_model_names_for_game = random.sample(available_model_names, k=4)
        
        random.shuffle(selected_model_names_for_game) # 打乱这4个选出的模型的顺序，实现对席位的随机分配

        self.current_game_policies.clear()
        self.current_game_policy_names.clear()

        for seat_idx in range(4):
            model_name_for_this_seat = selected_model_names_for_game[seat_idx]
            self.current_game_policies[seat_idx] = self.model_zoo[model_name_for_this_seat]
            self.current_game_policy_names[seat_idx] = model_name_for_this_seat
        
        self.logger.info("已为当前评估对局随机分配策略到席位：")
        for seat_idx in range(4):
            self.logger.info(f"  席位 {seat_idx}: 使用模型 '{self.current_game_policy_names[seat_idx]}'")
        return True

    def _run_single_episode(self, env: MahjongGBEnv, episode_num_global: int):
        """运行一局麻将评估游戏。"""
        try:
            obs_dict_env = env.reset() 
        except Exception as e:
            self.logger.error(f"评估局 {episode_num_global}: 环境重置失败: {e}. 跳过此局。", exc_info=True)
            return

        done_episode = False
        episode_steps = 0
        
        while not done_episode:
            episode_steps += 1
            actions_to_submit_to_env = {}
            
            current_agents_to_act = list(obs_dict_env.keys())
            if not current_agents_to_act and not done_episode:
                self.logger.warning(f"评估局 {episode_num_global}, 步骤 {episode_steps}: obs_dict 为空但游戏未结束。强制结束。")
                done_episode = True; break

            for agent_name_from_env in current_agents_to_act:
                try:
                    player_seat_idx = env.agent_names.index(agent_name_from_env)
                    policy_to_use = self.current_game_policies[player_seat_idx]
                    current_agent_obs_data = obs_dict_env[agent_name_from_env]
                    
                    obs_tensor = torch.tensor(current_agent_obs_data['observation'], dtype=torch.float).unsqueeze(0).to(self.device)
                    mask_tensor = torch.tensor(current_agent_obs_data['action_mask'], dtype=torch.float).unsqueeze(0).to(self.device)
                    model_input_for_nn = {'obs': {'observation': obs_tensor, 'action_mask': mask_tensor}}

                    with torch.no_grad():
                        logits, _ = policy_to_use(model_input_for_nn)
                        masked_logits = logits + torch.clamp(torch.log(mask_tensor), min=-1e9) 
                        action_idx = torch.argmax(masked_logits, dim=1).item()
                    actions_to_submit_to_env[agent_name_from_env] = action_idx
                except Exception as e_inf:
                    self.logger.error(f"评估局 {episode_num_global} 步骤 {episode_steps}: 玩家 {agent_name_from_env} (席位 {player_seat_idx}) 推理时出错: {e_inf}", exc_info=True)
                    raw_mask = current_agent_obs_data.get('action_mask', [])
                    valid_indices = [idx for idx, val in enumerate(raw_mask) if val == 1]
                    actions_to_submit_to_env[agent_name_from_env] = random.choice(valid_indices) if valid_indices else 0 
            
            if done_episode: break 
            if not actions_to_submit_to_env:
                if not done_episode: self.logger.warning(f"评估局 {episode_num_global} 步骤 {episode_steps}: 没有为环境收集到任何动作。强制结束。"); done_episode=True
                break
            
            try:
                next_obs_dict, rewards_from_step, done_info_dict = env.step(actions_to_submit_to_env)
            except Exception as e_step:
                self.logger.error(f"评估局 {episode_num_global} 步骤 {episode_steps}: env.step() 执行出错: {e_step}. 强制结束。", exc_info=True)
                done_episode = True

            if not done_episode:
                obs_dict_env = next_obs_dict
                if isinstance(done_info_dict, dict):
                    if done_info_dict.get("__all__", False): done_episode = True
                    else: 
                        active_agents_still = {k for k, v_done in done_info_dict.items() if not v_done and k in obs_dict_env}
                        obs_dict_env = {k: v for k,v_obs in next_obs_dict.items() if k in active_agents_still}
                        if not obs_dict_env and not done_info_dict.get("__all__", False): done_episode = True 
                elif isinstance(done_info_dict, bool): done_episode = done_info_dict
                else: self.logger.error(f"未知的 done_info 类型: {type(done_info_dict)}"); done_episode = True
        
        self.logger.info(f"评估局 {episode_num_global} 在 {episode_steps} 步后完成。")
        
        # --- 修改点：使用 env._reward() 获取最终得分 ---
        final_scores_dict = {}
        if hasattr(env, '_reward') and callable(env._reward):
            try:
                # 调用环境的 _reward 方法。需要确保此时 env.reward 和 env.obs (如果被 _reward 方法使用) 是正确的状态。
                # 通常，在 done_episode 为 True 后，环境内部应该已经计算并存储了最终奖励。
                final_scores_dict = env._reward() 
                if not isinstance(final_scores_dict, dict):
                    self.logger.warning(f"env._reward() 返回的不是字典类型 (类型: {type(final_scores_dict)}). 无法处理得分。")
                    final_scores_dict = {} # 置为空字典以避免后续错误
            except Exception as e_reward:
                self.logger.error(f"调用 env._reward() 时出错: {e_reward}", exc_info=True)
        else:
            self.logger.error("环境实例 env 没有可调用的 _reward 方法来获取最终得分。")

        if final_scores_dict: # 确保 final_scores_dict 是一个非空字典
            for agent_name_key, score_val in final_scores_dict.items():
                try:
                    # 将 agent_name_key (如 "player_0") 映射回席位索引
                    seat_idx_score = env.agent_names.index(agent_name_key)
                    model_name_at_this_seat = self.current_game_policy_names[seat_idx_score]
                    
                    self.model_stats[model_name_at_this_seat]['total_reward'] += score_val
                    self.model_stats[model_name_at_this_seat]['games_played'] += 1
                    self.model_stats[model_name_at_this_seat]['cumulative_score_sq'] += score_val**2
                    
                    if score_val > 0: 
                        self.model_stats[model_name_at_this_seat]['wins'] += 1
                    
                    self.logger.info(f"  席位 {seat_idx_score} ({agent_name_key}, 模型: '{model_name_at_this_seat}'): 本局得分 = {score_val}")
                except ValueError:
                    self.logger.warning(f"在 env.agent_names 中未找到从 env._reward() 返回的 agent_name '{agent_name_key}'。")
                except KeyError:
                    self.logger.warning(f"在 current_game_policy_names 中未找到席位索引 '{seat_idx_score}' (来自 agent_name '{agent_name_key}')。")
        else:
            self.logger.warning(f"未能从 env._reward() 获取到评估局 {episode_num_global} 的有效最终得分。")


    def run(self):
        self._setup_logging_and_tensorboard()
        try:
            torch.set_num_threads(1)
        except Exception as e:
            self.logger.warning(f"设置 PyTorch 评估器线程数失败: {e}")

        if not self._load_models():
            self.logger.error("一个或多个模型加载失败。中止评估。")
            return

        try:
            env_config = self.config.get('env_config', {})
            if 'agent_clz' not in env_config or not callable(env_config['agent_clz']):
                env_config['agent_clz'] = FeatureAgentTimeSeries
            self.logger.info(f"评估器正在使用环境配置: {env_config}")
            env = MahjongGBEnv(config=env_config)
            self.logger.info(f"麻将环境已创建，使用智能体类: {env_config['agent_clz'].__name__}.")
        except Exception as e:
            self.logger.error(f"创建麻将环境失败: {e}. 评估器正在退出。", exc_info=True)
            return

        num_evaluation_episodes = self.config.get('evaluation_episodes', 10)
        self.logger.info(f"将运行 {num_evaluation_episodes} 局评估游戏。")
        
        for episode_idx_global in range(num_evaluation_episodes):
            # 每隔 seat_assignment_interval 局或在第一局重新分配席位
            if episode_idx_global % self.seat_assignment_interval == 0:
                self.logger.info(f"--- 第 {episode_idx_global + 1} 局: 正在重新随机分配席位 ---")
                if not self._randomly_assign_policies_to_seats():
                    self.logger.error("未能为对局分配策略。中止评估。")
                    return
            else:
                self.logger.info(f"--- 开始评估局 {episode_idx_global + 1}/{num_evaluation_episodes} (使用上一轮席位分配) ---")
                # 打印当前席位分配（如果不是每局都换，这有助于跟踪）
                for seat_idx_log in range(4):
                     self.logger.info(f"  席位 {seat_idx_log}: 模型 '{self.current_game_policy_names.get(seat_idx_log, '未分配')}'")


            self._run_single_episode(env, episode_idx_global + 1)
            
            log_interval = self.config.get("log_stats_interval_episodes", max(1, num_evaluation_episodes // 10)) # 每10%或至少每1局
            if (episode_idx_global + 1) % log_interval == 0 or (episode_idx_global + 1) == num_evaluation_episodes:
                self._log_final_statistics(current_episode_count=episode_idx_global + 1)

        # _log_final_statistics 已经在循环末尾或按间隔调用，这里不再重复调用
        
        if self.writer:
            self.writer.close()
        self.logger.info(f"离线评估器 {self.name} (ID: {self.evaluator_id}) 已完成所有评估局。")

    def _log_final_statistics(self, current_episode_count: int):
        """记录并打印（中间或最终的）评估统计数据。"""
        self.logger.info(f"--- 评估统计 (截至 {current_episode_count} 局) ---")
        for model_name_stats, stats_data in self.model_stats.items():
            games_played_stat = stats_data['games_played']
            if games_played_stat > 0:
                avg_reward_stat = stats_data['total_reward'] / games_played_stat
                win_rate_stat = stats_data['wins'] / games_played_stat
                mean_sq_stat = stats_data['cumulative_score_sq'] / games_played_stat
                score_std_dev_stat = np.sqrt(max(0, mean_sq_stat - avg_reward_stat**2)) #确保非负

                self.logger.info(
                    f"模型: '{model_name_stats}' | 已玩局数: {games_played_stat} | "
                    f"平均得分: {avg_reward_stat:.2f} (StdDev: {score_std_dev_stat:.2f}) | 胜率: {win_rate_stat:.2%}"
                )
                if self.writer: 
                    sane_model_name_tag = "".join(c if c.isalnum() else '_' for c in model_name_stats)
                    self.writer.add_scalar(f'OfflineEval_Summary/{sane_model_name_tag}/AvgReward_vs_MixedPool', avg_reward_stat, current_episode_count)
                    self.writer.add_scalar(f'OfflineEval_Summary/{sane_model_name_tag}/WinRate_vs_MixedPool', win_rate_stat, current_episode_count)
                    self.writer.add_scalar(f'OfflineEval_Summary/{sane_model_name_tag}/GamesPlayed_Total', games_played_stat, current_episode_count)
                    self.writer.add_scalar(f'OfflineEval_Summary/{sane_model_name_tag}/ScoreStdDev_vs_MixedPool', score_std_dev_stat, current_episode_count)
            else:
                self.logger.info(f"模型: '{model_name_stats}' | 已玩局数: 0")
        if self.writer:
            self.writer.flush()


def main():
    parser = argparse.ArgumentParser(description="离线麻将模型评估脚本 (动态席位)")
    parser.add_argument('--config_path', type=str, required=True, 
                        help="评估配置文件的JSON路径。")
    
    args = parser.parse_args()

    if not os.path.isfile(args.config_path):
        print(f"错误: 配置文件未在路径 {args.config_path} 找到。")
        sys.exit(1)

    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"错误: 加载或解析配置文件 {args.config_path} 失败: {e}")
        sys.exit(1)

    config.setdefault('evaluator_id', f"offline_dyn_{int(time.time())}")
    config.setdefault('log_base_dir', './eval_results_offline_dynamic')
    config.setdefault('experiment_name', 'dynamic_seat_evaluation')
    config.setdefault('in_channels', 14) 
    config.setdefault('device', 'cpu')
    config.setdefault('evaluation_episodes', 10)
    config.setdefault('seat_assignment_interval', 1) # 每局都重新分配席位
    config.setdefault('log_stats_interval_episodes', max(1, config.get('evaluation_episodes', 10) // 10))


    env_config = config.setdefault('env_config', {})
    agent_clz_name = env_config.get('agent_clz', 'FeatureAgentTimeSeries') 
    
    if isinstance(agent_clz_name, str):
        if agent_clz_name == 'FeatureAgentTimeSeries':
            env_config['agent_clz'] = FeatureAgentTimeSeries
        elif agent_clz_name == 'FeatureAgent': 
             env_config['agent_clz'] = FeatureAgent
        else:
            print(f"错误: 未知的 agent_clz 名称 '{agent_clz_name}' 在配置中。")
            sys.exit(1)
    elif not callable(env_config.get('agent_clz')):
        print(f"错误: env_config 中的 'agent_clz' 必须是字符串名称或类对象。得到: {env_config.get('agent_clz')}")
        sys.exit(1)

    if 'model_details_dict' not in config or not isinstance(config['model_details_dict'], dict) or not config['model_details_dict']:
        print("错误: 配置中必须定义 'model_details_dict'，它应该是一个非空字典。")
        sys.exit(1)
    
    for model_name_val, model_detail_val in config['model_details_dict'].items():
        if 'model_dir' not in model_detail_val or not model_detail_val['model_dir']:
            print(f"错误: 模型 '{model_name_val}' 在 'model_details_dict' 中的 'model_dir' 未定义或为空。")
            sys.exit(1)
    
    # 不再需要 player_seat_assignments 的验证
    # if 'player_seat_assignments' in config:
    # print("注意: 配置中的 'player_seat_assignments' 将被忽略，因为此脚本使用动态席位分配。")


    print(f"启动离线评估器 (动态席位)，使用配置: {json.dumps(config, indent=2, default=lambda o: str(o) if not isinstance(o, (dict,list,str,int,float,bool,type(None))) else o, ensure_ascii=False)}")

    evaluator = EvaluatorOffline(config=config)
    evaluator.run()

    print(f"离线评估 (动态席位) 完成。结果已记录到: {os.path.join(config['log_base_dir'], config['experiment_name'])}")

if __name__ == '__main__':
    main()
