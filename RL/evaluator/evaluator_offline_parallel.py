import torch
from multiprocessing import Process
import numpy as np
import time
import os
import json
from collections import defaultdict

# 导入自定义模块
from models.actor import ResNet34Actor
from utils import setup_process_logging_and_tensorboard
from env.env_wrapper import SubprocVecEnv
from env.env import MahjongGBEnv
from agent.feature import FeatureAgent

DEFAULT_AGENT_NAMES_LIST = ['player_1', 'player_2', 'player_3', 'player_4']

class EvaluatorOfflineParallel(Process):
    """
    简化的并行离线评估器类，基于Actor的代码风格实现。
    专注于收集reward和胜负数据，只使用本地模型加载，不需要InferenceServer。
    """
    
    def __init__(self, config: dict):
        """
        初始化并行离线评估器。

        Args:
            config (dict): 配置字典，包含评估参数和模型信息。
        """
        super(EvaluatorOfflineParallel, self).__init__()
        self.config = config
        self.evaluator_id = config.get('evaluator_id', f"offline_parallel_{os.getpid()}")
        self.name = config.get('name', f'EvaluatorOfflineParallel-{self.evaluator_id}')
        
        self.logger = None
        self.writer = None 
        
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.in_channels = self.config.get('in_channels', 187)
        
        # 模型缓存 - 每个智能体一个模型
        self.loaded_models = {}  # {agent_name: model_instance}
        
        # 评估结果统计
        self.episode_results = []  # 存储每局的结果
        self.agent_stats = defaultdict(lambda: {'total_reward': 0.0, 'wins': 0, 'games': 0})
    
    def _load_model(self, model_path: str, agent_name: str) -> ResNet34Actor:
        """
        加载模型实例。
        
        Args:
            model_path (str): 模型文件路径
            agent_name (str): 智能体名称
            
        Returns:
            ResNet34Actor: 加载的模型实例，失败时返回None
        """
        try:
            # 创建模型实例
            out_channels = self.config.get('out_channels', 235)  # 动作数量
            model = ResNet34Actor(self.in_channels, out_channels).to(self.device)
            
            # 处理相对路径
            if not os.path.isabs(model_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                model_path = os.path.join(project_root, model_path)
            
            if os.path.isfile(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 处理不同的checkpoint格式
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                model.load_state_dict(state_dict)
                model.eval()
                
                self.logger.info(f"Successfully loaded model for '{agent_name}' from {model_path}")
                return model
            else:
                self.logger.error(f"Model file not found: {model_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading model for '{agent_name}' from {model_path}: {e}")
            return None
    
    def _get_action_from_model(self, agent_name: str, observation_dict: dict) -> int:
        """
        使用本地模型获取动作。
        
        Args:
            agent_name (str): 智能体名称
            observation_dict (dict): 观测字典
            
        Returns:
            int: 选择的动作
        """
        try:
            model = self.loaded_models[agent_name]
            
            with torch.no_grad():
                obs = torch.from_numpy(observation_dict['observation']).unsqueeze(0).to(self.device, dtype=torch.float32)
                mask = torch.from_numpy(observation_dict['action_mask']).unsqueeze(0).to(self.device, dtype=torch.float32)
                
                state_dict = {'obs': {'observation': obs, 'action_mask': mask}}
                logits = model(state_dict)
                
                # 应用mask并选择动作（贪婪策略）
                masked_logits = logits + (mask - 1) * 1e9
                action = torch.argmax(masked_logits, dim=1).item()
                
                return action
                
        except Exception as e:
            self.logger.error(f"Error getting action from model for {agent_name}: {e}")
            # 返回随机合法动作作为后备
            valid_actions = np.where(observation_dict['action_mask'] == 1)[0]
            return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
    
    def _create_env_fn(self):
        """创建环境函数，用于SubprocVecEnv"""
        def _env():
            env_config = self.config.get('env_config', {'agent_clz': FeatureAgent})
            return MahjongGBEnv(config=env_config)
        return _env
    
    def _process_episode_results(self, episode_rewards: dict, episode_num: int):
        """
        处理单局结果。
        
        Args:
            episode_rewards (dict): 每个智能体的奖励
            episode_num (int): 局数编号
        """
        agent_names = self.config.get('agent_names', DEFAULT_AGENT_NAMES_LIST)
        
        # 记录结果
        episode_result = {
            'episode': episode_num,
            'rewards': episode_rewards.copy(),
            'timestamp': time.time()
        }
        self.episode_results.append(episode_result)
        
        # 更新统计数据
        max_reward = max(episode_rewards.values()) if episode_rewards else 0
        for agent_name in agent_names:
            reward = episode_rewards.get(agent_name, 0.0)
            self.agent_stats[agent_name]['total_reward'] += reward
            self.agent_stats[agent_name]['games'] += 1
            
            # 判断是否获胜（获得最高奖励）
            if reward == max_reward and reward > 0:
                self.agent_stats[agent_name]['wins'] += 1
        
        # 定期日志记录
        log_interval = self.config.get('log_interval', 10)
        if episode_num % log_interval == 0:
            self.logger.info(f"Episode {episode_num} completed:")
            for agent_name, reward in episode_rewards.items():
                self.logger.info(f"  {agent_name}: {reward:.2f}")
    
    def _log_final_statistics(self):
        """记录最终统计结果"""
        self.logger.info("=== Final Evaluation Statistics ===")
        
        agent_names = self.config.get('agent_names', DEFAULT_AGENT_NAMES_LIST)
        
        for agent_name in agent_names:
            stats = self.agent_stats[agent_name]
            if stats['games'] > 0:
                avg_reward = stats['total_reward'] / stats['games']
                win_rate = stats['wins'] / stats['games']
                
                self.logger.info(f"{agent_name}:")
                self.logger.info(f"  Games: {stats['games']}")
                self.logger.info(f"  Total Reward: {stats['total_reward']:.2f}")
                self.logger.info(f"  Average Reward: {avg_reward:.4f}")
                self.logger.info(f"  Wins: {stats['wins']}")
                self.logger.info(f"  Win Rate: {win_rate:.2%}")
                
                # TensorBoard 记录
                if self.writer:
                    safe_agent_name = agent_name.replace('/', '_').replace('\\', '_')
                    self.writer.add_scalar(f'Evaluation/{safe_agent_name}/AvgReward', avg_reward, stats['games'])
                    self.writer.add_scalar(f'Evaluation/{safe_agent_name}/WinRate', win_rate, stats['games'])
                    self.writer.add_scalar(f'Evaluation/{safe_agent_name}/TotalReward', stats['total_reward'], stats['games'])
        
        # 检查零和性质
        total_rewards = [stats['total_reward'] for stats in self.agent_stats.values()]
        sum_all_rewards = sum(total_rewards)
        self.logger.info(f"Sum of all agent rewards: {sum_all_rewards:.4f}")
        
        if abs(sum_all_rewards) > 1e-6:
            self.logger.warning(f"Non-zero sum detected: {sum_all_rewards:.4f}. This may indicate an issue with reward assignment.")
        else:
            self.logger.info("Zero-sum property verified: rewards sum to approximately zero.")
        
        if self.writer:
            self.writer.add_scalar('Evaluation/SumAllRewards', sum_all_rewards, len(self.episode_results))
            self.writer.flush()
    
    def run(self):
        """主运行函数"""
        # 初始化日志和 TensorBoard
        log_base_dir = self.config.get('log_base_dir', './test_logs')
        experiment_name = self.config.get('experiment_name', 'evaluator_offline_parallel')
        
        self.logger, self.writer = setup_process_logging_and_tensorboard(
            log_base_dir, experiment_name, self.name
        )
        
        if not self.logger:
            print(f"CRITICAL: Logger for {self.name} could not be initialized. Exiting.")
            if self.writer:
                self.writer.close()
            return
        
        self.logger.info(f"EvaluatorOfflineParallel {self.name} started")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2, default=str)}")
        
        try:
            # 设置 PyTorch 线程数
            torch.set_num_threads(1)
            self.logger.info("PyTorch num_threads set to 1 for evaluator")
        except Exception as e:
            self.logger.warning(f"Failed to set torch num_threads: {e}")
        
        try:
            # 加载所有智能体的模型
            model_paths = self.config.get('model_paths', {})
            agent_names = self.config.get('agent_names', DEFAULT_AGENT_NAMES_LIST)
            
            if not model_paths:
                self.logger.error("No model paths provided in config")
                return
            
            # 加载模型
            for agent_name in agent_names:
                if agent_name in model_paths:
                    model = self._load_model(model_paths[agent_name], agent_name)
                    if model is not None:
                        self.loaded_models[agent_name] = model
                    else:
                        self.logger.error(f"Failed to load model for {agent_name}")
                        return
                else:
                    self.logger.error(f"No model path provided for {agent_name}")
                    return
            
            self.logger.info(f"Successfully loaded models for {len(self.loaded_models)} agents")
            
            # 创建并行环境
            num_envs = self.config.get('num_envs', 4)
            total_episodes = self.config.get('total_episodes', 100)
            
            self.logger.info(f"Creating {num_envs} parallel environments...")
            env_fns = [self._create_env_fn() for _ in range(num_envs)]
            vec_env = SubprocVecEnv(env_fns)
            
            # 运行评估
            self.logger.info(f"Starting evaluation for {total_episodes} episodes...")
            
            completed_episodes = 0
            episode_rewards = [{agent_name: 0.0 for agent_name in agent_names} for _ in range(num_envs)]
            
            # 重置环境
            observations = vec_env.reset()
            
            while completed_episodes < total_episodes:
                # 收集所有环境的动作
                actions_batch = {}
                
                for env_idx in range(num_envs):
                    if env_idx < len(observations):
                        env_obs = observations[env_idx]
                        env_actions = {}
                        
                        # 处理观测格式 - 可能是列表或字典
                        if isinstance(env_obs, dict):
                            obs_dict = env_obs
                        elif isinstance(env_obs, (list, tuple)) and len(env_obs) == 2:
                            # 假设是 (obs_dict, global_obs) 的格式
                            obs_dict = env_obs[0] if isinstance(env_obs[0], dict) else {}
                        else:
                            self.logger.warning(f"Unexpected observation format for env {env_idx}: {type(env_obs)}")
                            continue
                        
                        for agent_name, obs_data in obs_dict.items():
                            if agent_name in agent_names and agent_name in self.loaded_models:
                                action = self._get_action_from_model(agent_name, obs_data)
                                env_actions[agent_name] = action
                        
                        actions_batch[env_idx] = env_actions
                
                # 执行环境步骤
                try:
                    next_observations, rewards_batch, dones, infos = vec_env.step(actions_batch)
                    
                    # 处理奖励和完成状态
                    for env_idx in range(num_envs):
                        if env_idx < len(rewards_batch):
                            env_rewards = rewards_batch[env_idx]
                            env_done = dones[env_idx] if isinstance(dones, (list, np.ndarray)) else dones
                            
                            # 累积奖励
                            for agent_name, reward in env_rewards.items():
                                if agent_name in episode_rewards[env_idx]:
                                    episode_rewards[env_idx][agent_name] += reward
                            
                            # 检查是否完成
                            if env_done:
                                completed_episodes += 1
                                
                                # 处理完成的局结果
                                self._process_episode_results(episode_rewards[env_idx], completed_episodes)
                                
                                # 重置奖励累积
                                episode_rewards[env_idx] = {agent_name: 0.0 for agent_name in agent_names}
                                
                                if completed_episodes % 10 == 0:
                                    self.logger.info(f"Progress: {completed_episodes}/{total_episodes} episodes completed")
                    
                    observations = next_observations
                    
                except Exception as e:
                    self.logger.error(f"Error during environment step: {e}", exc_info=True)
                    break
            
            # 关闭环境
            vec_env.close()
            
            # 记录最终统计
            self._log_final_statistics()
            
        except Exception as e:
            self.logger.error(f"Error in evaluator run: {e}", exc_info=True)
        
        finally:
            if self.writer:
                self.writer.close()
            self.logger.info(f"EvaluatorOfflineParallel {self.name} finished")

    def get_results_summary(self):
        """
        获取评估结果摘要。
        
        Returns:
            dict: 包含统计结果的字典
        """
        summary = {
            'total_episodes': len(self.episode_results),
            'agent_stats': dict(self.agent_stats),
            'episode_results': self.episode_results
        }
        
        # 计算总体统计
        if self.episode_results:
            agent_names = self.config.get('agent_names', DEFAULT_AGENT_NAMES_LIST)
            
            for agent_name in agent_names:
                stats = self.agent_stats[agent_name]
                if stats['games'] > 0:
                    summary['agent_stats'][agent_name]['avg_reward'] = stats['total_reward'] / stats['games']
                    summary['agent_stats'][agent_name]['win_rate'] = stats['wins'] / stats['games']
        
        return summary


def main():
    """示例使用方法"""
    config = {
        'name': 'EvaluatorOfflineParallelTest',
        'evaluator_id': 'test_001',
        'log_base_dir': './test_logs',
        'experiment_name': 'offline_parallel_evaluation',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'in_channels': 187,
        'num_envs': 4,
        'total_episodes': 20,
        'agent_names': DEFAULT_AGENT_NAMES_LIST,
        'model_paths': {
            'player_1': '/root/autodl-tmp/MahjongMAS/RL/initial_models/actor_from_sl.pth',
            'player_2': '/root/autodl-tmp/MahjongMAS/RL/initial_models/actor_from_sl.pth',
            'player_3': '/root/autodl-tmp/MahjongMAS/RL/initial_models/actor_from_sl.pth',
            'player_4': '/root/autodl-tmp/MahjongMAS/RL/initial_models/actor_from_sl.pth',
        },
        'env_config': {
            'agent_clz': FeatureAgent
        }
    }
    
    print("Creating and starting EvaluatorOfflineParallel...")
    evaluator = EvaluatorOfflineParallel(config)
    evaluator.start()  # 启动进程
    evaluator.join()   # 等待进程完成
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
