
# --- 使用示例 ---
if __name__ == '__main__':
    # 这是一个如何使用 MahjongVecEnv 的示例

    # 1. 创建一个辅助函数来生成环境实例
    #    这个函数会被序列化并发送到子进程
    def make_env_func(config):
        def _init():
            # 确保导入在函数内部，以便 cloudpickle 能正确处理
            from env.env import MahjongGBEnv 
            from agent.feature import FeatureAgent # 假设使用这个
            
            env_config = config.get('env_config', {})
            # 如果 'agent_clz' 是字符串，将其转换为类对象
            if isinstance(env_config.get('agent_clz'), str):
                 if env_config['agent_clz'] == 'FeatureAgent':
                     env_config['agent_clz'] = FeatureAgent
                 # ... 其他 agent 类的处理 ...
            
            return MahjongGBEnv(config=env_config)
        return _init

    # 2. 设置配置
    num_parallel_envs = 4 # 同时运行4个麻将环境
    main_config = {
        'env_config': {
            'agent_clz': 'FeatureAgent' # 在 make_env_func 中会被解析
        }
    }

    print("创建向量化环境...")
    # 创建一个环境创建函数的列表
    env_functions = [make_env_func(main_config) for _ in range(num_parallel_envs)]
    
    # 实例化向量化环境
    vec_env = SubprocVecEnv(env_functions)
    print(f"向量化环境已创建，包含 {vec_env.num_envs} 个并行环境。")
    print(f"观测空间 (示例): {vec_env.observation_space}")
    print(f"动作空间 (示例): {vec_env.action_space}")

    # 3. 与环境交互
    print("\n重置所有环境...")
    # reset() 返回一个列表，每个元素是对应环境的初始观测字典
    initial_obs_list = vec_env.reset()
    print(f"收到 {len(initial_obs_list)} 个初始观测。")
    print(f"第一个环境的初始观测 (活跃玩家): {list(initial_obs_list[0].keys())}")

    print("\n执行一步随机动作...")
    # 为每个环境的当前活跃玩家生成一个随机动作
    random_actions = []
    for obs_dict in initial_obs_list:
        # 在多智能体环境中，观测字典通常只有一个键（当前行动的玩家）
        agent_to_act = list(obs_dict.keys())[0]
        # 从该玩家的有效动作中随机选择一个
        valid_actions_indices = np.where(obs_dict[agent_to_act]['action_mask'] == 1)[0]
        random_action_idx = np.random.choice(valid_actions_indices) if len(valid_actions_indices) > 0 else 0
        random_actions.append({agent_to_act: random_action_idx})
    
    print(f"发送的动作批次: {random_actions}")
    
    # 调用 step()，它会异步发送动作，然后等待所有环境完成
    next_obs_list, rewards_list, dones_array, infos_list = vec_env.step(random_actions)

    print("\n收到一步之后的结果:")
    print(f"  - 下一步观测列表长度: {len(next_obs_list)}")
    print(f"  - 奖励列表长度: {len(rewards_list)}")
    print(f"  - 完成状态数组: {dones_array}")
    print(f"  - 信息列表长度: {len(infos_list)}")
    print(f"  - 第一个环境的下一步观测 (活跃玩家): {list(next_obs_list[0].keys())}")
    print(f"  - 第一个环境的奖励字典: {rewards_list[0]}")


    # 4. 关闭环境
    print("\n正在关闭向量化环境...")
    vec_env.close()
    print("环境已关闭。")
