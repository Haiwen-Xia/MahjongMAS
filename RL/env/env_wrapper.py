import numpy as np
import torch # 仅用于类型提示
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import copy # 用于在 done 时深度复制最后的信息
import os, traceback
# CloudpickleWrapper 用于序列化环境创建函数，确保它能被传递到子进程
class CloudpickleWrapper(object):
    """
    使用 cloudpickle 来序列化内容 (否则 multiprocessing 会尝试使用 pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    """
    在子进程中运行的目标函数。
    它创建环境实例，并通过管道与主进程通信。
    """
    # 关闭父进程端的管道，因为子进程只使用自己的 remote 端
    parent_remote.close()
    env = env_fn_wrapper.x()
    
    last_cmd, last_data = None, None # 用于记录最后一次收到的命令

    while True:
        try:
            cmd, data = remote.recv()
            last_cmd, last_data = cmd, data # 在执行前记录

            if cmd == "step":
                # --- 麻将环境适配 ---
                # 'data' 是一个动作字典，例如 {'player_1': 123}
                # env.step 返回的是 (obs_dict, reward_dict, done_bool, global_obs)
                obs, reward, done, global_obs = env.step(data)
                
                if done:
                    # 如果对局结束，将结束前的最后观测和信息保存在 info 中，然后重置环境
                    # 这使得主进程可以访问到导致 episode 结束的最后状态
                    # terminal_global_obs = global_obs
                    # 重置环境，获取新一局的初始观测
                    obs, global_obs = env.reset()
                
                # 发送 (obs, reward, done, global_obs) 到主进程
                remote.send((obs, reward, done, global_obs))

            elif cmd == "reset":
                # 主进程请求重置环境
                obs, global_obs = env.reset()
                remote.send((obs, global_obs))

            elif cmd == "close":
                # 主进程请求关闭环境
                env.close()
                remote.close()
                break # 退出循环，结束子进程

            elif cmd == "get_spaces":
                # 主进程请求获取环境的空间信息
                remote.send((env.observation_space, env.action_space))

            else:
                raise NotImplementedError(f"未知命令: {cmd}")
        except EOFError:
            print(f"Worker (PID: {os.getpid()}) detected pipe closure. Exiting.")
            break
            
        except Exception as e:
            # --- 这是关键的修改部分 ---
            # 捕获到任何异常时，打印详细的错误信息、traceback 和导致错误的命令/数据
            print("="*60)
            print(f"!!! Worker (PID: {os.getpid()}) encountered a critical error !!!")
            print(f"    - Last received command: {last_cmd}")
            print(f"    - Data for last command: {last_data}")
            print(f"    - Error Type: {type(e).__name__}")
            print(f"    - Error Message: {e}")
            print("    - Traceback:")
            # 使用 traceback.print_exc() 打印完整的 traceback 到标准错误流
            traceback.print_exc()
            print("="*60)
            # --- 修改结束 ---
            
            remote.close() # 关闭管道
            break # 退出循环

class VecEnv(ABC):
    """
    向量化环境的抽象基类，定义了基本接口。
    """
    closed = False
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True
    
    def close_extras(self):
        pass

    def step(self, actions):
        """同步执行 step，方便使用。"""
        self.step_async(actions)
        return self.step_wait()


class SubprocVecEnv(VecEnv):
    """
    一个在独立子进程中运行多个环境的向量化环境实现。
    
    Args:
        env_fns (list): 一个函数列表，每个函数在被调用时会创建一个新的环境实例。
                        例如: [lambda: MahjongGBEnv(config) for _ in range(num_envs)]
    """
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        num_envs = len(env_fns)
        
        # 为每个环境创建一个管道用于进程间通信
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        
        # 创建并启动子进程
        self.processes = [
            Process(
                target=worker, 
                args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        
        for p in self.processes:
            p.daemon = True  # 设置为守护进程，如果主进程退出，子进程也会被终止
            p.start()
        
        # 在主进程中关闭子进程端的管道
        for remote in self.work_remotes:
            remote.close()

        # 初始化时从第一个子环境中获取空间信息
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        
        super(SubprocVecEnv, self).__init__(num_envs, observation_space, action_space)

    def step_async(self, actions):
        """
        异步发送动作到所有子进程。

        Args:
            actions (list): 包含 num_envs 个动作的列表。对于麻将，每个动作是一个字典，
                            例如 [{'player_1': 34}, {'player_3': 12}, ...]。
        """
        if self.waiting:
            raise RuntimeError("Already waiting for a step to complete.")
        
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        """
        等待并收集所有子进程的 step 结果。

        Returns:
            tuple: (obs, rewards, dones, global_obs_list)
                - obs (list): 包含 num_envs 个观测字典的列表。
                - rewards (list): 包含 num_envs 个奖励字典的列表。
                - dones (np.ndarray): 一个布尔数组，形状为 (num_envs,)，标记哪些环境已结束。
                - global_obs_list (list): 包含 num_envs 个全局观测数组的列表。
        """
        if not self.waiting:
            raise RuntimeError("Trying to wait for a step that was not A-sync'd.")
        
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        # 将结果解包
        obs, rewards, dones, global_obs_list = zip(*results)
        
        # dones 可以直接堆叠成一个 NumPy 数组
        # obs, rewards, global_obs_list 保持为列表，因为在多智能体回合制环境中，
        # 它们的结构（例如，字典的键）在每一步都可能不同。
        # Actor 端需要遍历这些列表来处理每个环境的数据。
        return list(obs), list(rewards), np.stack(dones), list(global_obs_list)

    def reset(self):
        """重置所有环境并返回初始观测和全局观测。"""
        for remote in self.remotes:
            remote.send(("reset", None))
        
        results = [remote.recv() for remote in self.remotes]
        obs_list, global_obs_list = zip(*results)
        return list(obs_list), list(global_obs_list)  # 返回观测字典列表和全局观测列表

    def close(self):
        """关闭所有子进程和管道。"""
        if self.closed:
            return
        
        if self.waiting:
            # 如果主进程在等待时被要求关闭，先接收完挂起的响应
            for remote in self.remotes:
                try:
                    remote.recv()
                except EOFError:
                    pass # 如果管道已经关闭，则忽略

        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
                pass # 如果子进程已经退出，则忽略

        for p in self.processes:
            p.join() # 等待所有子进程结束
            
        self.closed = True
