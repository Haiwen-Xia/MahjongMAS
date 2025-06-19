import torch
import torch.multiprocessing as mp # 为更好地处理CUDA，使用torch的多进程
import time
import os
import logging
import sys # 用于 sys.exit 在信号处理器中
from queue import Empty as QueueEmpty, Full as QueueFull # 用于处理队列操作
from threading import Lock # 用于保护模型并发访问
from collections import defaultdict, OrderedDict # 用于模型分组和LRU缓存

# 假设这些模块可被此文件访问
try:
    # 您的模型类现在应该在 models/ 目录下
    from models.actor import ResNet34Actor
    from models.critic import ResNet34CentralizedCritic 
    from utils import setup_process_logging_and_tensorboard # 您的日志工具
except ImportError as e:
    print(f"导入模块失败: {e}。请确保模型和工具类可访问。为演示将使用占位符。")
    class ResNet34Actor(torch.nn.Module): pass
    class ResNet34CentralizedCritic(torch.nn.Module): pass
    def setup_process_logging_and_tensorboard(*args, **kwargs): return logging.getLogger(), None

# --- 配置默认值 ---
DEFAULT_IN_CHANNELS = 187
DEFAULT_CRITIC_EXTRA_IN_CHANNELS = 16 # 示例值
DEFAULT_OUT_CHANNELS = 235
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class InferenceServer(mp.Process):
    """
    InferenceServer (方案 B: 推理服务器 + 单一学习器 架构)
    职责：
    1. 在GPU上托管用于推理的最新Actor和唯一的中心化Critic。
    2. 托管一组固定的基准模型。
    3. 响应 Learner 的命令，用 Learner 训练好的模型参数更新自己持有的模型。
    4. 响应 Actors 的推理请求，并进行批处理以提高效率。
    """
    def __init__(self, 
                 config: dict, 
                 cmd_queue_from_learner: mp.Queue, 
                 inference_req_queue_from_actors: mp.Queue, 
                 inference_resp_queues_to_actors: dict):
        """
        初始化 InferenceServer。
        此方法在父进程 (train.py) 中调用，应避免任何不可序列化的操作。
        """
        super(InferenceServer, self).__init__()
        
        # --- 1. 基本配置和身份信息 ---
        self.config = config
        self.server_id = config.get('server_id', f"server_{os.getpid()}")
        self.name = config.get('name', f'InferenceServer-{self.server_id}')
        
        # --- 2. 日志、设备和模型类 (将在 run() 中实际使用) ---
        self.logger = None 
        self.writer = None
        self.device = torch.device(self.config.get('device', DEFAULT_DEVICE))
        
        # 从配置中获取模型类定义
        self.ActorModelClass = self.config.get('actor_class', ResNet34Actor)
        self.CriticModelClass = self.config.get('critic_class', ResNet34CentralizedCritic)
        # 从配置中获取模型输入/输出维度
        self.actor_in_channels = self.config.get('in_channels', DEFAULT_IN_CHANNELS)
        self.actor_out_channels = self.config.get('out_channels', DEFAULT_OUT_CHANNELS)
        self.critic_obs_in_channels = self.config.get('in_channels', DEFAULT_IN_CHANNELS) # Critic的局部观测通道数
        self.critic_extra_in_channels = self.config.get('critic_extra_in_channels', DEFAULT_CRITIC_EXTRA_IN_CHANNELS) # Critic的额外信息通道数


        # --- 3. 通信队列 ---
        self.cmd_queue_from_learner = cmd_queue_from_learner
        self.inference_req_queue_from_actors = inference_req_queue_from_actors
        self.inference_resp_queues_to_actors = inference_resp_queues_to_actors

        # --- 4. 模型占位符 (将在 run() 中创建实例) ---
        # self.model_train 被移除，因为训练发生在 Learner 端
        self.latest_actor_for_inference = None # 用于 Actor 推理的、定期同步的策略网络
        self.benchmark_actors = {}           # 基准策略网络: {name: model_instance}
        self.critic_network = None           # 唯一的、中心化的 Critic 网络


        # --- 6. 同步和控制属性 ---
        self.model_lock = None # 锁，将在 run() 中初始化
        self.shutting_down = False 

        # --- 7. 推理批处理 (Batching) 参数 ---
        self.max_inference_batch_size = self.config.get('inference_batch_size', 8)
        self.max_inference_batch_wait_time_sec = self.config.get('inference_max_wait_ms', 10) / 1000.0
        self.inference_request_buffer = [] 
        self.last_batch_processed_time = None 

        print(f"InferenceServer '{self.name}' __init__ completed in parent process.")

    
    # _setup_logging, _load_state_dict_to_model, _prepare_batch_from_requests, _process_inference_batch
    # 等辅助方法与之前版本类似，但需要确保它们引用正确的模型属性 (例如 self.latest_actor_for_inference)
    # 这里将重点展示 _initialize_models 和 _handle_learner_command 的修改

    def _initialize_models(self):
        """
        初始化服务器上托管的所有模型实例并加载初始权重。
        此方法在子进程的 run() 方法中被调用。
        """
        self.logger.info(f"Initializing models on device: {self.device}")
        
        # 1. 初始化最新的评估 Actor (latest_actor_for_inference)
        self.logger.info(f"Creating latest_actor_for_inference instance of {self.ActorModelClass.__name__}...")
        self.latest_actor_for_inference = self.ActorModelClass(self.actor_in_channels, self.actor_out_channels).to(self.device)
        
        # 从配置中获取初始 Actor 模型路径
        initial_actor_path = self.config.get('initial_actor_eval_path') 
        if initial_actor_path:
            self._load_state_dict_to_model(self.latest_actor_for_inference, initial_actor_path, "latest_actor_for_inference (initial)")
        else:
            self.logger.warning("No initial_actor_eval_path provided. Using randomly initialized weights for latest_actor_for_inference.")
        
        self.latest_actor_for_inference.eval()  # 推理模式

        # 2. 初始化中心化的 Critic 网络
        self.logger.info(f"Creating centralized critic_network instance of {self.CriticModelClass.__name__}...")
        self.critic_network = self.CriticModelClass(
            in_channels_obs=self.critic_obs_in_channels,
            in_channels_extra=self.critic_extra_in_channels
        ).to(self.device)

        # 从配置中获取初始 Critic 模型路径
        initial_critic_path = self.config.get('initial_critic_eval_path')
        if initial_critic_path:
            self._load_state_dict_to_model(self.critic_network, initial_critic_path, "critic_network (initial)")
        else:
            self.logger.warning("No initial_critic_eval_path provided. Using randomly initialized weights for critic_network.")

        self.critic_network.eval() # 推理时 Critic 也处于评估模式

        # 3. 初始化基准 Actors (Benchmark Actors)
        benchmark_infos_dict = self.config.get('benchmark_models_info', {})
        self.logger.info(f"Found {len(benchmark_infos_dict)} benchmark model(s) in config.")
        for bm_name, bm_info_dict in benchmark_infos_dict.items():
            bm_path = bm_info_dict.get('path')
            if bm_path:
                bm_instance = self.ActorModelClass(self.actor_in_channels, self.actor_out_channels).to(self.device)
                if self._load_state_dict_to_model(bm_instance, bm_path, f"benchmark model '{bm_name}'"):
                    bm_instance.eval()
                    self.benchmark_actors[bm_name] = bm_instance
                    self.logger.info(f"Benchmark actor '{bm_name}' loaded successfully from {bm_path}.")
                else:
                    self.logger.error(f"Failed to load benchmark actor '{bm_name}' from path {bm_path}.")
        
        self.logger.info(f"Finished initializing models. Loaded {len(self.benchmark_actors)} benchmark actor(s).")


    def _handle_learner_command(self, command_tuple: tuple):
        """
        处理来自 Learner 进程的命令。
        现在只处理模型更新和关闭命令。
        """
        try:
            cmd_type, cmd_data = command_tuple[0], command_tuple[1] if len(command_tuple) > 1 else None
            self.logger.info(f"Received command from Learner: Type='{cmd_type}'")

            if cmd_type == "UPDATE_ACTOR_MODEL":
                # Learner 发送了新的 Actor 权重
                new_actor_state_dict = cmd_data
                if isinstance(new_actor_state_dict, dict):
                    with self.model_lock:
                        self.logger.info("Attempting to update latest_actor_for_inference with new state_dict from Learner...")
                        if self._load_state_dict_to_model(self.latest_actor_for_inference, new_actor_state_dict, "latest_actor (update)"):
                            self.logger.info("latest_actor_for_inference successfully updated.")
                        else:
                            self.logger.error("Failed to update latest_actor_for_inference.")
                else:
                    self.logger.error(f"Invalid data for UPDATE_ACTOR_MODEL: expected dict, got {type(new_actor_state_dict)}. Ignored.")
            
            elif cmd_type == "UPDATE_CRITIC_MODEL":
                # Learner 发送了新的 Critic 权重
                new_critic_state_dict = cmd_data
                if isinstance(new_critic_state_dict, dict):
                    with self.model_lock:
                        self.logger.info("Attempting to update critic_network with new state_dict from Learner...")
                        if self._load_state_dict_to_model(self.critic_network, new_critic_state_dict, "critic_network (update)"):
                            self.logger.info("critic_network successfully updated.")
                        else:
                            self.logger.error("Failed to update critic_network.")
                else:
                    self.logger.error(f"Invalid data for UPDATE_CRITIC_MODEL: expected dict, got {type(new_critic_state_dict)}. Ignored.")

            elif cmd_type == "SHUTDOWN":
                self.logger.info("Received SHUTDOWN command. InferenceServer will stop.")
                self.shutting_down = True 
            
            else:
                self.logger.warning(f"Received unknown command type from Learner: '{cmd_type}'. Ignored.")

        except Exception as e_cmd_handle:
            cmd_type_log = command_tuple[0] if command_tuple and len(command_tuple) > 0 else 'UnknownCmd'
            self.logger.error(f"Error handling learner command '{cmd_type_log}': {e_cmd_handle}", exc_info=True)


    def _setup_logging(self):
        """初始化日志记录器和 TensorBoard writer。"""
        # 根据配置决定使用主要日志还是详细日志
        log_type = 'detailed' if self.config.get('enable_detailed_logging', True) else 'main'
        
        try:
            # 使用新的日志设置函数
            self.logger, self.writer, server_log_paths = setup_process_logging_and_tensorboard(
                self.config['log_base_dir'], self.config, self.name, log_type=log_type
            )
        except Exception as e_log:
            print(f"CRITICAL [{self.name}]: Failed to initialize logger/writer via setup_tool: {e_log}")
            self.logger = logging.getLogger(f"{__name__}.{self.name}") 
            if not self.logger.hasHandlers(): 
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter('%(asctime)s [%(name)s/%(levelname)s] %(message)s') # 保持日志格式一致
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO) 
        
        if not self.logger: 
             print(f"CRITICAL [{self.name}]: Logger is None after setup attempt. InferenceServer cannot proceed.")
             raise RuntimeError("Logger initialization failed in InferenceServer.")

    def _load_state_dict_to_model(self, model_instance: torch.nn.Module, state_dict_path_or_data, model_description: str) -> bool:
        """将 state_dict (来自路径或直接数据) 加载到给定的模型实例中。"""
        try:
            actual_state_dict_to_load = None
            if isinstance(state_dict_path_or_data, str): 
                if not os.path.isfile(state_dict_path_or_data):
                    self.logger.error(f"Model file not found for {model_description}: {state_dict_path_or_data}")
                    return False
                self.logger.info(f"Loading state_dict for {model_description} from path: {state_dict_path_or_data}")
                checkpoint = torch.load(state_dict_path_or_data, map_location=self.device) # 直接加载到目标设备
                actual_state_dict_to_load = checkpoint
            elif isinstance(state_dict_path_or_data, dict): 
                self.logger.info(f"Loading provided state_dict directly for {model_description}.")
                actual_state_dict_to_load = state_dict_path_or_data
            else:
                self.logger.error(f"Invalid type for state_dict_path_or_data for {model_description}: {type(state_dict_path_or_data)}")
                return False

            if isinstance(actual_state_dict_to_load, dict) and 'model_state_dict' in actual_state_dict_to_load:
                actual_state_dict_to_load = actual_state_dict_to_load['model_state_dict']
            
            model_instance.load_state_dict(actual_state_dict_to_load)
            self.logger.info(f"Successfully loaded state_dict into {model_description}.")
            return True
        except Exception as e_load:
            self.logger.error(f"Failed to load state_dict for {model_description}: {e_load}", exc_info=True)
            return False

    def _prepare_batch_from_requests(self, requests_for_batch: list) -> tuple:
        """
        一个辅助函数，将一组针对同一模型的推理请求打包成批处理张量。
        现在它会同时处理局部观测和全局状态。

        Args:
            requests_for_batch (list): 一个列表，每个元素是元组 
                                       (actor_id, request_id, model_key, observation_data_cpu)。
                                       observation_data_cpu 现在应包含全局状态。

        Returns:
            tuple or None: 如果成功，返回 (batch_local_obs_dict, batch_global_obs_tensor, original_indices_map)。
                           如果准备失败，返回 None。
        """
        if not requests_for_batch:
            return None

        # 用于收集批次中各个部分的列表
        local_obs_list = []
        mask_list = []
        global_obs_list = [] # 新增：用于收集全局状态
        original_indices_map = [] # (actor_id, request_id)

        for actor_id, request_id, model_key, obs_data_cpu in requests_for_batch:
            # 提取局部观测
            local_obs = obs_data_cpu.get('obs', {})
            obs_np = local_obs.get('observation')
            mask_np = local_obs.get('action_mask')
            
            # 新增：提取全局状态 (centralized_extra_info)
            # Actor 在发送请求时，需要将全局状态包含在 observation_data_cpu 中
            global_obs_np = obs_data_cpu.get('centralized_extra_info')

            # 数据有效性检查
            if obs_np is None or mask_np is None or global_obs_np is None:
                self.logger.error(f"Invalid obs_data in batch for actor '{actor_id}', req_id {request_id}. "
                                  f"Missing 'observation', 'action_mask', or 'centralized_extra_info'. Skipping this request.")
                # 给这个 Actor 发送一个错误/默认响应
                if actor_id in self.inference_resp_queues_to_actors:
                    try:
                        self.inference_resp_queues_to_actors[actor_id].put((request_id, 0, 0.0, 0.0), block=False)
                    except QueueFull:
                        self.logger.warning(f"Response queue for actor {actor_id} is full when sending error for req_id {request_id}.")
                continue # 跳过这个无效的请求

            local_obs_list.append(torch.from_numpy(obs_np))
            mask_list.append(torch.from_numpy(mask_np))
            global_obs_list.append(torch.from_numpy(global_obs_np))
            original_indices_map.append((actor_id, request_id))
        
        if not original_indices_map: # 如果所有请求都无效
            return None

        try:
            # 将列表堆叠成批处理张量并移动到目标设备
            collated_obs_tensor = torch.stack(local_obs_list).to(self.device, dtype=torch.float)
            collated_mask_tensor = torch.stack(mask_list).to(self.device, dtype=torch.float)
            collated_global_obs_tensor = torch.stack(global_obs_list).to(self.device, dtype=torch.float)

            # 将局部观测和掩码重新打包成字典格式，以匹配模型输入
            batch_local_obs_dict = {
                'obs': {
                    'observation': collated_obs_tensor,
                    'action_mask': collated_mask_tensor
                }
            }
            return batch_local_obs_dict, collated_global_obs_tensor, original_indices_map
        except Exception as e_stack:
            self.logger.error(f"Error stacking tensors for batch inference: {e_stack}. Shapes might be inconsistent.", exc_info=True)
            # 为这个批次中所有有效的请求发送错误响应
            for actor_id, request_id in original_indices_map:
                if actor_id in self.inference_resp_queues_to_actors:
                    try: self.inference_resp_queues_to_actors[actor_id].put((request_id, 0, 0.0, 0.0), block=False)
                    except QueueFull: pass
            return None


    def _process_inference_batch(self):
        """
        处理累积的推理请求批次。
        此方法实现了按模型键名分组、批处理，并使用中心化Critic计算价值。
        """
        if not self.inference_request_buffer:
            self.logger.debug("_process_inference_batch called with empty request buffer, skipping.")
            return

        total_requests = len(self.inference_request_buffer)
        self.logger.debug(f"处理推理批次: 共 {total_requests} 个请求需要处理")
        
        requests_by_model_key = defaultdict(list)
        for req_tuple in self.inference_request_buffer:
            requests_by_model_key[req_tuple[2]].append(req_tuple) # 按 model_key 分组
        
        # 记录每种模型类型的请求数量
        model_key_counts = {key: len(reqs) for key, reqs in requests_by_model_key.items()}
        self.logger.debug(f"批次请求模型分布: {model_key_counts}")
        
        self.inference_request_buffer.clear() # 清空主缓冲

        # 为每个模型键的组执行批推理
        for model_key, requests_for_this_model in requests_by_model_key.items():
            self.logger.debug(f"处理模型'{model_key}'的批次: {len(requests_for_this_model)}个请求")
            
            # 记录一些请求的actor_id，帮助跟踪
            sample_actor_ids = [req[0] for req in requests_for_this_model[:min(3, len(requests_for_this_model))]]
            self.logger.debug(f"模型'{model_key}'批次中的样本actor_ids: {sample_actor_ids}...")
            
            prepared_batch = self._prepare_batch_from_requests(requests_for_this_model)
            if prepared_batch is None:
                self.logger.error(f"Failed to prepare batch for model_key '{model_key}'.")
                continue

            batch_local_obs_dict, batch_global_obs_tensor, original_indices = prepared_batch
            batch_size = len(original_indices)
            self.logger.debug(f"为模型'{model_key}'准备了批次，有效大小={batch_size}")
            
            # --- 在锁保护下，执行两次模型前向传播 ---
            with self.model_lock, torch.no_grad():
                # 1. 获取要使用的 Actor (策略) 模型
                actor_to_use = None
                if model_key == "latest_eval":
                    actor_to_use = self.latest_actor_for_inference
                elif model_key in self.benchmark_actors:
                    actor_to_use = self.benchmark_actors[model_key]
                else:
                    self.logger.warning(f"Unknown model_key '{model_key}' in batch. Using 'latest_eval' as fallback.")
                    actor_to_use = self.latest_actor_for_inference
                
                # 确保模型实例存在
                if actor_to_use is None or self.critic_network is None:
                    self.logger.error(f"Model instance for inference is None (actor or critic). Cannot process batch for key '{model_key}'.")
                    # ... (发送错误响应) ...
                    continue 

                actor_to_use.eval()
                self.critic_network.eval() # 推理时 Critic 也应在评估模式

                # 2. 执行 Actor 和 Critic 的前向传播
                try:
                    inference_start = time.time()
                    self.logger.debug(f"模型'{model_key}'前向传播开始，批次大小={len(batch_local_obs_dict['obs']['observation'])}")
                    
                    # 使用 Actor 网络获取 logits
                    batch_logits = actor_to_use(batch_local_obs_dict)
                    
                    # 始终使用唯一的中心化 Critic 网络获取价值
                    # 假设 CentralizedCritic 的 forward 方法接收 (local_obs_dict, global_obs_tensor)
                    batch_values = self.critic_network(batch_local_obs_dict, batch_global_obs_tensor)
                    
                    inference_time = (time.time() - inference_start) * 1000  # 转换为毫秒
                    self.logger.debug(f"模型'{model_key}'前向传播完成，耗时={inference_time:.2f}ms，批次大小={len(batch_values)}")

                except Exception as e_batch_infer:
                    self.logger.error(f"Error during batch inference for model_key '{model_key}': {e_batch_infer}", exc_info=True)
                    # ... (为这个批次的所有请求发送错误响应) ...
                    continue 
            
            # --- 处理批处理结果 ---
            try:
                dispatch_start = time.time()
                self.logger.debug(f"开始处理批次结果并分发给Actors，批次大小={len(original_indices)}")
                
                # 从 logits 计算动作和 log_prob
                batch_mask_tensor = batch_local_obs_dict['obs']['action_mask']
                masked_batch_logits = batch_logits + torch.clamp(torch.log(batch_mask_tensor), min=-1e9)
                batch_action_dist = torch.distributions.Categorical(logits=masked_batch_logits)
                
                batch_actions_sampled = batch_action_dist.sample()
                batch_log_probs = batch_action_dist.log_prob(batch_actions_sampled)
                
                # 统计值域范围，帮助调试
                min_value = batch_values.min().item()
                max_value = batch_values.max().item()
                mean_value = batch_values.mean().item()
                self.logger.debug(f"批次值函数统计: 最小={min_value:.4f}, 最大={max_value:.4f}, 平均={mean_value:.4f}")

                # 记录一些动作样本
                action_samples = batch_actions_sampled[:min(5, len(batch_actions_sampled))].tolist()
                self.logger.debug(f"批次动作样本: {action_samples}")

                # 将结果分发回各自的 Actor
                response_success = 0
                response_fail = 0
                
                for i in range(len(original_indices)):
                    actor_id_resp, request_id_resp = original_indices[i]
                    
                    response_payload = (
                        request_id_resp,
                        batch_actions_sampled[i].item(),
                        batch_values[i].item(), # 来自中心化 Critic 的价值
                        batch_log_probs[i].item() # 来自所选 Actor 的 log_prob
                    )
                    
                    # (发送响应到 actor_id_resp 的队列)
                    if actor_id_resp in self.inference_resp_queues_to_actors:
                        try:
                            self.inference_resp_queues_to_actors[actor_id_resp].put(response_payload, block=False)
                            response_success += 1
                            self.logger.debug(f"分发响应: request_id={request_id_resp}, actor_id={actor_id_resp}, "
                                              f"action={response_payload[1]}, value={response_payload[2]:.4f}")
                        except QueueFull:
                            response_fail += 1
                            self.logger.warning(f"响应队列已满! actor_id={actor_id_resp}, request_id={request_id_resp}")
                
                dispatch_time = (time.time() - dispatch_start) * 1000  # 毫秒
                self.logger.debug(f"批次响应分发完成: 成功={response_success}, 失败={response_fail}, 耗时={dispatch_time:.2f}ms")
            except Exception as e_dispatch:
                self.logger.error(f"批次结果分发过程中出错: model_key='{model_key}', 错误={e_dispatch}", exc_info=True)
                # 记录更多上下文信息，帮助调试
                self.logger.error(f"批次分发错误上下文: 批次大小={len(original_indices)}, "
                                  f"批次形状={batch_local_obs_dict['obs']['observation'].shape if 'obs' in batch_local_obs_dict else 'Unknown'}")

        batch_end_time = time.time()
        total_batch_time = (batch_end_time - self.last_batch_processed_time) * 1000  # 毫秒
        self.logger.debug(f"批次处理总耗时: {total_batch_time:.2f}ms (包含等待时间)")
        self.last_batch_processed_time = batch_end_time  # 更新批处理计时器

 
    def run(self):
        """InferenceServer 进程的主执行循环。"""
        # 1. 初始化
        self._setup_logging() 
        self.logger.info(f"InferenceServer process {self.name} (PID {os.getpid()}) has started on device {self.device}.")
        
        self.model_lock = Lock() 
        self.logger.info("Model lock initialized within InferenceServer run method.")

        try:
            torch.set_num_threads(1) 
            self.logger.info("PyTorch num_threads set to 1 for InferenceServer.")
        except Exception as e_threads:
            self.logger.warning(f"Failed to set torch num_threads for InferenceServer: {e_threads}")

        try:
            self._initialize_models() # 初始化模型和算法实例
        except Exception as e_init_models:
            self.logger.critical(f"CRITICAL error during model initialization: {e_init_models}. InferenceServer cannot start.", exc_info=True)
            if self.writer: self.writer.close()
            return 

        last_status_report_time = time.time()
        status_report_interval = 10.0  # 每10秒记录一次状态
        self.logger.info("InferenceServer entering main event loop...")
        self.last_batch_processed_time = time.time() # 初始化批处理计时器

        # 2. 主事件循环
        while not self.shutting_down:
            current_time = time.time()
            processed_in_this_cycle = False # 标记本轮循环是否处理了任何事情
            
            # 定期记录系统状态
            if current_time - last_status_report_time > status_report_interval:
                # 尝试获取请求队列的大小
                req_queue_size = "N/A"
                try:
                    req_queue_size = self.inference_req_queue_from_actors.qsize()
                except Exception:
                    pass
                    
                # 计算处理请求的速率
                buffer_size = len(self.inference_request_buffer)
                time_since_last_batch = current_time - self.last_batch_processed_time
                
                # 记录响应队列状态
                resp_queue_sizes = {}
                for actor_id, queue in self.inference_resp_queues_to_actors.items():
                    try:
                        resp_queue_sizes[actor_id] = queue.qsize()
                    except Exception:
                        resp_queue_sizes[actor_id] = "N/A"
                
                # 只记录几个代表性的响应队列信息，避免日志过长
                resp_queue_sample = {k: resp_queue_sizes[k] for k in list(resp_queue_sizes.keys())[:3]} if resp_queue_sizes else {}
                
                self.logger.debug(f"推理服务器状态: 请求队列大小={req_queue_size}, 缓冲区大小={buffer_size}, "
                                f"上次批处理距今={time_since_last_batch:.1f}秒, "
                                f"响应队列样本={resp_queue_sample}")
                
                last_status_report_time = current_time

            # a. 优先处理所有来自 Learner 的命令
            try:
                while not self.cmd_queue_from_learner.empty():
                    learner_cmd_tuple = self.cmd_queue_from_learner.get_nowait()
                    self._handle_learner_command(learner_cmd_tuple)
                    processed_in_this_cycle = True
                    if self.shutting_down: break # 如果收到关闭命令，立即跳出命令处理循环
            except QueueEmpty:
                pass # Learner 命令队列为空是正常情况
            except Exception as e_cmd_loop_main:
                self.logger.error(f"Error processing command from Learner queue in main loop: {e_cmd_loop_main}", exc_info=True)
            
            if self.shutting_down: break # 检查是否需要关闭

            # b. 从 Actors 收集推理请求到缓冲区
            try:
                # 尽可能多地从请求队列中获取请求，直到达到批次大小上限
                while len(self.inference_request_buffer) < self.max_inference_batch_size:
                    actor_req_tuple = self.inference_req_queue_from_actors.get_nowait() # 非阻塞获取
                    self.inference_request_buffer.append(actor_req_tuple)
                    processed_in_this_cycle = True 
            except QueueEmpty:
                pass # Actor 请求队列为空是正常的
            except Exception as e_actor_q_get:
                 self.logger.error(f"从Actor请求队列获取数据时出错: {e_actor_q_get}", exc_info=True)
                 # 记录当前缓冲区状态以帮助调试
                 self.logger.error(f"错误时缓冲区状态: 请求缓冲区大小={len(self.inference_request_buffer)}, "
                                  f"最大批次大小={self.max_inference_batch_size}")

            # c. 检查是否满足批处理条件
            current_time = time.time()
            time_since_last_batch = current_time - self.last_batch_processed_time
            
            buffer_is_full = len(self.inference_request_buffer) >= self.max_inference_batch_size
            timeout_reached = time_since_last_batch >= self.max_inference_batch_wait_time_sec

            # 记录满足的批处理条件
            if self.inference_request_buffer and (buffer_is_full or timeout_reached):
                reason = "缓冲区已满" if buffer_is_full else "等待超时"
                self.logger.debug(f"触发批处理: 原因={reason}, 缓冲区大小={len(self.inference_request_buffer)}, "
                                 f"等待时间={time_since_last_batch:.4f}秒")
                self._process_inference_batch() # 处理并清空 self.inference_request_buffer
                processed_in_this_cycle = True # 标记已处理

            # d. 如果本轮什么都没做，短暂休眠以减少CPU空转
            if not processed_in_this_cycle:
                time.sleep(0.0001) # 100微秒

        # 3. 循环结束后的清理工作
        self.logger.info(f"InferenceServer {self.name} (PID {os.getpid()}) is shutting down its main loop.")
        if self.writer:
            self.logger.info("Closing InferenceServer's TensorBoard writer.")
            self.writer.close()
        self.logger.info(f"InferenceServer {self.name} (PID {os.getpid()}) has shut down completely.")
