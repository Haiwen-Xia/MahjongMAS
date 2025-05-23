import torch
import torch.multiprocessing as mp # 为更好地处理CUDA，使用torch的多进程
import time
import os
import logging
import sys # 用于 sys.exit 在信号处理器中
from queue import Empty as QueueEmpty # 用于队列的非阻塞 get
from threading import Lock # 用于保护模型并发访问
from collections import defaultdict # 用于按模型键对请求进行分组

# 假设这些模块可被此文件访问
try:
    from model import ResNet34AC # 您的模型类
    from utils import setup_process_logging_and_tensorboard # 您的日志工具
except ImportError as e:
    print(f"导入模块失败: {e}. 请确保 model.py 和 utils.py 可访问。")
    raise # 重新抛出异常，因为这些是核心依赖

# --- 配置默认值 (可以被外部配置覆盖) ---
DEFAULT_BENCHMARK_MODEL_INFO = { # 示例基准模型信息
    # "benchmark_il_policy": {"path": "/path/to/your/il_model.pth"},
}
DEFAULT_IN_CHANNELS = 187 # 根据您的模型调整 (之前是14，现在根据您的代码片段设为187)
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 默认设备

class InferenceServer(mp.Process):
    """
    InferenceServer 作为一个独立的进程运行，负责：
    1. 在GPU上维护一个最新的评估模型 (model_eval)。
    2. 在GPU上托管一组固定的基准模型 (benchmark_models)。
    3. 响应 Learner 的命令，用 Learner 训练好的模型参数更新 model_eval。
    4. 响应 Actors 的推理请求，使用指定的模型 (model_eval 或某个 benchmark_model) 进行推理并返回结果。
       (理想情况下，会进行批处理以提高效率)
    """
    def __init__(self, 
                 config: dict, 
                 cmd_queue_from_learner: mp.Queue, 
                 inference_req_queue_from_actors: mp.Queue, 
                 inference_resp_queues_to_actors: dict):
        """
        初始化 InferenceServer。
        (参数描述与之前相同)
        """
        super(InferenceServer, self).__init__() # 调用父类 Process 的初始化
        self.config = config
        self.server_id = config.get('server_id', f"server_{os.getpid()}") # 如果未指定server_id，则使用当前进程ID
        self.name = config.get('name', f'InferenceServer-{self.server_id}') # 设置进程名称，便于日志识别
        
        self.logger = None 
        self.writer = None 

        self.device = torch.device(self.config.get('device', DEFAULT_DEVICE)) 
        self.in_channels = self.config.get('in_channels', DEFAULT_IN_CHANNELS) 
        self.ModelClass = self.config.get('model_definition_class', ResNet34AC) 

        # 通信队列
        self.cmd_queue_from_learner = cmd_queue_from_learner
        self.inference_req_queue_from_actors = inference_req_queue_from_actors
        self.inference_resp_queues_to_actors = inference_resp_queues_to_actors

        # GPU 上的模型实例
        self.model_eval = None      
        self.benchmark_models = {}  

        self.model_lock = None # 将在 run() 方法中初始化
        self.shutting_down = False 

        # 用于批处理推理的参数
        self.max_inference_batch_size = self.config.get('inference_batch_size', 8) # 默认批处理大小增加到8
        self.max_inference_batch_wait_time_sec = self.config.get('inference_max_wait_ms', 10) / 1000.0 # 等待时间（秒），例如10ms
        self.inference_request_buffer = [] 
        self.last_batch_processed_time = time.time()

    def _setup_logging(self):
        """初始化日志记录器和 TensorBoard writer。"""
        log_base_dir = self.config.get('log_base_dir', './logs_inference_server') 
        experiment_name = self.config.get('experiment_name', 'default_experiment') 
        try:
            # 注意: setup_process_logging_and_tensorboard 的第三个参数之前是 process_name
            # 如果它期望的是一个通用名称，那么 self.name (例如 'InferenceServer-12345') 是合适的
            # 如果它期望的是一个固定的类型名，例如 'inference_server'，则需要调整
            # 根据之前的 utils.py，它接受 process_type 和 process_id
            self.logger, self.writer = setup_process_logging_and_tensorboard(
                log_base_dir, experiment_name, 
                process_name='inference_server'
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

    def _initialize_models(self):
        """初始化服务器上托管的所有模型实例并加载初始权重。"""
        self.logger.info(f"Initializing models on device: {self.device}")
        
        self.logger.info(f"Creating model_eval instance of {self.ModelClass.__name__}...")
        self.model_eval = self.ModelClass(self.in_channels).to(self.device)
        initial_eval_model_path = self.config.get('initial_model_eval_path', self.config.get('supervised_model_path'))
        
        if initial_eval_model_path:
            if self._load_state_dict_to_model(self.model_eval, initial_eval_model_path, "model_eval (initial)"):
                 self.logger.info(f"model_eval initialized with weights from: {initial_eval_model_path}")
            else:
                self.logger.warning(f"Failed to load initial weights for model_eval from {initial_eval_model_path}. Using random init.")
        else:
            self.logger.info("model_eval initialized with random weights. Awaiting first update from Learner.")
        self.model_eval.eval()

        benchmark_infos_dict = self.config.get('benchmark_models_info', DEFAULT_BENCHMARK_MODEL_INFO)
        self.logger.info(f"Attempting to load {len(benchmark_infos_dict)} benchmark model(s) from config.")
        for bm_name, bm_info_dict in benchmark_infos_dict.items():
            bm_model_path = bm_info_dict.get('path')
            if bm_model_path:
                self.logger.info(f"Creating benchmark model '{bm_name}' instance...")
                bm_instance = self.ModelClass(self.in_channels).to(self.device)
                if self._load_state_dict_to_model(bm_instance, bm_model_path, f"benchmark model '{bm_name}'"):
                    bm_instance.eval() 
                    self.benchmark_models[bm_name] = bm_instance
                    self.logger.info(f"Benchmark model '{bm_name}' loaded successfully from {bm_model_path}.")
                else:
                    self.logger.error(f"Failed to load benchmark model '{bm_name}' from path {bm_model_path}.")
            else:
                self.logger.warning(f"No path provided for benchmark model '{bm_name}' in config. Skipping.")
        self.logger.info(f"Finished initializing models. Loaded {len(self.benchmark_models)} benchmark model(s).")

    def _handle_learner_command(self, command_tuple):
        """处理来自 Learner 进程的命令。"""
        try:
            cmd_type, cmd_data = command_tuple
            self.logger.info(f"Received command from Learner: Type='{cmd_type}'")

            if cmd_type == "UPDATE_EVAL_MODEL":
                new_state_dict_for_eval = cmd_data
                if not isinstance(new_state_dict_for_eval, dict):
                    self.logger.error(f"Invalid data type for UPDATE_EVAL_MODEL: expected dict, got {type(new_state_dict_for_eval)}. Ignored.")
                    return

                with self.model_lock: 
                    self.logger.info("Attempting to update model_eval with new state_dict from Learner...")
                    if self._load_state_dict_to_model(self.model_eval, new_state_dict_for_eval, "model_eval (update)"):
                        self.logger.info("model_eval successfully updated by Learner.")
                    else:
                        self.logger.error("Failed to update model_eval with state_dict from Learner.")
            
            elif cmd_type == "SHUTDOWN":
                self.logger.info("Received SHUTDOWN command. InferenceServer will stop.")
                self.shutting_down = True 
            
            else:
                self.logger.warning(f"Received unknown command type from Learner: '{cmd_type}'. Ignored.")
        except Exception as e_cmd_handle:
            cmd_type_log = command_tuple[0] if command_tuple and len(command_tuple) > 0 else 'UnknownCmd'
            self.logger.error(f"Error handling learner command '{cmd_type_log}': {e_cmd_handle}", exc_info=True)

    def _prepare_batch_from_requests(self, requests_for_batch: list) -> tuple or None:
        """
        将一组请求的观测数据打包成批处理张量。
        Args:
            requests_for_batch (list): 包含元组 (actor_id, request_id, model_key, observation_data_cpu) 的列表。
                                       这里假设所有请求都针对同一个 model_key。
        Returns:
            tuple: (collated_obs_tensor, collated_mask_tensor, original_indices_map) or None
                   original_indices_map 是一个列表，包含 (actor_id, request_id) 以便将结果映射回去。
        """
        if not requests_for_batch:
            return None

        obs_list = []
        mask_list = []
        original_indices_map = [] # 用于将批处理结果映射回原始请求

        for actor_id, request_id, model_key, obs_data_cpu in requests_for_batch:
            obs_np = obs_data_cpu.get('obs', {}).get('observation')
            mask_np = obs_data_cpu.get('obs', {}).get('action_mask')

            if obs_np is None or mask_np is None:
                self.logger.error(f"Invalid obs_data in batch for actor '{actor_id}', req_id {request_id}. Skipping this request.")
                # 给这个 Actor 发送一个错误/默认响应
                if actor_id in self.inference_resp_queues_to_actors:
                    try:
                        self.inference_resp_queues_to_actors[actor_id].put((request_id, 0, 0.0, 0.0), block=False)
                    except QueueFull:
                        self.logger.warning(f"Response queue for actor {actor_id} is full when sending error for req_id {request_id}.")
                continue 

            obs_list.append(torch.tensor(obs_np, dtype=torch.float)) # 暂时不上 unqueeze 和 to(device)
            mask_list.append(torch.tensor(mask_np, dtype=torch.float))
            original_indices_map.append((actor_id, request_id))
        
        if not obs_list: # 如果所有请求都无效
            return None

        try:
            # 在 .to(device) 之前堆叠，可以减少CPU到GPU的传输次数
            collated_obs_tensor = torch.stack(obs_list).to(self.device) # (B, C, H, W) or (B, Features)
            collated_mask_tensor = torch.stack(mask_list).to(self.device) # (B, NumActions)
            # 注意：unsqueeze(0) 在这里不需要了，因为 stack 会创建批次维度
            return collated_obs_tensor, collated_mask_tensor, original_indices_map
        except Exception as e_stack:
            self.logger.error(f"Error stacking tensors for batch inference: {e_stack}", exc_info=True)
            # 给所有相关的 Actor 发送错误响应
            for actor_id, request_id in original_indices_map:
                if actor_id in self.inference_resp_queues_to_actors:
                    try:
                        self.inference_resp_queues_to_actors[actor_id].put((request_id, 0, 0.0, 0.0), block=False)
                    except QueueFull: pass # 忽略队列满
            return None


    def _process_inference_batch(self):
        """处理累积的推理请求批次。实现真正的批处理。"""
        if not self.inference_request_buffer:
            return

        self.logger.debug(f"Attempting to process a batch of {len(self.inference_request_buffer)} inference requests.")
        
        # 1. 按 model_key 对请求进行分组
        requests_by_model_key = defaultdict(list)
        for req_tuple in self.inference_request_buffer:
            # req_tuple is (actor_id, request_id, model_key, observation_data_cpu)
            model_key = req_tuple[2]
            requests_by_model_key[model_key].append(req_tuple)
        
        self.inference_request_buffer.clear() # 清空主缓冲

        # 2. 为每个模型键的组执行批推理
        for model_key, requests_for_this_model in requests_by_model_key.items():
            self.logger.debug(f"Processing batch for model_key='{model_key}', num_requests={len(requests_for_this_model)}")
            
            model_to_use = None
            with self.model_lock: # 在选择模型时加锁
                if model_key == "latest_eval":
                    model_to_use = self.model_eval
                elif model_key in self.benchmark_models:
                    model_to_use = self.benchmark_models[model_key]
                else:
                    self.logger.warning(f"Unknown model_key '{model_key}' in batch. Using 'latest_eval' as fallback for {len(requests_for_this_model)} requests.")
                    model_to_use = self.model_eval
                
                if model_to_use is None:
                    self.logger.error(f"No model instance available for model_key '{model_key}' (fallback also None). Cannot process this batch.")
                    # 为这组请求发送错误响应
                    for actor_id_err, req_id_err, _, _ in requests_for_this_model:
                        if actor_id_err in self.inference_resp_queues_to_actors:
                            try: self.inference_resp_queues_to_actors[actor_id_err].put((req_id_err, 0, 0.0, 0.0), block=False)
                            except QueueFull: pass
                    continue # 处理下一个模型键的批次

                model_to_use.eval() # 确保评估模式

            # 准备批处理数据
            prepared_batch = self._prepare_batch_from_requests(requests_for_this_model)
            if prepared_batch is None:
                self.logger.error(f"Failed to prepare batch for model_key '{model_key}'. Requests in this sub-batch will get error responses.")
                continue # _prepare_batch_from_requests 内部已处理错误响应

            batch_obs_tensor, batch_mask_tensor, original_indices = prepared_batch
            batch_model_input = {'obs': {'observation': batch_obs_tensor, 'action_mask': batch_mask_tensor}}

            # 执行批推理
            try:
                with torch.no_grad(): # 推理时不需要梯度
                    batch_logits, batch_values = model_to_use(batch_model_input) # (B, NumActions), (B, 1)
                
                # 处理批处理结果
                batch_action_dist = torch.distributions.Categorical(logits=batch_logits)
                batch_actions_sampled = batch_action_dist.sample() # (B,)
                batch_log_probs = batch_action_dist.log_prob(batch_actions_sampled) # (B,)

                for i in range(len(original_indices)):
                    actor_id_resp, request_id_resp = original_indices[i]
                    action_resp = batch_actions_sampled[i].item()
                    value_resp = batch_values[i].item()
                    log_prob_resp = batch_log_probs[i].item()

                    if actor_id_resp in self.inference_resp_queues_to_actors:
                        try:
                            self.inference_resp_queues_to_actors[actor_id_resp].put(
                                (request_id_resp, action_resp, value_resp, log_prob_resp), timeout=0.1 # 短超时
                            )
                        except QueueFull:
                            self.logger.warning(f"Response queue full for actor {actor_id_resp} (req_id {request_id_resp}) after batch inference.")
                    else:
                        self.logger.warning(f"Response queue not found for actor {actor_id_resp} after batch inference.")
            except Exception as e_batch_infer:
                self.logger.error(f"Error during batch inference for model_key '{model_key}': {e_batch_infer}", exc_info=True)
                # 为这个批次的所有请求发送错误响应
                for actor_id_err, req_id_err, _, _ in requests_for_this_model:
                    if actor_id_err in self.inference_resp_queues_to_actors:
                        try: self.inference_resp_queues_to_actors[actor_id_err].put((req_id_err, 0, 0.0, 0.0), block=False)
                        except QueueFull: pass
        
        self.last_batch_processed_time = time.time()

    # _handle_single_inference_request_logic 方法可以移除或保留为单请求处理的内部逻辑（如果需要）
    # 当前 _process_inference_batch 的目标是取代它，实现真正的批处理。
    # 为保持接口，但实际由 _process_inference_batch 中的循环调用
    def _handle_single_inference_request_logic(self, actor_id: str, request_id: int, model_key: str, observation_data_cpu: dict):
         # 这个方法的逻辑现在被包含在 _process_inference_batch 和 _prepare_batch_from_requests 中了
         # 为了避免代码重复和混淆，这个方法可以被标记为废弃，或者其内容被完全整合。
         # 当前的 _process_inference_batch 的简化版实现就是逐个调用此方法。
         # 如果要实现真正的批处理，此方法不再被主循环直接调用。
         # 这里暂时保留其结构，但实际的批处理发生在 _process_inference_batch。
         # (为了简洁，我将移除这个旧的单请求处理逻辑，因为它被新的批处理逻辑覆盖)
         pass


    def run(self):
        """InferenceServer 进程的主执行循环。"""
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
            self._initialize_models() 
        except Exception as e_init_models:
            self.logger.critical(f"CRITICAL error during model initialization: {e_init_models}. InferenceServer cannot start.", exc_info=True)
            if self.writer: self.writer.close()
            return 

        self.logger.info("InferenceServer entering main event loop...")
        self.last_batch_processed_time = time.time()

        while not self.shutting_down:
            processed_in_this_cycle = False # 标记本轮循环是否处理了任何事情

            # 1. 处理来自 Learner 的命令 (通常数量较少，可以优先处理)
            try:
                while not self.cmd_queue_from_learner.empty():
                    learner_cmd_tuple = self.cmd_queue_from_learner.get_nowait()
                    self._handle_learner_command(learner_cmd_tuple)
                    processed_in_this_cycle = True
                    if self.shutting_down: break 
            except QueueEmpty:
                pass 
            except Exception as e_cmd_loop_main:
                self.logger.error(f"Error processing command from Learner queue in main loop: {e_cmd_loop_main}", exc_info=True)
            
            if self.shutting_down: break

            # 2. 从 Actors 收集推理请求到缓冲区
            try:
                # 尽可能多地从请求队列中获取请求，直到达到批次大小上限或队列为空
                while len(self.inference_request_buffer) < self.max_inference_batch_size and \
                      not self.inference_req_queue_from_actors.empty():
                    actor_req_tuple = self.inference_req_queue_from_actors.get_nowait()
                    self.inference_request_buffer.append(actor_req_tuple)
                    processed_in_this_cycle = True 
            except QueueEmpty:
                pass 
            except Exception as e_actor_q_get:
                 self.logger.error(f"Error getting request from Actor queue in main loop: {e_actor_q_get}", exc_info=True)

            # 3. 检查是否满足批处理条件
            current_time = time.time()
            if self.inference_request_buffer and \
               (len(self.inference_request_buffer) >= self.max_inference_batch_size or \
                (current_time - self.last_batch_processed_time) >= self.max_inference_batch_wait_time_sec):
                self._process_inference_batch() # 处理并清空 self.inference_request_buffer
                processed_in_this_cycle = True # 标记已处理

            if not processed_in_this_cycle: # 如果本轮什么都没做
                time.sleep(0.0001) # 短暂休眠100微秒，减少CPU空转 (或者使用更长的带超时的队列get)

        self.logger.info(f"InferenceServer {self.name} (PID {os.getpid()}) is shutting down its main loop.")
        if self.writer:
            self.logger.info("Closing InferenceServer's TensorBoard writer.")
            self.writer.close()
        self.logger.info(f"InferenceServer {self.name} (PID {os.getpid()}) has shut down completely.")

