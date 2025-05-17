from multiprocessing.shared_memory import SharedMemory, ShareableList
import _pickle as cPickle # 使用 cPickle 以获得更好的性能
import time
import numpy as np # 之后采样模型时可能用到

class ModelPoolServer:
    """
    模型池服务器端。
    负责接收新的模型参数 (state_dict)，将其存储在共享内存中，并管理模型元数据。
    它使用一个固定大小的循环缓冲区来存储模型。当缓冲区满时，新的模型会覆盖最旧的模型 (FIFO)。
    """
    
    def __init__(self, capacity, name):
        """
        初始化模型池服务器。

        Args:
            capacity (int): 模型池可以存储的最大模型数量。
            name (str): 用于创建或连接共享内存列表的唯一名称。
        """
        self.capacity = capacity # 模型池的容量
        self.n = 0 # 已推送到池中的模型总数 (用于生成唯一的模型 ID 和计算在循环缓冲区中的位置)
        self.model_list = [None] * capacity # 服务器端维护的实际模型元数据列表 (包含共享内存对象本身，用于释放)
        
        # shared_model_list: 用于在进程间共享模型元数据。
        # 结构: [metadata_0, metadata_1, ..., metadata_capacity-1, current_total_model_count]
        # 每个 metadata 包含模型的 ID 和共享内存地址 ('_addr')。
        # 最后一个元素存储的是 self.n (已推送模型的总数)。
        self.metadata_slot_bytes = 1024 # 为每个序列化后的元数据条目预留的字节大小 (需要足够大以容纳 cPickle.dumps(metadata))
        
        # 创建一个 ShareableList，包含 capacity 个元数据槽位和 1 个计数器槽位。
        # 用空格字符串初始化元数据槽位，用 self.n 初始化计数器。
        try:
            # 尝试先删除可能已存在的同名共享内存，避免因未正确清理导致的错误
            # 这在开发和调试时很有用，但在生产环境中可能需要更谨慎的处理
            _ = ShareableList(name=name) # 尝试连接
            _.shm.unlink() # 如果连接成功，说明已存在，则尝试删除
            # print(f"ModelPoolServer: Unlinked existing ShareableList '{name}'.")
        except FileNotFoundError:
            # print(f"ModelPoolServer: No existing ShareableList '{name}' to unlink.")
            pass # 如果不存在，则忽略

        self.shared_model_list = ShareableList([' ' * self.metadata_slot_bytes] * capacity + [self.n], name=name)
        # print(f"ModelPoolServer: Created ShareableList '{name}' with capacity {capacity}.")

    def push(self, state_dict, metadata=None):
        """
        将新的模型状态字典推送到模型池。

        Args:
            state_dict (dict): PyTorch 模型的 state_dict。
            metadata (dict, optional): 与此模型相关的额外元数据 (例如训练步数、胜率等)。默认为空字典。

        Returns:
            dict: 包含模型 ID、共享内存地址和其他传入元数据的完整元数据字典。
                  还包含一个 'memory' 键，指向服务器端的 SharedMemory 对象 (用于后续可能的 unlink)。
        """
        if metadata is None:
            metadata = {}

        # 计算当前模型在循环缓冲区中的索引位置
        idx_in_buffer = self.n % self.capacity
        
        # 如果该位置已有旧模型 (缓冲区已满一轮)，则需要释放旧模型占用的共享内存
        if self.model_list[idx_in_buffer] is not None and 'memory' in self.model_list[idx_in_buffer]:
            try:
                old_shm_name = self.model_list[idx_in_buffer]['memory'].name
                self.model_list[idx_in_buffer]['memory'].close() # 先 close
                self.model_list[idx_in_buffer]['memory'].unlink() # 再 unlink
                # print(f"ModelPoolServer: Unlinked old shared memory {old_shm_name} at index {idx_in_buffer}.")
            except FileNotFoundError:
                # print(f"ModelPoolServer: Old shared memory {old_shm_name} already unlinked or not found.")
                pass # 可能已经被其他方式释放或从未成功创建
            except Exception as e:
                # print(f"ModelPoolServer: Error unlinking old shared memory {old_shm_name}: {e}")
                pass


        # 序列化模型参数
        data_bytes = cPickle.dumps(state_dict)
        
        
        # 创建共享内存块来存储序列化后的模型参数
        # 使用唯一的名称，通常基于 self.n 和基础名称，但 SharedMemory 内部会处理命名
        # 这里直接让 SharedMemory 生成名称，然后将该名称存入元数据
        try:
            shm = SharedMemory(create=True, size=len(data_bytes))
        except FileExistsError: # 如果碰巧名称冲突 (极小概率，但为了健壮性)
            # print(f"ModelPoolServer: SharedMemory collision, trying to clean up and retry for model {self.n}")
            # 尝试清理可能的残留 (这部分逻辑比较复杂，简单处理是附加唯一后缀)
            try:
                # 尝试删除同名共享内存，如果它是一个未被跟踪的残留
                temp_shm = SharedMemory(name=shm.name) # 尝试连接
                temp_shm.close()
                temp_shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e_clean:
                # print(f"ModelPoolServer: Error cleaning conflicting SHM {shm.name}: {e_clean}")
                pass
            # 重试一次或使用更鲁棒的唯一名称生成策略
            shm = SharedMemory(create=True, size=len(data_bytes))


        shm.buf[:] = data_bytes[:] # 将序列化数据写入共享内存
        # print(f'ModelPoolServer: Created model {self.n} in shared memory {shm.name}, size {len(data_bytes)} bytes.')
        
        # 准备元数据
        current_metadata = metadata.copy() # 复制传入的元数据，以避免修改原始字典
        current_metadata['_addr'] = shm.name # 存储共享内存的地址 (名称)
        current_metadata['id'] = self.n # 分配全局唯一的模型 ID
        current_metadata['timestamp'] = time.time() # 添加时间戳
        current_metadata['size_bytes'] = len(data_bytes) # 记录模型大小

        # 更新服务器本地的模型列表 (主要用于管理 SharedMemory 对象以便后续释放)
        self.model_list[idx_in_buffer] = current_metadata.copy() # 存储元数据副本
        self.model_list[idx_in_buffer]['memory'] = shm # 存储 SharedMemory 对象本身

        # 将序列化后的元数据写入 ShareableList，供客户端读取
        # 需要确保序列化后的元数据不超过预设的 self.metadata_slot_bytes

        serialized_metadata = cPickle.dumps(current_metadata)

        # 使用实例变量 self.metadata_slot_bytes 进行比较
        if len(serialized_metadata) > self.metadata_slot_bytes:
            print(f"Warning: Serialized metadata for model {self.n} (actual size: {len(serialized_metadata)}) "
                f"is too large for the allocated slot (max size: {self.metadata_slot_bytes})! It will be truncated.")
            # 截断序列化后的元数据以适应槽位大小
            serialized_metadata = serialized_metadata[:self.metadata_slot_bytes]

        # 将可能已截断的元数据写入共享列表
        self.shared_model_list[idx_in_buffer] = serialized_metadata
        
        # 更新已推送模型的总数，并将其写入 ShareableList 的最后一个元素
        self.n += 1
        self.shared_model_list[-1] = self.n
        
        # print(f"ModelPoolServer: Pushed model {current_metadata['id']} (addr: {current_metadata['_addr']}) to shared_model_list index {idx_in_buffer}. Total models: {self.n}")
        
        # 返回包含 SharedMemory 对象的元数据给调用者 (Learner)，虽然 Learner 通常不需要直接操作 shm 对象
        return self.model_list[idx_in_buffer] 

    def __del__(self):
        """
        在服务器对象被销毁时，尝试清理所有创建的共享内存块和 ShareableList。
        """
        # print("ModelPoolServer is being deleted. Cleaning up shared resources...")
        # 清理 ShareableList
        try:
            self.shared_model_list.shm.close()
            self.shared_model_list.shm.unlink()
            # print(f"ModelPoolServer: Unlinked ShareableList '{self.shared_model_list.shm.name}'.")
        except Exception as e:
            # print(f"ModelPoolServer: Error unlinking ShareableList: {e}")
            pass

        # 清理所有模型占用的共享内存
        for item in self.model_list:
            if item and 'memory' in item and isinstance(item['memory'], SharedMemory):
                try:
                    # print(f"ModelPoolServer: Cleaning up SHM for model id {item.get('id', 'N/A')}, name {item['memory'].name}")
                    item['memory'].close()
                    item['memory'].unlink()
                except FileNotFoundError:
                    # print(f"ModelPoolServer: SHM {item['memory'].name} already unlinked or not found.")
                    pass # 可能已经被释放
                except Exception as e:
                    # print(f"ModelPoolServer: Error cleaning up SHM {item['memory'].name}: {e}")
                    pass


class ModelPoolClient:
    """
    模型池客户端。
    负责从模型池服务器获取模型元数据和加载模型参数。
    """
    
    def __init__(self, name, retry_interval=0.1, max_retries=300): # 默认等待30秒
        """
        初始化模型池客户端。

        Args:
            name (str): 用于连接到共享内存列表的唯一名称 (必须与服务器端一致)。
            retry_interval (float): 连接失败时的重试间隔（秒）。
            max_retries (int): 最大重试次数。
        """
        connected = False
        retries = 0
        while not connected and retries < max_retries:
            try:
                # 连接到服务器创建的 ShareableList
                self.shared_model_list = ShareableList(name=name)
                # 尝试读取模型总数，以验证连接和 ShareableList 的有效性
                _ = self.shared_model_list[-1] 
                connected = True
                # print(f"ModelPoolClient: Successfully connected to ShareableList '{name}'.")
            except FileNotFoundError:
                # print(f"ModelPoolClient: ShareableList '{name}' not found. Retrying in {retry_interval}s... (Attempt {retries+1}/{max_retries})")
                time.sleep(retry_interval)
                retries += 1
            except Exception as e: # 其他可能的异常，例如 ShareableList 格式不匹配等
                # print(f"ModelPoolClient: Error connecting to ShareableList '{name}': {e}. Retrying... (Attempt {retries+1}/{max_retries})")
                time.sleep(retry_interval)
                retries += 1
        
        if not connected:
            raise ConnectionError(f"ModelPoolClient: Failed to connect to ShareableList '{name}' after {max_retries} retries.")

        # ShareableList 的容量是其实际长度减 1 (因为最后一个元素是计数器)
        self.capacity = len(self.shared_model_list) - 1
        # 客户端本地维护的模型元数据列表的缓存
        self.local_model_metadata_cache = [None] * self.capacity
        # 客户端本地记录的已同步到的模型总数
        self.n_synced = 0 
        # 初始化时尝试更新一次本地模型元数据列表
        self._update_local_metadata_cache()
    
    def _update_local_metadata_cache(self):
        """
        从共享的 ShareableList 更新客户端本地的模型元数据缓存。
        只在服务器端的模型总数 (self.shared_model_list[-1]) 大于客户端已同步的模型总数 (self.n_synced) 时执行更新。
        """
        # 从 ShareableList 获取服务器端的模型总数
        n_server = self.shared_model_list[-1]
        
        if n_server > self.n_synced:
            # print(f"ModelPoolClient: Server has {n_server} models, client synced to {self.n_synced}. Updating cache.")
            # 有新模型可用，更新本地元数据列表
            # 更新范围：从 (本地已同步数) 或 (服务器总数 - 容量) 中的较大者，到 (服务器总数 - 1)
            # 这样可以避免读取已经被覆盖的非常旧的元数据槽位（虽然 ShareableList 本身不保证内容一致性，依赖服务器正确写入）
            start_idx_global = max(self.n_synced, n_server - self.capacity)
            
            for i in range(start_idx_global, n_server):
                idx_in_buffer = i % self.capacity # 计算在循环缓冲区中的索引
                try:
                    # 从 ShareableList 读取序列化后的元数据并反序列化
                    # 需要处理可能的 cPickle.loads 错误，例如元数据为空白或损坏
                    raw_metadata_bytes = self.shared_model_list[idx_in_buffer]
                    # 确保不是初始的空白字符串
                    if isinstance(raw_metadata_bytes, str) and raw_metadata_bytes.isspace():
                        # print(f"ModelPoolClient: Skipping empty metadata slot at index {idx_in_buffer} for global model id {i}.")
                        continue # 跳过空白槽
                    
                    metadata = cPickle.loads(raw_metadata_bytes)
                    self.local_model_metadata_cache[idx_in_buffer] = metadata
                    # print(f"ModelPoolClient: Updated cache for model id {metadata.get('id')} at buffer index {idx_in_buffer}")
                except EOFError: # cPickle.loads 可能因为数据不完整（例如被截断）抛出EOFError
                    # print(f"ModelPoolClient: EOFError loading metadata for model (global_id approx {i}) at buffer index {idx_in_buffer}. Slot might be partially written or corrupted.")
                    self.local_model_metadata_cache[idx_in_buffer] = None #标记为无效
                except Exception as e:
                    # print(f"ModelPoolClient: Error loading metadata for model (global_id approx {i}) at buffer index {idx_in_buffer}: {e}")
                    self.local_model_metadata_cache[idx_in_buffer] = None # 标记为无效

            # 更新客户端已同步的模型总数
            self.n_synced = n_server
        # else:
            # print(f"ModelPoolClient: No new models from server (Server: {n_server}, Client synced: {self.n_synced}). Cache up to date.")

    def get_all_valid_model_metadata(self):
        """
        （扩展接口）获取当前模型池中所有有效（非None）的模型元数据列表。
        返回的列表是本地缓存的副本，顺序可能不完全代表推送顺序，但包含了所有可访问模型的元数据。
        """
        self._update_local_metadata_cache() # 确保缓存是相对最新的
        valid_metadata_list = []
        # 遍历本地缓存，收集所有非 None 的元数据
        # 顺序是基于循环缓冲区的索引，可以通过 model 'id' 排序（如果需要严格的推送顺序）
        # 但对于采样，当前顺序可能也适用
        
        # 为了尽可能按推送顺序（从旧到新），如果池已满，我们从 (n_synced % capacity) 开始遍历
        # 如果池未满，则从 0 到 (n_synced - 1)
        
        if self.n_synced == 0:
            return []

        if self.n_synced >= self.capacity:
            # 池已满或已覆盖一轮
            start_buffer_idx = self.n_synced % self.capacity
            for i in range(self.capacity):
                current_buffer_idx = (start_buffer_idx + i) % self.capacity
                metadata = self.local_model_metadata_cache[current_buffer_idx]
                if metadata is not None and isinstance(metadata, dict) and 'id' in metadata: # 确保是有效的元数据字典
                    valid_metadata_list.append(metadata)
        else:
            # 池未满
            for i in range(self.n_synced):
                metadata = self.local_model_metadata_cache[i]
                if metadata is not None and isinstance(metadata, dict) and 'id' in metadata:
                    valid_metadata_list.append(metadata)
        
        # print(f"ModelPoolClient: get_all_valid_model_metadata returning {len(valid_metadata_list)} items. Synced count: {self.n_synced}")
        return valid_metadata_list

    def get_latest_model_metadata(self): # Renamed from get_latest_model for clarity
        """
        获取最新的模型元数据。
        会阻塞直到至少有一个模型可用。
        """
        self._update_local_metadata_cache()
        # 循环等待直到至少有一个模型被推送到池中
        while self.n_synced == 0:
            # print("ModelPoolClient: get_latest_model_metadata - No models available yet. Waiting...")
            time.sleep(0.1) # 等待一小段时间
            self._update_local_metadata_cache() # 再次尝试更新
        
        # 最新的模型在循环缓冲区中的位置是 (总数 - 1) % 容量
        latest_idx_in_buffer = (self.n_synced - 1) % self.capacity
        latest_metadata = self.local_model_metadata_cache[latest_idx_in_buffer]

        if latest_metadata is None or not (isinstance(latest_metadata, dict) and 'id' in latest_metadata):
            # print(f"ModelPoolClient: get_latest_model_metadata - Latest metadata at index {latest_idx_in_buffer} is None or invalid. This might indicate a sync issue or rapid overwriting.")
            # 这种情况理论上不应频繁发生，如果 n_synced > 0。可能是元数据损坏或更新逻辑问题。
            # 尝试重新获取所有元数据并返回最后一个有效的
            all_meta = self.get_all_valid_model_metadata()
            if all_meta:
                return all_meta[-1] # 返回最后一个有效的
            else: # 极端情况，无法获取任何有效元数据
                # print("ModelPoolClient: get_latest_model_metadata - Critical: No valid metadata found even after refresh.")
                return None # 或者抛出异常
        
        # print(f"ModelPoolClient: get_latest_model_metadata returning model id {latest_metadata.get('id')}")
        return latest_metadata
        
    def load_model_parameters(self, metadata): # Renamed from load_model for clarity
        """
        根据提供的元数据加载模型参数 (state_dict)。

        Args:
            metadata (dict): 包含模型 'id' 和共享内存地址 '_addr' 的元数据。

        Returns:
            dict or None: 模型的 state_dict；如果模型太旧已被覆盖或元数据无效，则返回 None。
        """
        if not metadata or '_addr' not in metadata or 'id' not in metadata:
            # print("ModelPoolClient: load_model_parameters - Invalid metadata provided.")
            return None

        self._update_local_metadata_cache() # 确保 self.n_synced 是最新的
        
        model_id_to_load = metadata['id']
        shm_addr = metadata['_addr']

        # 检查请求的模型是否因为太旧而被覆盖
        # 如果 (当前同步到的模型总数 - 请求的模型ID) > 容量，则说明模型应该已经被覆盖了
        # (self.n_synced - 1) 是最新模型的ID。 (self.n_synced - self.capacity) 是当前池中理论上最老模型的ID。
        if model_id_to_load < self.n_synced - self.capacity:
            # print(f"ModelPoolClient: Model id {model_id_to_load} is too old (current oldest approx: {self.n_synced - self.capacity}) and likely overwritten. Cannot load.")
            return None
        
        try:
            # 连接到指定的共享内存块
            memory = SharedMemory(name=shm_addr)
            # 从共享内存反序列化模型参数
            state_dict = cPickle.loads(memory.buf)
            memory.close() # 读取完毕后关闭连接 (但不 unlink，unlink 由服务器负责)
            # print(f"ModelPoolClient: Successfully loaded model id {model_id_to_load} from SHM {shm_addr}.")
            return state_dict
        except FileNotFoundError:
            # print(f"ModelPoolClient: SharedMemory '{shm_addr}' for model id {model_id_to_load} not found. It might have been unlinked by the server.")
            return None
        except Exception as e:
            # print(f"ModelPoolClient: Error loading model id {model_id_to_load} from SHM '{shm_addr}': {e}")
            return None

    # --- 扩展接口 ---
    def get_model_metadata_by_id(self, model_id):
        """
        （扩展接口）根据模型 ID 获取特定模型的元数据。

        Args:
            model_id (int): 要获取元数据的模型的 ID。

        Returns:
            dict or None: 如果找到模型且其元数据有效，则返回元数据字典；否则返回 None。
        """
        self._update_local_metadata_cache()
        if model_id < 0 or model_id >= self.n_synced:
            # print(f"ModelPoolClient: Requested model_id {model_id} is out of range (current max id: {self.n_synced - 1}).")
            return None
        
        # 检查请求的模型是否因为太旧而被覆盖
        if model_id < self.n_synced - self.capacity:
            # print(f"ModelPoolClient: Model id {model_id} is too old and likely overwritten. Cannot get metadata.")
            return None
            
        idx_in_buffer = model_id % self.capacity
        metadata = self.local_model_metadata_cache[idx_in_buffer]
        
        # 验证元数据是否确实是请求的 ID (因为缓冲区是循环的)
        if metadata and isinstance(metadata, dict) and metadata.get('id') == model_id:
            # print(f"ModelPoolClient: Found metadata for model id {model_id}.")
            return metadata
        else:
            # print(f"ModelPoolClient: Metadata for model id {model_id} not found or ID mismatch at buffer index {idx_in_buffer} (found: {metadata.get('id') if metadata else 'None'}).")
            # 可能是因为模型被快速覆盖，或者请求的ID在该槽位上但内容已变
            return None

    def sample_model_metadata(self, strategy='latest', k=1, exclude_ids=None, require_distinct_from_latest=False):
        """
        （扩展接口）根据指定策略从模型池中采样一个或多个模型元数据。

        Args:
            strategy (str): 采样策略。可选值:
                'latest': 获取最新的模型元数据 (等同于 get_latest_model_metadata)。
                'uniform': 从当前池中所有有效模型中均匀随机采样一个。
                'latest_k': 从最新的 k 个有效模型中均匀随机采样一个。
                'nth_latest': 获取倒数第 N 新的模型 (k=1 表示最新, k=2 表示第二新, ...)。
                'specific_id': 尝试获取具有特定ID的模型 (k 此时为 model_id)。
            k (int, optional): 根据策略使用的参数。
                对于 'latest_k', k 是指最新的模型数量。
                对于 'nth_latest', k 是指倒数第几个 (1-indexed)。
                对于 'specific_id', k 是 model_id。
                默认为 1。
            exclude_ids (list of int, optional): 需要从采样中排除的模型 ID 列表。
            require_distinct_from_latest (bool): 如果为 True，并且采样策略不是 'latest' 或 'specific_id'，
                                                 则确保采样到的模型与当前最新模型不同（如果池中有多于一个模型）。

        Returns:
            dict or None: 单个采样到的模型元数据；如果无法根据策略采样或池为空，则返回 None。
                          (未来可以扩展为返回元数据列表，如果 k > 1 且策略支持多采样)
        """
        self._update_local_metadata_cache()
        all_valid_metadata = self.get_all_valid_model_metadata()

        if not all_valid_metadata:
            # print("ModelPoolClient: sample_model_metadata - No valid models in the pool to sample from.")
            return None

        # 根据ID排序，最新的在最后
        all_valid_metadata.sort(key=lambda m: m['id'])
        
        # 处理排除ID
        if exclude_ids:
            all_valid_metadata = [m for m in all_valid_metadata if m['id'] not in exclude_ids]
            if not all_valid_metadata:
                # print(f"ModelPoolClient: sample_model_metadata - No models left after excluding IDs: {exclude_ids}.")
                return None
        
        latest_model_meta = all_valid_metadata[-1] if all_valid_metadata else None

        if require_distinct_from_latest and latest_model_meta:
            # 如果需要与最新不同，先从池中移除最新模型（仅为本次采样逻辑）
            candidate_pool = [m for m in all_valid_metadata if m['id'] != latest_model_meta['id']]
            if not candidate_pool: # 如果移除后池为空（即池中只有一个模型）
                 # print(f"ModelPoolClient: sample_model_metadata - Cannot sample distinct from latest, only one model (id: {latest_model_meta['id']}) in pool or after exclusion.")
                 return None # 无法满足要求
            source_pool = candidate_pool
        else:
            source_pool = all_valid_metadata


        if not source_pool: # 再次检查源池是否为空
            # print(f"ModelPoolClient: sample_model_metadata - Source pool for sampling is empty after filters.")
            return None
            
        num_available = len(source_pool)

        if strategy == 'latest':
            return latest_model_meta # 直接返回原始的最新模型，不受 require_distinct_from_latest 影响
        
        elif strategy == 'specific_id':
            model_id_to_get = k
            # 从原始的 all_valid_metadata (包含所有，未排除) 中查找
            for meta in all_valid_metadata: # 确保从完整列表（排序后）查找
                if meta['id'] == model_id_to_get:
                    return meta
            # print(f"ModelPoolClient: sample_model_metadata (specific_id) - Model ID {model_id_to_get} not found.")
            return None

        # 对于 'uniform', 'latest_k', 'nth_latest'，它们都从 source_pool 中选择
        if strategy == 'uniform':
            if num_available == 0: return None
            chosen_meta = np.random.choice(source_pool)
            # print(f"ModelPoolClient: sample_model_metadata (uniform) - Sampled model id {chosen_meta.get('id')}")
            return chosen_meta
        
        elif strategy == 'latest_k':
            if k <= 0: k = 1
            # 从 source_pool 中取最新的 k 个
            latest_k_candidates = source_pool[-min(k, num_available):]
            if not latest_k_candidates: return None
            chosen_meta = np.random.choice(latest_k_candidates)
            # print(f"ModelPoolClient: sample_model_metadata (latest_k, k={k}) - Sampled model id {chosen_meta.get('id')} from {len(latest_k_candidates)} candidates.")
            return chosen_meta

        elif strategy == 'nth_latest':
            if k <= 0 : k = 1 # 1-indexed
            if k > num_available:
                # print(f"ModelPoolClient: sample_model_metadata (nth_latest, k={k}) - Requested k is greater than available models ({num_available}).")
                return None # k 超出范围
            # source_pool 已经按 ID 升序排序，所以 source_pool[-k] 是倒数第 k 个
            chosen_meta = source_pool[-k]
            # print(f"ModelPoolClient: sample_model_metadata (nth_latest, k={k}) - Selected model id {chosen_meta.get('id')}.")
            return chosen_meta
            
        else:
            # print(f"ModelPoolClient: sample_model_metadata - Unknown strategy '{strategy}'.")
            return None # 未知策略

    def __del__(self):
        """
        在客户端对象被销毁时，关闭与 ShareableList 的连接。
        """
        if hasattr(self, 'shared_model_list') and self.shared_model_list:
            try:
                self.shared_model_list.shm.close()
                # print(f"ModelPoolClient: Closed connection to ShareableList '{self.shared_model_list.shm.name}'.")
            except Exception as e:
                # print(f"ModelPoolClient: Error closing ShareableList connection: {e}")
                pass