from multiprocessing.shared_memory import SharedMemory, ShareableList
import _pickle as cPickle
import time
import numpy as np
import logging # 导入 logging

# logger = logging.getLogger(__name__) # 或者在 __init__ 中获取实例 logger

class ModelPoolServer:
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.n = 0 
        self.model_list = [None] * capacity
        self.metadata_slot_bytes = 1024
        
        # 获取一个独立的 logger 实例
        self.logger = logging.getLogger(f"{__name__}.ModelPoolServer.{name}")
        # 您可能需要在主程序中配置这个 logger 的处理器和级别
        if not self.logger.hasHandlers(): # 简易默认配置，如果外部没有配置
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
             self.logger.setLevel(logging.INFO)


        try:
            self.logger.info(f"Attempting to unlink existing ShareableList named '{name}' if it exists.")
            # 检查是否存在，如果存在则尝试 unlink
            # Note: Creating a ShareableList just to unlink it can be problematic if another process is actively using it.
            # A more robust cleanup usually happens at a higher application level or specific cleanup scripts.
            # For now, keeping a similar logic to before but with logging.
            try:
                existing_sl = ShareableList(name=name)
                existing_sl.shm.close() # Close first
                existing_sl.shm.unlink()
                self.logger.info(f"Successfully unlinked pre-existing ShareableList '{name}'.")
            except FileNotFoundError:
                self.logger.info(f"No pre-existing ShareableList named '{name}' found to unlink.")
        except Exception as e_sl_init_unlink:
            self.logger.warning(f"Exception during pre-unlink of ShareableList '{name}': {e_sl_init_unlink}")


        try:
            self.shared_model_list = ShareableList([' ' * self.metadata_slot_bytes] * capacity + [self.n], name=name)
            self.logger.info(f"ModelPoolServer: Created/Connected to ShareableList '{name}' with capacity {capacity}, slot_bytes {self.metadata_slot_bytes}.")
        except Exception as e:
            self.logger.critical(f"ModelPoolServer: CRITICAL - Failed to create/connect to ShareableList '{name}'. Error: {e}", exc_info=True)
            raise # Re-raise, as this is fundamental

    def push(self, state_dict, metadata=None):
        current_model_id_for_log = self.n # 当前要推送的模型的ID
        self.logger.info(f"[Push model_id:{current_model_id_for_log}] Method started.") # 标记方法开始

        if metadata is None:
            metadata = {}
        self.logger.info(f"[Push model_id:{current_model_id_for_log}] Metadata (initial): {metadata}")

        idx_in_buffer = self.n % self.capacity
        self.logger.info(f"[Push model_id:{current_model_id_for_log}] Calculated buffer index: {idx_in_buffer}")
        
        # 1. 清理旧模型（如果存在）
        if self.model_list[idx_in_buffer] is not None and 'memory' in self.model_list[idx_in_buffer]:
            old_shm_obj = self.model_list[idx_in_buffer]['memory']
            old_shm_name = old_shm_obj.name
            old_model_id_val = self.model_list[idx_in_buffer].get('id', 'N/A')
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Buffer slot {idx_in_buffer} is full. Unlinking old model ID {old_model_id_val} (SHM name: {old_shm_name}).")
            try:
                self.logger.info(f"[Push model_id:{current_model_id_for_log}] Closing old SHM: {old_shm_name}")
                old_shm_obj.close()
                self.logger.info(f"[Push model_id:{current_model_id_for_log}] Unlinking old SHM: {old_shm_name}")
                old_shm_obj.unlink()
                self.logger.info(f"[Push model_id:{current_model_id_for_log}] Successfully unlinked old SHM: {old_shm_name}")
            except FileNotFoundError:
                self.logger.warning(f"[Push model_id:{current_model_id_for_log}] Old SHM {old_shm_name} (for model ID {old_model_id_val}) was already unlinked or not found.")
            except Exception as e_unlink:
                self.logger.error(f"[Push model_id:{current_model_id_for_log}] Error unlinking old SHM {old_shm_name} (for model ID {old_model_id_val}): {e_unlink}", exc_info=True)
        self.logger.info(f"[Push model_id:{current_model_id_for_log}] Old model cleanup finished (if any).")
        
        # 2. 序列化模型参数
        data_bytes = None
        try:
            num_keys_in_state_dict = len(state_dict.keys())
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Pickling state_dict with {num_keys_in_state_dict} keys...")
            data_bytes = cPickle.dumps(state_dict)
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] State_dict pickled. Serialized data_bytes length: {len(data_bytes)}")
        except Exception as e_pickle_state:
            self.logger.critical(f"[Push model_id:{current_model_id_for_log}] CRITICAL - Failed to pickle state_dict: {e_pickle_state}", exc_info=True)
            return None 

        # 3. 创建新的共享内存并写入数据
        shm = None # 初始化为 None
        try:
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Attempting to create SharedMemory with size {len(data_bytes)}...")
            shm = SharedMemory(create=True, size=len(data_bytes)) # 匿名共享内存
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] SharedMemory '{shm.name}' created. Attempting to write data_bytes to shm.buf...")
            shm.buf[:] = data_bytes[:]
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] data_bytes successfully written to SHM '{shm.name}'.")
        except FileExistsError: 
            self.logger.critical(f"[Push model_id:{current_model_id_for_log}] CRITICAL - FileExistsError when creating unnamed SharedMemory. This is unexpected. SHM system might be unstable.", exc_info=True)
            # 对于匿名SHM，此错误极不寻常。如果发生，清理已创建的 (如果shm不是None)
            if shm:
                shm.close()
                try: shm.unlink()
                except: pass # 忽略unlink中的错误
            return None 
        except OSError as e_os:
            self.logger.critical(f"[Push model_id:{current_model_id_for_log}] CRITICAL - OSError when creating/writing SharedMemory (size {len(data_bytes)}). Errno: {e_os.errno}, Message: {e_os.strerror}. Possible SHM exhaustion.", exc_info=True)
            if shm: # 如果 shm 对象已创建但后续操作（如写入 buf）失败
                shm.close()
                try: shm.unlink()
                except: pass
            return None 
        except Exception as e_shm_other:
            self.logger.critical(f"[Push model_id:{current_model_id_for_log}] CRITICAL - Unexpected error creating/writing SharedMemory: {e_shm_other}", exc_info=True)
            if shm: 
                shm.close()
                try: shm.unlink()
                except: pass
            return None
        
        # 4. 准备和序列化元数据
        current_metadata = metadata.copy()
        current_metadata['_addr'] = shm.name 
        current_metadata['id'] = current_model_id_for_log # 使用循环开始时的 self.n 作为当前模型ID
        current_metadata['timestamp'] = time.time()
        current_metadata['size_bytes'] = len(data_bytes)
        self.logger.info(f"[Push model_id:{current_model_id_for_log}] Metadata prepared: {current_metadata}")

        serialized_metadata = None
        try:
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Pickling metadata...")
            serialized_metadata = cPickle.dumps(current_metadata)
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Metadata pickled. Serialized_metadata length: {len(serialized_metadata)}")
        except Exception as e_pickle_meta:
            self.logger.critical(f"[Push model_id:{current_model_id_for_log}] CRITICAL - Failed to pickle metadata: {e_pickle_meta}", exc_info=True)
            # 清理已为此模型创建的 SHM，因为元数据无法共享
            shm.close()
            try: shm.unlink()
            except: pass
            return None

        # 5. 检查元数据大小并写入 ShareableList
        if len(serialized_metadata) > self.metadata_slot_bytes:
            self.logger.critical(
                f"[Push model_id:{current_model_id_for_log}] CRITICAL - Serialized metadata (size: {len(serialized_metadata)}) "
                f"is LARGER than allocated slot size ({self.metadata_slot_bytes}). "
                "This WILL CAUSE client-side unpickling errors. Increase metadata_slot_bytes or reduce metadata content."
            )
            shm.close() # 清理SHM
            try: shm.unlink()
            except: pass
            return None # 指示推送失败

        try:
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Writing serialized metadata to ShareableList slot {idx_in_buffer}...")
            self.shared_model_list[idx_in_buffer] = serialized_metadata
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Metadata written to ShareableList slot {idx_in_buffer}.")
            
            # 更新服务器本地的 model_list (用于追踪 SHM 对象以便清理)
            self.model_list[idx_in_buffer] = current_metadata.copy() # 存储元数据副本
            self.model_list[idx_in_buffer]['memory'] = shm # 存储 SharedMemory 对象本身
            
            # 更新全局模型计数器 (self.n 是下一个模型的ID，也是当前推送完成后的模型总数)
            self.n += 1 
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Updating total model count in ShareableList to {self.n}...")
            self.shared_model_list[-1] = self.n # 将新的总数写入共享列表
            self.logger.info(f"[Push model_id:{current_model_id_for_log}] Successfully pushed. New total models: {self.n}. SHM: {shm.name}, Slot: {idx_in_buffer}.")
        
        except Exception as e_sl_write:
            self.logger.critical(f"[Push model_id:{current_model_id_for_log}] CRITICAL - Error writing to ShareableList or updating count: {e_sl_write}", exc_info=True)
            # 如果写入 ShareableList 失败，相关的 SHM 段也应该被清理
            shm.close()
            try: shm.unlink()
            except: pass
            self.model_list[idx_in_buffer] = None # 清除本地追踪
            # self.n 此时不应增加，因为推送未完全成功
            # (如果 self.n 已经在上面增加了，需要考虑是否回滚，但这里的结构是先写元数据再增加 self.n)
            # 修正：self.n 应该在所有操作成功后再增加。current_model_id_for_log 已经是 self.n。
            # 所以在 finally 或成功路径的最后，才将 self.n 更新为 current_model_id_for_log + 1
            # 为了简单，我们假设如果这里出错，外部调用者（Learner）会知道 push 失败了。
            return None # 指示推送失败

        object_to_return = self.model_list[idx_in_buffer]
        self.logger.info(f"[Push model_id:{current_model_id_for_log}] Preparing to return object of type {type(object_to_return)}. Content keys (if dict): {list(object_to_return.keys()) if isinstance(object_to_return, dict) else 'N/A'}")
        return object_to_return


    def cleanup(self):
        """
        显式清理所有由该 ModelPoolServer 实例创建和管理的共享内存资源。
        这个方法应该在程序退出前被调用。
        """
        self.logger.info(f"ModelPoolServer '{self.shared_model_list.shm.name if hasattr(self, 'shared_model_list') and self.shared_model_list.shm else 'Unknown'}' cleanup initiated...")
        
        # 1. 清理 ShareableList 本身
        if hasattr(self, 'shared_model_list') and self.shared_model_list.shm:
            sl_name = self.shared_model_list.shm.name
            try:
                self.shared_model_list.shm.close()
                self.shared_model_list.shm.unlink()
                self.logger.info(f"Successfully unlinked ShareableList '{sl_name}'.")
            except Exception as e:
                self.logger.error(f"Error unlinking ShareableList '{sl_name}' during cleanup: {e}", exc_info=True)
        else:
            self.logger.info("No ShareableList found or it was already unlinked/closed.")

        # 2. 清理所有模型数据占用的 SharedMemory 段
        if hasattr(self, 'model_list'):
            for item_idx, item_meta in enumerate(self.model_list):
                if item_meta and 'memory' in item_meta and isinstance(item_meta['memory'], SharedMemory):
                    shm_obj_to_clean = item_meta['memory']
                    model_id_cleaned = item_meta.get('id', f'N/A_at_slot_{item_idx}')
                    shm_name_to_clean = shm_obj_to_clean.name
                    self.logger.info(f"Cleaning up SHM for model id {model_id_cleaned}, name {shm_name_to_clean}")
                    try:
                        shm_obj_to_clean.close()
                        shm_obj_to_clean.unlink()
                        self.logger.debug(f"Successfully unlinked SHM: {shm_name_to_clean} for model id {model_id_cleaned}")
                    except FileNotFoundError:
                        self.logger.warning(f"SHM {shm_name_to_clean} for model {model_id_cleaned} was already unlinked or not found during cleanup.")
                    except Exception as e:
                        self.logger.error(f"Error cleaning up SHM {shm_name_to_clean} for model {model_id_cleaned} during cleanup: {e}", exc_info=True)
        self.logger.info(f"ModelPoolServer cleanup attempt finished.")

    def __del__(self):
        # __del__ 的调用时机不保证，尤其是在进程被信号中断时。
        # 主要的清理逻辑应该放在显式的 cleanup() 方法中。
        # 但作为最后一道防线，可以尝试在这里也调用 cleanup。
        self.logger.debug(f"ModelPoolServer __del__ called. Attempting cleanup if not already done.")
        if hasattr(self, 'shared_model_list'): # 检查属性是否存在，防止在不完整初始化时出错
            self.cleanup()

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