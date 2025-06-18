from multiprocessing import Queue
from collections import deque
import numpy as np
import random
import time # 需要 time 模块来计算时间差

class ReplayBuffer:

    def __init__(self, capacity, episode_queue_capacity): # 参数名 episode_queue_capacity 保持清晰
        self.queue = Queue(episode_queue_capacity)
        self.capacity = capacity
        self.buffer = None

        # Cumulative counters
        self.cumulative_sample_in = 0
        self.cumulative_sample_out = 0
        self.cumulative_episode_in = 0

        # For rate calculation in stats() method
        self.last_stat_query_time = time.time()
        # 使用与 stats() 方法中一致的属性名进行初始化
        self.prev_cumulative_sample_in_for_rate = 0
        self.prev_cumulative_sample_out_for_rate = 0

    def push(self, samples): # only called by actors
        self.queue.put(samples) # samples is an entire episode
    
    def _flush(self):
        if self.buffer is None: # called first time by learner, or if buffer was cleared
            self.buffer = deque(maxlen=self.capacity)
            # Stats are initialized in __init__ now, no need to re-init here
            # unless a clear operation also resets cumulative stats, which might be desirable.
            # For now, assume cumulative stats persist across buffer clears unless explicitly reset.

        while not self.queue.empty():
            episode_data = self.queue.get()
            unpacked_data = self._unpack(episode_data) # List of timestep dicts
            
            if unpacked_data: # Ensure there's something to add
                self.buffer.extend(unpacked_data)
                self.cumulative_sample_in += len(unpacked_data)
                self.cumulative_episode_in += 1
    
    def sample(self, batch_size): # only called by learner
        self._flush() # Ensure all pending episodes are processed into the buffer

        if self.buffer is None or not self.buffer:
            # Log a warning or return None if no data is available
            # print("Warning: Replay buffer is empty, cannot sample.") # Or use a logger
            return None 

        current_buffer_size = len(self.buffer)
        if current_buffer_size == 0:
            return None

        actual_sample_size = min(batch_size, current_buffer_size)
        
        if actual_sample_size == current_buffer_size:
            # If sampling the whole buffer, or a large portion, taking a list copy first might be safer
            # depending on `random.sample`'s behavior with deques if it modifies or needs list.
            # `random.sample` works directly on deques and returns a new list.
            samples = random.sample(self.buffer, actual_sample_size) # Samples all available
        else:
            samples = random.sample(self.buffer, actual_sample_size)
        
        self.cumulative_sample_out += actual_sample_size # Update with actual number of samples taken
        
        batch = self._pack(samples)
        return batch
    
    def size(self): # only called by learner
        self._flush() # Ensure size reflects episodes potentially just moved from queue
        return len(self.buffer) if self.buffer is not None else 0
    
    def clear(self): # only called by learner
        # Option 1: Only clear the deque
        if self.buffer is not None:
            self.buffer.clear()
        # Option 2: Also reset cumulative stats if clear means a full reset
        # self.cumulative_sample_in = 0
        # self.cumulative_sample_out = 0
        # self.cumulative_episode_in = 0
        # self.prev_cumulative_sample_in_for_rate = 0 # Reset for rate calculation
        # self.prev_cumulative_sample_out_for_rate = 0
        # self.last_stat_query_time = time.time()
        
        # For now, clear() only clears the deque content, cumulative stats persist.
        # This means rates will reflect activity since the start unless Learner resets.

    def stats(self):
        current_time = time.time()
        time_delta = current_time - self.last_stat_query_time

        rate_in = 0
        rate_out = 0

        if time_delta > 1e-6: # Avoid division by zero or tiny time deltas
            samples_in_this_period = self.cumulative_sample_in - self.prev_cumulative_sample_in_for_rate
            samples_out_this_period = self.cumulative_sample_out - self.prev_cumulative_sample_out_for_rate

            rate_in = samples_in_this_period / time_delta
            rate_out = samples_out_this_period / time_delta
        
        self.last_stat_query_time = current_time
        self.prev_cumulative_sample_in_for_rate = self.cumulative_sample_in
        self.prev_cumulative_sample_out_for_rate = self.cumulative_sample_out

        return {
            'samples_in_per_second': rate_in,
            'samples_out_per_second': rate_out,
            'samples_in_per_second_smoothed': rate_in, 
            'samples_out_per_second_smoothed': rate_out,
            'current_buffer_size': len(self.buffer) if self.buffer is not None else 0,
            'queue_size': self.queue.qsize(),
            'cumulative_sample_in': self.cumulative_sample_in,
            'cumulative_sample_out': self.cumulative_sample_out,
            'cumulative_episode_in': self.cumulative_episode_in,
        }
    
    def _unpack(self, data):
        """
        将完整批次数据拆分为单独的样本。
        递归地处理嵌套字典结构，返回一个样本列表。
        
        Args:
            data: 要拆分的数据，可以是字典、列表或单个值
            
        Returns:
            list: 样本列表
        """
        try:
            if isinstance(data, dict):
                res = []
                # 找出episode的长度（从第一个list类型的值）
                episode_len = 0
                for value in data.values():
                    if isinstance(value, (list, np.ndarray)):
                        episode_len = len(value)
                        break
                
                # 特殊情况处理
                if episode_len == 0:
                    if not data:
                        # 空字典
                        return []
                    else:
                        # 非空字典但没有列表类型的值，可能是单个时间步
                        # 作为单个样本返回
                        return [data]
                
                # 正常情况：创建结果列表
                res = [{} for _ in range(episode_len)]
                
                for key, value_list in data.items():
                    try:
                        unpacked_values = self._unpack(value_list)
                        
                        # 检查数据一致性
                        if len(unpacked_values) != episode_len and episode_len != 0:
                            # 数据不一致，记录并尝试修复
                            print(f"Warning: Key {key} has {len(unpacked_values)} items, expected {episode_len}")
                            
                            if len(unpacked_values) > 0:
                                # 有一些值，通过重复或截断来调整长度
                                if len(unpacked_values) < episode_len:
                                    # 重复最后一个值
                                    last_value = unpacked_values[-1]
                                    unpacked_values.extend([last_value] * (episode_len - len(unpacked_values)))
                                else:
                                    # 截断
                                    unpacked_values = unpacked_values[:episode_len]
                            else:
                                # 跳过这个键
                                continue
                        
                        # 填充结果
                        for i, v_item in enumerate(unpacked_values):
                            if i < episode_len:  # 安全检查
                                res[i][key] = v_item
                    except Exception as e:
                        # 处理单个键的错误，但不中断整个过程
                        print(f"Error unpacking key {key}: {e}")
                        # 跳过这个键
                        continue
                
                return res
            
            elif isinstance(data, (list, tuple, deque)):
                # 确保返回普通列表
                return list(data)
            
            elif isinstance(data, np.ndarray):
                # 这可能是一个批次的特征，我们希望返回单个样本的列表
                if data.ndim > 0:
                    # 多维数组，可能包含多个样本
                    return [data[i] for i in range(data.shape[0])]
                else:
                    # 标量数组
                    return [data]
            
            else:
                # 单个值（数字、字符串等）
                try:
                    # 尝试将其作为可迭代对象
                    return list(data)
                except:
                    # 不可迭代，作为单个元素的列表返回
                    return [data]
                    
        except Exception as e:            # 记录错误并返回空列表
            print(f"General error in _unpack: {e}")
            return []
            
    def _pack(self, data_list_of_dicts): 
        """
        将单独的样本打包成一个批次。
        递归地处理嵌套字典结构，返回合并后的批次数据。
        
        Args:
            data_list_of_dicts: 要打包的数据列表，通常是一个字典列表
            
        Returns:
            dict/ndarray: 打包后的批次数据
        """
        # 处理边缘情况
        if not data_list_of_dicts:
            return {}
        
        try:
            # 字典列表：每个字典是一个时间步
            if isinstance(data_list_of_dicts[0], dict):
                # 获取所有键，使用集合合并所有可能的键
                all_keys = set()
                for d in data_list_of_dicts:
                    all_keys.update(d.keys())
                
                res_dict = {}
                for key in all_keys:
                    try:
                        # 收集该键的所有值，对缺失值使用None
                        values_for_key = []
                        for timestep_dict in data_list_of_dicts:
                            if key in timestep_dict:
                                values_for_key.append(timestep_dict[key])
                            else:
                                # 键缺失，使用None占位
                                values_for_key.append(None)
                        
                        # 递归打包，如果全是None则跳过
                        if any(v is not None for v in values_for_key):
                            # 过滤掉None并记录其索引
                            valid_values = []
                            valid_indices = []
                            for i, v in enumerate(values_for_key):
                                if v is not None:
                                    valid_values.append(v)
                                    valid_indices.append(i)
                            
                            # 递归处理有效值
                            if valid_values:
                                packed_valid = self._pack(valid_values)
                                
                                # 如果所有值都有效，直接使用结果
                                if len(valid_values) == len(values_for_key):
                                    res_dict[key] = packed_valid
                                else:
                                    # 否则需要重建完整列表，填充None的位置
                                    print(f"Warning: Key {key} has {len(values_for_key) - len(valid_values)} None values")
                                    # 这里的处理取决于packed_valid的类型
                                    if isinstance(packed_valid, np.ndarray):
                                        # 为None值创建适当形状的零数组
                                        full_shape = list(packed_valid.shape)
                                        full_shape[0] = len(values_for_key)  # 调整批次维度
                                        full_array = np.zeros(full_shape, dtype=packed_valid.dtype)
                                        
                                        # 将有效值放入对应位置
                                        for src_idx, dst_idx in enumerate(valid_indices):
                                            if src_idx < len(packed_valid):  # 安全检查
                                                full_array[dst_idx] = packed_valid[src_idx]
                                        
                                        res_dict[key] = full_array
                                    else:
                                        # 对于其他类型（如列表），直接替换
                                        # 这种情况较少见，因为大多数有效值会被转换为ndarray
                                        res_dict[key] = packed_valid
                    except Exception as e:
                        print(f"Error packing key {key}: {e}")
                        # 跳过这个键但继续处理其他键
                        continue
                
                return res_dict
            
            # 数组的列表：打包成一个大数组
            elif isinstance(data_list_of_dicts[0], np.ndarray):
                try:
                    # 过滤掉所有None值
                    valid_arrays = [arr for arr in data_list_of_dicts if arr is not None]
                    if not valid_arrays:
                        return np.array([])  # 全部是None
                    
                    # 检查所有数组的形状是否一致
                    first_shape = valid_arrays[0].shape
                    all_same_shape = all(arr.shape == first_shape for arr in valid_arrays)
                    
                    if all_same_shape:
                        # 所有数组形状一致，可以直接堆叠
                        return np.stack(valid_arrays)
                    else:
                        # 形状不一致，尝试使用战略
                        print(f"Warning: Arrays have inconsistent shapes for stacking")
                        
                        # 策略1：使用最常见的形状
                        from collections import Counter
                        shapes = [arr.shape for arr in valid_arrays]
                        most_common_shape = Counter(shapes).most_common(1)[0][0]
                        
                        # 只保留符合最常见形状的数组
                        filtered_arrays = [arr for arr in valid_arrays if arr.shape == most_common_shape]
                        if filtered_arrays:
                            return np.stack(filtered_arrays)
                        else:
                            # 如果过滤后为空，退回到列表
                            return valid_arrays
                except ValueError as e:
                    print(f"Warning: Could not stack arrays: {e}")
                    # 退回到列表
                    return [arr for arr in data_list_of_dicts if arr is not None]
            
            # 基本类型列表（数字、布尔值等）：转换为数组
            else:
                try:
                    # 过滤掉None
                    valid_items = [item for item in data_list_of_dicts if item is not None]
                    if not valid_items:
                        return np.array([])  # 全是None
                    
                    return np.array(valid_items)
                except Exception as e:
                    print(f"Warning: Could not convert to array: {e}")
                    # 退回到列表
                    return [item for item in data_list_of_dicts if item is not None]
                    
        except Exception as e:
            print(f"General error in _pack: {e}")
            # 返回原始列表作为最后的退路
            return data_list_of_dicts