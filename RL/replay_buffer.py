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
        if isinstance(data, dict): # Changed type() to isinstance() for robustness
            res = []
            # Find the length of the episode from the first list-like value
            # This assumes all features in an episode have the same number of timesteps
            episode_len = 0
            for value in data.values():
                if isinstance(value, (list, np.ndarray)): # Should be list or ndarray of features
                    episode_len = len(value)
                    break
            if episode_len == 0 and data: # Non-empty dict but no list-like values or empty lists
                 # This might indicate an issue with how episode data is structured or if it's a single step
                 # For now, if no length, assume it's a single step dict to be wrapped in a list
                # Or handle as an error / special case depending on expected data format
                # Let's assume for now _unpack is always called with list-like structures for episode data
                # If 'data' can be a single timestep dict, _unpack logic needs adjustment.
                # Given it's called on 'episode_data', it's likely a collection of timesteps.
                pass


            if not data: return [] # Handle empty dict case for episode_data

            res = [{} for _ in range(episode_len)]
            for key, value_list in data.items():
                unpacked_values = self._unpack(value_list) # value_list should be a list of items for this key
                if len(unpacked_values) != episode_len and episode_len !=0 :
                    # This would be an error: data inconsistency within an episode
                    print(f"Warning: Key {key} has {len(unpacked_values)} items, expected {episode_len}")
                    # Handle error or skip this key
                    continue 
                for i, v_item in enumerate(unpacked_values):
                    if i < episode_len: # safety for res index
                        res[i][key] = v_item
            return res
        elif isinstance(data, (list, tuple, deque)): # if it's already a list of items (e.g. list of rewards)
            return list(data) # Ensure it's a plain list, or unpack further if elements are dicts
        else: # individual numbers, strings, or ndarrays (not representing multiple timesteps for this key)
            # This branch is tricky for _unpack. _unpack's goal is to yield a LIST of timestep dicts.
            # If 'data' is a single ndarray for a single timestep, it should be wrapped.
            # The original logic `return list(data)` for non-dicts is likely for when `value` is a list of numbers/arrays already.
            # Example: data['rewards'] = [r1, r2, r3] -> _unpack([r1,r2,r3]) -> [r1,r2,r3] (correct)
            return list(data) # This assumes if not dict, it's already an iterable of per-timestep features
                                # or a single feature that should be iterated by the caller's _unpack.

    def _pack(self, data_list_of_dicts): # data is a list of timestep dicts
        if not data_list_of_dicts:
            return {} # Or handle as an error / return None

        if isinstance(data_list_of_dicts[0], dict):
            keys = data_list_of_dicts[0].keys()
            res_dict = {}
            for key in keys:
                # Collect all values for this key from each timestep dict in the list
                values_for_key = [timestep_dict[key] for timestep_dict in data_list_of_dicts]
                res_dict[key] = self._pack(values_for_key) # Recursively pack these collected values
            return res_dict
        # Base case for recursion: list of numbers or ndarrays
        elif isinstance(data_list_of_dicts[0], np.ndarray):
            try:
                return np.stack(data_list_of_dicts)
            except ValueError as e: # Handle cases where arrays can't be stacked (e.g. different shapes not along axis 0)
                # print(f"Warning: Could not stack ndarrays for packing, returning as list. Error: {e}")
                return list(data_list_of_dicts) # Fallback to list of arrays
        else: # list of numbers, bools, etc.
            return np.array(data_list_of_dicts)