import ipdb.stdout
import torch
import numpy as np
import os
import glob
import random
from torch.utils.data import Dataset, DataLoader, Sampler
import ipdb
from torch.nn.utils.rnn import pad_sequence
#* key padding mask 由false到true
class TimeMajongDataset(Dataset):
    """
    麻将数据集，用于加载预处理后的麻将对局数据
    """
    def __init__(self, data_dir, split_ratio=(0.8, 0.1, 0.1), split='train', random_seed=42):
        """
        初始化数据集
        
        Args:
            data_dir: 存储.npz数据文件的目录
            split_ratio: (训练集, 验证集, 测试集)的比例
            split: 'train', 'val', 或 'test'
            random_seed: 随机种子，确保可重现性
        """
        self.data_dir = data_dir
        self.split = split
        #TODO: 加载数据一次加载，过于浪费，可能需要缓存
        # 获取所有npz文件
        npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        
        # 设置随机种子以确保可重现性
        random.seed(random_seed)
        random.shuffle(npz_files)
        
        # 划分数据集
        train_size = int(len(npz_files) * split_ratio[0])
        val_size = int(len(npz_files) * split_ratio[1])
        
        if split == 'train':
            self.files = npz_files[:train_size]
        elif split == 'val':
            self.files = npz_files[train_size:train_size+val_size]
        else:  # 'test'
            self.files = npz_files[train_size+val_size:]
            
        # 加载所有数据以获取样本数量
        self.data = []
        self.total_samples = 0
        
        for file in self.files:
            data = np.load(file)
            history = data['history']
            global_state = data['global_state']
            mask = data['mask']
            act = data['act']
            lengths = data['lengths']
            for i, length in enumerate(lengths):
                self.data.append({
                    'history': history[:length],
                    'global_state': global_state[i],
                    'action_mask': mask[i],
                    'action': act[i]
                })
                
            self.total_samples += len(act)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.FloatTensor(sample['history']),
            torch.FloatTensor(sample['global_state']),
            torch.FloatTensor(sample['action_mask']),
            torch.LongTensor([sample['action']])
        )
    def get_sample_length(self, idx):
        """仅返回样本历史序列的长度，而不加载完整数据"""
        return len(self.data[idx]['history'])


from collections import OrderedDict

# 1. 自定义简单缓存类(可序列化)
class SimpleCache:
    def __init__(self, maxsize=100):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        
    def get(self, key):
        if key in self.cache:
            # 将访问的项移到末尾(最近使用)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    def set(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            # 移除最早添加的项(最少使用)
            self.cache.popitem(last=False)
        self.cache[key] = value

class EfficientMahjongDataset(Dataset):
    """
    麻将数据集的高效实现，使用索引+缓存方式按需加载数据
    """
    def __init__(self, data_dir, split_ratio=(0.8, 0.1, 0.1), split='train', random_seed=42, cache_size=100):
        """初始化数据集"""
        self.data_dir = data_dir
        self.split = split
        
        # 获取所有npz文件
        npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        
        # 设置随机种子以确保可重现性
        random.seed(random_seed)
        random.shuffle(npz_files)
        
        # 划分数据集
        train_size = int(len(npz_files) * split_ratio[0])
        val_size = int(len(npz_files) * split_ratio[1])
        
        if split == 'train':
            self.files = npz_files[:train_size]
        elif split == 'val':
            self.files = npz_files[train_size:train_size+val_size]
        else:  # 'test'
            self.files = npz_files[train_size+val_size:]
        
        # 使用自定义缓存替代lru_cache
        self.cache = SimpleCache(maxsize=cache_size)
        
        # 创建索引列表 [(file_idx, idx, time_idx), ...]
        self.sample_indices = []
        self._build_indices()
    
    def _build_indices(self):
        """构建所有样本的索引"""
        for file_idx, file_path in enumerate(self.files):
            # 只加载lengths数据来构建索引
            try:
                length = np.load(file_path)['lengths'].tolist() 
                for idx, time_idx in enumerate(length):
                    self.sample_indices.append((file_idx, idx, time_idx))
            except Exception as e:
                print(f"无法加载文件 {file_path}: {e}")
    
    def _load_file(self, file_idx):
        """加载指定索引的文件并返回"""
        cached_data = self.cache.get(file_idx)
        if cached_data is not None:
            return cached_data
            
        data = np.load(self.files[file_idx])
        self.cache.set(file_idx, data)
        return data
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        # 获取样本索引
        file_idx, sample_idx, time_idx = self.sample_indices[idx]
        
        try:
            # 加载文件(通过缓存机制)
            data = self._load_file(file_idx)
            
            # 提取数据
            history = data['history'][:time_idx]  # 只获取到当前时间步的历史
            global_state = data['global_state'][sample_idx] 
            action_mask = data['mask'][sample_idx] 
            action = data['act'][sample_idx]
            return (
                torch.FloatTensor(history),
                torch.FloatTensor(global_state),
                torch.FloatTensor(action_mask),
                torch.LongTensor([action])
            )
        except Exception as e:
            print(f"加载样本错误 idx={idx}, file_idx={file_idx}, sample_idx={sample_idx}, time_idx={time_idx}: {e}")
            # 返回一个空样本作为备用
            raise e
    
    def get_sample_length(self, idx):
        """返回样本历史序列的长度"""
        _, _, time_idx = self.sample_indices[idx]
        return time_idx  # 时间索引即为长度

class BatchFileMahjongDataset(Dataset):
    """
    基于文件批处理的高效麻将数据集实现，支持多进程
    """
    def __init__(self, data_dir, split_ratio=(0.8, 0.1, 0.1), split='train', random_seed=42, cache_size=100):
        """初始化数据集"""
        self.data_dir = data_dir
        self.split = split
        
        # 获取所有npz文件
        npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        
        # 设置随机种子以确保可重现性
        random.seed(random_seed)
        random.shuffle(npz_files)
        
        # 划分数据集
        train_size = int(len(npz_files) * split_ratio[0])
        val_size = int(len(npz_files) * split_ratio[1])
        
        if split == 'train':
            self.files = npz_files[:train_size]
        elif split == 'val':
            self.files = npz_files[train_size:train_size+val_size]
        else:  # 'test'
            self.files = npz_files[train_size+val_size:]
        
        # 全局索引和块映射
        self.sample_indices = []  # 全局索引 [(file_idx, sample_idx, time_idx), ...]
        self.file_to_indices = {}  # 文件索引到样本索引的映射
        self.file_to_chunk = {}    # 文件索引到块索引的映射
        self.chunk_to_files = {}   # 块索引到文件列表的映射
        
        # 构建索引
        self._build_indices()
        
        # 工作进程安全的缓存
        self.cache = SimpleCache(maxsize=cache_size)
        
        # 兼容旧接口，但改为不实际存储数据的字典
        self.current_file_data = {}
        
    def _build_indices(self):
        """构建索引和块映射"""
        global_idx = 0
        for file_idx, file_path in enumerate(self.files):
            try:
                # 加载长度信息以构建索引
                lengths = np.load(file_path)['lengths'].tolist()
                
                # 初始化这个文件的索引列表
                self.file_to_indices[file_idx] = []
                
                for sample_idx, time_idx in enumerate(lengths):
                    # 添加到全局索引
                    self.sample_indices.append((file_idx, sample_idx, time_idx))
                    
                    # 添加到文件索引映射
                    self.file_to_indices[file_idx].append((sample_idx, time_idx))
                    
                    global_idx += 1
            except Exception as e:
                print(f"无法加载文件 {file_path}: {e}")
    
    def load_files(self, file_indices):
        """
        兼容旧接口，但不实际清空缓存
        预加载指定文件到缓存(如果尚未加载)
        """
        # 记录当前正在使用的文件集合(为了兼容)
        self.current_file_data = {idx: True for idx in file_indices}
        
        # 预加载文件到缓存
        for file_idx in file_indices:
            if file_idx not in self.file_to_indices:
                continue
                
            # 检查是否已缓存
            if self.cache.get(file_idx) is None:
                try:
                    data = np.load(self.files[file_idx])
                    self.cache.set(file_idx, data)
                except Exception as e:
                    print(f"加载文件 {self.files[file_idx]} 失败: {e}")
    
    def _get_file_data(self, file_idx):
        """获取文件数据，优先从缓存读取"""
        # 从缓存获取
        data = self.cache.get(file_idx)
        if data is not None:
            return data
            
        # 如果缓存中没有，加载文件
        try:
            data = np.load(self.files[file_idx])
            self.cache.set(file_idx, data)
            return data
        except Exception as e:
            print(f"加载文件 {self.files[file_idx]} 失败: {e}")
            raise e
    
    def get_file_sample_map(self):
        """返回文件索引到样本索引的映射，用于采样器"""
        return self.file_to_indices
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # 兼容旧的(file_idx, local_idx)元组索引
            file_idx, local_idx = idx
            sample_idx, time_idx = self.file_to_indices[file_idx][local_idx]
            
            try:
                # 检查文件是否已在current_file_data中(兼容老逻辑)
                if self.current_file_data and file_idx not in self.current_file_data:
                    # 自动加载文件
                    self.load_files([file_idx])
                
                # 获取文件数据
                data = self._get_file_data(file_idx)
                
                # 提取所需数据
                history = data['history'][:time_idx]
                global_state = data['global_state'][sample_idx]
                action_mask = data['mask'][sample_idx]
                action = data['act'][sample_idx]
                
                return (
                    torch.FloatTensor(history),
                    torch.FloatTensor(global_state),
                    torch.FloatTensor(action_mask),
                    torch.LongTensor([action])
                )
            except Exception as e:
                print(f"加载样本错误 file_idx={file_idx}, local_idx={local_idx}: {e}")
                raise e
        else:
            # 支持使用全局整数索引(用于多进程)
            file_idx, sample_idx, time_idx = self.sample_indices[idx]
            
            try:
                # 获取文件数据
                data = self._get_file_data(file_idx)
                
                # 提取所需数据
                history = data['history'][:time_idx]
                global_state = data['global_state'][sample_idx]
                action_mask = data['mask'][sample_idx]
                action = data['act'][sample_idx]
                
                return (
                    torch.FloatTensor(history),
                    torch.FloatTensor(global_state),
                    torch.FloatTensor(action_mask),
                    torch.LongTensor([action])
                )
            except Exception as e:
                print(f"加载样本错误 idx={idx}, file_idx={file_idx}, sample_idx={sample_idx}: {e}")
                raise e
    
    def get_sample_length(self, idx):
        """返回样本历史序列的长度"""
        if isinstance(idx, tuple):
            # 兼容旧索引
            file_idx, local_idx = idx
            _, time_idx = self.file_to_indices[file_idx][local_idx]
        else:
            # 全局整数索引
            _, _, time_idx = self.sample_indices[idx]
        return time_idx
# 2. 修复collate_fn中的键名
def collate_fn(batch):
    """将一个batch的样本填充到相同长度，并返回符合模型要求的字典格式"""
    histories, global_states, action_masks, actions = zip(*batch)
    
    # 计算最大长度并预分配内存
    max_len = max(len(h) for h in histories)
    batch_size = len(histories)
    history_dim = histories[0].shape[1]
    
    # 预分配填充张量
    histories_padded = torch.zeros((batch_size, max_len, history_dim), dtype=torch.float32)
    pad_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    # 手动填充
    for i, h in enumerate(histories):
        length = len(h)
        histories_padded[i, :length] = h
        pad_mask[i, length:] = True
    
    # 不需要pad的tensors
    global_states = torch.stack(global_states)
    action_masks = torch.stack(action_masks)
    actions = torch.cat(actions)
    
    return {
        "history": histories_padded,
        "global_state": global_states,  # 修改为train.py中使用的键名
        "action_mask": action_masks,
        "action": actions,
        "pad_mask": pad_mask
    }

class FileBatchSampler(Sampler):
    """
    基于文件批处理的采样器，支持多进程
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, files_per_chunk=50):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.files_per_chunk = files_per_chunk
        
        # 获取文件到样本的映射
        self.file_sample_map = dataset.get_file_sample_map()
        self.file_indices = list(self.file_sample_map.keys())
        
        # 创建文件到全局索引的映射
        self.file_to_global_indices = {}
        for global_idx, (file_idx, _, _) in enumerate(dataset.sample_indices):
            if file_idx not in self.file_to_global_indices:
                self.file_to_global_indices[file_idx] = []
            self.file_to_global_indices[file_idx].append(global_idx)
    
    def __iter__(self):
        # 文件级别的随机打乱
        if self.shuffle:
            random.shuffle(self.file_indices)
        
        # 将文件分成多个chunk
        file_chunks = [
            self.file_indices[i:i+self.files_per_chunk]
            for i in range(0, len(self.file_indices), self.files_per_chunk)
        ]
        
        for chunk in file_chunks:
            # 预加载当前chunk中的所有文件
            self.dataset.load_files(chunk)
            
            # 创建全局索引列表
            all_indices = []
            for file_idx in chunk:
                if file_idx in self.file_to_global_indices:
                    all_indices.extend(self.file_to_global_indices[file_idx])
            
            # 随机打乱样本顺序
            if self.shuffle:
                random.shuffle(all_indices)
            
            # 生成批次
            for i in range(0, len(all_indices), self.batch_size):
                batch = all_indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or i + self.batch_size >= len(all_indices):
                    yield batch
    
    def __len__(self):
        total_samples = sum(len(samples) for samples in self.file_sample_map.values())
        return (total_samples + self.batch_size - 1) // self.batch_size
class LocalSampler():
    '''与dataset相配合的采样器'''
    '''利用在一个epoch内每个样本只会生成一次的特点，每个epoch，每个文件会被读入一次；小批量读入，小批量采样'''
    # 读取 dataset SimpleCache中的文件下标, 记为cache_idx
        #loop:
        # 保证 dataset.sample_indices[current samples] 的 第一位都在cache_idx中
    # 采样完成后，清空cache_idx
    #注：这样的话,cache_idx甚至不需要维护LRU的逻辑;唯一需要维护的是，确保文件不会重复读取
def get_mahjong_dataloader(data_dir, name='efficient', batch_size=32, split='train', shuffle=True, 
                          files_per_chunk=50, num_workers=4):
    """创建麻将数据加载器"""
    # 创建数据集
    if name == 'efficient':
        dataset = EfficientMahjongDataset(data_dir, split=split)
    elif name == 'batch_file':
        dataset = BatchFileMahjongDataset(data_dir, split=split)
        sampler = FileBatchSampler(dataset, 
                                   batch_size=batch_size, 
                                   shuffle=shuffle, 
                                   files_per_chunk=files_per_chunk)
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=True# 由于我们的采样策略，这里不使用多进程
        )        
        return dataloader
    else:
        dataset = TimeMajongDataset(data_dir, split=split)
         # 使用DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,  # 多进程加载
        persistent_workers=True  # 保持工作进程存活，减少创建开销
    )

    return dataloader

if __name__ == "__main__":
    # 测试数据集
    data_dir = "data"
    dataloader = get_mahjong_dataloader(data_dir,
                                        name='batch_file',
                                        batch_size=32,
                                        split='train',
                                        shuffle=True,
                                        num_workers=8)
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, batch in enumerate(tqdm(dataloader)):
        history = batch["history"].to(device)
        global_state = batch["global_state"].to(device)
        action_mask = batch["action_mask"].to(device)
        action = batch["action"].to(device)
        pad_mask = batch["pad_mask"].to(device)