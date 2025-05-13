import torch
import numpy as np
import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
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
            hand = data['hand']
            mask = data['mask']
            act = data['act']
            
            for i in range(len(act)):
                self.data.append({
                    'history': history[:i+1],
                    'hand': hand[i],
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
            torch.FloatTensor(sample['hand']),
            torch.FloatTensor(sample['action_mask']),
            torch.LongTensor([sample['action']])
        )
    def get_sample_length(self, idx):
        """仅返回样本历史序列的长度，而不加载完整数据"""
        return len(self.data[idx]['history'])

def create_bucket_sampler(dataset:TimeMajongDataset, bucket_sizes, batch_size, shuffle=True):
    """
    创建基于bucket的采样器
    
    Args:
        dataset: 要采样的数据集
        bucket_sizes: bucket的大小列表
        batch_size: 每个batch的样本数
        shuffle: 是否打乱样本
    
    Returns:
        BucketSampler对象
    """
    lengths = []
    for idx in range(len(dataset)):
        # 只获取history长度，不加载完整数据
        sample = dataset.get_sample_length(idx)  # 需要添加此方法到数据集类
        lengths.append(sample)
    
    # 创建buckets
    buckets = [[] for _ in bucket_sizes]
    for idx, length in enumerate(lengths):
        for i, max_len in enumerate(bucket_sizes):
            if length <= max_len:
                buckets[i].append(idx)
                break
    
    return BucketSampler(buckets, batch_size, shuffle)


class BucketSampler(torch.utils.data.Sampler):
    """
    基于bucket的采样器，将相似长度的序列放在一起，减少填充,目的就是返回每次采样需要的下标
    """
    def __init__(self, buckets, batch_size, shuffle=True):
        self.buckets = buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = []
        self.create_indices()
    
    def create_indices(self):
        self.indices = []
        for bucket in self.buckets:
            # Skip empty buckets
            if not bucket:
                continue
                
            if self.shuffle:
                random.shuffle(bucket)
                
            # Create batches from the bucket
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i+self.batch_size]
                # Only add batches with at least one sample
                if batch:
                    self.indices.append(batch)
                    
        # Shuffle the batches if needed
        if self.shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self):
        self.create_indices()
        for batch in self.indices:
            yield batch
    
    def __len__(self):
        return sum(len(bucket)//self.batch_size + (1 if len(bucket)%self.batch_size >0 else 0) for bucket in self.buckets)


def collate_fn(batch):
    """
    将一个batch的样本填充到相同长度，并返回符合模型要求的字典格式
    """
    histories, hands, action_masks, actions = zip(*batch)
    
    # 填充history到相同长度
    histories_padded = pad_sequence(histories, batch_first=True)
    
    # 创建padding mask (True表示要mask的位置)
    lengths = [len(h) for h in histories]
    max_len = max(lengths)
    pad_mask = torch.zeros(len(histories), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        pad_mask[i, length:] = True
    
    # 不需要pad的tensors
    hands = torch.stack(hands)
    action_masks = torch.stack(action_masks)
    actions = torch.cat(actions)
    
    return {
        "history": histories_padded,
        "hand": hands,
        "action_mask": action_masks,
        "action": actions,
        "pad_mask": pad_mask
    }

def get_mahjong_dataloader(data_dir, batch_size=32, split='train', shuffle=True, bucket_sizes=[20 * i for i in range(1,21)]):
    """
    创建麻将数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: batch大小
        split: 'train', 'val', 或 'test'
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader对象
    """
    # 创建数据集
    dataset = TimeMajongDataset(data_dir, split=split)
    
    # 根据新的特征设计调整bucket大小
    # 基础特征为6维，历史特征为95维，总共101维

    
    # 创建bucket采样器
    sampler = create_bucket_sampler(dataset, bucket_sizes, batch_size, shuffle)
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return dataloader