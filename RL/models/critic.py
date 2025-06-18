import torch
from torch import nn
from .base_model import ResNet34
from collections import OrderedDict
from .base_model import ResNetFeatureExtractor
from .base_model import ExtraInfoFeatureExtractor

import os
import sys

class CriticBase:
    def __init__(self):
        super(CriticBase, self).__init__()

    def forward(self, state, action):
        raise NotImplementedError("This method should be overridden by subclasses")

    def update(self, state, action, target):
        raise NotImplementedError("This method should be overridden by subclasses")

    def save(self, filepath):
        """保存模型权重。"""
        print(f"正在将 Critic 权重保存到: {filepath}")
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """加载模型权重。"""
        print(f"正在从 '{filepath}' 加载 Critic 权重...")
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        
class ResNet34CentralizedCritic(CriticBase, nn.Module):
    """
    一个采用后期融合 (Late Fusion) 架构的中心化 Critic。
    """
    def __init__(self, in_channels_obs: int, in_channels_extra: int, 
                 obs_feature_dim: int = 2304, extra_feature_dim: int = 128, mlp_hidden_dim: int = 512):
        """
        初始化中心化 Critic。

        Args:
            in_channels_obs (int): 局部观测的输入通道数。
            in_channels_extra (int): 额外全局信息的输入通道数。
            obs_feature_dim (int): ResNetFeatureExtractor 输出的展平后的特征维度。
            extra_feature_dim (int): 额外信息特征提取器输出的维度。
            mlp_hidden_dim (int): 最终融合 MLP 的隐藏层维度。
        """
        super(ResNet34CentralizedCritic, self).__init__()
        print("正在创建 ResNet34CentralizedCritic 实例...")

        # 1. 处理标准局部观测的特征提取器 (这个可以用IL权重初始化)
        self.feature_extractor_obs = ResNetFeatureExtractor(in_channels_obs)
        
        # 2. 处理额外全局信息的特征提取器 (这个将从头训练)
        self.feature_extractor_extra = ExtraInfoFeatureExtractor(in_channels_extra, extra_feature_dim)
        
        self.flatten = nn.Flatten()
        
        # 3. 融合后的 MLP 头部，用于计算最终的价值
        combined_feature_dim = obs_feature_dim + extra_feature_dim
        self.critic_head_mlp = nn.Sequential(
            nn.Linear(combined_feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1) # 输出单个标量价值
        )
        print("ResNet34CentralizedCritic 实例创建完毕。")

    def forward(self, input_dict: dict, extra_info_tensor: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            input_dict (dict): 包含局部观测的字典，与 Actor 使用的格式相同。
            extra_info_tensor (torch.Tensor): 包含额外中心化信息的张量。

        Returns:
            torch.Tensor: 计算出的状态价值。
        """
        # 提取局部观测特征
        local_features = self.feature_extractor_obs(input_dict)
        local_features_flat = self.flatten(local_features)

        # 提取额外全局信息特征
        extra_features_flat = self.feature_extractor_extra(extra_info_tensor)

        # 拼接 (融合) 特征
        combined_features = torch.cat([local_features_flat, extra_features_flat], dim=1)

        # 通过 MLP 头部得到最终价值
        value = self.critic_head_mlp(combined_features)
        return value

    def update(self, input_dict: dict, extra_info_tensor: torch.Tensor, target: torch.Tensor):
        pass