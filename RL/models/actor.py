from .base_model import ResNet34
import torch
from torch import nn

class ActorBase:
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, input_dict):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def update(self, input_dict, target):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def save(self, filepath):
        """保存模型权重。"""
        print(f"正在将 Actor 权重保存到: {filepath}")
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """加载模型权重。"""
        print(f"正在从 '{filepath}' 加载 Actor 权重...")
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

        
class ResNet34Actor(ActorBase, nn.Module): # Actor 也应该是 nn.Module
    def __init__(self, in_channels, out_channels):
        # 正确调用父类 __init__
        ActorBase.__init__(self, in_channels, out_channels=235)
        nn.Module.__init__(self)
        self.resnet34 = ResNet34(in_channels)

    def forward(self, input_dict):
        return self.resnet34(input_dict)

    def update(self, input_dict, target):
        pass