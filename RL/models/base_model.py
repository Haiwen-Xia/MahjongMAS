import torch
from torch import nn
from torch.nn import functional as F


# 降采样模块
def downsample_block(in_channels, out_channels, stride):

    layers = []
    # 使用 1x1 卷积来匹配通道数和进行空间下采样
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
    # 通常也会在下采样路径中加入 BatchNorm
    layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

# 残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样层 (downsample)
        if in_channels != out_channels or stride != 1:
            self.downsample = downsample_block(in_channels, out_channels, stride)
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out) # 通常在相加之后再应用 ReLU

        return out
    

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(ResNetFeatureExtractor, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.residuals1 = nn.Sequential(
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )
        self.residuals2 = nn.Sequential(
            BasicBlock(256, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        self.residuals3 = nn.Sequential(
            BasicBlock(128, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.residuals4 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )

    def forward(self, input_dict):
        obs = input_dict["obs"]["observation"].float()

        x = self.layer1(obs)
        x = self.residuals1(x)
        x = self.residuals2(x)
        x = self.residuals3(x)
        x = self.residuals4(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34, self).__init__()

        self.feature_extractor = ResNetFeatureExtractor(in_channels)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 9 * 64, 235)

    def _extract_feature(self, input_dict):  
        x = self.feature_extractor(input_dict)
        return x
    
    def forward(self, input_dict):
        x = self._extract_feature(input_dict)

        action_logits = self.fc(self.flatten(x))

        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)

        return action_logits + inf_mask

class ExtraInfoFeatureExtractor(nn.Module):
    """
    一个用于处理额外中心化信息的特征提取器模块。
    现在专门为处理形状为 (batch_size, 16, 4, 9) 的全局观测而设计。
    """
    def __init__(self, in_channels_extra, output_dim):
        super(ExtraInfoFeatureExtractor, self).__init__()
        self.in_channels_extra = in_channels_extra  # 应该是 16
        self.output_dim = output_dim
        
        # 验证输入通道数
        if in_channels_extra != 16:
            print(f"Warning: ExtraInfoFeatureExtractor 期望 in_channels_extra=16，但收到 {in_channels_extra}")
        
        # 为 (16, 4, 9) 输入设计的网络
        self.network = nn.Sequential(
            # 输入: (batch_size, 16, 4, 9)
            nn.Conv2d(in_channels_extra, 64, kernel_size=3, padding=1),  # -> (batch_size, 64, 4, 9)
            BasicBlock(64, 64),  # -> (batch_size, 64, 4, 9)
            BasicBlock(64, 32),  # -> (batch_size, 32, 4, 9)
            BasicBlock(32, 16),  # -> (batch_size, 16, 4, 9)
            # nn.AdaptiveAvgPool2d((1, 1)),  # -> (batch_size, 16, 1, 1)
            nn.Flatten(),  # -> (batch_size, 16)
            nn.Linear(16 * 4 * 9, output_dim)  # -> (batch_size, output_dim)
        )

    def forward(self, extra_info_tensor):
        """
        前向传播
        
        Args:
            extra_info_tensor: 形状为 (batch_size, 16, 4, 9) 的张量
            
        Returns:
            输出特征，形状为 (batch_size, output_dim)
        """
        # 验证输入形状
        if len(extra_info_tensor.shape) != 4:
            raise ValueError(f"ExtraInfoFeatureExtractor 期望4D输入 (batch_size, 16, 4, 9)，但收到形状 {extra_info_tensor.shape}")
        
        if extra_info_tensor.shape[1:] != (16, 4, 9):
            raise ValueError(f"ExtraInfoFeatureExtractor 期望输入形状 (batch_size, 16, 4, 9)，但收到 {extra_info_tensor.shape}")
        
        return self.network(extra_info_tensor)