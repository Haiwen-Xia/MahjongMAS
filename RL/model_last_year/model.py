import torch
from torch import nn

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


# 首先我们定义了卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# 进一步抽象,定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 一个残差块中包含两个前面定义的卷积块
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)

        # 由于我们的in_channels和out_channels未必匹配,因此我们考虑downsampling操作
        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    # 前向传播
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        residual = self.downsample(x)
        out = residual + out
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1_1 = ConvBlock(203, 256)
        self.layer1_2 = ResidualBlock(256, 256)
        self.layer1_3 = ResidualBlock(256, 256)
        self.layer1_4 = ResidualBlock(256, 128)

        self.layer2_1 = ResidualBlock(128, 128)
        self.layer2_2 = ResidualBlock(128, 128)
        self.layer2_3 = ResidualBlock(128, 128)
        self.layer2_4 = ResidualBlock(128, 64)
        
        self.layer3_1 = ResidualBlock(64, 64)
        self.layer3_2 = ResidualBlock(64, 64)
        self.layer3_3 = ResidualBlock(64, 64)
        self.layer3_4 = ResidualBlock(64, 64)

        # 将特征图展平
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 9 * 64, 256)
        self.fc2 = nn.Linear(256,235)

    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()

        obs = self.layer1_4(self.layer1_3(self.layer1_2(self.layer1_1(obs))))
        obs = self.layer2_4(self.layer2_3(self.layer2_2(self.layer2_1(obs))))
        obs = self.layer3_4(self.layer3_3(self.layer3_2(self.layer3_1(obs))))
        
        obs = self.flatten(obs)

        action_logits = self.fc2(F.relu(self.fc1(obs)))
        
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask