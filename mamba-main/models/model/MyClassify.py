import torch
import torch.nn as nn
import torch.nn.functional as F


# Swish 激活函数的实现
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 自定义的 Residual Block（带跳跃连接）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=Swish):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.activation = activation()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels)
            )

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return self.activation(out + self.shortcut(x))  # 添加跳跃连接


# 强化版分类网络
class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim, number_outputs):
        super(EnhancedClassifier, self).__init__()
        self.number_outputs = number_outputs

        # 自定义分类层，使用更强的模块
        self.classify_layer = nn.Sequential(
            ResidualBlock(input_dim, 1024),  # 使用 Residual Block
            nn.LayerNorm(1024),  # 使用 LayerNorm 代替 BatchNorm
            Swish(),  # 使用 Swish 激活函数
            nn.Dropout(0.5),  # 增加 Dropout 防止过拟合
            ResidualBlock(1024, 512),
            nn.LayerNorm(512),  # LayerNorm
            Swish(),
            ResidualBlock(512, 256),
            nn.LayerNorm(256),  # LayerNorm
            Swish(),
            nn.Linear(256, self.number_outputs)  # 输出层
        )

    def forward(self, x):
        return self.classify_layer(x)

if __name__ == '__main__':
    input_tensor = torch.randn(64, 768)
    class_faiy = EnhancedClassifier(768,9)
    res = class_faiy(input_tensor)
    print(res.shape)