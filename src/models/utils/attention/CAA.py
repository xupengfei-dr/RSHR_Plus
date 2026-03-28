import os
from typing import Optional

import matplotlib
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import torchvision.transforms.functional as TF
# 论文地址：https://arxiv.org/pdf/2403.06258
# 论文：Poly Kernel Inception Network for Remote Sensing Detection(CVPR 2024)
# Github地址：https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
# Context Anchor Attention (CAA) module
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        # Convolution Layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # Normalization Layer
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # Activation Layer
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # Combine all layers
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # Add more normalization types if needed
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # Add more activation types if needed
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

# Example usage to print input and output shapes
if __name__ == "__main__":

    layers_caa = nn.ModuleList([CAA(channels=3) for _ in range(12)])

    img = Image.open('demo.jpeg').convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # [0,1]
    ])

    img_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 128, 128)
    res = img_tensor
    # === 如果图像不是64通道，我们需要升维（例如通过1x1卷积） ===
    # img_tensor = nn.Conv2d(3, 64, kernel_size=1)(img_tensor)  # Shape: (1, 64, 128, 128)

    # === 应用CAA模块 ===
    caa = CAA(channels=3)
    with torch.no_grad():
        attn = caa(img_tensor)  # Shape: (1, 64, 128, 128)
        # for layers in layers_caa:
        #     img_tensor = layers(img_tensor)
        attn = attn+res
    # === 保存处理后的图像 ===
    processed = attn.squeeze(0).clamp(0, 1)  # 去掉 batch 维度
    processed_img = TF.to_pil_image(processed)  # 转换为 PIL 图像
    processed_img.save("processed__1_+.jpg")  # 保存
    # === 可视化一个通道的注意力图 ===
    # # === 可视化并保存注意力图 ===
    # attn_map = attn[0, 0].cpu().numpy()  # 取第一个通道进行可视化
    # save_path = "attention_map3l.jpg"
    # # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.figure(figsize=(6, 5))
    # plt.imshow(attn_map, cmap='jet')
    # plt.title("CAA Attention Map (Channel 0)")
    # plt.axis('off')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"Attention map saved to {save_path}")

    # input = torch.randn(64, 3, 128, 128) #输入 B C H W
    # block = CAA(3)
    # output = block(input)
    # print(input.size())
    # print(output.size())