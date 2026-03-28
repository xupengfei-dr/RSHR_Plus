from torch import nn


class In2(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # 序列维度平均池化

    def forward(self, x):
        # 输入 x: [B, 40, 768]
        x = x.permute(0, 2, 1)  # → [B, 768, 40]
        pooled = self.pool(x)  # → [B, 768, 1]
        return pooled.squeeze(-1)  # → [B, 768]