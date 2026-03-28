import torch
from torch import nn
from torch.nn import functional as F
class In2Conv1d(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # 序列维度平均池化
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.convd1 =nn.Conv1d(in_channels=768, out_channels=768, kernel_size=40, stride=40, groups=768)


    def forward(self, x):
        # 输入 x: [B, 40, 768]
        x = x.permute(0, 2, 1)
        res = self.convd1(x)
        return res.squeeze(-1)

class In2(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # 序列维度平均池化

    def forward(self, x,mask=None):
        # 输入 x: [B, 40, 768]
        x = x.permute(0, 2, 1)  # → [B, 768, 40]
        pooled = self.pool(x)  # → [B, 768, 1]
        return pooled.squeeze(-1)  # → [B, 768]

class GlobalQueryAttentionPool(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 可学习的全局Query [1, D]
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim))
        nn.init.xavier_uniform_(self.global_query)

        # Key-Value投影层（不再需要Q投影）
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出层 & 正则化
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: x [B, L, D]
        输出: pooled [B, D]
        """
        B, L, D = x.shape

        # 生成全局Query → [B, 1, D]
        q = self.global_query.expand(B, -1, -1)  # 复制到批次大小

        # 生成Key-Value → [B, L, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 多头切分 → [B, H, Q_L, d]
        q = q.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, 1, d]
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, d]
        v = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, d]

        # 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, 1, L]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合 → [B, H, 1, d]
        context = torch.matmul(attn_weights, v)  # [B, H, 1, d]

        # 多头合并 → [B, 1, D]
        context = context.transpose(1, 2).contiguous().view(B, 1, D)

        # 池化：直接取加权后的结果
        pooled = context.squeeze(1)  # [B, D]

        return self.out_proj(pooled)


class myClassify(nn.Module):

    def __init__(self, number_outputs) -> None:
        super().__init__()
        self.number_outputs =number_outputs

    def forward(self,x):
       class_res = nn.Sequential(
            nn.Linear(768, 1024),  # 调整为更合适的隐藏层大小
            nn.BatchNorm1d(1024),  # 添加 BatchNorm
            nn.GELU(),  # 使用更高效的激活函数
            nn.Dropout(0.5),  # 增大正则化强度，减少过拟合
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # 再次使用 BatchNorm
            nn.GELU(),
            nn.Dropout(0.3),  # 减少后续 Dropout 强度
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, self.number_outputs)  # 输出层
        )
       res = class_res(x)
       return res