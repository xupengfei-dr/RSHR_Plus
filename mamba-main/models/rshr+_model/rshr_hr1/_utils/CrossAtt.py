import torch
from torch import nn


import torch
import torch.nn as nn
#
# class CrossAttentionBlock(nn.Module):
#
#     def __init__(self, dim, num_heads=8, dropout=0.0):
#         super().__init__()
#         self.norm_q  = nn.LayerNorm(dim)
#         self.norm_kv = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
#         )
#
#     def forward(self, x_q, x_kv, kv_mask=None, attn_mask=None):
#         assert x_q.dim() == 3 and x_kv.dim() == 3, f"x_q/x_kv must be [B,L,H], got {x_q.shape}/{x_kv.shape}"
#         Bq, Lq, Hq = x_q.shape
#         Bk, Lk, Hk = x_kv.shape
#         assert Bq == Bk, "Batch size mismatch between Q and KV"
#         assert Hq == Hk, f"Hidden size mismatch: {Hq} vs {Hk}"
#
#         # ---- 归一化特征（注意：只对特征做LN，不对mask做LN）----
#         q  = self.norm_q(x_q)     # [B, Lq, H]
#         kv = self.norm_kv(x_kv)   # [B, Lk, H]
#
#         # ---- 处理 KV padding 掩码（每样本不同：传给 key_padding_mask）----
#         key_padding_mask = None   # True 表示 pad（屏蔽）
#         if kv_mask is not None:
#             # 支持 [B,1,1,Lk] 或 [B,Lk]
#             if kv_mask.dim() == 4:
#                 kv_mask = kv_mask.squeeze(1).squeeze(1)     # -> [B,Lk]
#             assert kv_mask.shape == (Bk, Lk), f"kv_mask expected [B,Lk], got {kv_mask.shape}"
#             # 约定：kv_mask=1/True 为有效，需要转成 True=pad
#             kv_valid = kv_mask.bool()
#             key_padding_mask = ~kv_valid                    # [B,Lk]，True=pad
#
#         # ---- 处理结构性 attn_mask（批共享的 2D 遮罩）----
#         # PyTorch MHA 期望 attn_mask 形状为 [Lq, Lk]（或 [B*heads, Lq, Lk] 高阶用法）
#         if attn_mask is not None:
#             assert attn_mask.dim() == 2 and attn_mask.shape == (Lq, Lk), \
#                 f"attn_mask must be [Lq, Lk]; got {attn_mask.shape}"
#             # 如果是布尔（True=屏蔽）或加性（-inf），MHA 都能接受；保持 dtype/shape 即可
#
#         # ---- 注意力 ----
#         out, _ = self.attn(
#             q, kv, kv,
#             attn_mask=attn_mask,                 # 批共享 2D 结构遮罩（可为 None）
#             key_padding_mask=key_padding_mask    # 每样本的 KV padding
#         )   # out: [B, Lq, H]
#
#         # ---- 残差 ----
#         return x_q + out

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm_q = torch.nn.LayerNorm(dim)
        self.norm_kv = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x_q, x_kv, attn_mask=None):
        kpm = extended_to_key_padding_mask(attn_mask)
        # print(attn_mask.shape)
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        # print(attn_mask.shape)
        out, _ = self.attn(q, kv, kv,key_padding_mask=kpm)
        x = x_q + out * 0.2
        return x


def extended_to_key_padding_mask(ext_mask: torch.Tensor) -> torch.Tensor:
    """
    ext_mask: (B,1,1,L) 或可被 squeeze 到 (B,L) 的张量
    返回: key_padding_mask (B,L) bool，True=mask/pad
    """
    if ext_mask is None:
        return None

    # squeeze 到 (B, L)
    m = ext_mask
    # 连续去掉前面的 size=1 维度（保留 B 和 L）
    while m.dim() > 2 and m.size(1) == 1:
        m = m.squeeze(1)
    while m.dim() > 2 and m.size(1) == 1:
        m = m.squeeze(1)
    # 现在应为 (B, L)
    assert m.dim() == 2, f"Expected (B,L) after squeeze, got {tuple(m.shape)}"

    if m.dtype == torch.bool:
        # 常见：True=keep，需要反转；做个启发式判断
        true_ratio = m.float().mean().item()
        return ~m if true_ratio > 0.5 else m  # 若大部分为True，认为True=keep

    if torch.is_floating_point(m):
        # 典型HF：keep=0.0，mask为负大数（如 -inf 或 -1e4）
        # 统一判定：小于0 或 非有限值 都视作 mask
        return (~torch.isfinite(m)) | (m < 0)

    # 整数类型：常见 1=keep, 0=mask
    return (m == 0)

# def make_key_padding_mask(mask: torch.Tensor, neg_threshold: float = -1e3) -> torch.Tensor:
#     """
#     将 HF 的 attention/extended_attention_mask 转为 PyTorch MHA 的 key_padding_mask。
#     返回: [B, L]，dtype=bool，True 表示 pad（需要屏蔽）
#     """
#     assert mask.dim() in (2, 4), f"expect [B,L] or [B,1,1,L], got {mask.shape}"
#
#     # 1) [B,1,1,L] → [B,L]
#     if mask.dim() == 4:
#         mask = mask.squeeze(1).squeeze(1)  # [B, L]
#
#     # 2) 不同 dtype 的判定
#     if mask.dtype == torch.bool:
#         # 约定 True=有效（大多数 HF attention_mask），转成 True=pad
#         key_padding_mask = ~mask
#     elif torch.is_floating_point(mask):
#         # extended_attention_mask 通常有效=0，pad=大负数（-1e4 或 -inf）
#         key_padding_mask = mask < neg_threshold
#     else:
#         # 整型 attention_mask：1=有效, 0=pad
#         key_padding_mask = (mask == 0)
#
#     return key_padding_mask  # [B, L], bool(True=pad)

class CrossAttentionBlock_orimask(torch.nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm_q = torch.nn.LayerNorm(dim)
        self.norm_kv = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x_q, x_kv, attn_mask=None):
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        out, _ = self.attn(q, kv, kv, attn_mask=attn_mask)
        x = x_q + out
        return x


class CrossAttentionBlock_mlp(torch.nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm_q = torch.nn.LayerNorm(dim)
        self.norm_kv = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, 2*dim),
            torch.nn.GELU(),
            torch.nn.Linear(2*dim, dim),
        )
    def forward(self, x_q, x_kv, attn_mask=None):
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        out, _ = self.attn(q, kv, kv, attn_mask=attn_mask)
        x = x_q + out
        x = x + self.mlp(x)
        return x

class KVPrompt(nn.Module):
    def __init__(self, h_in, h_kv, M=8, dropout=0.1):
        super().__init__()
        self.M = M
        self.proj = nn.Sequential(nn.Linear(h_in, h_kv), nn.GELU(), nn.Dropout(dropout), nn.Linear(h_kv, h_kv))
        self.pos_bias = nn.Parameter(torch.zeros(M, h_kv))
    def forward(self, img_cls_vec):  # [B,H_in]
        base = self.proj(img_cls_vec)                               # [B,H_kv]
        kv   = base.unsqueeze(1).expand(-1, self.M, -1) + self.pos_bias.unsqueeze(0)  # [B,M,H_kv]
        return kv