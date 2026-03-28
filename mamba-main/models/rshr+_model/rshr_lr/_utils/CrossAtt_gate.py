import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadGatedCrossAttention(nn.Module):
    """
    Head-Gated Cross-Attention (HGCA)
    - Q: visual tokens (e.g., CLS or all patches)
    - K,V: text tokens
    创新点：
      1) Head-wise gate：每个注意力头有一个“文本驱动”的门控系数 g_h ∈ (0,1)，用点积注意力从文本里提取，无MLP。
      2) Top-k token routing：每个头只看自己认为最相关的 k 个文本token，降噪更稳。
      3) Skip-init 残差门：输出= x_q + α*(跨注意力输出)，α可学习，初始极小，避免破坏原表征。
    """
    def __init__(self, dim=768, num_heads=8, attn_drop=0.1, proj_drop=0.1,
                 rope=None, topk=0, gate_init=-4.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.h = num_heads
        self.dh = dim // num_heads
        self.topk = topk  # 每个头选取的文本token数；0表示不过滤

        # Q/K/V/O 线性投影（注意：不是MLP，只是标准注意力投影）
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 每个头一个可学习“查询原型”（不经MLP），用来从文本中抽取gate信号
        self.head_queries = nn.Parameter(torch.randn(self.h, self.dh) / (self.dh ** 0.5))

        # Skip-init 残差门（标量），初值很小 -> 稳定涨点
        self.resid_logit = nn.Parameter(torch.tensor(gate_init))  # α = sigmoid(logit) ≈ ~0

        # 可选RoPE（若你已有实现，可注入一个函数：apply_rope(q/k)）
        self.rope = rope

    def _reshape_heads(self, x):
        # [B, L, D] -> [B, H, L, Dh]
        B, L, D = x.shape
        x = x.view(B, L, self.h, self.dh).transpose(1, 2)
        return x

    def _merge_heads(self, x):
        # [B, H, L, Dh] -> [B, L, D]
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, x_q, x_kv, kv_attn_mask=None):
        """
        x_q: [B, Lq, D]   视觉tokens（CLS或patch）
        x_kv:[B, Lt, D]   文本tokens
        kv_attn_mask: [B, Lt]  1=有效，0=pad
        """
        B, Lq, _ = x_q.shape
        _, Lt, _ = x_kv.shape

        q = self.q_proj(self.ln_q(x_q))
        k = self.k_proj(self.ln_kv(x_kv))
        v = self.v_proj(self.ln_kv(x_kv))

        q = self._reshape_heads(q)   # [B,H,Lq,Dh]
        k = self._reshape_heads(k)   # [B,H,Lt,Dh]
        v = self._reshape_heads(v)   # [B,H,Lt,Dh]

        # 可选RoPE
        if self.rope is not None:
            q = self.rope(q)  # 需自带RoPE实现：保持shape不变
            k = self.rope(k)

        # ====== 1) 每头Gate：用 head_queries 与文本做点积注意力，得到 g_h ∈ (0,1) ======
        # head_queries: [H, Dh] -> [B,H,1,Dh] 便于批处理
        hq = self.head_queries.unsqueeze(0).unsqueeze(2).expand(B, self.h, 1, self.dh)  # [B,H,1,Dh]
        # 对文本tokens做注意力：score_h,t = <hq_h, k_h,t> / sqrt(Dh)
        gate_scores = (hq * k).sum(-1) / (self.dh ** 0.5)  # [B,H,Lt]
        if kv_attn_mask is not None:
            mask = (kv_attn_mask == 0).unsqueeze(1)  # [B,1,Lt]
            gate_scores = gate_scores.masked_fill(mask, float('-inf'))

        gate_attn = gate_scores.softmax(dim=-1)  # [B,H,Lt]
        # 门控上下文（每头一个向量）：c_h = Σ_t attn_h,t * k_h,t
        c_h = torch.einsum('bhl,bhld->bh d', gate_attn, k)  # [B,H,Dh]
        # 将 c_h 与 head_queries 做相似度，生成标量门控 g_h
        g_logits = (c_h * self.head_queries.unsqueeze(0)).sum(-1) / (self.dh ** 0.5)  # [B,H]
        g_h = torch.sigmoid(g_logits).unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]

        # ====== 2) Top-k 文本路由（可选）：每个头仅保留最相关的 k 个文本token ======
        if self.topk and self.topk < Lt:
            # 用 gate_scores 作为相关度度量
            topk_val, topk_idx = torch.topk(gate_scores, k=self.topk, dim=-1)  # [B,H,K]
            # 构建mask
            keep_mask = torch.zeros_like(gate_scores, dtype=torch.bool)  # [B,H,Lt]
            keep_mask.scatter_(dim=-1, index=topk_idx, value=True)
            # 扩展到 [B,H,Lt,1] 以筛K/V
            keep_mask_ = keep_mask.unsqueeze(-1)
            k = torch.where(keep_mask_, k, torch.zeros_like(k))
            v = torch.where(keep_mask_, v, torch.zeros_like(v))

        # ====== 3) 标准缩放点积注意力（按头） ======
        attn_logits = torch.einsum('bhqd,bhkd->bhqk', q, k) / (self.dh ** 0.5)  # [B,H,Lq,Lt]
        if kv_attn_mask is not None:
            mask = (kv_attn_mask == 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,Lt]
            attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)  # [B,H,Lq,Dh]

        # ====== 4) Head-wise Gate 调制 + 残差门（无MLP） ======
        # 每头缩放
        out = out * g_h  # [B,H,Lq,Dh]

        out = self._merge_heads(out)           # [B,Lq,D]
        out = self.o_proj(out)                 # 线性投影回D
        out = self.proj_drop(out)

        # Skip-init 残差门：α很小，训练中逐步“开门”
        alpha = torch.sigmoid(self.resid_logit)
        y = x_q + alpha * out
        return y
