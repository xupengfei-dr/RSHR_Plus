import torch
import torch.nn as nn
import torch.nn.functional as F

class In2(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x:    [B, L, H]
        mask: [B, L]  (1=有效, 0=无效)；可为 bool/long/float
        return: [B, H]
        """
        if mask is None:

            x_perm = x.permute(0, 2, 1)
            pooled = self.pool(x_perm).squeeze(-1)
            return pooled

        # --- 掩码平均池化 ---
        mask = mask.to(dtype=x.dtype, device=x.device)  # [B, L]
        weights = mask.unsqueeze(-1)                    # [B, L, 1]
        num = (x * weights).sum(dim=1)                  # [B, H]
        den = weights.sum(dim=1).clamp_min(1.0)         # [B, 1] 避免除0
        pooled = num / den                               # [B, H]
        return pooled
