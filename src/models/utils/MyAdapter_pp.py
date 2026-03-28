import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdapterPlusPlus(nn.Module):
    def __init__(self, dim, rank=32, film_hidden_dim=128):
        super().__init__()
        # LoRA 低秩注入: 下采样到 rank，再上采样回 dim
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        # GLU 门控 (参考 CROME): 双线性下采样与 Sigmoid 门控，再上采样
        self.glu_down1 = nn.Linear(dim, rank)
        self.glu_down2 = nn.Linear(dim, rank)
        self.glu_up = nn.Linear(rank, dim)
        # FiLM 特征调制: 从文本上下文生成缩放系数和偏移系数
        self.film_gamma = nn.Linear(dim, dim)
        self.film_beta = nn.Linear(dim, dim)
        # NeAT 风格非线性小网络: 两层全连接（含 ReLU）
        self.neat_fc1 = nn.Linear(dim, film_hidden_dim)
        self.neat_fc2 = nn.Linear(film_hidden_dim, dim)

    def forward(self, fused_feats, text_feats, image_feats=None):
        # 融合特征形状 [B, L1, D]
        x = fused_feats  # [B, L1, D]

        # LoRA 低秩注入: 学习微调增量
        # (等价于将冻结的线性权重加上低秩矩阵 A·B)
        lora_down = self.lora_down(x)            # [B, L1, rank]
        lora_up = self.lora_up(lora_down)        # [B, L1, D]
        x = x + lora_up                          # 残差连接注入 LoRA 更新

        # GLU 门控 (CROME): 对 x 进行双线性变换和 Sigmoid 门控
        glu_down1 = self.glu_down1(x)            # [B, L1, rank]
        glu_gate = torch.sigmoid(self.glu_down2(x))  # [B, L1, rank]
        glu_hidden = glu_down1 * glu_gate        # [B, L1, rank]（逐元素相乘门控）
        glu_out = self.glu_up(glu_hidden)        # [B, L1, D]
        x = x + glu_out                          # 残差连接加入门控输出

        # FiLM 特征调制: 使用文本全局上下文生成尺度和平移
        # 这里以文本特征均值作为全局提示 (prompt) 信息
        text_context = text_feats.mean(dim=1)    # [B, D]
        gamma = self.film_gamma(text_context)    # [B, D]
        beta = self.film_beta(text_context)      # [B, D]
        # 融合特征按通道进行仿射调制 (特征线性调制)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # [B, L1, D]

        # NeAT 非线性网络: 两层 MLP 提供非线性映射
        neat_hidden = F.relu(self.neat_fc1(x))   # [B, L1, film_hidden_dim]
        neat_out = self.neat_fc2(neat_hidden)    # [B, L1, D]
        x = x + neat_out                         # 残差连接融合非线性变化

        return x  # 输出形状 [B, L1, D]

if __name__ == '__main__':
    # 超参数定义
    batch_size = 64
    text_len = 40
    image_len = 197
    dim = 768

    # 模拟特征输入（可替换为 BLIP 或 ViLT 等模型的输出）
    text_feats = torch.randn(batch_size, text_len, dim)
    image_feats = torch.randn(batch_size, image_len, dim)
    fused_feats = torch.randn(batch_size, text_len, dim)  # 融合后的特征与文本长度对齐

    # 实例化适配器
    adapter = MyAdapterPlusPlus(dim=dim, rank=32, film_hidden_dim=128)

    # 前向传播
    output_feats = adapter(fused_feats, text_feats, image_feats)

    # 输出检查
    print("输出特征形状：", output_feats.shape)  # 应为 [batch_size, text_len, dim]