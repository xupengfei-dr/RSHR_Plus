import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制的完整实现
    """
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv_layer = nn.Linear(embed_dim, 3 * embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_len, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        query, key, value = qkv.chunk(3, dim=-1)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.reshape(batch_size, seq_len, self.embed_dim)

        output = self.fc_out(attention)
        return output


# --- 使用示例 ---
if __name__ == '__main__':
    # 定义超参数
    embed_dim = 512  # 模型的总维度 (d_model)
    n_heads = 8  # 注意力头的数量
    seq_length = 100  # 序列长度
    batch_size = 32  # 批量大小

    # 创建多头自注意力层实例
    multi_head_attention = MultiHeadAttention(embed_dim, n_heads)

    # 创建虚拟输入数据
    # x 的形状: [32, 100, 512]
    x = torch.randn(batch_size, seq_length, embed_dim)

    # 前向传播
    output = multi_head_attention(x)

    print("--- 多头注意力模块 ---")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 注意输出维度和输入维度相同

    # 验证输出维度是否正确
    assert output.shape == x.shape
    print("\n代码运行成功，输出维度正确！")