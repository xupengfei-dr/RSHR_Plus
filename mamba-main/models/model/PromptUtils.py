from torch import nn

import torch

class FiLM(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(condition_dim, feature_dim)  # 学习缩放参数 (gamma)
        self.beta_proj = nn.Linear(condition_dim, feature_dim)   # 学习平移参数 (beta)

    def forward(self, features, condition):
        """
        Args:
            features:  要调制的特征, shape: [B, ..., feature_dim]
                       (可以是任意形状, 只要最后一个维度是 feature_dim)
            condition: 条件向量 (例如，答案 embedding), shape: [B, condition_dim]
        Returns:
            modulated_features: 调制后的特征, shape 与 features 相同
        """
        gamma = self.gamma_proj(condition)  # [B, feature_dim]
        beta = self.beta_proj(condition)     # [B, feature_dim]

        # 扩展 gamma 和 beta 的维度，以匹配 features 的形状 (广播机制)
        shape = [-1] + [1] * (features.dim() - 2) + [gamma.size(-1)] #例如 [B, 1, D]
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)

        # 应用 FiLM 调制:  features * gamma + beta
        return features * gamma + beta
class AnswerConditionalClassifierWithFiLM(nn.Module):
    def __init__(self, hidden_dim, num_candidates, answer_embedding_dim=64, answer_dropout=0.5):
        super().__init__()
        self.answer_embedding = nn.Embedding(num_candidates, answer_embedding_dim)
        self.answer_dropout = nn.Dropout(answer_dropout)  # 可选的 Dropout
        self.film = FiLM(feature_dim=hidden_dim, condition_dim=answer_embedding_dim)  # FiLM 层
        self.classify_layer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5), # 可以放在激活函数之后
            nn.Linear(512, num_candidates)
        )

    def forward(self, features, answer_indices):
        answer_embed = self.answer_embedding(answer_indices)
        answer_embed = self.answer_dropout(answer_embed)  # 可选

        # 使用 FiLM 调制 features
        modulated_features = self.film(features, answer_embed)
        logits = self.classify_layer(modulated_features)

        return logits

# class WeakAnswerConditionalClassifier(nn.Module):
#     def __init__(self, hidden_dim, num_candidates, answer_embed_dim=32):
#         super().__init__()
#         # 弱提示组件
#         self.answer_embed = nn.Embedding(num_candidates, answer_embed_dim)
#         self.attention_gate = nn.Sequential(
#             nn.Linear(hidden_dim + answer_embed_dim, 768),  # 轻量级门控
#             nn.Sigmoid()  # 输出0-1的注意力权重
#         )
#
#         # 简化分类层
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_candidates)
#         )
#
#     def forward(self, features, answer_indices=None):
#         """
#         若answer_indices为None（测试模式），使用默认门控值1.0
#         """
#         if answer_indices is not None and self.training:
#             # 训练时：使用真实答案生成门控
#             ans_emb = self.answer_embed(answer_indices)  # [B, E]
#             gate_input = torch.cat([features, ans_emb], dim=1)
#             gate = self.attention_gate(gate_input)  # [B, 1]
#         else:
#             # 测试时：无答案信息，门控全开
#             gate = torch.ones(features.size(0), 1).to(features.device)
#
#         # 弱提示：通过门控调整特征强度
#         adjusted_features = features * gate
#         return self.classifier(adjusted_features)

class AnswerConditionalClassifier(nn.Module):
    def __init__(self, hidden_dim,num_candidates, answer_embedding_dim):
        super().__init__()
        self.answer_embedding = nn.Embedding(num_candidates, answer_embedding_dim)
        self.classify_layer = nn.Sequential(
            nn.Linear(hidden_dim+answer_embedding_dim , 1024),  # 调整为更合适的隐藏层大小
            nn.BatchNorm1d(1024),  # 添加 BatchNorm
            torch.nn.GELU(),  # 使用更高效的激活函数
            nn.Dropout(0.5),  # 增大正则化强度，减少过拟合
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # 再次使用 BatchNorm
            torch.nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, num_candidates)  # 输出层
        )

    def forward(self, features, answer_indices):  # 注意：现在 answer_indices 在测试时也需要
        answer_embed = self.answer_embedding(answer_indices)
        combined_features = torch.cat([features, answer_embed], dim=1)
        logits = self.classify_layer(combined_features)
        return logits

class AnswerGate(nn.Module):
    def __init__(self, num_classes, hidden_dim=768):
        super().__init__()
        # 答案嵌入层 + 门控生成器
        self.answer_embed = nn.Embedding(num_classes, hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 融合特征与答案
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        x: 融合后的特征 [B, D]
        y: 真实答案标签 [B]
        """
        # 获取答案嵌入 [B, D]
        ans_emb = self.answer_embed(y)

        # 拼接特征与答案嵌入 → [B, 2D]
        gate_input = torch.cat([x, ans_emb], dim=-1)

        # 生成门控向量 → [B, D]
        gate = self.gate_net(gate_input)

        # 特征加权：增强与答案相关的维度
        return x * gate




if __name__ == '__main__':
    ansG = AnswerGate(9)


