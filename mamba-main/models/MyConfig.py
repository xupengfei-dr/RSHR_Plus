import torch
import torch.nn as nn

from models.t.trans import Blip2QFormerConfig


class QFormerModelConfig:
    def __init__(
            self,
            num_query_tokens_val: int = 32,
            # Blip2QFormerConfig 参数
            qformer_vocab_size: int = 30522,
            qformer_hidden_size: int = 768,  # <--- 修改这里 (例如，从 512 改为 768)
            qformer_num_hidden_layers: int = 6,
            qformer_num_attention_heads: int = 12,  # <--- 保持或根据 hidden_size 调整
            qformer_intermediate_size: int = 3072,  # 通常是 hidden_size * 4, (768 * 4 = 3072)
            qformer_hidden_act: str = "gelu",
            qformer_hidden_dropout_prob: float = 0.1,
            qformer_attention_probs_dropout_prob: float = 0.1,
            qformer_max_position_embeddings: int = 512,
            qformer_initializer_range: float = 0.02,
            qformer_layer_norm_eps: float = 1e-12,
            qformer_cross_attention_frequency: int = 2,
            qformer_encoder_hidden_size: int = 1024,
            qformer_add_cross_attention: bool = True,
    ):
        self.num_query_tokens = num_query_tokens_val

        # 确保 hidden_size 和 num_attention_heads 兼容
        if qformer_hidden_size % qformer_num_attention_heads != 0:
            raise ValueError(
                f"The Q-Former hidden size ({qformer_hidden_size}) is not a multiple of the number of attention "
                f"heads ({qformer_num_attention_heads})"
            )

        # 如果 intermediate_size 依赖于 hidden_size, 最好也动态计算或检查
        expected_intermediate_size = qformer_hidden_size * 4
        if qformer_intermediate_size != expected_intermediate_size:
            print(f"Warning: qformer_intermediate_size ({qformer_intermediate_size}) "
                  f"is not 4 * qformer_hidden_size ({expected_intermediate_size}). This might be intentional.")

        self.qformer_config = Blip2QFormerConfig(
            vocab_size=qformer_vocab_size,
            hidden_size=qformer_hidden_size,
            num_hidden_layers=qformer_num_hidden_layers,
            num_attention_heads=qformer_num_attention_heads,
            intermediate_size=qformer_intermediate_size,
            hidden_act=qformer_hidden_act,
            hidden_dropout_prob=qformer_hidden_dropout_prob,
            attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            max_position_embeddings=qformer_max_position_embeddings,
            initializer_range=qformer_initializer_range,
            layer_norm_eps=qformer_layer_norm_eps,
            cross_attention_frequency=qformer_cross_attention_frequency,
            encoder_hidden_size=qformer_encoder_hidden_size,
            add_cross_attention=qformer_add_cross_attention,
            # Blip2QFormerConfig 继承自 PretrainedConfig, 以下参数有默认值
            # output_attentions=False,
            # output_hidden_states=False,
            # use_return_dict=True,
        )
