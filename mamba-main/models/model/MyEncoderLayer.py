from typing import Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers import BlipConfig
import sys

import models.flash_attention2_test

sys.path.append("../..")  # 将项目的根目录添加到 sys.path
from transformers.activations import ACT2FN
from models.model.MyAdapter import MyAdapter


# from src.mamba_ssm.modules.mamba2 import Mamba2
# from src.mamba_ssm.modules.mamba2_other import Mamba2 as Mamba2o
# from src.mamba_ssm.modules.mamba2_other import Mamba2Config
class MyEncoderLayer(nn.Module):
    def __init__(self, config: BlipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # self.self_attn = Blip2Attention()
        self.self_attn = BlipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = BlipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 以下是新增
        self.inter = MyFormerIntermediate()
        self.output = MyOutput()
        self.attn_adapter = MyAdapter(config.hidden_size, D_dim=1024, skip_connect=False)
        self.mlp_adapter = MyAdapter(config.hidden_size, D_dim=1024, skip_connect=False)
        self.attn_adapter_scale = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.mlp_adapter_scale = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.flash_aten = models.flash_attention2_test.MyFlashAttention2()
        # mamba
        # self.mamba = Mamba2(d_model=768)
        # self.mamba2 = Mamba2o(Mamba2Config(768,1,64))
        # self.mambaconfig =Mamba2Config()
        # self.mamba = Mamba2Model(self.mambaconfig)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        # residual_end = hidden_states
        # mamba_out = self.mamba2(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        #####################################################

        #####################################################

        # self-attention
        # hidden_states, attn_weights = self.self_attn(
        #     hidden_states=hidden_states,
        #     head_mask=attention_mask,
        #     output_attentions=output_attentions,
        # )


        #todo:改为动态获取

        bcz = hidden_states.size(0)
        dim = hidden_states.size(1)
        # position_ids = torch.arange(40).unsqueeze(0).expand(20, -1)
        position_ids = torch.arange(dim).unsqueeze(0).expand(bcz, -1)
        position_ids = position_ids.to(hidden_states.device)
        dtype_1 = hidden_states.dtype
        with autocast():
            hidden_states, attn_weights, _ = self.flash_aten(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        # hidden_states = attention_output
        #################
        # hidden_states = hidden_states+mamba_out[0]
        #################

        # outputs = attn_weights  todo:   mamba+ zhe 8905
        hidden_states = hidden_states.to(dtype_1)
        end_residual = hidden_states

        # 在这加为 0= 85307  1 不变

        # -------第一个残差链接--------
        # todo:
        attn_adapt = self.attn_adapter(hidden_states)
        hidden_states = hidden_states + residual + attn_adapt * self.attn_adapter_scale
        residual = hidden_states
        residual2 = hidden_states

        hidden_states = self.layer_norm2(hidden_states)
        hidden_states_val = hidden_states

        # mamba
        # mamba_out = self.mamba2(hidden_states)
        # hidden_states = hidden_states+mamba_out
        # todo：  使用adapter    hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # todo: 用这个+那个
        # res2 = self.mlp_adapter(hidden_states_val)
        # hidden_states = hidden_states + res2 * self.mlp_adapter_scale
        # todo:新增
        # mlp_adapter = self.mlp_adapter(hidden_states_val)
        layer_output = self.inter(hidden_states_val)
        # todo : 应该是residual
        layer_output1 = self.output(layer_output, residual2)
        # layer_output = layer_output + hidden_states
        # hidden_states = hidden_states +layer_output

        # todo：加后面
        # hidden_states = hidden_states * self.mlp_adapter_scale + layer_output1 + residual
        hidden_states = hidden_states * self.mlp_adapter_scale + layer_output1 + residual
        # hidden_states = hidden_states + residual+layer_output1
        outputs = (hidden_states,)
        # if output_attentions:
        #     outputs += (attn_weights,)
        outputs += (attn_weights,)

        return outputs


class BlipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.attention_dropout)

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = (
            self.qkv(hidden_states)
                .reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


# todo :加个残差
class BlipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MyFormerIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 3072)
        if isinstance("gelu", str):
            self.intermediate_act_fn = ACT2FN["gelu"]
        else:
            self.intermt_ediate_acfn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MyOutput(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.dense = nn.Linear(3072, 768)
        # self.dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.0)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states
# todo：多加个output2
# class MyOutput2(nn.Module):
#     def __init__(self, input_dim: int = 3072, output_dim: int = 768, dropout_prob: float = 0.0) -> None:
#         super().__init__()
#         self.dense = nn.Linear(input_dim, output_dim)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.layer_norm = nn.LayerNorm(output_dim, eps=1e-6)  # 加入 LayerNorm
#         self.activation = nn.GELU()  # 激活函数
#         self.residual_scale = nn.Parameter(torch.ones(1))  # Learnable Residual Scaling
#
#     def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
#         # 前馈计算
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.activation(hidden_states)  # 非线性激活
#         hidden_states = self.dropout(hidden_states)
#
#         # 加入残差连接和层归一化
#         hidden_states = hidden_states * self.residual_scale + input_tensor
#         hidden_states = self.layer_norm(hidden_states)
#         return hidden_states


# class BlipMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.activation_fn = ACT2FN[config.hidden_act]
#         self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         # Xavier initialization for layers
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.zeros_(self.fc2.bias)
#
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         residual = hidden_states
#         hidden_states = self.fc1(hidden_states)
#         hidden_states = self.activation_fn(hidden_states)
#         hidden_states = self.fc2(hidden_states)
#         hidden_states = hidden_states + residual  # Adding residual connection
#         return hidden_states
