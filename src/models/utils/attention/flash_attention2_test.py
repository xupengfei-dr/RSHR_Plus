import math
from typing import Optional, Tuple, TypedDict, Union, List

import torch
from torch import nn
from transformers import Cache, StaticCache, ROPE_INIT_FUNCTIONS, PretrainedConfig
from transformers.processing_utils import Unpack
from transformers.utils import logging
from torch.cuda.amp import autocast

from src.t.trans import DynamicCache
from src.t.trans.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


class FlashAttConfig:
    def __init__(self,hidden_size=768):
        self.num_heads = 16
        self.num_attention_heads = 16
        self.num_key_value_heads = 8
        self._name_or_path = "/home"
        self.architectures = ["xpf"]
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.bos_token_id = 128000
        self.eos_token_id = 128009
        self.hidden_act = "gelu"
        self.hidden_size = hidden_size
        self.initializer_range = 0.02
        # self.intermediate_size = 14336
        self.intermediate_size = 3072
        self.is_bitnet_config = True
        self.max_position_embeddings = 8192
        self.mlp_bias = False
        self.model_type = "llama"
        self.pretraining_tp = 1
        self.quantization_config = {
            "modules_to_not_convert": None,
            "quant_method": "bitnet"
        }
        self.rms_norm_eps = 1e-05
        self.rope_interleaved = False
        self.rope_scaling = None
        self.rope_theta = 500000.0
        self.tie_word_embeddings = False
        self.torch_dtype = "bfloat16"
        self.transformers_version = "4.44.0.dev0"
        self.use_cache = True
        self.vocab_size = 30522
        self.quantization_config = {
            "quant_method": "bitnet"
        }


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumlative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumlative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
            config: Optional[FlashAttConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MyAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: FlashAttConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MyFlashAttention2(MyAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )
        # position_embeddings = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)  # 32 40 768
        key_states = self.k_proj(hidden_states)  # 32 40 384
        value_states = self.v_proj(hidden_states)  # 32 40 384

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # 32 16 40 48
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # 32 8 40 48
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,
                                                                                                        2)  # 32 8 40 48

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        with autocast():
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
                **kwargs,
            )


        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# ================================ Test =========================================


class MyMLP(nn.Module):
    def __init__(self, config: FlashAttConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.act_fn = nn.GELU()  # 使用配置中的激活函数
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


class MyLayerNorm(nn.Module):
    # 简单实现，可以使用 torch.nn.LayerNorm 或 RMSNorm 等
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x):
        return self.norm(x)


class MyAttentionLayer(nn.Module):
    def __init__(self, config: FlashAttConfig, layer_idx: int):
        super().__init__()
        # --- 在这里实例化注意力模块，并传入 layer_idx ---
        self.self_attn = MyFlashAttention2(config, layer_idx=layer_idx)
        self.mlp = MyMLP(config)
        self.input_layernorm = MyLayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MyLayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 接收预计算的 RoPE
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,  # Will be ignored by FlashAttention anyway
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:  # Return hidden_states, None (attn_weights), cache

        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states_norm,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,  # 传递 RoPE
            past_key_value=past_key_value,
            output_attentions=output_attentions,  # Passed but likely ignored
            use_cache=use_cache,
            cache_position=cache_position,
        )
        attn_output = attn_outputs[0]
        present_key_value = attn_outputs[2]  # Get cache state

        hidden_states = residual + attn_output

        # Fully Connected
        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states_norm)

        # Return structure matches typical HF layer output when cache is used
        outputs = (hidden_states,)
        # No attentions weights to return with Flash
        outputs += (None,)  # Placeholder for attn_weights
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MyTransformerModel(nn.Module):
    def __init__(self, config: FlashAttConfig, num_layers: int = 12):
        super().__init__()
        self.config = config
        self.num_layers = num_layers

        # --- 词嵌入层 (示例，根据你的需要添加或删除) ---
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # --- RoPE 计算模块 (实例化一次) ---
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # --- 核心：实例化多层 AttentionLayer ---
        self.layers = nn.ModuleList([
            # --- 在这里传递正确的 layer_idx ---
            MyAttentionLayer(config, layer_idx=i)
            for i in range(num_layers)
        ])
        self.norm = MyLayerNorm(config.hidden_size, eps=config.rms_norm_eps)  # Final LayerNorm

    def forward(
            self,
            inputs_embeds: torch.FloatTensor,  # 直接接收 embeddings
            attention_mask: Optional[torch.Tensor] = None,  # 用于 padding 或特定模式
            position_ids: Optional[torch.LongTensor] = None,  # 用于计算 RoPE
            past_key_values: Optional[List[torch.Tensor]] = None,  # 用于 KV 缓存
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,  # 通常为 False
            output_hidden_states: Optional[bool] = None,  # 是否输出所有层隐藏状态
            cache_position: Optional[torch.LongTensor] = None,  # 主要用于静态缓存或精确定位
            return_dict: Optional[bool] = True,  # 推荐使用字典
    ) -> Union[Tuple, dict]:

        use_cache = use_cache if use_cache is not None else getattr(self.config, 'use_cache', False)
        output_attentions = False  # FlashAttention 通常不输出注意力权重
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.shape

        # --- KV Cache / Position Handling ---
        if use_cache:
            if past_key_values is None:
                # Initialize based on actual Cache type used in MyFlashAttention2
                # Example placeholder using DynamicCache logic
                past_key_values = DynamicCache(self.config, bsz, seq_len, hidden_states.device, hidden_states.dtype)
            # Determine cache_position if not provided (for dynamic cache)
            if cache_position is None:
                cache_position = torch.arange(past_key_values.get_seq_length(self.num_layers - 1),
                                              # Get current cache length
                                              past_key_values.get_seq_length(self.num_layers - 1) + seq_len,
                                              device=hidden_states.device)
        else:
            past_key_values = None
            cache_position = None  # Not needed if not using cache effectively

        # --- Position IDs for RoPE ---
        if position_ids is None:
            # If not using cache, simple range; if using cache, need positions for *current* tokens
            # This uses cache_position which should correspond to absolute positions
            position_ids = cache_position if cache_position is not None else torch.arange(seq_len,
                                                                                          device=hidden_states.device).unsqueeze(
                0)

        # --- 预计算 RoPE ---
        # Ensure RoPE module is on the correct device
        self.rotary_emb = self.rotary_emb.to(hidden_states.device)
        # Calculate RoPE based on the necessary positions (might be longer than seq_len if using cache)
        # Use cache_position if available and relevant for RoPE indices
        rope_position_ids = cache_position if use_cache else position_ids
        cos, sin = self.rotary_emb(hidden_states, rope_position_ids)
        position_embeddings = (cos, sin)

        # --- 存储中间结果 ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = ()  # FlashAttention doesn't return attentions
        next_decoder_cache = () if use_cache else None

        # --- 循环通过所有层 ---
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get cache for the current layer
            current_layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,  # 传递预计算的 RoPE
                past_key_value=current_layer_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                # layer_outputs should be (hidden_states, None, cache)
                next_decoder_cache += (layer_outputs[2],)  # Append updated cache state

        # --- 最终处理 ---
        hidden_states = self.norm(hidden_states)  # Apply final layer norm

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # --- 处理输出格式 ---
        if not return_dict:
            outputs = [hidden_states]
            if use_cache: outputs.append(next_decoder_cache)
            if output_hidden_states: outputs.append(all_hidden_states)
            # No attentions to add
            return tuple(v for v in outputs if v is not None)
        else:
            # 使用 transformers 的标准输出来模拟
            from transformers.modeling_outputs import BaseModelOutputWithPast
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=None,  # FlashAttention doesn't return attentions
            )


class RSFlashAttention(nn.Module):
    """
    结合 RoPE 和 FlashAttention 2 的注意力模块。

    接收单模态特征序列、可选掩码和位置ID (用于内部 RoPE 计算)。
    """

    def __init__(self, config: FlashAttConfig, layer_idx: Optional[int] = None,
                 is_causal: bool = False):  # 添加 is_causal 控制
        """
        初始化 RSFlashAttention 层。

        Args:
            config (FlashAttConfig): 包含注意力参数的配置对象。
            layer_idx (Optional[int]): 当前层的索引，主要用于 KV 缓存。
            is_causal (bool, optional): 是否应用因果掩码。默认为 False。
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and may "
                "cause errors if KV caching is used."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Ensure head_dim is in config or calculate it
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = is_causal  # 设置是否为因果注意力

        # --- 初始化线性投影层 ---
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # --- 初始化 RoPE 计算模块 ---
        # RoPE 是在 forward 中动态计算的，基于传入的 position_ids
        # 但为了遵循 HF 模式，可以在 __init__ 中创建实例
        # 注意: 如果 LlamaRotaryEmbedding 本身状态不依赖输入 x，可以在此创建
        self.rotary_emb = LlamaRotaryEmbedding(config=config)  # 使用你的配置
        # FlashAttention 内部标记 (通常由底层库处理，但可以保留)
        self._flash_attn_uses_top_left_mask = False  # Assuming flash_attn >= 2.1

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,  # e.g., for padding
            position_ids: Optional[torch.LongTensor] = None,  # Essential for RoPE calculation
            past_key_value: Optional[Cache] = None,  # For KV Caching
            output_attentions: Optional[bool] = False,  # Typically False for FlashAttention
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,  # For precise cache indexing
            **kwargs,  # Allows passing extra args like cu_seqlens if needed by _flash_attention_forward
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """
        前向传播函数。

        Args:
            hidden_states (torch.Tensor): 输入特征，形状 (batch_size, seq_len, hidden_size)。
            attention_mask (Optional[torch.Tensor]): 注意力掩码，例如处理 padding。
                                                  FlashAttention 对掩码有特定要求，可能需要处理。
            position_ids (Optional[torch.LongTensor]): 位置 ID，用于计算 RoPE。形状 (batch_size, seq_len)。
                                                    如果为 None，需要在此处创建。
            past_key_value (Optional[Cache]): 用于 KV 缓存的先前状态。
            output_attentions (Optional[bool]): 是否输出注意力权重（FlashAttention 通常不支持）。
            use_cache (Optional[bool]): 是否使用并返回 KV 缓存。
            cache_position (Optional[torch.LongTensor]): 缓存位置索引。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
                - 处理后的特征张量。
                - 注意力权重（通常为 None）。
                - 更新后的 KV 缓存（如果 use_cache=True）。
        """
        if output_attentions:
            logger.warning_once(f"{self.__class__.__name__} does not support output_attentions.")
            output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # --- QKV 投影 ---
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # --- 调整形状以适应多头和 GQA ---
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # --- KV 缓存处理 ---
        kv_seq_len = q_len  # Sequence length for K/V before cache update
        if past_key_value is not None:
            if self.layer_idx is None: raise ValueError("layer_idx must be provided for caching.")
            # Use cache_position if provided, otherwise calculate dynamically
            if cache_position is None:
                kv_seq_len = past_key_value.get_seq_length(self.layer_idx)  # Get current cache length
                cache_position = torch.arange(kv_seq_len, kv_seq_len + q_len, device=hidden_states.device)
            else:
                kv_seq_len = cache_position[-1].item() + 1  # Get length based on max position

            # Update cache (assuming cache expects non-RoPE'd K/V and handles RoPE internally, OR update *after* RoPE)
            # Let's follow Llama: update *after* RoPE
        else:
            kv_seq_len = q_len  # If no cache, KV length is query length

        # --- 计算并应用 RoPE ---
        if position_ids is None:
            # Create default position_ids if not provided
            position_ids = torch.arange(kv_seq_len, kv_seq_len + q_len, device=hidden_states.device).unsqueeze(0)
            # Need shape [bsz, q_len] for LlamaRotaryEmbedding forward maybe? Adjust if needed.
            # position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        # Ensure rotary_emb is on the correct device
        self.rotary_emb = self.rotary_emb.to(hidden_states.device)

        # Calculate RoPE embeddings based on the required positions
        # If using cache, position_ids should reflect the absolute positions of the current query tokens
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)  # Pass value_states for dtype/device info
        # Apply RoPE - cos/sin shape should match for broadcasting with Q/K [B, H, L, D]
        # The apply_rotary_pos_emb expects cos/sin that can broadcast, e.g., [B, 1, L, D]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin,
                                                        unsqueeze_dim=1)  # Use unsqueeze_dim=1

        # --- 更新 KV 缓存 (在 RoPE 之后) ---
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx,
                                                             {"cache_position": cache_position})
            # Update kv_seq_len *after* updating cache
            kv_seq_len = key_states.shape[-2]  # K shape is [B, H_kv, L_kv, D]

        # --- GQA: 重复 K/V 头 ---
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # --- 调整形状以适应 FlashAttention ---
        # Layout: [batch_size, sequence_length, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # --- 计算 Varlen 参数 (如果需要且可用) ---
        # 这是为了优化处理 padding 的情况，如果确定无 padding 或 Flash 库能处理掩码，可能不需要
        cu_seqlens_q = kwargs.get("cu_seqlens_q", None)
        cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
        max_length_q = kwargs.get("max_length_q", None)
        max_length_k = kwargs.get("max_length_k", None)
        if cu_seqlens_q is None:  # Example calculation for no padding
            cu_seqlens_q = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=query_states.device)
            max_length_q = q_len
        if cu_seqlens_k is None:  # Example calculation for no padding
            cu_seqlens_k = torch.arange(0, (bsz + 1) * kv_seq_len, step=kv_seq_len, dtype=torch.int32,
                                        device=key_states.device)
            max_length_k = kv_seq_len

        dropout_rate = self.attention_dropout if self.training else 0.0

        # --- 调用 FlashAttention 核心函数 ---
        # 确保你的 _flash_attention_forward 接口能接收这些参数
        attn_output = _flash_attention_forward(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attention_mask,  # Pass the original mask if needed by _flash_attention_forward
            q_len,  # Explicitly pass q_len? Maybe not needed if using max_length_q
            # Varlen parameters for the underlying kernel
            position_ids=position_ids,
            cu_seq_lens_q=cu_seqlens_q,
            cu_seq_lens_k=cu_seqlens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            dropout=dropout_rate,
            is_causal=self.is_causal and use_cache is False,
            # Causal mask usually only needed for training/full sequence
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            # Remove position_ids if not directly used by _flash_attention_forward's underlying call

        )

        # --- Reshape 输出和应用输出投影 ---
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        # --- 处理返回值 ---
        attn_weights = None  # FlashAttention 通常不返回权重
        present_key_value = past_key_value if use_cache else None  # 返回更新后的缓存（如果使用了）

        return attn_output, attn_weights, present_key_value


# --- 测试代码 (使用 MyTransformerModel) ---
if __name__ == "__main__":
    torch.cuda.set_device(1)  # 设置使用的 GPU 索引
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_config = FlashAttConfig()  # 使用你的配置
    num_layers = 12  # 设置层数

    # --- 实例化完整的 Transformer 模型 ---
    print(f"Instantiating MyTransformerModel with {num_layers} layers...")
    model = MyTransformerModel(config=model_config, num_layers=num_layers).to(device)
    print("Model instantiated.")
    # model.eval() # 如果模型包含 Dropout 等，测试时设置为评估模式

    # 准备输入数据 (假设输入是 embeddings)
    batch_size = 64  # 使用更小的 batch size 以防 OOM
    seq_length = 35  # 使用一个典型的序列长度
    print(f"Preparing input tensor with shape: ({batch_size}, {seq_length}, {model_config.hidden_size})")
    input_embeds = torch.rand(batch_size, seq_length, model_config.hidden_size).to(device)
    dtype = torch.bfloat16 if model_config.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    input_embeds = input_embeds.to(dtype)
    print(f"Input tensor dtype: {input_embeds.dtype}")

    # 创建 position_ids (模型内部会根据 seq_length 创建)
    position_ids = None  # 让模型内部创建

    # Attention Mask (示例：无 padding)
    attention_mask = None

    from torch.cuda.amp import autocast

    print(f"Running inference with {num_layers} layers using FlashAttention2 + RoPE...")
    try:
        with autocast(enabled=True, dtype=dtype):  # 使用 autocast

            outputs = model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,  # 让模型生成 RoPE
                use_cache=False,  # 不使用 KV 缓存
                return_dict=True  # 获取字典输出
            )
        print("Inference finished successfully.")

        # --- 检查输出 ---
        print(f"Output keys: {outputs.keys()}")
        if "last_hidden_state" in outputs:
            print(f"Output 'last_hidden_state' shape: {outputs.last_hidden_state.shape}")
            # 检查输出形状是否正确 (batch_size, seq_length, hidden_size)
            assert outputs.last_hidden_state.shape == (batch_size, seq_length, model_config.hidden_size)
            print("Output shape check passed.")
        else:
            print("Output dictionary did not contain 'last_hidden_state'")

    except Exception as e:
        print("\n !!! An error occurred during inference !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback

        traceback.print_exc()  # 打印详细的错误追踪信息

    print("\nScript finished.")

# if __name__ == "__main__":
#     torch.cuda.set_device(1)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Renamed devie to device
#     model_config = FlashAttConfig()
#
#     # --- 处理警告 1：传递 layer_idx ---
#     # 假设这是第 0 层，您可以根据实际情况调整
#     layer_idx_to_pass = 0
#     # 在实例化时传递 layer_idx
#     model = MyFlashAttention2(config=model_config, layer_idx=layer_idx_to_pass).to(device)
#
#     # 准备输入数据
#     batch_size = 32
#     seq_length = 40
#     # tensor = torch.rand(64, 40, 768).to(device) # 转换为 float16
#     tensor = torch.rand(batch_size, seq_length, model_config.hidden_size).to(device) # 使用 config 中的 hidden_size
#     tensor = tensor.type(torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16) # 使用 config 中的 dtype
#
#     # 创建 position_ids
#     position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
#
#     # --- 处理警告 2：预计算 RoPE position_embeddings ---
#     # 1. 实例化 RoPE 模块 (与模型内部使用的相同逻辑)
#     #    注意: 这里我们直接用 model_config 来初始化，确保参数一致
#     rotary_emb_calculator = LlamaRotaryEmbedding(config=model_config, device=device)
#
#     # 2. 计算 cos 和 sin 缓存
#     #    forward 方法需要一个 dummy tensor x 来获取 dtype 和 device，
#     #    以及实际的 position_ids
#     #    这里的 tensor 可以是 value_states，或者直接用输入的 tensor 也可以，因为它只关心 dtype 和 device
#     #    我们直接用输入 tensor
#     cos_cached, sin_cached = rotary_emb_calculator(tensor, position_ids=position_ids)
#     #    将计算结果打包成元组，这是 forward 方法期望的格式
#     position_embeddings_to_pass = (cos_cached, sin_cached)
#
#     from torch.cuda.amp import autocast
#     # 在 autocast 上下文管理器中进行推理
#     print("Running inference with precomputed position embeddings...")
#     with autocast(dtype=torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16): # 显式指定 autocast 类型
#         # --- 在调用 forward 时传递 position_embeddings，不再传递 position_ids ---
#         result = model(
#             hidden_states=tensor,
#             position_ids=None, # 不再需要传递 position_ids 给 attention 层
#             position_embeddings=position_embeddings_to_pass # 传递预计算的 cos/sin
#             # 如果您的模型 forward 还需要 position_ids 用于其他目的（例如KV缓存的索引），则仍需传递
#             # 但对于 RoPE 计算本身，应该优先使用 position_embeddings
#             # 检查 MyAttention/MyFlashAttention2 forward 签名，确认 position_ids 是否仅用于 RoPE
#             # 根据当前的 Llama 实现，position_ids 主要就是给 RoPE 用的，所以设为 None 应该是对的
#         )
#     print("Inference finished.")
#     print(result[0].shape)
#
#     # --- (可选) 验证不传递 layer_idx 和使用 position_ids 的旧方式是否仍能运行 (但会产生警告) ---
#     # print("\nRunning inference with internal RoPE calculation (will produce warnings)...")
#     # model_old = MyFlashAttention2(model_config).to(device) # 不传递 layer_idx
#     # with autocast(dtype=torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16):
#     #     result_old = model_old(tensor, position_ids=position_ids) # 只传递 position_ids
#     # print("Old way finished.")
#     # print(result_old[0].shape)
#
# #
# if __name__ == "__main__":
#     torch.cuda.set_device(1)
#     devie = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model_config = FlashAttConfig()
#     model = MyFlashAttention2(model_config).to(devie)
#     # tensor = torch.rand(64, 40, 768).to(devie)# 转换为 float16
#     tensor = torch.rand(32, 40, 768).to(devie)# 转换为 float16
#     tensor = tensor.type(torch.float16)
#     position_ids = torch.arange(40,device=devie).unsqueeze(0).expand(32, -1)
#     from torch.cuda.amp import autocast
#     # 在 autocast 上下文管理器中进行推理
#     with autocast():
#         result = model(tensor, position_ids=position_ids)
#     print(result[0].shape)
