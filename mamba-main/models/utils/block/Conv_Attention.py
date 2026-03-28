import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Tuple
import torch
from torch import nn, autocast
from transformers import Cache, ROPE_INIT_FUNCTIONS
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import logging

logger = logging.get_logger(__name__)


class FlashAttConfig:
    def __init__(self, hidden_size=768):
        # self.num_heads = 16
        self.num_attention_heads = 16
        self.num_key_value_heads = 8
        self.hidden_size = hidden_size
        self.initializer_range = 0.02
        self._name_or_path = "/home"
        self.architectures = ["xpf"]
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.bos_token_id = 128000
        self.eos_token_id = 128009
        self.hidden_act = "gelu"
        # self.intermediate_size = 14336
        self.intermediate_size = 3072
        self.is_bitnet_config = True
        self.max_position_embeddings = 8192
        self.mlp_bias = False
        self.model_type = "llama"
        self.pretraining_tp = 1
        self.rms_norm_eps = 1e-05
        self.rope_interleaved = False
        self.rope_scaling = None
        self.rope_theta = 500000.0
        self.tie_word_embeddings = False
        self.torch_dtype = "bfloat16"
        self.use_cache = True
        self.vocab_size = 30522
        self.quantization_config = {
            "modules_to_not_convert": None,
            "quant_method": "bitnet"
        }


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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


class RSRotaryEmbedding(nn.Module):
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
                "`RSRotaryEmbedding` can now be fully parameterized by passing the model config through the "
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


class RSFlashAttention(nn.Module):
    """
    结合 RoPE 和 FlashAttention 2 ,GQA 的注意力模块。
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
        self.num_attention_heads = config.num_attention_heads
        # Ensure head_dim is in config or calculate it
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = is_causal  # 设置是否为因果注意力

        # --- 初始化线性投影层 ---
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # --- 初始化 RoPE 计算模块 ---
        # RoPE 是在 forward 中动态计算的，基于传入的 position_ids
        # 但为了遵循 HF 模式，可以在 __init__ 中创建实例
        # 注意: 如果 RSRotaryEmbedding 本身状态不依赖输入 x，可以在此创建
        self.rotary_emb = RSRotaryEmbedding(config=config)  # 使用你的配置
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
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
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
            # Need shape [bsz, q_len] for RSRotaryEmbedding forward maybe? Adjust if needed.
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


class Mu_Conv(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=(3, 3), padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=(5, 5), padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=(7, 7), padding=7 // 2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=(1, 1), )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x


class Conv_Attention(nn.Module):
    def __init__(self,
                 in_dim, mid_dim=192):
        super().__init__()

        # ------------------------卷积------------------------
        # self.dropout = nn.Dropout(p=0.0)
        self.project1 = nn.Linear(in_dim, mid_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        self.adapter_conv = Mu_Conv(192)
        self.project2 = nn.Linear(mid_dim, in_dim)
        # ----------------------注意力-------------------------
        self.attn_project_in = nn.Linear(in_dim, mid_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=mid_dim,
            num_heads=8,
            dropout=0.0,
            batch_first=True
        )
        self.attn_project_out = nn.Linear(mid_dim, in_dim)
        # -----------------RSFlashAttention--------------
        self.model_config_fla = FlashAttConfig(hidden_size=192)
        self.attn_flash_ = RSFlashAttention(config=self.model_config_fla, layer_idx=0, is_causal=False)
        # ------------------------------权重参数------------------------
        # self.alpha = nn.Parameter(torch.ones(in_dim))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))
        # ------------------------激活-----------------------------
        self.nonlinear = nn.GELU()

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        # ---------------卷积-------------------
        cls_token = x[:, 0:1, :]  # 64,1,768
        conv_mid_ = x[:, 1:, :]  # 64,256,768
        project1 = self.project1(conv_mid_)
        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)
        project2 = self.project2(nonlinear)
        conv_result_ = torch.cat([cls_token, project2], dim=1)
        # --------------注意力------------------
        atten_in_ = self.attn_project_in(x)
        # ----------------MHSA-------------------
        # atten_out, _ = self.attn(atten_in_, atten_in_, atten_in_)
        # --------------RSFlashAttention-------------
        position_ids_input = torch.arange(atten_in_.shape[1]).unsqueeze(0).to(device)  # Shape [1, seq_len]
        attention_mask_input = None
        atten_out, _, _ = self.attn_flash_(
            hidden_states=atten_in_,
            attention_mask=attention_mask_input,
            position_ids=position_ids_input,
            use_cache=False
        )
        # --------------------------
        atten_out = self.nonlinear(atten_out)
        atten_out = self.attn_project_out(atten_out)

        return identity + conv_result_ * self.alpha + atten_out * self.beta


# --- 测试代码 ---
if __name__ == "__main__":
    # --- 设置环境 ---
    torch.cuda.set_device(1)
    device = torch.device('cuda')
    print(f"Using device: {device}")
    # --- 实例化配置 ---
    model_config = FlashAttConfig()
    # 确定数据类型
    dtype = torch.bfloat16 if model_config.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")

    # --- 执行前向传播 ---
    print(f"\nRunning forward pass...")

    with torch.no_grad():  # 推理时不需要梯度
        with autocast(device_type='cuda', enabled=True, dtype=dtype):
            mona_model = Conv_Attention(768).to(device).to(dtype)
            feature = torch.randn(64, 257, 768).to(device).to(dtype)
            result = mona_model(feature, (16, 16))
            print(result.shape)
    print("\nScript finished.")

