from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache, Mamba2Mixer

# torch.cuda.set_device(3)

from models.t.trans.models.mamba2.modeling_mamba2 import Mamba2RMSNorm


class Mamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config, layer_idx=layer_idx)

    def forward(
            self,
            hidden_states,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        return hidden_states


class RSE_Mamba2(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
            self,
            hidden_states,
            cache_params: Optional[Mamba2Cache] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        all_hidden_states = () if output_hidden_states else None

        for mixer_block in self.layers:
            # if self.gradient_checkpointing and self.training:
            if False:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        return hidden_states


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def Config_rse_mamba2(config_all):
    config_ALL_ = {
        "architectures": [
            "RSE_"
        ],
        "bos_token_id": 0,
        "conv_kernel": 4,
        "eos_token_id": 0,
        "expand": 2,
        "fused_add_norm": True,
        "hidden_act": "silu",
        "hidden_size": 768,
        "initializer_range": 0.1,
        "intermediate_size": 2560,
        "layer_norm_epsilon": 1e-05,
        "model_type": "mamba",
        "pad_token_id": 0,
        "pad_vocab_size_multiple": 8,
        "rescale_prenorm_residual": False,
        "residual_in_fp32": True,
        "rms_norm": True,
        "state_size": 16,
        "time_step_floor": 0.0001,
        "time_step_init_scheme": "random",
        "time_step_max": 0.1,
        "time_step_min": 0.001,
        "time_step_rank": 160,
        "time_step_scale": 1.0,
        "torch_dtype": "float32",
        "use_bias": False,
        "use_cache": True,
        "use_conv_bias": True,
        "vocab_size": 50280,
        # "n_layer": 0,
        "num_hidden_layers": 12,
        "num_heads": 24,
        "n_groups": 1,
        "head_dim": 64,
        "chunk_size": 256,
        # "time_step_limit": 0.0
        "time_step_limit": (0.0, float("inf"))

    }
    config_ns = dict_to_namespace(config_ALL_)
    return config_ns


if __name__ == '__main__':
    # config_ = {
    #     "d_model": 768,
    #     "hidden_size": 768,
    #     "d_intermediate": 0,
    #     "num_hidden_layers": 24,
    #     "vocab_size": 50277,
    #     "ssm_cfg": {
    #         "layer": "Mamba2"
    #     },
    #     "attn_layer_idx": [],
    #     "attn_cfg": {},
    #     "rms_norm": True,
    #     "residual_in_fp32": True,
    #     "fused_add_norm": True,
    #     "pad_vocab_size_multiple": 16,
    #     "tie_embeddings": True
    # }
    config_ = {
        "architectures": [
            "RSE_"
        ],
        "bos_token_id": 0,
        "conv_kernel": 4,
        "eos_token_id": 0,
        "expand": 2,
        "fused_add_norm": True,
        "hidden_act": "silu",
        "hidden_size": 768,
        "initializer_range": 0.1,
        "intermediate_size": 2560,
        "layer_norm_epsilon": 1e-05,
        "model_type": "mamba",
        "pad_token_id": 0,
        "pad_vocab_size_multiple": 8,
        "rescale_prenorm_residual": False,
        "residual_in_fp32": True,
        "rms_norm": True,
        "state_size": 16,
        "time_step_floor": 0.0001,
        "time_step_init_scheme": "random",
        "time_step_max": 0.1,
        "time_step_min": 0.001,
        "time_step_rank": 160,
        "time_step_scale": 1.0,
        "torch_dtype": "float32",
        "use_bias": False,
        "use_cache": True,
        "use_conv_bias": True,
        "vocab_size": 50280,
        # "n_layer": 0,
        "num_hidden_layers": 12,
        "num_heads": 24,
        "n_groups": 1,
        "head_dim": 64,
        "chunk_size": 256,
        # "time_step_limit": 0.0
        "time_step_limit": (0.0, float("inf"))

    }

    config_ns = dict_to_namespace(config_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rse = RSE_Mamba2(config_ns).to(device, dtype=torch.float16)
    print(rse)

    dummy_input = torch.randn(64, 40, 768, dtype=torch.float16).to(device)

    # 前向传播测试
    with torch.no_grad():
        output = rse(dummy_input)
        print("输出形状:", output.shape)
