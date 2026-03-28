import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

import torch
from torch import nn, autocast
import torch.nn.functional as F

from mamba_ssm.modules.mamba2 import Mamba2
from models.utils.attention.RSSRAttention import FlashAttConfig, RSFlashAttention

flash_model_config_RSEFlash = FlashAttConfig(hidden_size=384)
flash_model_config_RSE = FlashAttConfig(hidden_size=1024)


class RSEFlashBlock(nn.Module):


    def __init__(self, hidden_dim, num_heads, layer_idx_test):

        super(RSEFlashBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.l1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.l2 = nn.Linear(hidden_dim // 2, hidden_dim)

        # Add multi-head attention
        # self.multihead_attention1 = nn.MultiheadAttention(hidden_dim // 2, num_heads)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)

        self.rsflash = RSFlashAttention(flash_model_config_RSEFlash, layer_idx=layer_idx_test, is_causal=False)
        self.init_weights()

    def init_weights(self):
        self.l2.weight.data.zero_()
        self.l2.bias.data.zero_()

    def forward(self, x, attention_mask=None, position_ids=None):
        xinit = x
        x = self.l1(x)
        x2 = x
        with autocast(device_type='cuda', enabled=True, dtype=torch.float16):
            attn_output, _, _ = self.rsflash(
                hidden_states=x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
        alpha = torch.sigmoid(self.gate1)
        attn = alpha * attn_output + (1 - alpha) * x2
        attn_activated  = F.gelu(attn)

        x = self.l2(attn_activated)
        return x + xinit


class RSE(nn.Module):


    # def LT(self, x, weight, bias):
    #     return x * weight + bias
    #
    # def init_LT(self, dim):
    #     weight = nn.Parameter(torch.ones(dim))
    #     bias = nn.Parameter(torch.zeros(dim))
    #
    #     nn.init.normal_(weight, mean=1, std=.02)
    #     nn.init.normal_(bias, std=.02)
    #
    #     return weight, bias

    def __init__(self, hidden_size, layer_id=0):
        super(RSE, self).__init__()
        self.hidden_dim = 1024

        self.img_proj_down = nn.Linear(hidden_size, self.hidden_dim)
        self.img_proj_up = nn.Linear(self.hidden_dim, hidden_size)
        # self.rsflash_M = RSFlashAttention(flash_model_config_RSE, layer_idx=layer_id, is_causal=False)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.init_weights()
        # self.flash_RSEF = nn.ModuleList([
        #     RSEFlashBlock(hidden_size, 8, i)
        #     for i in range(6)
        # ])

        self.mamba2 = Mamba2(
            d_model=768,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=1,
            dtype=torch.float32
        )

    def init_weights(self):
        self.img_proj_up.weight.data.zero_()
        self.img_proj_up.bias.data.zero_()

    def forward(self, feature, attention_mask=None, position_ids=None):
        x_init = feature
        # x = self.img_proj_down(feature)
        # x = F.gelu(x)
        # mid_feature = x_init
        mid_feature_mamba = x_init
        # current_features = mid_feature
        with autocast(device_type='cuda', enabled=True, dtype=torch.float16):
            mamba2_output_ = self.mamba2(mid_feature_mamba)


            # for i, layers in enumerate(self.flash_RSEF):
            #     caiyang_out_mid = layers(current_features)
            #     # caiyang_out_mid = caiyang_out_mid.to(x.dtype)
            #     # if i < len(self.caiyang_down_wt_norm_acts):
            #     #     caiyang_out_mid = self.caiyang_down_wt_norm_acts[i](caiyang_out_mid)
            #     current_features = caiyang_out_mid
            # mid_feature = current_features

        # mamba2_output_ = self.img_proj_up(mamba2_output_)
        alpha = torch.sigmoid(self.gate1)
        # x = alpha * mid_feature + (1 - alpha) * mamba2_output_ +x_init
        x = alpha * mamba2_output_
        return x


if __name__ == '__main__':
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    seqlen = 40
    d_model = 768
    dummy_data = torch.randn(batch_size, seqlen, d_model, dtype=torch.float16).to(device)
    RSE_text = RSE(768, 12).to(device, dtype=torch.float16)
    out_rse = RSE_text(dummy_data)
    print(out_rse.shape)
