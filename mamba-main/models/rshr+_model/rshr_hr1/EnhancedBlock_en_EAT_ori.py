import torch
from torch import nn, autocast
from mamba_ssm.modules.mamba2 import Mamba2


class RSE(nn.Module):

    def __init__(self, hidden_size, layer_id=0, mid_hidden_dim=256):
        super(RSE, self).__init__()
        self.mid_hidden_dim = mid_hidden_dim
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.mamba2 = Mamba2(
            d_model=768,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=1,
            dtype=torch.float32,
            use_mem_eff_path=False
        )
        self.proj_down = nn.Linear(hidden_size, self.mid_hidden_dim)
        self.proj_up = nn.Linear(self.mid_hidden_dim, hidden_size)
        self.init_weights()
        self.act = nn.GELU()
        self.weight_1, self.bias_1 = self.init_EAT(self.mid_hidden_dim)
        self.weight_2, self.bias_2 = self.init_EAT(hidden_size)
        self.gate2 = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def init_weights(self):
        self.proj_down.weight.data.zero_()
        self.proj_up.bias.data.zero_()

    def init_EAT(self, dim):
        weight = nn.Parameter(torch.ones(dim))
        bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(weight, mean=1, std=.02)
        nn.init.normal_(bias, std=.02)

        return weight, bias

    def EAT(self, x, weight, bias):
        return x * weight + bias

    def forward(self, feature, attention_mask=None, position_ids=None):
        x_init = feature
        # todo:-----------SSD----------------------
        mid_feature_mamba = x_init
        with autocast(device_type='cuda', enabled=True, dtype=torch.float16):
            mamba2_output_ = self.mamba2(mid_feature_mamba)


        alpha = torch.sigmoid(self.gate1)
        # todo:-----------EAT----------------------
        # eat_feature = x_init
        # eat_feature = self.proj_down(eat_feature)
        # eat_feature = self.EAT(eat_feature, self.weight_1, self.bias_1)
        # eat_feature = self.act(eat_feature)
        # eat_feature = self.proj_up(eat_feature)
        # eat_feature = self.EAT(eat_feature, self.weight_2, self.bias_2)
        # beta = torch.sigmoid(self.gate2)

        # TODO:-----------ADD--------------------
        x = alpha * mamba2_output_ + x_init
        # x = alpha * mamba2_output_ + x_init + eat_feature * beta

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
