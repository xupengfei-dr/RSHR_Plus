# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch

# torch.cuda.set_device(0)
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin


class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=4,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        # todo: ---------------------in_proj-------------------------
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)   #torch.Size([64, 40, 3352])
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state) 24
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)   #0

        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            # todo: ---------------------split-------------------------
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2   # 0
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            # zxbcdt :torch.Size([64, 40, 3352])dt :torch.Size([64, 40, 24])  x0:torch.Size([64, 40, 0]) xbc:torch.Size([64, 40, 1792])   z:torch.Size([64, 40, 1536]) z0:torch.Size([64, 40, 0])
            #zxbcdt : [64, 40, |-----------------------------3352-----------------------------|]
                     # |    |    |         |------------------|         |----|
                     # z0   x0    z        xBC                      dt
                    # (0)  (0)  (1536)     (1792)                  (24)
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                #xBC : torch.Size([64, 40, 1792])
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            # X :torch.Size([64, 40, 1536]) ,B=C=torch.Size([64, 40, 128])
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            #Y:torch.Size([64, 40, 1536])
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state





def main():
    # 定义模型参数 (应与 Mamba2 的 __init__ 方法中的参数匹配)
    d_model = 768        # 输入和输出的维度
    d_state = 128        # 状态空间模型的维度
    d_conv = 4           # 卷积核的大小
    expand = 2           # 内部维度的扩展因子
    headdim = 64         # SSM 部分的 head 维度
    ngroups = 1          # SSM 的分组数量 (对于默认的回退单步推理，保持为 1 比较简单)
    seqlen = 40        # 序列长度
    batch_size = 64      # Batch 大小

    # 确定使用的设备 (GPU 或 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 实例化 Mamba2 模型
    # 注意: 为了获得最佳性能，这段代码高度依赖于 causal-conv1d 和 mamba-ssm 提供的 CUDA/Triton 内核。
    # 如果这些库未安装或没有可用的 CUDA 设备，它可能会回退到较慢的 PyTorch 实现，或者可能失败。
    # 如果遇到与内核相关的错误，可以尝试设置 use_mem_eff_path=False，但即使是回退路径也可能需要 causal_conv1d。
    try:
        print("尝试实例化 Mamba2...")
        model = Mamba2(
            d_model=768,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=1, # ngroups=1 对于默认的单步回退是必需的
            device=device,
            use_mem_eff_path=False,
            dtype=torch.float32 if device.type == 'cpu' else torch.float16 # 在 GPU 上通常使用 float16 以提高效率
        ).to(device)
        print("Mamba2 模型实例化成功.")
        print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,} ")

    except Exception as e:
        print(f"实例化 Mamba2 模型时出错，可能是由于缺少依赖或 CUDA 问题: {e}")
        print("请确保已安装 causal-conv1d 和 mamba-ssm，并在需要内核时拥有支持 CUDA 的 GPU。")
        return # 如果模型实例化失败则退出

    print("\n--- 运行标准前向传播 ---")

    # 创建模拟输入数据 (batch, seqlen, d_model)
    # 使用与模型参数相同的 dtype
    dummy_input = torch.randn(batch_size, seqlen, d_model, device=device, dtype=model.in_proj.weight.dtype)

    # 执行模型的前向传播
    # 注意: 如果 use_mem_eff_path=True 且相关 Triton 内核失败，这里可能会抛出你在前面看到的错误
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状 (前向传播): {output.shape}")


    print("\n--- 运行单步推理示例 ---")
    # 这是为了生成任务进行的单步 (token-by-token) 推理

    # 需要一个简单的对象来保存缓存状态 (conv_state, ssm_state)
    class InferenceParams:
        def __init__(self):
            self.key_value_memory_dict = {} # 用于存储每一层的状态
            self.seqlen_offset = 0 # 记录已经处理的 token 数量

    inference_params = InferenceParams()

    # 为推理分配缓存 (如果 model.layer_idx 未设置)
    # model.step() 方法会使用 inference_params 中的缓存
    # model.allocate_inference_cache 方法会在推理模式下 (inference_params 不为 None)
    # 处理第一个 token 时隐式调用。但我们也可以手动分配。
    # Mamba2 的 allocate_inference_cache 使用 layer_idx，如果模型没有设置，这里指定一个假的。
    if model.layer_idx is None:
        model.layer_idx = 0 # 分配一个假的 layer_idx

    try:
        print("分配推理缓存...")
        # 状态将存储在 inference_params.key_value_memory_dict[model.layer_idx] 中
        conv_state, ssm_state = model.allocate_inference_cache(batch_size, seqlen, dtype=model.in_proj.weight.dtype)
        print("缓存分配成功.")
        print(f"初始 conv_state 形状: {conv_state.shape}")
        print(f"初始 ssm_state 形状: {ssm_state.shape}")

        print("\n处理第一个 token...")
        # 模拟处理第一个 token (索引为 0)
        first_token_input = dummy_input[:, 0:1, :] # (batch, 1, d_model)
        with torch.no_grad(): # 推理通常在 torch.no_grad() 环境下进行
            # step 方法处理一个 token 的输入和当前状态
            # 它会 原地 更新状态
            output_token, updated_conv_state, updated_ssm_state = model.step(
                first_token_input, conv_state, ssm_state
            )

        print(f"输入形状 (step): {first_token_input.shape}")
        print(f"输出形状 (处理第一个 token 的 step): {output_token.shape}")
        # 注意: updated_conv_state 和 updated_ssm_state 是与 conv_state 和 ssm_state 相同的对象，
        # 因为它们是原地更新的。

        print("\n处理第二个 token...")
        # 模拟使用已更新的状态处理第二个 token (索引为 1)
        second_token_input = dummy_input[:, 1:2, :] # (batch, 1, d_model)
        # 如果使用 inference_params 结构，这里会增加 seqlen_offset。
        # 对于这种手动调用 step 的情况，我们只需传入状态即可。
        with torch.no_grad():
            output_token_2, _, _ = model.step(
                second_token_input, conv_state, ssm_state # 传入 *已经更新* 的状态
            )
        print(f"输出形状 (处理第二个 token 的 step): {output_token_2.shape}")


    except Exception as e:
         print(f"\n单步推理时出错: {e}")
         print("单步推理高度依赖特定的内核 (causal_conv1d_update, selective_state_update)。")
         print("如果这些内核不可用，step 方法可能会引发错误。")
         print("请检查模型实例化期间打印的警告，看哪些内核未找到。")
if __name__ == "__main__":
    main()

# def main():
#     # Define model parameters (should match those in your Mamba2 __init__)
#     d_model = 768        # Dimension of the input and output
#     d_state = 128        # Dimension of the state space model
#     d_conv = 4           # Kernel size for the convolution
#     expand = 2           # Expansion factor for the inner dimension
#     headdim = 64         # Head dimension for the SSM part
#     ngroups = 1          # Number of groups for SSM (keep at 1 for simplicity in fallback step)
#     seqlen = 24        # Sequence length
#     batch_size = 64       # Batch size
#     # Determine the device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Instantiate the Mamba2 model
#     # Note: For optimal performance, this code heavily relies on CUDA/Triton kernels
#     # from causal-conv1d and mamba-ssm. If these are not installed or no CUDA device
#     # is available, it might fall back to slower PyTorch implementations or fail.
#     # If you encounter errors related to kernels, you might try setting
#     # use_mem_eff_path=False, but the fallback might also require causal_conv1d.
#     try:
#         print("Attempting to instantiate Mamba2...")
#         model = Mamba2(
#             d_model=d_model,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#             headdim=headdim,
#             ngroups=ngroups, # ngroups=1 is required for the default step fallback
#             device=device,
#             dtype=torch.float32 if device.type == 'cpu' else torch.float16 # Use float16 on GPU for efficiency typically
#             # Add other parameters as needed based on the Mamba2.__init__ signature
#         ).to(device)
#         print("Mamba2 model instantiated successfully.")
#         print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} total")
#
#     except Exception as e:
#         print(f"Error instantiating Mamba2 model, likely due to missing dependencies or CUDA issues: {e}")
#         print("Please ensure you have installed causal-conv1d and mamba-ssm and have a CUDA-enabled GPU if required by the kernels.")
#         return
#
#     print("\n--- Running standard forward pass ---")
#
#     dummy_input = torch.randn(batch_size, seqlen, d_model, device=device, dtype=model.in_proj.weight.dtype)
#
#     output = model(dummy_input)
#
#     print("\n--- Running single-step inference example ---")
#
#     class InferenceParams:
#         def __init__(self):
#             self.key_value_memory_dict = {}
#             self.seqlen_offset = 0
#
#     inference_params = InferenceParams()
#
#
#     if model.layer_idx is None:
#         model.layer_idx = 0 # Assign a dummy layer_idx
#
#     try:
#         print("Allocating inference cache...")
#         # The states will be stored in inference_params.key_value_memory_dict[model.layer_idx]
#         conv_state, ssm_state = model.allocate_inference_cache(batch_size, seqlen, dtype=model.in_proj.weight.dtype)
#         print("Cache allocated.")
#         print(f"Initial conv_state shape: {conv_state.shape}")
#         print(f"Initial ssm_state shape: {ssm_state.shape}")
#
#         print("\nProcessing first token...")
#         # Simulate processing the first token (index 0)
#         first_token_input = dummy_input[:, 0:1, :] # (batch, 1, d_model)
#         with torch.no_grad():
#             # The step method takes the input for ONE token and the states
#             # It updates the states IN PLACE
#             output_token, updated_conv_state, updated_ssm_state = model.step(
#                 first_token_input, conv_state, ssm_state
#             )
#
#         print(f"Input shape (step): {first_token_input.shape}")
#         print(f"Output shape (step for 1st token): {output_token.shape}")
#         # Note: updated_conv_state and updated_ssm_state are the same objects as conv_state and ssm_state
#         # as they are updated in place.
#
#         print("\nProcessing second token...")
#         # Simulate processing the second token (index 1) using the updated states
#         second_token_input = dummy_input[:, 1:2, :] # (batch, 1, d_model)
#         # Increment seqlen_offset in inference_params (if you were using that structure)
#         # For this manual step call, we just need to pass the states.
#         with torch.no_grad():
#             output_token_2, _, _ = model.step(
#                 second_token_input, conv_state, ssm_state # Pass the *already updated* states
#             )
#         print(f"Output shape (step for 2nd token): {output_token_2.shape}")
#
#
#     except Exception as e:
#          print(f"\nError during step inference: {e}")
#          print("Step inference is highly dependent on specific kernels (causal_conv1d_update, selective_state_update).")
#          print("If these kernels are not available, the step method might raise errors.")
#          print("Check the printed warnings during model instantiation to see which kernels were not found.")
#

