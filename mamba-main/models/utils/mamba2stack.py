import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2


class Mamba2Stack(nn.Module):
    """
    一个 Mamba2 层的堆叠模块，其行为类似于 TransformerEncoder 堆叠 TransformerEncoderLayer。
    它假设 'Mamba2' 类是一个已定义的、功能齐全的 nn.Module。
    """

    def __init__(self, num_layers: int, d_model: int, norm: nn.Module = None, **mamba2_layer_args):
        """
        初始化 Mamba2Stack。

        参数:
            num_layers (int): 要堆叠的 Mamba2 层的数量。
            d_model (int): 输入和输出特征的维度。这也是每个 Mamba2 层的 d_model。
            norm (nn.Module, 可选): 在所有 Mamba2 层处理完毕后应用的可选归一化层。
                                     默认为 None，表示不进行最终归一化。
            **mamba2_layer_args: 传递给每个 Mamba2 层构造函数的额外关键字参数。
                                 例如: d_state, d_conv, expand, use_mem_eff_path, device, dtype 等。
                                 `d_model` 和 `layer_idx` 会被此模块自动设置。
        """
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        # 创建 Mamba2 层的列表
        # 每个 Mamba2 层都会接收 d_model, layer_idx 以及来自 mamba2_layer_args 的参数
        self.layers = nn.ModuleList([
            Mamba2(d_model=d_model, layer_idx=i, **mamba2_layer_args)
            for i in range(num_layers)
        ])

        self.norm = norm  # 可选的输出归一化

    def forward(self, src: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        通过堆叠的 Mamba2 层处理输入序列。

        参数:
            src (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, d_model)。
            inference_params (可选): 用于有状态推理的参数对象或字典。
                                     此参数会直接传递给每个 Mamba2 层的 forward 方法。
                                     Mamba2 层内部应通过其 layer_idx 从此对象中管理其状态。
                                     如果 Mamba2 层在特定模式下不需要此参数，可以为 None。

        返回:
            torch.Tensor: 输出序列，形状与 src 相同 (batch_size, seq_len, d_model)。
        """
        output = src

        for layer in self.layers:
            output = layer(output, inference_params=inference_params)  # 将 inference_params 传递给每个 Mamba2 层

        if self.norm is not None:
            output = self.norm(output)

        res_net_ = src + output
        return res_net_


if __name__ == '__main__':
    num_mamba_layers = 6
    feature_dim = 768  #  (d_model)
    batch_size = 4
    seq_length = 128

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mamba2_specific_params = {
        'd_state': 128,
        'd_conv': 4,
        'expand': 2,
        'headdim': 64,
        'd_ssm': None,
        'rmsnorm': True,
        'norm_before_gate': False,
        'bias': False,
        'conv_bias': True,
        'use_mem_eff_path': False,
        'chunk_size': 256,
        'device': device,
        'dtype': dtype
    }

    # output_normalization_layer = nn.LayerNorm(feature_dim, device=device, dtype=dtype)
    output_normalization_layer = None  # 如果不需要最终归一化
    # ----  创建 Mamba2Stack 实例 ----
    mamba_encoder_model = Mamba2Stack(
        num_layers=num_mamba_layers,
        d_model=feature_dim,
        norm=output_normalization_layer,
        **mamba2_specific_params  # 将 Mamba2 特定参数解包传入
    ).to(device)

    print("Mamba2Stack 模型结构:")
    # 打印模型结构，会显示 ModuleList 中的 Mamba2 层
    # (具体打印内容取决于你的 Mamba2 类的 __repr__ 方法)
    print(mamba_encoder_model)

    # ---- 步骤 5: 创建虚拟输入数据并进行前向传播 ----
    dummy_input_features = torch.randn(batch_size, seq_length, feature_dim, device=device, dtype=dtype)
    print(f"\n输入特征形状: {dummy_input_features.shape}")

    try:
        # 在训练或非自回归推理时，inference_params 通常可以为 None
        output_features = mamba_encoder_model(dummy_input_features, inference_params=None)
        print(f"输出特征形状: {output_features.shape}")

        # 验证输出形状是否正确
        assert output_features.shape == dummy_input_features.shape, "输出形状与输入形状不匹配！"
        print("Mamba2Stack 前向传播成功！")

    except Exception as e:
        print(f"Mamba2Stack 前向传播时发生错误: {e}")
        print("请检查：")
        print("1. 你的 Mamba2 类定义是否完整且无误。")
        print("2. Mamba2 所需的所有辅助类 (如 RMSNormGated) 是否已定义或导入。")
        print("3. 如果 use_mem_eff_path=True，确保相关 CUDA 内核已编译并可用。")
        print("4. 传递给 Mamba2Stack 的 mamba2_specific_params 是否与你的 Mamba2 类构造函数匹配。")
        import traceback

        traceback.print_exc()




    # ---- 关于推理 (Autoregressive Generation) 的说明 ----
    # 如果你需要使用 Mamba2Stack 进行自回归生成 (例如，逐个 token 生成文本)，
    # 那么 `inference_params` 的角色将非常关键。
    # 你需要设计一个 `inference_params` 对象 (通常是一个类的实例或字典)，
    # Mamba2 层可以使用它来存储和检索其内部状态 (如卷积状态和SSM状态)。
    #
    # 典型的流程可能如下：
    # 1. 初始化 `inference_params` 对象，其中包含一个空的字典 (例如 `key_value_memory_dict`)
    #    来存储每层的状态，以及一个 `seqlen_offset` 来跟踪生成进度。
    # 2. (可选，但推荐) 调用每个 Mamba2 层的 `allocate_inference_cache` 方法，
    #    并将返回的状态存储到 `inference_params.key_value_memory_dict` 中，以层索引为键。
    # 3. 在生成循环的每一步：
    #    a. 获取当前 token 的 embedding。
    #    b. 调用 `mamba_encoder_model(current_token_embedding, inference_params=inference_params_obj)`。
    #       - Mamba2 层的 `forward` 或 `step` 方法会使用 `inference_params_obj` 中的状态，
    #         并更新它。
    #    c. 更新 `inference_params_obj.seqlen_offset`。
    #
    # `Mamba2Stack` 本身只是简单地将 `inference_params` 对象传递给每一层。
    # 状态管理的具体实现细节位于 `Mamba2` 层内部。
