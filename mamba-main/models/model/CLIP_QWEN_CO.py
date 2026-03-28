import requests
import torch
import torch
import torch.nn as nn

from PIL import Image
from torch import nn
device = torch.device("cuda:4")

# from transformers import AutoModel, AutoConfig, CLIPConfig, PretrainedConfig  # 导入需要的类
# 假设已经本地化, 或者把之前的代码拷贝过来,
from transformers import CLIPConfig, PretrainedConfig, AutoModel, AutoConfig, AutoModelForCausalLM, \
    AutoTokenizer, CLIPProcessor, AutoProcessor, CLIPVisionModel, Qwen2ForCausalLM, Qwen2Model


class MyVLQConfig(PretrainedConfig):
    def __init__(
            self,
            vision_path,
            text_path,
            fusion_hidden_dim=512,
            num_classes=10,
            pretrained_vision_name_or_path=None,  # 新增：预训练视觉模型的名称或路径
            **kwargs
    ):

        super().__init__(**kwargs)
        # 如果传入字典类型, 用CLIPConfig进行配置
        self.vision_model = CLIPVisionModel.from_pretrained(vision_path)
        self.vision_config = self.vision_model.vision_model.config

        self.text_model = Qwen2Model.from_pretrained(text_path)
        self.text_config = self.text_model.config
        self.vision_feature_layer = -2
        self.fusion_hidden_dim = fusion_hidden_dim  # 维度信息
        self.num_classes = num_classes  # 分类信息
        self.pretrained_vision_name_or_path = pretrained_vision_name_or_path

    def to_dict(self):  # 转成dict格式
        output = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "to_dict"):
                output[key] = value.to_dict()
            else:
                output[key] = value

        return output

    def __repr__(self) -> str:  # 方便print
        return str(self.to_dict())  #


class MultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_dim, text_hidden_dim):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_dim, text_hidden_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_dim, text_hidden_dim)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states




class CoAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=1536, num_heads=8, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 图像到文本的注意力
        self.img2text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 文本到图像的注意力
        self.text2img_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # 残差连接的Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_features, text_features):
        """
        输入:
        img_features:   [batch_size, 576, 1536]
        text_features: [batch_size, 40, 1536]

        输出:
        fused_features: [batch_size, 576+40, 1536]
        """
        # --------------------------
        # 第一阶段：图像到文本的注意力
        # --------------------------
        # 使用图像作为Query，文本作为Key/Value
        img_attn_out, _ = self.img2text_attn(
            query=img_features,
            key=text_features,
            value=text_features
        )
        img_attn_out = self.norm1(img_features + self.dropout(img_attn_out))

        # --------------------------
        # 第二阶段：文本到图像的注意力
        # --------------------------
        # 使用文本作为Query，图像作为Key/Value
        text_attn_out, _ = self.text2img_attn(
            query=text_features,
            key=img_attn_out,
            value=img_attn_out
        )
        text_attn_out = self.norm2(text_features + self.dropout(text_attn_out))

        # --------------------------
        # 第三阶段：特征拼接与融合
        # --------------------------
        # 拼接两种模态特征
        combined = torch.cat([img_attn_out, text_attn_out], dim=1)

        # 前馈网络增强
        fused_features = self.feed_forward(combined)
        fused_features = self.norm3(combined + self.dropout(fused_features))

        return fused_features



class VLQ(nn.Module):
    def __init__(self, config: MyVLQConfig):
        super().__init__()
        self.config = config
        self.vision_tower = config.vision_model.vision_model
        self.language_model = config.text_model
        self.conattention = CoAttentionFusion(hidden_dim=config.text_config.hidden_size)
        self.multi_modal_projector = MultiModalProjector(
            vision_hidden_dim=config.vision_config.hidden_size,
            text_hidden_dim=config.text_config.hidden_size  # config.text_config.hidden_size,  #  text_config 如果为None,就随意传入一个int
        )


        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else 0.02  #
        )

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        vision_outputs = self.vision_tower(pixel_values,output_hidden_states=True)
        image_features = vision_outputs.hidden_states[self.config.vision_feature_layer] #1,577,1024
        image_features = image_features[:, 1:]# [CLS] token  1,576,1024

        if input_ids is not None:
            text_outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True)


        text_features = text_outputs.last_hidden_state  # 或使用其他层的输出
        # text_features = torch.mean(text_features, dim=1)  # [batch_size, hidden_size] 简单的全局平均池化
        # 3. 特征对齐 ，将视觉的大小投影到文本的长度上
        fused_features = self.multi_modal_projector(image_features)
        # print(fused_features.shape) # 1 576 1536
        # print(text_outputs.hidden_states[0].shape)  # 1 40 1536
        # fused_image,fused_text = self.hyconattention(fused_features,text_features)
        # 4. 特征融合
        result_feature = self.conattention(fused_features,text_features)    # 1 ,616,1536

        return result_feature



if __name__ == '__main__':
    qwen_model_path = "/home/pengfei/Qwen2.5-1.5B-Instruct"
    clip_vit_model_Path = "/home/pengfei/clip-vit-large-patch14-336"

    # --- 虚拟输入 ---
    batch_size = 2
    image_size = (336, 336)
    input_ids_question = "how are you"

    llm_model_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
    inputs = llm_model_tokenizer(input_ids_question, return_tensors="pt", max_length=40,padding="max_length")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    labels = torch.randint(0, 9, (batch_size,)).to(device)  #

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processor = AutoProcessor.from_pretrained(clip_vit_model_Path)
    pixel_values = processor(images=image, return_tensors="pt")
    pixel_values = pixel_values['pixel_values'].to(device)
    config = MyVLQConfig(vision_path=clip_vit_model_Path, text_path=qwen_model_path)
    model = VLQ(config)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    print(model)  # 输出模型结构
    # --- 前向传播 ---
    logits= model(pixel_values, input_ids, attention_mask,labels=labels)
    print("Logits:", logits.shape)

