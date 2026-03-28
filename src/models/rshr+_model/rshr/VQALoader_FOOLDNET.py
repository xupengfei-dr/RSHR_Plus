import json
import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class VQALoader(Dataset):
    """
    Args:
        json_path (str): JSON注释文件的路径。
        img_folder_path (str): 存放所有图像文件的具体文件夹路径。
        ans2idx (dict): 【必须提供】的答案到索引的映射字典。
                        通常从官方的 class_to_label.json 文件加载。
        img_transform (callable, optional): 应用于图像的转换。
    """

    def __init__(self, json_path, img_folder_path,answer_path,tokenizer,image_processor,sequence_length=40,img_transform=None,is_train=False):

        print(f"Loading annotations from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.img_folder_path = img_folder_path
        self.img_transform = img_transform

        # 2. 直接使用外部传入的答案词汇表
        print("=" * 20 + " Loading Official Answer Space " + "=" * 20)
        with open(answer_path, 'r', encoding='utf-8') as f:
            self.ans2idx = json.load(f)
        self.idx2ans = {idx: ans for ans, idx in self.ans2idx.items()}
        self.num_answers = len(self.ans2idx)
        print(f"Initialized dataset with {len(self.annotations)} annotations.")
        print(f"Using a shared vocabulary of {self.num_answers} answers.")
        print(f"Using a shared vocabulary of {self.ans2idx} answers.")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sequence_length = sequence_length
        self.train = is_train

    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        获取一个样本。
        """
        # 1. 获取指定索引的注释
        annotation = self.annotations[idx]
        # 2. 构建图像路径并加载
        image_id = annotation['Image_ID']
        image_id = image_id.lower()

        img_path = os.path.join(self.img_folder_path, image_id)

        try:
            image = Image.open(img_path).convert('RGB')
            if self.img_transform:
                # image = self.img_transform(image)
                imgT = self.image_processor(image, return_tensors="pt", do_resize=True)
                # imgT = self.image_processor(image, return_tensors="pt")
            else:
                imgT = self.image_processor(image, return_tensors="pt")
            pixel_values = imgT['pixel_values'][0]
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}. Skipping this item.")
            return self.__getitem__((idx + 1) % len(self))

        # 3. 获取问题文本
        question = annotation['Question']

        language_feats = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            max_length=self.sequence_length
        )
        input_ids = language_feats['input_ids'][0]
        attention_mask = language_feats['attention_mask'][0]
        # print('inputids',input_ids.shape)
        # print('attentionmask',attention_mask.shape)
        # print('pixv',pixel_values.shape)
        # 4. 获取答案并使用提供的词汇表转换为标签
        label_str  = annotation['Question_Type']
        label_f = self.tokenizer(
            label_str,
            return_tensors='pt',
            padding='max_length',
            max_length=self.sequence_length
        )
        label = label_f['input_ids'][0]
        label_attention_mask = label_f['attention_mask'][0]
        answer_str = annotation['Ground_Truth']
        # print(question,'-----------',answer_str)
        answer_label = self.ans2idx.get(answer_str)
        # print(answer_label,'------------------')
        # 5. 处理在词汇表中找不到的答案
        if answer_label is None:
            # 这种情况理论上不应该发生，因为官方列表是完整的。
            # 但作为健壮性代码，我们还是处理一下。
            print(f"Warning: Answer '{answer_str}' from annotation {idx} not found in the provided vocabulary!")
            # 选项1: 抛出错误，强制修复数据或词汇表
            # raise KeyError(f"Answer '{answer_str}' not in vocabulary!")
            # 选项2: 映射到一个默认值，例如 0
            answer_label = 0
            # 选项3: 如果词汇表里有 <UNK> 标记，则映射到它
            # answer_label = self.ans2idx.get('<UNK>', 0)

        return {
            "pixel_values": pixel_values,  # 图像的特征
            "input_ids": input_ids,  # 语言特征
            "attention_mask": attention_mask,  # 语言的 attention mask
            'image': imgT,
            'question': question,
            'answer_str':answer_str,
            'answer': torch.tensor(answer_label, dtype=torch.long),
            "labels": label,  # 答案的标签
            "label_attention_mask": label_attention_mask,
            "question_type":label_str
        }