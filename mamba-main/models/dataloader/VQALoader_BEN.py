import os.path
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoImageProcessor, BlipProcessor
import concurrent.futures

class VQALoader_BEN_Efficient(Dataset):
    """
    Efficient Data Loader for RSVQAxBEN dataset.
    """

    def __init__(self,
                 imgFolder,
                 images_file,
                 questions_file,
                 answers_file,
                 tokenizer,
                 image_processor,
                 train=True,
                 selected_answers=None,
                 sequence_length=40,
                 transform=None,
                 preload_image_ids=True):  # 添加预加载图像 ID 的选项

        self.train = train
        self.imgFolder = imgFolder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.transform = transform
        self.sequence_length = sequence_length
        self.preload_image_ids = preload_image_ids
        self.image_paths = {}  # 用于存储图像路径

        # 加载 JSON 文件
        print("Loading JSONs...")
        with open(questions_file, 'r') as f:
            self.questionsJSON = json.load(f)
        with open(answers_file, 'r') as f:
            self.answersJSON = json.load(f)
        with open(images_file, 'r') as f:
            self.imagesJSON = json.load(f)
        print("Done.")

        # 构建索引字典
        self.question_id_to_question = {q.get('id'): q for q in self.questionsJSON['questions']}
        self.answer_id_to_answer = {a.get('id'): a for a in self.answersJSON['answers']}

        # 构建 image_id 到 question_ids 的映射, 以及 active 的图片
        self.image_id_to_question_ids = {}
        self.active_images = []
        for img in self.imagesJSON['images']:
            if img.get('active', False):
                self.image_id_to_question_ids[img['id']] = img.get('questions_ids', [])
                self.active_images.append(img['id'])

        # 构建 question_id 到 answer_ids 的映射 (只考虑 active 的答案)
        self.question_id_to_answer_ids = {}
        for ans in self.answersJSON['answers']:
            if ans.get('active', False):
                question_id = ans.get('question_id')
                if question_id is not None:
                    self.question_id_to_answer_ids.setdefault(question_id, []).append(ans.get('id'))
        # 预加载图像路径
        if self.preload_image_ids:
            self.preload_image_paths_()

        # 训练阶段构建答案集 (和之前一样)
        if train:
             print("Building answer set (Train Mode)...")
             self.freq_dict = {}
             for img_id in tqdm(self.active_images, desc="Processing Images (Train)"):
                for question_id in self.image_id_to_question_ids.get(img_id, []):
                    question = self.question_id_to_question.get(question_id)  # 直接查字典
                    if question:
                        answer_ids = self.question_id_to_answer_ids.get(question_id, [])
                        for answer_id in answer_ids:
                            answer = self.answer_id_to_answer.get(answer_id)  # 直接查字典
                            if answer:
                                answer_str = answer.get('answer')
                                if answer_str:
                                    if answer_str not in self.freq_dict:
                                         self.freq_dict[answer_str] = 1
                                    else:
                                        self.freq_dict[answer_str] += 1

             # 按频率排序
             self.freq_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)
             self.selected_answers = [item[0] for item in self.freq_dict]
             print("Selected Answers (Top N):", self.selected_answers)

        else:
            if selected_answers is None:
                print("Building answer set (Test Mode - No pre-selected answers)...")
                self.freq_dict = {}
                for img_id in tqdm(self.active_images, desc="Processing Images (Test)"):
                    for question_id in self.image_id_to_question_ids.get(img_id, []):
                        question = self.question_id_to_question.get(question_id)   # O(1)
                        if question:
                            answer_ids = self.question_id_to_answer_ids.get(question_id, [])
                            for answer_id in answer_ids:
                                answer = self.answer_id_to_answer.get(answer_id)   # O(1)
                                if answer:
                                    answer_str = answer.get('answer')
                                    if answer_str:
                                         if answer_str not in self.freq_dict:
                                            self.freq_dict[answer_str] = 1
                                         else:
                                            self.freq_dict[answer_str] += 1

                self.freq_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)
                self.selected_answers = [item[0] for item in self.freq_dict]
                print("Selected Answers (from Test Data):", self.selected_answers)
            else: #selected_answer 不为None.
                 self.selected_answers = selected_answers
                 print("Using provided selected answers:", self.selected_answers)
         # 构建数据样本列表
        self.data_samples = []
        print("Building data samples...")
        for img_id in tqdm(self.active_images, desc="Building Data Samples"):
            for question_id in self.image_id_to_question_ids.get(img_id, []):
                 question = self.question_id_to_question.get(question_id)
                 if question:
                     answer_ids = self.question_id_to_answer_ids.get(question_id, [])
                     for answer_id in answer_ids:
                        answer = self.answer_id_to_answer.get(answer_id)
                        if answer:
                            answer_str = answer.get('answer')
                            if answer_str:
                                try:
                                    answer_index = self.selected_answers.index(answer_str)
                                except ValueError:
                                    answer_index = -1

                                self.data_samples.append({
                                        'image_id': img_id,
                                        'question': question.get('question'),  #
                                        'answer_index': answer_index,
                                        'answer_str': answer_str,  # 原始答案字符串
                                         'question_id':question_id
                                    })
        print("Done.")
    def preload_image_paths_(self):
        """预加载所有激活图像的路径"""
        print("Pre-loading image paths...")
        for img_id in tqdm(self.active_images, desc="Pre-loading Image Paths"):
            self.image_paths[img_id] = os.path.join(self.imgFolder, f"{img_id}.tif")
        print("Done pre-loading image paths.")

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        image_id = sample['image_id']
        question_text = sample['question']
        answer_index = sample['answer_index']
        answer_str = sample['answer_str']
        question_id = sample['question_id']

        # 加载图像 (根据是否预加载路径采取不同策略)
        if self.preload_image_ids:
            image_path = self.image_paths[image_id]
        else:
            image_path = os.path.join(self.imgFolder, f"{image_id}.tif")

        try:
            image = io.imread(image_path)
            if self.transform is not None:
                image = Image.fromarray(image).convert('RGB') #不用unit8
                image = self.transform(image)
            image = self.image_processor(images=image, return_tensors="pt")['pixel_values'][0]

        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            image = torch.zeros((3, 256, 256))  # 使用占位符
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros((3, 256, 256))


        # 处理文本
        encoding = self.tokenizer(
            question_text,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # labels
        labels = self.tokenizer(answer_str, padding='max_length', truncation=True, max_length=9, return_tensors="pt")
        label = labels['input_ids'].squeeze(0) # (seq_len)
        label_attention_mask = labels['attention_mask'].squeeze(0)

        # 构建返回字典
        if self.train:
            return {
                'pixel_values': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label,
                'answer': torch.tensor(answer_index, dtype=torch.long),
                "label_attention_mask": label_attention_mask,
                 'question_id':question_id
            }
        else:
            return {
                'pixel_values': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label,
                'answer': torch.tensor(answer_index, dtype=torch.long),
                'image_id': image_id,
                'question': question_text,
                'answer_str': answer_str,
                "label_attention_mask": label_attention_mask,
                 'question_id':question_id
            }
def create_ben_data_loader_efficient(img_folder, images_file, questions_file, answers_file,
                                     tokenizer, image_processor, batch_size,
                                     is_train=True, selected_answers=None, sequence_length=40,
                                     transform=None, num_workers=0, shuffle=None,
                                     preload_image_ids=True):  # 添加预加载选项
    dataset = VQALoader_BEN_Efficient(
        imgFolder=img_folder,
        images_file=images_file,
        questions_file=questions_file,
        answers_file=answers_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        train=is_train,
        selected_answers=selected_answers,
        sequence_length=sequence_length,
        transform=transform,
        preload_image_ids=preload_image_ids  # 传递预加载选项
    )

    if shuffle is None:
        shuffle = is_train

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return data_loader

# 示例用法
if __name__ == '__main__':
    # 假设你的模型和数据集路径
    dataset_path = "/home/pengfei/RSVQABEN"  # 数据集根目录
    processor = BlipProcessor.from_pretrained("/home/pengfei/rsvqa/src/Salesforce/blip-vqa-base")
    tokenizer = processor.tokenizer

    # 加载分词器和图像处理器
    image_processor = processor

    # 训练集 DataLoader (启用多进程和图像ID预加载)
    train_dataloader = create_ben_data_loader_efficient(
         img_folder=os.path.join(dataset_path, "images"),
         images_file=os.path.join(dataset_path, "RSVQAxBEN_split_train_images.json"),
         questions_file=os.path.join(dataset_path, "RSVQAxBEN_split_train_questions.json"),
         answers_file=os.path.join(dataset_path, "RSVQAxBEN_split_train_answers.json"),
         tokenizer=tokenizer,
         image_processor=image_processor,
         batch_size=32,
         is_train=True,
         num_workers=4,         # 使用 4 个进程
         preload_image_ids=True  # 启用图像 ID 预加载
     )
      # 验证集 DataLoader (通常只需要多进程)
    val_dataloader = create_ben_data_loader_efficient(
         img_folder=os.path.join(dataset_path, "images"),
         images_file=os.path.join(dataset_path, "RSVQAxBEN_split_val_images.json"),
         questions_file=os.path.join(dataset_path, "RSVQAxBEN_split_val_questions.json"),
         answers_file=os.path.join(dataset_path, "RSVQAxBEN_split_val_answers.json"),
         tokenizer=tokenizer,
         image_processor=image_processor,
         batch_size=32,
         is_train=False,
         shuffle=False,
         num_workers=4,          # 使用多进程
         preload_image_ids=True # 验证集通常也预加载 ID
     )
      # 遍历 DataLoader (示例)
    for batch in train_dataloader:
         pixel_values = batch["pixel_values"]
         input_ids = batch["input_ids"]
         attention_mask = batch["attention_mask"]
         labels = batch["labels"]
         answer = batch['answer']
         print(f"Pixel values shape: {pixel_values.shape}")
         print(f"Input IDs shape: {input_ids.shape}")
         print(f"Attention mask shape: {attention_mask.shape}")
         print(f"Labels shape: {labels.shape}")
         print(f"Answer shape: {answer.shape}")
         break