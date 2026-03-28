import json
import os
import sys
import torch
import numpy as np
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import sys
torch.set_float32_matmul_precision('high')
# from models.model.MyEncoder import MyEncoder
from models.model.MyUtils import GlobalQueryAttentionPool, In2Conv1d, In2
from models.t.trans.models.blip.modeling_blip_test import BlipForQuestionAnswering

sys.path.append("../../")
sys.path.append("../")

# from transformers import BlipConfig

path = ''


class VQAModel(pl.LightningModule):
    def __init__(self, batch_size=None, lr=None, number_outputs=None):
        super(VQAModel, self).__init__()

        self.save_hyperparameters()
        self.number_outputs = number_outputs
        self.loss = F.cross_entropy
        self.lr = lr
        self.batch_size = batch_size
        self.validation_step_outputs = []
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=number_outputs)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=number_outputs)
        self.results = {}
        self.res = []
        self.blip = BlipForQuestionAnswering.from_pretrained("/home/pengfei/blip-vqa-capfilt-large")

        # self.myencoder = MyEncoder(BlipConfig.from_dict(json.load(open('model/layer_config.json'))))
        self.in2 = In2()

        for name, param in self.blip.named_parameters():
            if 'adapter' not in name and 'text_encoder.' not in name:
                param.requires_grad = False
                print(name)
            # param.requires_grad = False
            # if 'adapter' not in name and 'text_encoder' not in name and 'vision_model.embeddings' not in name and 'vision_model.post_layernorm' not in name:
            #     # if 'vision_model.encoder.layers.0' not in name and 'vision_model.encoder.layers.1' not in name and'vision_model.encoder.layers.10' not in name and'vision_model.encoder.layers.11' not in name \
            #     #         and'vision_model.encoder.layers.5' not in name and'vision_model.encoder.layers.6' not in name and'vision_model.encoder.layers.7' not in name:

        self.classify_layer = nn.Sequential(
            nn.Linear(768, 1024),  # 调整为更合适的隐藏层大小
            nn.BatchNorm1d(1024),  # 添加 BatchNorm
            torch.nn.GELU(),  # 使用更高效的激活函数
            nn.Dropout(0.5),  # 增大正则化强度，减少过拟合
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # 再次使用 BatchNorm
            torch.nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, self.number_outputs)  # 输出层
        )

    def forward(self, pixel_values, input_ids, attention_mask, labels, labels_attention_mask):

        out = self.blip(pixel_values=pixel_values, input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels
                        )


        question_embedding = torch.squeeze(out['last_hidden_state_vero'])
        result = question_embedding

        # result = (self.myencoder(inputs_embeds=question_embedding))['last_hidden_state']

        res = self.in2(result)

        logist = self.classify_layer(res)

        return logist

    def configure_optimizers(self):
        # configuration of the optimizer
        def rule(epoch):
            if self.number_outputs == 9:
                if epoch <= 3:
                    lamda = 1
                else:
                    lamda = 0.01
                return lamda
            else:

                # 3 1  ,10 0.1 0.06
                if epoch <= 3:
                    lamda = 1 + epoch
                elif epoch <= 6:
                    lamda = 0.08
                else:
                    lamda = 0.01
                return lamda
            # else:
            #     if epoch <= 2:
            #         lamda = 1
            #     elif epoch <= 6:  #10
            #         lamda = 0.08
            #     else:
            #         lamda = 0.01
            #     return lamda

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        # optimizer = torch.optim.AdamW()
        scheduler = LambdaLR(optimizer, lr_lambda=rule)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        # performs the training steps

        # pixel_values, input_ids,  attention_mask, labels,answer = batch
        # 推理时解包 batch
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # tishi_language_feats = batch["tishi_language_feats"]
        # tishi_attention_mask = batch["tishi_attention_mask"]
        # print(tishi_language_feats)
        # exit()
        # print("label",labels)
        # exit()
        answer = batch["answer"]
        # print(answer,'---------------------------------------------answer')
        # print(answer.shape,'---------------------------------------------answer')
        # exit()
        # print(answer,"answer----------------------")
        labels_attention_mask = batch["label_attention_mask"]
        pred = self(pixel_values, input_ids, attention_mask, labels, labels_attention_mask)
        # pred = self(pixel_values, tishi_language_feats, tishi_attention_mask, labels, labels_attention_mask)
        # print(answer,"pred----------------------")
        # exit()
        self.train_acc(pred, answer)
        train_loss = self.loss(pred, answer)
        self.log("train_loss", train_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=True, sync_dist=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # pixel_values, input_ids, token_type_ids, attention_mask, answer, question_type, img_id, question, answer_str = batch
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        answer = batch["answer"]
        question_type = batch["question_type"]
        labels = batch["labels"]
        labels_attention_mask = batch["label_attention_mask"]
        # tishi_language_feats = batch["tishi_language_feats"]
        # tishi_attention_mask = batch["tishi_attention_mask"]

        pred = self(pixel_values, input_ids, attention_mask, labels, labels_attention_mask)
        # pred = self(pixel_values, tishi_language_feats, tishi_attention_mask, labels, labels_attention_mask)

        self.valid_acc(pred, answer)
        valid_loss = self.loss(pred, answer)

        self.log("valid_loss", valid_loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=self.batch_size)
        self.log("valid_acc", self.valid_acc, on_epoch=True, on_step=False, sync_dist=True, batch_size=self.batch_size)

        pred_arg = torch.argmax(pred, axis=1)
        for i in range(pred.shape[0]):
            if pred_arg[i] == answer[i]:
                self.validation_step_outputs.append([1, question_type[i]])
            else:
                self.validation_step_outputs.append([0, question_type[i]])

    def on_validation_epoch_end(self):
        outputs = np.stack(self.validation_step_outputs)

        total_rural_urban, total_presence, total_count, total_comp = 0, 0, 0, 0
        right_rural_urban, right_presence, right_count, right_comp = 0, 0, 0, 0
        acc_rural_urban, acc_presence, acc_count, acc_comp = 0, 0, 0, 0
        AA, OA, right, total = 0, 0, 0, 0

        for i in range(outputs.shape[0]):
            if outputs[i][1] == 'comp':
                total_comp += 1
                if outputs[i][0] == '1':
                    right_comp += 1
            elif outputs[i][1] == 'presence':
                total_presence += 1
                if outputs[i][0] == '1':
                    right_presence += 1
            elif outputs[i][1] == 'count':
                total_count += 1
                if outputs[i][0] == '1':
                    right_count += 1
            else:
                total_rural_urban += 1
                if outputs[i][0] == '1':
                    right_rural_urban += 1

        # Note that for RSVQA_HR, there's no 'rural_urban' question type 
        # so 'rural_urban' in RSVQA_HR represent for 'area' question type
        acc_rural_urban = right_rural_urban / total_rural_urban
        acc_presence = right_presence / total_presence
        acc_count = right_count / total_count
        acc_comp = right_comp / total_comp

        right = right_rural_urban + right_presence + right_count + right_comp
        total = total_rural_urban + total_presence + total_count + total_comp

        AA = (acc_rural_urban + acc_presence + acc_count + acc_comp) / 4
        OA = right / total

        self.log("acc_rural_urban", acc_rural_urban, sync_dist=True)
        self.log("acc_presence", acc_presence, sync_dist=True)
        self.log("acc_count", acc_count, sync_dist=True)
        self.log("acc_comp", acc_comp, sync_dist=True)
        # self.log("total", total, sync_dist=True)
        self.log('valid_AA', AA, sync_dist=True)
        self.log('valid_OA', OA, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # Unpack the batch data
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # tishi_language_feats = batch["tishi_language_feats"]
        answer = batch["answer"]
        questions = batch["question"]

        # print(questions[359])
        # exit()
        labels_attention_mask = batch["label_attention_mask"]
        question_type = batch['question_type']

        pred = self(pixel_values, input_ids, attention_mask, labels, labels_attention_mask)
        # pred = self(pixel_values, tishi_language_feats, attention_mask, labels, labels_attention_mask)

        # Compute the test loss
        test_loss = self.loss(pred, answer)

        # Compute the test accuracy
        test_acc = (pred.argmax(dim=1) == answer).float().mean()

        # Compute accuracy for each question type (e.g., yes/no)
        question_type_acc = {}
        for q_type in set(question_type):  # Iterate over unique question types
            indices = [i for i, q in enumerate(question_type) if q == q_type]
            if indices:  # If there are any examples for this question type
                q_type_preds = pred[indices].argmax(dim=1)
                q_type_answers = answer[indices]
                q_type_acc = (q_type_preds == q_type_answers).float().mean().item()
                question_type_acc[q_type] = round(q_type_acc * 100, 2)

        # Log the overall metrics
        self.log("test_loss", test_loss, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test_acc", test_acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # Log per-question-type accuracy (as numeric values, not formatted strings)
        for q_type, acc in question_type_acc.items():
            self.log(f"test_acc_{q_type}", acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # Log the formatted accuracy (for display purposes)
        for q_type, acc in question_type_acc.items():
            self.log(f"{q_type}_accuracy", acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # Return predictions and metrics
        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "question_type_acc": question_type_acc,
            "predictions": pred,
            "answers": answer,
        }
# 'Is it a rural or an urban area', 'Is there a grass area?', 'What is the number of roads?', 'Is there a road?', 'Is a large road present?', 'Are there less buildings than farmlands?', 'Is a residential building present?', 'Are there more commercial buildings than roads?', 'Is a forest present in the image?', 'What is the amount of farmlands?', 'How many grass areas are there?', 'What is the amount of residential buildings?', 'What is the number of circular commercial buildings in the image?', 'What is the number of buildings?', 'What is the number of water areas in the image?', 'Are there more residential buildings than water areas?', 'Are there more water areas than small commercial buildings?', 'Is a water area present?', 'Is a commercial building present in the image?', 'Are there less commercial buildings than water areas?', 'Is the number of residential buildings equal to the number of grass areas in the image?', 'Is a parking present?', 'Are there more small roads than residential buildings?', 'What is the number of rectangular commercial buildings?', 'Is there a medium building?', 'Is a rectangular road present?', 'Is the number of water areas equal to the number of residential buildings?', 'Is there a circular road?', 'Is there a building?', 'What is the amount of square buildings?', 'Are there more forests than residential buildings?', 'What is the number of orchards?', 'What is the amount of commercial buildings?', 'Is a square building present?', 'How many small residential buildings are there?', 'What is the amount of square water areas?', 'Is a small commercial building present?', 'How many small farmlands are there in the image?', 'Is a farmland next to a  commercial building present?', 'Are there less roads than buildings in the image?', 'Is there a small water area?', 'Is there a circular building?', 'Is there a farmland in the image?', 'Is there a circular water area?', 'Is a commercial building on the right of a  water area present?', 'Are there more roads than residential buildings?', 'Is a medium farmland present?', 'What is the number of circular roads?', 'Is the number of commercial buildings equal to the number of grass areas?', 'Is the number of roads equal to the number of commercial buildings?', 'How many forests are there?', 'How many pitchs are there?', 'What is the number of residential areas?', 'Are there more water areas than commercial buildings?', 'How many medium roads are there?', 'Is the number of grass areas equal to the number of water areas?', 'Is a circular residential building present?', 'Is there a square road?', 'How many small roads are there?', 'What is the amount of square grass areas?', 'Are there more rectangular residential buildings than place of worships?', 'Is there a medium water area?', 'Is the number of residential buildings equal to the number of water areas in the image?', 'Is there a large water area?']
# ['Is it a rural or an urban area', 'Is there a grass area?', 'What is the number of roads?', 'Is there a road?', 'Is a large road present?', 'Are there less buildings than farmlands?', 'Is a residential building present?', 'Are there more commercial buildings than roads?', 'Is a forest present in the image?', 'What is the amount of farmlands?', 'How many grass areas are there?', 'What is the amount of residential buildings?', 'What is the number of circular commercial buildings in the image?', 'What is the number of buildings?', 'What is the number of water areas in the image?', 'Are there more residential buildings than water areas?', 'Are there more water areas than small commercial buildings?', 'Is a water area present?', 'Is a commercial building present in the image?', 'Are there less commercial buildings than water areas?', 'Is the number of residential buildings equal to the number of grass areas in the image?', 'Is a parking present?', 'Are there more small roads than residential buildings?', 'What is the number of rectangular commercial buildings?', 'Is there a medium building?', 'Is a rectangular road present?', 'Is the number of water areas equal to the number of residential buildings?', 'Is there a circular road?', 'Is there a building?', 'What is the amount of square buildings?', 'Are there more forests than residential buildings?', 'What is the number of orchards?', 'What is the amount of commercial buildings?', 'Is a square building present?', 'How many small residential buildings are there?', 'What is the amount of square water areas?', 'Is a small commercial building present?', 'How many small farmlands are there in the image?', 'Is a farmland next to a  commercial building present?', 'Are there less roads than buildings in the image?', 'Is there a small water area?', 'Is there a circular building?', 'Is there a farmland in the image?', 'Is there a circular water area?', 'Is a commercial building on the right of a  water area present?', 'Are there more roads than residential buildings?', 'Is a medium farmland present?', 'What is the number of circular roads?', 'Is the number of commercial buildings equal to the number of grass areas?', 'Is the number of roads equal to the number of commercial buildings?', 'How many forests are there?', 'How many pitchs are there?', 'What is the number of residential areas?', 'Are there more water areas than commercial buildings?', 'How many medium roads are there?', 'Is the number of grass areas equal to the number of water areas?', 'Is a circular residential building present?', 'Is there a square road?', 'How many small roads are there?', 'What is the amount of square grass areas?', 'Are there more rectangular residential buildings than place of worships?', 'Is there a medium water area?', 'Is the number of residential buildings equal to the number of water areas in the image?', 'Is there a large water area?']
