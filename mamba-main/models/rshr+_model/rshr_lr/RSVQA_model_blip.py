import sys
import torch
import numpy as np
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from EnhancedBlock_en_EAT import RSE
from models.t.trans.models.blip.modeling_blip_ori_text_image_conv_attn_mid_rsf_rshr_1o_lr import \
    BlipForQuestionAnswering
from rshr.In2Mask import In2

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
torch.set_float32_matmul_precision('high')
path = ''


class VQAModel(pl.LightningModule):
    def __init__(self, batch_size=None, lr=None, number_outputs=None,
                 # 新增：用于传递测试数据加载器参数的字典
                 test_loader_params=None):
        super(VQAModel, self).__init__()
        self.save_hyperparameters(ignore=['test_loader_params'])

        # 从 hparams 获取参数，如果它们被保存了
        self.number_outputs = self.hparams.get("number_outputs", number_outputs)
        self.lr = self.hparams.get("lr", lr)
        self.batch_size = self.hparams.get("batch_size", batch_size)

        self.loss_fn = F.cross_entropy  # 之前用 self.loss，改为 self.loss_fn 避免与 pl.LightningModule.loss 冲突
        self.validation_step_outputs = []
        self.test_step_outputs = []  # 新增，用于收集测试步骤的输出

        # torchmetrics 实例
        self.train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_outputs)
        self.valid_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_outputs)
        self.test_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_outputs)

        # 存储测试加载器参数以供 test_dataloader() 使用
        # 警告：如果 params 包含 tokenizer/processor 实例，保存/加载 checkpoint 可能出问题
        self._test_loader_params = test_loader_params
#
        # 模型组件
        self.blip = BlipForQuestionAnswering.from_pretrained("/home/pengfei/blip-vqa-capfilt-large")
        # self.blip = BlipForQuestionAnswering.from_pretrained("/home/pengfei/blip-vqa-base")
        self.in2 = In2()
        self.RSE_text = RSE(768, 12, mid_hidden_dim=128)
        for name, param in self.blip.named_parameters():
            if 'adapter' not in name and \
                    'mona' not in name and \
                    'text_encoder.' not in name and \
                    'mamba2' not in name and \
                    'RSE_text' not in name and \
                    'classify_layer' not in name:
                param.requires_grad = False

        # self.classify_layer = nn.Sequential(
        #     nn.Linear(768, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.GELU(),
        #     nn.Linear(256, self.number_outputs)
        # )
        self.classify_layer = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.number_outputs)
        )


    def forward(self, pixel_values, input_ids, attention_mask, labels, labels_attention_mask):
        exit()
        out = self.blip(pixel_values=pixel_values, input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels, labels_attention_mask=labels_attention_mask)
        question_embedding = torch.squeeze(out['last_hidden_state_vero'])
        result = self.RSE_text(question_embedding)
        res_processed = self.in2(result, mask=attention_mask)
        logits = self.classify_layer(res_processed)
        return logits
    #
    def configure_optimizers(self):
        def rule(epoch):
            if self.number_outputs == 9:
                if epoch <= 3:
                    lamda = 1
                elif epoch <= 5:
                    lamda = 0.5
                else:
                    lamda = 0.1
            else:
                if epoch <= 3:
                    lamda = 1 + epoch
                elif epoch <= 6:
                    lamda = 0.08
                else:
                    lamda = 0.01
            return lamda

    # def configure_optimizers(self):
    #     def rule(epoch):
    #         if self.number_outputs == 9:
    #             if epoch <= 3:
    #                 lamda = 1
    #             elif epoch <= 5:
    #                 lamda = 0.5
    #             elif epoch <= 8: #9->8
    #                 lamda = 0.1
    #             else:
    #                 lamda = 0.05
    #         else:
    #             if epoch <= 3:
    #                 lamda = 1 + epoch
    #             elif epoch <= 6:
    #                 lamda = 0.08
    #             else:
    #                 lamda = 0.01
    #         return lamda

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=rule)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def _shared_step(self, batch, batch_idx, stage_prefix):  # stage_prefix 现在主要用于区分指标更新
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels_for_blip = batch["labels"]
        labels_attention_mask = batch["label_attention_mask"]
        true_answers = batch["answer"]

        pred_logits = self(pixel_values, input_ids, attention_mask, labels_for_blip, labels_attention_mask)
        loss = self.loss_fn(pred_logits, true_answers)

        # 更新相应的准确率指标
        if stage_prefix == "train":
            self.train_acc_metric.update(pred_logits, true_answers)
        elif stage_prefix == "valid":
            self.valid_acc_metric.update(pred_logits, true_answers)
        elif stage_prefix == "test" or stage_prefix.startswith("test_manual"):  # 捕获手动测试的前缀
            self.test_acc_metric.update(pred_logits, true_answers)

        # 不再在这里 log on_step=True 的指标
        # self.log(f"{stage_prefix}_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        return loss, pred_logits, true_answers

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "train")
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=self.batch_size,
                 sync_dist=True)  # 在这里log step loss
        self.log("train_acc_step", self.train_acc_metric, on_step=True, on_epoch=False, prog_bar=True,
                 batch_size=self.batch_size, sync_dist=True)  # 在这里log step acc

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=self.batch_size,
                 sync_dist=True)
        self.log("train_acc", self.train_acc_metric, on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_logits, true_answers = self._shared_step(batch, batch_idx, "valid")
        # 不需要在这里 log valid_loss_step，因为 on_validation_epoch_end 是 epoch 级别
        # self.log("valid_loss_step", loss, on_step=True, ...) # 这行也是错的，如果有的话

        # epoch 级别的日志 (这些会累积)
        self.log("valid_loss_epoch", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=self.batch_size,
                 sync_dist=True)
        self.log("valid_acc_epoch", self.valid_acc_metric, on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=self.batch_size, sync_dist=True)

        pred_arg = torch.argmax(pred_logits, axis=1)
        question_type = batch["question_type"]
        for i in range(pred_logits.shape[0]):
            is_correct = 1 if pred_arg[i] == true_answers[i] else 0
            self.validation_step_outputs.append([is_correct, question_type[i]])
        return loss

    def test_step(self, batch, batch_idx):  # 这个是 trainer.test() 调用的
        loss, pred_logits, true_answers = self._shared_step(batch, batch_idx, "test")
        # epoch 级别的日志
        self.log("test_loss_epoch", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=self.batch_size,
                 sync_dist=True)
        self.log("test_acc_epoch_overall", self.test_acc_metric, on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=self.batch_size, sync_dist=True)

        pred_arg = torch.argmax(pred_logits, axis=1)
        question_type = batch["question_type"]
        for i in range(pred_logits.shape[0]):
            is_correct = 1 if pred_arg[i] == true_answers[i] else 0
            self.test_step_outputs.append([is_correct, question_type[i]])
        return loss

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

        # --- 打印验证结果总结到控制台 ---
        print(f"\nEpoch {self.current_epoch + 1} VALIDATION SUMMARY:")
        print(f"  Overall Accuracy (OA): {OA:.4f} ")
        print(f"  Average Accuracy (AA): {AA:.4f}")
        print(
            f"  Rural/Urban Acc: {acc_rural_urban:.4f} ({right_rural_urban}/{total_rural_urban if total_rural_urban > 0 else 'N/A'})")
        print(
            f"  Presence Acc: {acc_presence:.4f} ({right_presence}/{total_presence if total_presence > 0 else 'N/A'})")
        print(f"  Count Acc: {acc_count:.4f} ({right_count}/{total_count if total_count > 0 else 'N/A'})")
        print(f"  Comparison Acc: {acc_comp:.4f} ({right_comp}/{total_comp if total_comp > 0 else 'N/A'})")
        print(f"--------------------------------------")  # 分隔符

        self.validation_step_outputs.clear()
        # ... (之前的验证指标计算和日志) ...
        current_val_acc = self.valid_acc_metric.compute()  # 确保这个在正确的地方
        self.log("valid_acc", current_val_acc, prog_bar=True, sync_dist=True)
        # ...

        # --- 手动执行测试逻辑 ---
        if self.trainer.is_global_zero:
            print(f"--- Epoch {self.current_epoch + 1} - Manually Running In-Epoch Test (Observation Only) ---")

        test_loader = self.test_dataloader()
        if test_loader:
            self.eval()
            self.test_acc_metric.reset()  # 重置测试指标
            self.test_step_outputs.clear()
            original_device = self.device
            use_amp_for_test = isinstance(self.trainer.precision,
                                          str) and 'mixed' in self.trainer.precision or self.trainer.precision == 16

            # 用于累积手动测试的损失
            manual_test_epoch_losses = []

            with torch.no_grad():
                autocast_context = torch.cuda.amp.autocast(enabled=use_amp_for_test and original_device.type == 'cuda')
                with autocast_context:
                    for batch_idx, batch in enumerate(test_loader):
                        batch = self.trainer.strategy.batch_to_device(batch, device=original_device)

                        # 调用 _shared_step，它现在不记录 on_step=True
                        loss, pred_logits, true_answers = self._shared_step(batch, batch_idx,
                                                                            "test_manual")  # 使用 "test_manual" 前缀

                        manual_test_epoch_losses.append(loss)  # 收集损失

                        pred_arg_test = torch.argmax(pred_logits, axis=1)
                        question_type_test = batch["question_type"]
                        for i in range(pred_logits.shape[0]):
                            is_correct_test = 1 if pred_arg_test[i] == true_answers[i] else 0
                            self.test_step_outputs.append([is_correct_test, question_type_test[i]])

            # 计算手动测试的平均损失 (可选)
            if manual_test_epoch_losses:
                avg_manual_test_loss = torch.stack(manual_test_epoch_losses).mean()
                self.log("in_epoch_test_loss", avg_manual_test_loss, prog_bar=False, sync_dist=True)  # 记录 epoch 级损失

            self.manual_on_test_epoch_end(prefix_for_log="in_epoch_test")

    def manual_on_test_epoch_end(self, prefix_for_log="test"):
        # 这个方法复制 on_test_epoch_end 的逻辑，但允许自定义日志前缀
        # 以避免与最终的 trainer.test() 的日志记录冲突
        if not self.test_step_outputs:
            if self.trainer.is_global_zero:
                print(f"No outputs in test_step_outputs for manual run. Skipping {prefix_for_log} summary.")
            self.log(f"{prefix_for_log}_acc", torch.tensor(0.0, device=self.device), prog_bar=False, sync_dist=True)
            # Log other metrics as 0
            return

        outputs = np.array(self.test_step_outputs, dtype=object)
        self.test_step_outputs.clear()  # 清空

        # ... (复制 on_test_epoch_end 中计算 OA, AA, 各类型准确率的逻辑) ...
        total_rural_urban, total_presence, total_count, total_comp = 0, 0, 0, 0
        right_rural_urban, right_presence, right_count, right_comp = 0, 0, 0, 0
        for i in range(outputs.shape[0]):
            is_correct = int(outputs[i, 0])
            q_type = outputs[i, 1]
            if q_type == 'comp':
                total_comp += 1;
                right_comp += is_correct
            elif q_type == 'presence':
                total_presence += 1;
                right_presence += is_correct
            elif q_type == 'count':
                total_count += 1;
                right_count += is_correct
            else:
                total_rural_urban += 1;
                right_rural_urban += is_correct

        acc_rural_urban = (right_rural_urban / total_rural_urban) if total_rural_urban > 0 else 0.0
        # ... (计算其他 acc_type) ...
        acc_presence = (right_presence / total_presence) if total_presence > 0 else 0.0
        acc_count = (right_count / total_count) if total_count > 0 else 0.0
        acc_comp = (right_comp / total_comp) if total_comp > 0 else 0.0

        current_test_acc = self.test_acc_metric.compute()  # 这是 torchmetrics 的总体准确率

        # 使用自定义前缀记录日志
        self.log(f"{prefix_for_log}_overall_acc", current_test_acc, prog_bar=False, sync_dist=True)
        self.log(f"{prefix_for_log}_acc_rural_urban", acc_rural_urban, sync_dist=True)
        # ... (log 其他 acc_type) ...
        self.log(f"{prefix_for_log}_acc_presence", acc_presence, sync_dist=True)
        self.log(f"{prefix_for_log}_acc_count", acc_count, sync_dist=True)
        self.log(f"{prefix_for_log}_acc_comp", acc_comp, sync_dist=True)

        right_total = right_rural_urban + right_presence + right_count + right_comp
        total_all = total_rural_urban + total_presence + total_count + total_comp
        OA_test = (right_total / total_all) if total_all > 0 else 0.0

        num_valid_types_test = sum(1 for x in [total_rural_urban, total_presence, total_count, total_comp] if x > 0)
        AA_test = ((
                           acc_rural_urban + acc_presence + acc_count + acc_comp) / num_valid_types_test) if num_valid_types_test > 0 else 0.0

        self.log(f'{prefix_for_log}_AA', AA_test, sync_dist=True)
        self.log(f'{prefix_for_log}_OA', OA_test, sync_dist=True)

        # 打印结果
        epoch_info = f"Epoch {self.current_epoch + 1} ({prefix_for_log.replace('_', ' ').title()})"
        print(f"\n--- {epoch_info} SUMMARY ---")
        print(f"  Overall Accuracy (OA): {OA_test:.4f} (TorchMetrics: {current_test_acc:.4f})")
        print(f"  Average Accuracy (AA): {AA_test:.4f}")
        # 你可能还想打印损失，但这需要你在 _shared_step 中也收集测试损失
        # avg_test_loss = ...
        # print(f"  Loss: {avg_test_loss:.4f}")
        print(f"  Rural/Urban Acc: {acc_rural_urban:.4f} ({right_rural_urban}/{total_rural_urban})")
        print(f"  Presence Acc: {acc_presence:.4f} ({right_presence}/{total_presence})")
        print(f"  Count Acc: {acc_count:.4f} ({right_count}/{total_count})")
        print(f"  Comparison Acc: {acc_comp:.4f} ({right_comp}/{total_comp})")
        print(f"--------------------------------------")

    def test_dataloader(self):
        if self._test_loader_params is None:
            if self.trainer.is_global_zero:
                print(
                    "Warning: _test_loader_params not set in VQAModel. Cannot create test_dataloader for in-epoch testing.")
            return None

        # 动态导入 VQALoader，或者确保它在模块级别可用
        try:
            from VQALoader_TestLR import VQALoader
        except ImportError:
            if self.trainer.is_global_zero:
                print("Error: Could not import VQALoader in VQAModel.test_dataloader(). Ensure it's in PYTHONPATH.")
            return None

        # 从 self._test_loader_params 中获取必要的参数
        # 确保这些 key 与你在主脚本中创建 _test_loader_params 字典时使用的 key 一致
        images_path = self._test_loader_params.get('images_path')
        images_json = self._test_loader_params.get('images_json_test')  # 确保key正确
        questions_json = self._test_loader_params.get('questions_json_test')  # 确保key正确
        answers_json = self._test_loader_params.get('answers_json_test')  # 确保key正确

        # 处理 tokenizer 和 image_processor
        # 方案1: 直接使用传递的实例 (如之前代码所示，有序列化风险)
        tokenizer_instance = self._test_loader_params.get('tokenizer_instance')
        image_processor_instance = self._test_loader_params.get('image_processor_instance')

        # 方案2: 从路径重新加载 (更稳健)
        # processor_path = self._test_loader_params.get('processor_path')
        # if processor_path:
        #     from transformers import BlipProcessor # 确保导入
        #     try:
        #         reloaded_processor = BlipProcessor.from_pretrained(processor_path)
        #         tokenizer_instance = reloaded_processor.tokenizer
        #         image_processor_instance = reloaded_processor.image_processor
        #     except Exception as e:
        #         if self.trainer.is_global_zero:
        #             print(f"Error reloading processor in test_dataloader: {e}")
        #         return None
        # elif not tokenizer_instance or not image_processor_instance: # 如果路径和实例都没有
        #      if self.trainer.is_global_zero:
        #         print("Error: Missing tokenizer/image_processor info for test_dataloader.")
        #      return None

        dataset_name = self._test_loader_params.get('dataset_name')
        sequence_length = self._test_loader_params.get('sequence_length')
        num_workers = self._test_loader_params.get('num_workers', 0)  # 提供默认值
        selected_answers = self._test_loader_params.get('selected_answers')
        # batch_size 从 self.hparams 获取，因为它是模型级别的配置
        batch_size_for_test = self.hparams.get('batch_size', 32)

        # 确保所有必要参数都存在
        required_keys = ['images_path', 'images_json_test', 'questions_json_test', 'answers_json_test',
                         'tokenizer_instance', 'image_processor_instance', 'dataset_name',
                         'sequence_length', 'selected_answers']
        missing_keys = [key for key in required_keys if self._test_loader_params.get(key) is None]
        if missing_keys:
            if self.trainer.is_global_zero:
                print(f"Error: Missing required parameters to build test_dataloader in VQAModel: {missing_keys}")
            return None

        test_data = VQALoader(
            images_path, images_json, questions_json, answers_json,
            tokenizer=tokenizer_instance, image_processor=image_processor_instance,
            Dataset=dataset_name, train=False, sequence_length=sequence_length,
            ratio_images_to_use=1.0,
            selected_answers=selected_answers,
            transform=None  # 测试时通常不使用训练时的随机增强
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size_for_test, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=True if num_workers > 0 else False
        )
        if self.trainer.is_global_zero:
            print(
                f"VQAModel: Reconstructed test_dataloader for epoch {self.current_epoch + 1} with {len(test_loader)} batches.")
        return test_loader

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            if self.trainer.is_global_zero:
                print("No outputs in test_step_outputs. Skipping test summary.")
            # Log default test metrics if needed by any external system
            self.log("test_acc", torch.tensor(0.0, device=self.device), prog_bar=True, sync_dist=True)
            self.log("test_AA", 0.0, sync_dist=True)
            self.log("test_OA", 0.0, sync_dist=True)
            return

        # 与 on_validation_epoch_end 类似地计算 AA, OA 等指标
        outputs = np.array(self.test_step_outputs, dtype=object)
        self.test_step_outputs.clear()

        total_rural_urban, total_presence, total_count, total_comp = 0, 0, 0, 0
        right_rural_urban, right_presence, right_count, right_comp = 0, 0, 0, 0

        for i in range(outputs.shape[0]):
            is_correct = int(outputs[i, 0])
            q_type = outputs[i, 1]
            if q_type == 'comp':
                total_comp += 1
                if is_correct: right_comp += 1
            elif q_type == 'presence':
                total_presence += 1
                if is_correct: right_presence += 1
            elif q_type == 'count':
                total_count += 1
                if is_correct: right_count += 1
            else:  # Assuming 'rural_urban' or other default
                total_rural_urban += 1
                if is_correct: right_rural_urban += 1

        acc_rural_urban = (right_rural_urban / total_rural_urban) if total_rural_urban > 0 else 0.0
        acc_presence = (right_presence / total_presence) if total_presence > 0 else 0.0
        acc_count = (right_count / total_count) if total_count > 0 else 0.0
        acc_comp = (right_comp / total_comp) if total_comp > 0 else 0.0

        # Logged by test_step: self.log("test_acc_epoch_overall", self.test_acc_metric, ...)
        current_test_acc = self.test_acc_metric.compute()  # 获取 torchmetrics 计算的总体准确率
        self.log("test_acc", current_test_acc, prog_bar=True, sync_dist=True)  # 用于最终报告

        self.log("test_acc_rural_urban", acc_rural_urban, sync_dist=True)
        self.log("test_acc_presence", acc_presence, sync_dist=True)
        self.log("test_acc_count", acc_count, sync_dist=True)
        self.log("test_acc_comp", acc_comp, sync_dist=True)

        right_total = right_rural_urban + right_presence + right_count + right_comp
        total_all = total_rural_urban + total_presence + total_count + total_comp
        OA_test = (right_total / total_all) if total_all > 0 else 0.0

        num_valid_types_test = sum(1 for x in [total_rural_urban, total_presence, total_count, total_comp] if x > 0)
        AA_test = ((
                           acc_rural_urban + acc_presence + acc_count + acc_comp) / num_valid_types_test) if num_valid_types_test > 0 else 0.0

        self.log('test_AA', AA_test, sync_dist=True)
        self.log('test_OA', OA_test, sync_dist=True)  # 与 current_test_acc 应该一致或非常接近

        # 根据是 epoch 内测试还是最终测试，打印不同的前缀
        epoch_prefix = f"Epoch {self.current_epoch + 1} (In-Epoch) " \
            if self.trainer and self.trainer.state.stage == pl.trainer.states.TrainerFn.VALIDATING \
            else "Final "

        print(f"\n--- {epoch_prefix}TEST SUMMARY ---")
        print(f"  Overall Accuracy (OA): {OA_test:.4f} (TorchMetrics: {current_test_acc:.4f})")
        print(f"  Average Accuracy (AA): {AA_test:.4f}")
        print(f"  Loss: {self.trainer.callback_metrics.get('test_loss_epoch', torch.tensor(0.0)):.4f}")
        print(f"  Rural/Urban Acc: {acc_rural_urban:.4f} ({right_rural_urban}/{total_rural_urban})")
        print(f"  Presence Acc: {acc_presence:.4f} ({right_presence}/{total_presence})")
        print(f"  Count Acc: {acc_count:.4f} ({right_count}/{total_count})")
        print(f"  Comparison Acc: {acc_comp:.4f} ({right_comp}/{total_comp})")
        print(f"--------------------------------------")
