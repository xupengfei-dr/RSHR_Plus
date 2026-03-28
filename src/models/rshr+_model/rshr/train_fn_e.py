import os
import copy
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from models.augment.aug_lr import AutoAugment
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import BlipImageProcessor, BlipProcessor
from FNRSVQA_model_blip import VQAModel
from VQALoader_FOOLDNET import VQALoader
SEED = 42
dir_path = './blip_no_adapter_3e5_b3_img_384_conv_mid_512_XTori_res_eatmid_384_t5_out_'

import pytorch_lightning as pl
import torch

# todo:512 256 384   TOP ALL
class EveryEpochTestCallback(pl.Callback):
    """在每个训练周期结束后运行测试集的回调。"""

    def __init__(self, test_loader):
        super().__init__()
        self.test_loader = test_loader

    def on_train_epoch_end(self, trainer, pl_module):
        print("\n" + "=" * 50)
        print(f"  RUNNING TEST AT THE END OF TRAIN EPOCH {trainer.current_epoch}  ")
        print("=" * 50 + "\n")
        model_copy = copy.deepcopy(pl_module)
        temp_trainer = pl.Trainer(
            devices=trainer.device_ids,
            accelerator=trainer.accelerator,
            precision=trainer.precision,  # 保持混合精度测试
            logger=False,
            callbacks=[],
            enable_progress_bar=True,
        )
        temp_trainer.test(model=model_copy, dataloaders=self.test_loader, verbose=True)

# ==============================================================================
# 主训练函数
# ==============================================================================
def main(num_workers: int = 10,
         ratio_images_to_use: float = 1,
         sequence_length: int = 40,
         num_epochs: int = 30,
         batch_size: int = 10,
         lr: float = 3e-5,
         Dataset='LR'):
    # --- 设置随机种子 ---
    random.seed(SEED)
    torch.manual_seed(SEED)
    pl.seed_everything(SEED, workers=True)

    # --- 初始化处理器和分词器 ---
    image_processor_instance = BlipImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225],
                                                  image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True,
                                                  size=384, size_divisor=32)
    blip_processor_path = "/home/pengfei/blip-vqa-capfilt-large"
    try:
        processor = BlipProcessor.from_pretrained(blip_processor_path)
        # image_processor_instance = processor.image_processor
        tokenizer_instance = processor.tokenizer
    except Exception as e:
        print(f"Error loading BlipProcessor from '{blip_processor_path}': {e}")
        return

    # --- 定义数据增强 ---
    transform_train_list = [
        transforms.RandomHorizontalFlip(),
        AutoAugment(),
    ]
    transform_train = transforms.Compose(transform_train_list)

    DATA_ROOT = '/home/pengfei/DataSet/FLOODNET/data'
    # IMG_ROOT = '/home/pengfei/FLOODNET/Original_Image_Final_Reshape'
    IMG_ROOT = '/home/pengfei/DataSet/FLOODNET/result_all'
    TRAIN_JSON_PATH = os.path.join(DATA_ROOT, 'train_annotations.json')
    VAL_JSON_PATH = os.path.join(DATA_ROOT, 'valid_annotations.json')
    TEST_JSON_PATH = os.path.join(DATA_ROOT, 'test_annotations.json')
    ANSWER_SPACE_PATH = os.path.join(DATA_ROOT, 'class_to_label.json')

    # --- 创建训练 DataLoader ---
    FN_train_dataset = VQALoader(
        json_path=TRAIN_JSON_PATH,
        img_folder_path=IMG_ROOT,
        img_transform=transform_train,
        answer_path=ANSWER_SPACE_PATH,
        image_processor=image_processor_instance,
        tokenizer=tokenizer_instance,
        is_train=True
    )
    LR_train_loader = torch.utils.data.DataLoader(
        FN_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )

    # --- 创建验证 DataLoader ---
    FN_val_dataset = VQALoader(
        json_path=VAL_JSON_PATH,
        img_folder_path=IMG_ROOT,
        img_transform=None,  # 验证集不应使用随机数据增强
        answer_path=ANSWER_SPACE_PATH,
        image_processor=image_processor_instance,
        tokenizer=tokenizer_instance
    )
    LR_val_loader = torch.utils.data.DataLoader(
        FN_val_dataset, batch_size=batch_size, shuffle=False,  # <<< 修正：验证集不打乱
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )

    # --- 创建测试 DataLoader (核心修改点) ---
    FN_test_dataset = VQALoader(
        json_path=TEST_JSON_PATH,
        img_folder_path=IMG_ROOT,
        img_transform=None,  # 测试集不使用数据增强
        answer_path=ANSWER_SPACE_PATH,
        image_processor=image_processor_instance,
        tokenizer=tokenizer_instance
    )
    LR_test_loader = torch.utils.data.DataLoader(
        FN_test_dataset, batch_size=batch_size, shuffle=False,  # 测试集不打乱
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )

    # --- 实例化模型 ---
    num_outputs = 51
    model = VQAModel(
        batch_size=batch_size,
        lr=lr,
        number_outputs=num_outputs,
    )

    # --- 配置回调函数 --- every_n_epochs=1,  # 每个 epoch 都触发
    #         save_top_k=-1,  # -1 通常表示保存所有满足 every_n_epochs 的
    #         save_on_train_epoch_end=True,
    #         monitor="valid_acc",  # VQAModel 必须 log 这个指标
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_OA",
        mode="max",
        dirpath=dir_path,
        filename=f"best-checkpoint-{{epoch}}-{{valid_OA:.4f}}",
    )
    early_stopping = EarlyStopping(
        monitor="valid_OA",  # 与 checkpoint 保持一致
        patience=30,  # 可以适当调整
        mode="max",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # --- 实例化自定义回调 (核心修改点) ---
    test_on_epoch_end_callback = EveryEpochTestCallback(LR_test_loader)
    # test_on_epoch_end_callback = TestAfterEpochCallback(LR_test_loader)

    # --- 配置 Logger ---
    tb_logger = TensorBoardLogger("lightning_logs", name=dir_path.split('/')[-2] or "VQA_run")

    # --- 实例化并配置 Trainer ---
    trainer = pl.Trainer(
        devices=1,
        accelerator='cuda',
        fast_dev_run=False,
        precision='16-mixed',
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            test_on_epoch_end_callback  # <<<< 添加我们的回调
        ],
    )

    print("--- Starting Training (with testing after each epoch) ---")
    trainer.fit(model, train_dataloaders=LR_train_loader, val_dataloaders=LR_val_loader)

    print("\n--- Training finished. Final testing with the best model. ---")
    # 训练结束后，使用最好的 checkpoint 进行最终测试
    trainer.test(model=model, dataloaders=LR_test_loader, ckpt_path='best')


if __name__ == "__main__":
    main()