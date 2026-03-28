import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import random
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from transformers import BlipImageProcessor, BlipProcessor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from models.augment.aug_lr import AutoAugment

from RSVQA_model_blip import VQAModel
from VQALoader_TestLR import VQALoader

SEED = 42

# todo:512 384

def main(num_workers: int = 10,
         ratio_images_to_use: float = 1,
         sequence_length: int = 40,
         num_epochs: int = 30,
         batch_size: int = 62,
         lr: float = 3e-5,
         Dataset='LR'):

    random.seed(SEED)
    torch.manual_seed(SEED)
    pl.seed_everything(SEED, workers=True)

    experiment_name = f"dataset_{Dataset}_lr{lr}_bs{batch_size}_epochs{num_epochs}_01"
    base_results_dir = "./lightning_runs_lr_3e5_ori_def_prosser_optim_5_ep8_pros_def_layernorm_b64_mid_512_eat512_topk5__textencoder_maskin2_r128_dr05_enhanceall_out58"

    dir_path = os.path.join(base_results_dir, experiment_name)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Results will be saved to: {dir_path}")

    data_path = '/home/pengfei/DataSet/RSVQA_LR'
    LR_questionsJSON = os.path.join(data_path, 'LR_split_train_questions.json')
    LR_answersJSON = os.path.join(data_path, 'LR_split_train_answers.json')
    LR_imagesJSON = os.path.join(data_path, 'LR_split_train_images.json')
    LR_questionsvalJSON = os.path.join(data_path, 'LR_split_val_questions.json')
    LR_answersvalJSON = os.path.join(data_path, 'LR_split_val_answers.json')
    LR_imagesvalJSON = os.path.join(data_path, 'LR_split_val_images.json')
    LR_questionstestJSON = os.path.join(data_path, 'LR_split_test_questions.json')
    LR_answerstestJSON = os.path.join(data_path, 'LR_split_test_answers.json')
    LR_imagestestJSON = os.path.join(data_path, 'LR_split_test_images.json')
    LR_images_path = os.path.join(data_path, 'Images_LR/')

    image_processor_instance = BlipImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225],
                                                  image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True,
                                                  size=384, size_divisor=32)
    blip_processor_path = "/home/pengfei/blip-vqa-capfilt-large"
    # blip_processor_path = "/home/pengfei/blip-vqa-base"
    try:
        processor = BlipProcessor.from_pretrained(blip_processor_path)
        # image_processor_instance = processor.image_processor
        tokenizer_instance = processor.tokenizer
    except Exception as e:
        print(f"Error loading BlipProcessor from '{blip_processor_path}': {e}")
        return

    transform_train_list = [
        transforms.RandomHorizontalFlip(),
    ]
    transform_train_list.append(AutoAugment())
    transform_train = transforms.Compose(transform_train_list)

    LR_data_train = VQALoader(
        LR_images_path, LR_imagesJSON, LR_questionsJSON, LR_answersJSON,
        tokenizer=tokenizer_instance, image_processor=image_processor_instance,
        Dataset=Dataset, train=True, sequence_length=sequence_length,
        ratio_images_to_use=ratio_images_to_use,
        transform=transform_train  # 使用你定义的 transform_train
    )
    LR_train_loader = torch.utils.data.DataLoader(
        LR_data_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    selected_answers_map = LR_data_train.selected_answers  # 获取 selected_answers

    LR_data_val = VQALoader(
        LR_images_path, LR_imagesvalJSON, LR_questionsvalJSON, LR_answersvalJSON,
        tokenizer=tokenizer_instance, image_processor=image_processor_instance,
        Dataset=Dataset, train=False, sequence_length=sequence_length,
        ratio_images_to_use=1.0,
        selected_answers=selected_answers_map,
        transform=None
    )
    LR_val_loader = torch.utils.data.DataLoader(
        LR_data_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )

    # 测试数据加载器参数 (用于传递给 VQAModel)
    # 键名需要与 VQAModel.test_dataloader() 中期望的一致
    test_loader_construction_params = {
        'images_path': LR_images_path,
        'images_json_test': LR_imagestestJSON,
        'questions_json_test': LR_questionstestJSON,
        'answers_json_test': LR_answerstestJSON,
        # 方案1: 直接传递实例 (如之前代码，有序列化风险)
        'tokenizer_instance': tokenizer_instance,
        'image_processor_instance': image_processor_instance,
        # 方案2: 传递路径 (更稳健, VQAModel.test_dataloader 需要相应修改以重新加载)
        # 'processor_path': blip_processor_path,
        'dataset_name': Dataset,  # 使用你原来的参数名
        'sequence_length': sequence_length,
        'num_workers': num_workers,
        'selected_answers': selected_answers_map,
        # batch_size 可以从模型的 hparams.batch_size 获取
    }
    # 注意：我们不再在这里创建 LR_test_loader 实例用于 trainer.test_dataloaders 属性

    # --- 3. 模型实例化 ---
    # 根据 Dataset 确定输出数量
    num_outputs = 9 if Dataset == 'LR' else 98

    # 实例化 VQAModel，并传递 test_loader_params
    model = VQAModel(
        batch_size=batch_size,
        lr=lr,
        number_outputs=num_outputs,
        test_loader_params=test_loader_construction_params  # 传递参数字典
    )

    # --- 4. 回调函数和 Logger ---
    # ModelCheckpoint: 确保 VQAModel log 了 "valid_acc" 以及文件名中用到的其他指标
    # 你的 VQAModel 在 on_validation_epoch_end 中记录了 "valid_acc", "acc_count", "acc_rural_urban", "acc_presence"
    # checkpoint_callback = ModelCheckpoint(
    #     save_top_k=1,
    #     monitor="valid_acc",  # VQAModel 必须 log 这个指标
    #     save_weights_only=False,  # 改为 False 以保存完整模型状态
    #     mode="max",
    #     dirpath=dir_path,  # 使用你定义的 dir_path
    #     filename=f"{{epoch}}_{{valid_acc:.5f}}_{{acc_count:.5f}}_{{acc_rural_urban:.5f}}_{{acc_presence:.5f}}"
    #     # 你原来的文件名格式
    # )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,  # -1 通常表示保存所有满足 every_n_epochs 的
        save_on_train_epoch_end=True,
        monitor="valid_acc",  # VQAModel 必须 log 这个指标
        save_weights_only=True,  # 改为 False 以保存完整模型状态
        mode="max",
        dirpath=dir_path,  # 使用你定义的 dir_path
        filename=f"{{epoch}}_{{valid_acc:.5f}}_{{acc_count:.5f}}_{{acc_rural_urban:.5f}}_{{acc_presence:.5f}}"
        # 你原来的文件名格式
    )

    early_stopping = EarlyStopping(
        monitor="valid_acc",
        patience=20,
        mode="max",
        verbose=True
    )


    lr_monitor = LearningRateMonitor(logging_interval='epoch')


    tb_logger = TensorBoardLogger(
        save_dir=base_results_dir,
        name=os.path.basename(dir_path)
    )

    # --- 5. Trainer 配置和运行 ---
    trainer = pl.Trainer(
        devices=1,  # 假设使用单个 GPU，你可以根据 os.environ['CUDA_VISIBLE_DEVICES'] 调整
        accelerator='cuda',
        fast_dev_run=False,  # 设置为 True 进行快速调试
        precision='16-mixed',
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=tb_logger,  # 使用配置的 logger
        # strategy='ddp_find_unused_parameters_true', # 如果使用 DDP
    )

    # 移除: trainer.test_dataloaders = LR_test_loader (不再需要)
    # 你原来的代码中有 LR_test_loader 的定义，但没有用在 trainer.fit 或 trainer.test_dataloaders
    # LR_data_test = VQALoader(...)
    # LR_test_loader = torch.utils.data.DataLoader(LR_data_test, ...)
    # 这部分现在由 VQAModel.test_dataloader() 处理，所以不需要在这里创建 LR_test_loader 实例给 Trainer

    print("--- Starting Training ---")
    trainer.fit(model, train_dataloaders=LR_train_loader, val_dataloaders=LR_val_loader)

    print("--- Training Finished ---")
    best_model_path = checkpoint_callback.best_model_path
    best_model_score = checkpoint_callback.best_model_score  # 获取最佳分数
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")
        if best_model_score is not None:  # 检查是否有分数
            print(f"Best validation score (valid_acc): {best_model_score:.5f}")  # 使用与文件名相同的格式
    else:
        print("No best model was saved by ModelCheckpoint.")

    # --- 6. 最终测试 ---
    # 使用 Trainer.test() 测试最佳模型
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n--- Performing FINAL test on the BEST model from: {best_model_path} ---")
        # Trainer 会自动调用 VQAModel.test_dataloader()
        # Trainer 会自动加载 ckpt_path="best" 指定的最佳模型
        final_test_results = trainer.test(ckpt_path="best", verbose=True)
        if final_test_results:
            print("Final test results (on best validation checkpoint):")
            # final_test_results 是一个包含字典的列表，每个字典对应一个测试数据加载器（通常只有一个）
            for res_dict in final_test_results:
                for key, value in res_dict.items():
                    # 尝试格式化浮点数，其他类型直接打印
                    if isinstance(value, float) or (hasattr(value, 'item') and isinstance(value.item(), float)):
                        val_to_print = value.item() if hasattr(value, 'item') else value
                        print(f"  {key}: {val_to_print:.4f}")
                    else:
                        print(f"  {key}: {value}")
    else:
        print("\n--- Skipping final test: Best model checkpoint not found or not saved. ---")
        print("If training completed, consider testing the last saved model state (if any) or an explicit checkpoint.")

    # 你原来的 trainer.test(model,LR_val_loader) 是用验证集测试，这里改为用测试集测试最佳模型
    print("\n--- Experiment Run Finished ---")


if __name__ == "__main__":
    main()
