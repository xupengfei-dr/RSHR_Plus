import os
import random
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
TRAIN_HR = False
import torch

random.seed(42)
# dir_path = './blip_hr_3e6_batch_64_ep3_conv_attn_image384_mid384_eatnone_batchnorm_out06_hr2'
dir_path = './blip_hr_4e6_batch_64_ep3_conv_attn_image384_mid384_eat384_QRTSNone_LayerNorm_hr2_out20'
torch.manual_seed(42)
import typer

import pytorch_lightning as pl
import torchvision.transforms as transforms
from models.augment.auto_augment import AutoAugment
from transformers import BlipProcessor, BlipImageProcessor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from RSVQA_model_blip import VQAModel
from VQALoader_HR import VQALoader

def main(num_workers: int = 10,
         ratio_images_to_use: int = 1,
         sequence_length: int = 40,
         num_epochs: int = 11,
         batch_size: int = 64,
         lr: float = 4e-6,
         # lr: float = 4e-6,
         # lr: float = 2e-6,
         Dataset='HR'):
    data_path = '/home/pengfei/DataSet/RSVQA_HR'

    HR_questionsJSON = os.path.join(data_path, 'USGS_split_train_questions.json')
    HR_answersJSON = os.path.join(data_path, 'USGS_split_train_answers.json')
    HR_imagesJSON = os.path.join(data_path, 'USGS_split_train_images.json')

    HR_questionsvalJSON = os.path.join(data_path, 'USGS_split_val_questions.json')
    HR_answersvalJSON = os.path.join(data_path, 'USGS_split_val_answers.json')
    HR_imagesvalJSON = os.path.join(data_path, 'USGS_split_val_images.json')

    HR_questionstestJSON = os.path.join(data_path, 'USGS_split_test_phili_questions.json')
    HR_answerstestJSON = os.path.join(data_path, 'USGS_split_test_phili_answers.json')
    HR_imagestestJSON = os.path.join(data_path, 'USGS_split_test_phili_images.json')
    HR_images_path = os.path.join(data_path, 'Data/')

    image_processor = BlipImageProcessor(padding=True, do_resize=True, image_std=[0.229, 0.224, 0.225],
                                         image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True, size=384,
                                         size_divisor=32)
    processor = BlipProcessor.from_pretrained("/home/pengfei/blip-vqa-capfilt-large")
    tokenizer = processor.tokenizer

    if Dataset == 'LR':
        model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=9)
    else:
        model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=94)

    transform_train = [
        transforms.RandomHorizontalFlip(),
    ]
    transform_train.append(AutoAugment())
    transform_train = transforms.Compose(transform_train)

    HR_data_train = VQALoader(HR_images_path, HR_imagesJSON, HR_questionsJSON, HR_answersJSON,
                              tokenizer=tokenizer, image_processor=image_processor,
                              Dataset='HR', train=True, sequence_length=sequence_length,
                              ratio_images_to_use=ratio_images_to_use, transform=transform_train)
    HR_train_loader = torch.utils.data.DataLoader(HR_data_train, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)

    HR_data_val = VQALoader(HR_images_path, HR_imagesvalJSON, HR_questionsvalJSON, HR_answersvalJSON,
                            tokenizer=tokenizer, image_processor=image_processor,
                            Dataset='HR', train=False, sequence_length=sequence_length,
                            ratio_images_to_use=ratio_images_to_use)
    HR_val_loader = torch.utils.data.DataLoader(HR_data_val, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)

    HR_data_test = VQALoader(HR_images_path, HR_imagestestJSON, HR_questionstestJSON, HR_answerstestJSON,
                             tokenizer=tokenizer, image_processor=image_processor,
                             Dataset='HR', train=False, sequence_length=sequence_length,
                             ratio_images_to_use=ratio_images_to_use)
    HR_test_loader = torch.utils.data.DataLoader(HR_data_test, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_acc/dataloader_idx_0",
        save_weights_only=True,
        mode="max",
        dirpath=dir_path,
        filename="{epoch}-{val1_acc:.5f}-{acc_count:.5f}-{acc_rural_urban:.5f}-{acc_presence:.5f}"
    )

    # early stopping
    early_stopping = EarlyStopping(monitor="val1_acc", patience=20, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(devices=1,
                         accelerator='cuda',
                         precision='16-mixed',
                         max_epochs=num_epochs,
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor])

    # 🚀
    trainer.fit(model, train_dataloaders=HR_train_loader, val_dataloaders=[HR_val_loader, HR_test_loader])

if __name__ == "__main__":
    typer.run(main)
