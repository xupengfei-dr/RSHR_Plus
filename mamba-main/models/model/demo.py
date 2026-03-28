import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import classification_report, confusion_matrix  # 新增混淆矩阵导入
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.trans.models.mamba2 import Mamba2Config, Mamba2Model

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AggregateClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AggregateClassifier, self).__init__()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x


from sklearn.preprocessing import LabelEncoder

class CustomPreprocessing:
    @staticmethod
    def prepro(file_path, split_rate):
        df = pd.read_csv(file_path,nrows=80000)
        # 检查列名是否存在（避免拼写错误）
        if 'Label' in df.columns:
            # 统计并打印各类别数量
            label_counts = df['Label'].value_counts()
            print("各Label类别的数量统计：")
            print(label_counts)
        else:
            print("错误：列名 'Label' 不存在，请检查列名拼写。")


        # 时间戳处理（假设原始时间戳列名为Timestamp）
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp'],
            dayfirst=True,  # 优先解析为 日/月/年
            errors='coerce'
        )
        df = df.sort_values('Timestamp').reset_index(drop=True)

        # 时间特征工程
        df['hour_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)
        df['time_since_start'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()

        # 端口分桶处理（示例处理目标端口）
        df['dst_port_type'] = df['Destination Port'].apply(
            lambda x: 2 if x in [80, 443, 22] else (1 if x <= 1023 else 0)
        )
        df = pd.get_dummies(df, columns=['dst_port_type'], prefix='port')

        # 删除原始时间戳和端口列
        df = df.drop(['Timestamp', 'Source Port', 'Destination Port'], axis=1)

        # 标签转换为整数（使用LabelEncoder）
        # label_encoder = LabelEncoder()
        # df['Label'] = label_encoder.fit_transform(df['Label'])

        # 数据分割（按时间顺序）
        split_idx = int(len(df) * split_rate)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # 标准化（仅在训练集拟合）
        numeric_cols = [col for col in df.columns if col != 'Label']
        scaler = StandardScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

        # 分离特征和标签
        X_train = train_df.drop('Label', axis=1).values
        y_train = train_df['Label'].values
        X_test = test_df.drop('Label', axis=1).values
        y_test = test_df['Label'].values

        # 返回数据
        print(f"特征维度验证 -> 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        return X_train, y_train, X_test, y_test


def acc_loss_line(train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    epochs = range(len(train_loss_list))
    plt.figure()
    plt.plot(epochs, train_acc_list, 'r-.', label="训练集准确率")
    plt.plot(epochs, val_acc_list, 'b--', label="验证集准确率")
    plt.title('训练集和验证集准确率曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.legend()

    plt.figure()
    plt.plot(epochs, train_loss_list, 'r-.', label="训练集损失")
    plt.plot(epochs, val_loss_list, 'b--', label="验证集损失")
    plt.title('训练集和验证集损失值曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.legend()
    plt.show()


def data_pre(file_path, split_rate, length=None):  # 改为可选参数
    X_train, y_train, X_test, y_test = CustomPreprocessing.prepro(file_path, split_rate)

    # 动态获取特征数
    if length is None:
        length = X_train.shape[1]  # 直接使用实际特征数作为序列长度

    # 调整形状 (样本数, 序列长度, 特征维度)
    X_train = X_train.reshape((-1, length, 1))
    X_test = X_test.reshape((-1, length, 1))

    return X_train, y_train, X_test, y_test


class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        # self.labels = torch.from_numpy(labels.astype(np.int64))  # 确保标签是整数类型
        # self.labels = torch.from_numpy(labels.astype(np.float32))
        self.labels = labels
        self.answers = ['BENIGN','DoS','PortScan','DDoS','Brute Force','Web Attack','Bot','Infiltration','Heartbleed']



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        answer_Index = self.answers.index(self.labels[idx])
        return {
            'inputs': self.features[idx],
            'targets': answer_Index
        }


def create_dataloaders(x_train, y_train, x_valid, y_val, batch_size=64):
    train_dataset = TimeSeriesDataset(x_train, y_train)
    val_dataset = TimeSeriesDataset(x_valid, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader
# self.mambaconfig =Mamba2Config()
        # self.mamba = Mamba2Model(self.mambaconfig)

def train_and_evaluate(x_train, y_train, x_valid, y_val, batch_size, epochs, input_length, device):
    train_loader, val_loader = create_dataloaders(x_train, y_train, x_valid, y_val, batch_size)

    model_config = Mamba2Config()
    model = Mamba2Model(model_config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Train", leave=False):
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs_embeds=inputs)[0]
            outputs = classifyl(outputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 验证阶段（新增混淆矩阵和分类报告）
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_targets = []
        all_val_preds = []

        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Val", leave=False):
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            outputs = model(inputs_embeds=inputs)[0]
            outputs = classifyl(outputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
            all_val_targets.extend(targets.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # 调整学习率
        scheduler.step(val_loss)

        # 打印验证结果
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # 打印混淆矩阵和分类报告
        print("\nValidation Confusion Matrix:")
        print(confusion_matrix(all_val_targets, all_val_preds))
        print("\nValidation Classification Report:")
        print(classification_report(all_val_targets, all_val_preds, digits=4, zero_division=0))

    history = {
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_acc": train_acc_list,
        "val_acc": val_acc_list
    }
    return history, model


if __name__ == '__main__':
    file_path = '/home/pengfei/cuda/new-cleaned_cic-ids2017.csv'
    batch_size = 4
    epochs = 20
    x_train, y_train, x_valid, y_val = data_pre(file_path, split_rate=0.7)
    length = x_train.shape[1]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    classifyl = AggregateClassifier(9).to(device)
    history, model = train_and_evaluate(x_train, y_train, x_valid, y_val, batch_size, epochs, length,
                                        device)
    acc_loss_line(history["train_loss"], history["val_loss"], history["train_acc"], history["val_acc"])
