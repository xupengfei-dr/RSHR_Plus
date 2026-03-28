import numpy as np

# # 替换为你的.npy文件路径
file_path = 'numYALLs.npy'# shape(181, 31)
# file_path = 'bushY2.npy'  # shape (181,)
# 10倍交折划分数据集 R2和RMSE评估
# 加载 .npy 文件
data = np.load(file_path, allow_pickle=True)

# 打印文件属性
print("文件类型:", type(data))
if isinstance(data, np.ndarray):
    print("数组形状:", data.shape)
    print("数据类型:", data.dtype)

# 打印全部数据
print("\n数据内容:")
print(data)
