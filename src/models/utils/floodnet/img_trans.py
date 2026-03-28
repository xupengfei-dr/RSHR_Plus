# 导入Pillow库中的Image模块，以及os库用于处理文件路径
from PIL import Image
import os

# --- 配置 ---
# 输入图片的文件名
input_filename = '10163.jpg'

# 输出结果存放的文件夹名称
output_dir = 'output_squashed'
output_quality = 95
# 您需要的目标尺寸列表，格式为 (宽度, 高度)
target_sizes = [
    (384, 384),

]

# --- 检查与准备 ---
# 检查输入文件是否存在
if not os.path.exists(input_filename):
    print(f"错误：找不到输入文件 '{input_filename}'。请确保文件与脚本在同一目录下。")
    exit()

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出文件夹: '{output_dir}'")

# --- 主处理逻辑 ---
try:
    # 使用 with 语句打开图片，确保处理完后自动关闭
    with Image.open(input_filename) as img:
        original_width, original_height = img.size
        print(f"成功打开图片 '{input_filename}'，原始尺寸: {original_width}x{original_height}")

        # 分离文件名和扩展名，方便构造新文件名
        base_name, extension = os.path.splitext(os.path.basename(input_filename))

        # 遍历所有目标尺寸进行压扁处理
        for width, height in target_sizes:
            print(f"--- 正在将图片压扁至: {width}x{height} ---")

            # 直接使用 resize() 方法进行压扁/拉伸 (不保持宽高比)
            # Image.Resampling.LANCZOS 是高质量的缩放算法，能让图片更清晰
            squashed_img = img.resize((width, height), Image.Resampling.LANCZOS)

            # 构造新的输出文件名，例如 "10163_384x384.jpg"
            output_filename = f"{base_name}_{width}x{height}{extension}"
            output_filepath = os.path.join(output_dir, output_filename)

            # 保存压扁后的图片
            squashed_img.save(output_filepath)
            print(f"已保存压扁后的图片到: {output_filepath}")

    print("\n所有任务处理完毕！")

except Exception as e:
    print(f"处理过程中发生错误: {e}")