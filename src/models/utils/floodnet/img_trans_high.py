from PIL import Image
import os

# --- 配置 ---
input_filename = '10163.jpg'
output_dir = 'output_squashed_high_quality'
target_sizes = [
    (384, 384)
]
# 新增：设置输出的JPEG质量 (1-100, 95为高质量)
output_quality = 100

# --- 检查与准备 ---
if not os.path.exists(input_filename):
    print(f"错误：找不到输入文件 '{input_filename}'。")
    exit()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出文件夹: '{output_dir}'")

# --- 主处理逻辑 ---
try:
    with Image.open(input_filename) as img:
        print(f"成功打开图片 '{input_filename}'，原始尺寸: {img.size[0]}x{img.size[1]}")
        base_name, extension = os.path.splitext(os.path.basename(input_filename))

        for width, height in target_sizes:
            print(f"--- 正在将图片压扁至: {width}x{height} (质量: {output_quality}) ---")

            squashed_img = img.resize((width, height), Image.Resampling.LANCZOS)

            output_filename = f"{base_name}_{width}x{height}_q{output_quality}{extension}"
            output_filepath = os.path.join(output_dir, output_filename)

            # 在保存时添加 quality 参数！
            squashed_img.save(output_filepath, quality=output_quality)

            # 获取并打印新文件的大小
            file_size_kb = os.path.getsize(output_filepath) / 1024
            print(f"已保存压扁后的图片到: {output_filepath} (大小: {file_size_kb:.2f} KB)")

    print("\n所有任务处理完毕！")

except Exception as e:
    print(f"处理过程中发生错误: {e}")