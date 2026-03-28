import os

# --- 配置区域 ---
# 请将下面的路径替换成你实际的文件夹 a 和文件夹 b 的路径
# 推荐使用正斜杠 '/'，这样在所有操作系统上都能正常工作
folder_a_path = '/home/pengfei/FLOODNET/result_all'
folder_b_path = '/home/pengfei/FLOODNET/Original_Image_Final_Reshape'


# --- 主逻辑 ---

def find_unique_images_case_insensitive(dir1, dir2):
    """
    查找在第一个目录 (dir1) 中存在，但在第二个目录 (dir2) 中不存在的 .jpg 图片。
    此版本不区分文件名的大小写。

    Args:
        dir1 (str): 第一个文件夹的路径。
        dir2 (str): 第二个文件夹的路径。
    """
    print("--- 开始比较 (不区分大小写) ---")
    print(f"文件夹 'a': {dir1}")
    print(f"文件夹 'b': {dir2}")

    # 1. 检查路径是否存在
    if not os.path.isdir(dir1):
        print(f"\n错误：文件夹 'a' 的路径 '{dir1}' 不存在或不是一个目录。")
        return
    if not os.path.isdir(dir2):
        print(f"\n错误：文件夹 'b' 的路径 '{dir2}' 不存在或不是一个目录。")
        return

    try:
        # 2. 获取文件名映射：{小写文件名: 原始文件名}
        # 这样做可以在不区分大小写的情况下进行比较，同时保留原始文件名用于输出
        files_map_a = {f.lower(): f for f in os.listdir(dir1) if f.lower().endswith('.jpg')}
        files_map_b = {f.lower(): f for f in os.listdir(dir2) if f.lower().endswith('.jpg')}

        print(f"\n在文件夹 'a' 中找到 {len(files_map_a)} 个 .jpg 图片。")
        print(f"在文件夹 'b' 中找到 {len(files_map_b)} 个 .jpg 图片。")

        # 3. 使用集合的差集运算找出只在 a 中存在的小写文件名
        unique_keys = files_map_a.keys() - files_map_b.keys()

        # 4. 输出结果
        print("\n--- 比较结果 ---")
        if not unique_keys:
            print("文件夹 'a' 中没有比文件夹 'b' 多的 .jpg 图片。")
        else:
            print(f"在文件夹 'a' 中，但不在文件夹 'b' 中的图片有 {len(unique_keys)} 张：")

            # 从映射中获取原始文件名并排序输出
            original_filenames = [files_map_a[key] for key in unique_keys]
            for filename in sorted(original_filenames):
                print(f"- {filename}")

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")


# --- 运行主程序 ---
if __name__ == "__main__":
    find_unique_images_case_insensitive(folder_a_path, folder_b_path)