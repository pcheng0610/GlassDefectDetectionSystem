import os
import shutil


def organize_dataset(source_dir, target_dir):
    """
    将源目录中的文件按照images/labels结构组织

    Parameters:
    source_dir (str): 源目录路径（包含jpg、png和txt文件）
    target_dir (str): 目标目录路径
    """

    # 创建目标文件夹
    images_dir = os.path.join(target_dir, 'images')
    labels_dir = os.path.join(target_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)

        # 只处理文件，忽略子目录
        if os.path.isfile(src_path):
            # 处理JPG和PNG文件（图像文件）
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                dest_path = os.path.join(images_dir, filename)
                shutil.copy2(src_path, dest_path)
                print(f'Copied image: {filename} -> images/')

            # 处理TXT文件（标签文件）
            elif filename.lower().endswith('.txt'):
                dest_path = os.path.join(labels_dir, filename)
                shutil.copy2(src_path, dest_path)
                print(f'Copied label: {filename} -> labels/')


if __name__ == "__main__":
    # 请根据实际情况修改路径
    source_directory = "path/to/your/source/files"  # 替换为图1文件所在路径
    target_directory = "path/to/your/target/dataset"  # 替换为目标数据集路径

    organize_dataset(source_directory, target_directory)
    print("数据集组织完成！")

    # 显示结果目录结构
    print("\n生成的文件结构:")
    for root, dirs, files in os.walk(target_directory):
        level = root.replace(target_directory, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")