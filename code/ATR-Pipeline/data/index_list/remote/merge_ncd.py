
import os
import shutil
import random


def sample_subfolders(input_folder, output_folder, file_count=float('inf')):
    """生成完全平衡的测试集文件夹
    """
    assert file_count > 0, "Must input a postive number"
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # 统计每个子文件夹中的文件数量
    file_counts = []
    for subfolder in subfolders:
        files = [f.name for f in os.scandir(subfolder) if f.is_file()]
        file_counts.append((subfolder, len(files)))
    print('sub-file in current folder: {}'.format(file_count))

    # 确定采样数量
    min_file_count = min(file_counts, key=lambda x: x[1])[1]
    sample_count = min(file_count, min_file_count)

    # 创建新文件夹用于存放采样后的文件
    if os.path.exists(output_folder):  
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # 从每个子文件夹中采样相同数量的文件，并复制到新文件夹中
    for subfolder, _ in file_counts:
        files = [f.name for f in os.scandir(subfolder) if f.is_file()]
        sampled_files = random.sample(files, sample_count)
        subfolder_name = os.path.basename(subfolder)
        new_subfolder = os.path.join(output_folder, subfolder_name)
        os.makedirs(new_subfolder, exist_ok=True)
        for file in sampled_files:
            src_path = os.path.join(subfolder, file)
            dst_path = os.path.join(new_subfolder, file)
            shutil.copy(src_path, dst_path)

    print(f'采样完成, 每一个平衡文件夹file数量: {sample_count}. 生成的文件夹位于: {output_folder}')