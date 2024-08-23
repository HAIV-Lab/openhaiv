
import os
import shutil
import random


def sample_subfolders(input_folder, output_folder, file_count=float('inf')):
    """������ȫƽ��Ĳ��Լ��ļ���
    """
    assert file_count > 0, "Must input a postive number"
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # ͳ��ÿ�����ļ����е��ļ�����
    file_counts = []
    for subfolder in subfolders:
        files = [f.name for f in os.scandir(subfolder) if f.is_file()]
        file_counts.append((subfolder, len(files)))
    print('sub-file in current folder: {}'.format(file_count))

    # ȷ����������
    min_file_count = min(file_counts, key=lambda x: x[1])[1]
    sample_count = min(file_count, min_file_count)

    # �������ļ������ڴ�Ų�������ļ�
    if os.path.exists(output_folder):  
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # ��ÿ�����ļ����в�����ͬ�������ļ��������Ƶ����ļ�����
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

    print(f'�������, ÿһ��ƽ���ļ���file����: {sample_count}. ���ɵ��ļ���λ��: {output_folder}')