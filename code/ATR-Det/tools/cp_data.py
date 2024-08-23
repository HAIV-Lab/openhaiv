import os
image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
import os
import shutil


def copy_dirs(src_path, target_path):
    file_count = 0
    source_path = os.path.abspath(src_path)
    target_path = os.path.abspath(target_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            for file in files:
                # if file.split('.')[0][0:1] == 'P':
                src_file = os.path.join(root, file)
                # file = file[1:]
                # target_path = os.path.join(target_path, file[1:])
                shutil.copy(src_file, target_path)
                file_count += 1
                print(src_file)
    return int(file_count)
if __name__ == "__main__":
    # dir  = "/data/xz2002/450data/JPEGImage/Ship/"
    # dir2= "/data/xz2002/450data/JPEGImage/Planenoa6/"
    # dir3 = "/data/xz2002/450data/JPEGImage/Plane&SHIPnoa6/"
    # dir4 = "/data/xz2002/450data/JPEGImage/Plane&ship/"
    # dir5 = "/data/xz2002/450data/JPEGImage/Plane/"
    copy_dirs("/data/xz2002/450data/Annotations/Plane&SHIPnoa6/val/","/data/xz2002/450data/Annotations/P&Schoose/val/")