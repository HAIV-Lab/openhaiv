import os
import shutil

# 想要移动文件所在的根目录
rootdir ="/new_data/dataset/450/RGB/Detection/VOC_8_CLS/Annoation_all/bndbox/013/"
sourcedir = "/new_data/dataset/450/RGB/Detection/013.Type_45_destroyer/"
name_list = []
for name in os.listdir(sourcedir):
    name_str = os.path.splitext(name)[0]
    name_list.append(name_str)
des_path = "/new_data/dataset/450/RGB/Detection/VOC_8_CLS/"
train = name_list[0:100]
test = name_list[100:]
for s in train:
    path = os.path.join(sourcedir, s)+'.jpg'
    des_img_path = os.path.join(des_path,'train','JPEGImages',s)+'.jpg'
    shutil.copy(path, des_img_path)
for s in test:
    path = os.path.join(sourcedir, s)+'.jpg'
    des_img_path = os.path.join(des_path,'test','JPEGImages',s)+'.jpg'
    shutil.copy(path, des_img_path)

des_path = "/new_data/dataset/450/RGB/Detection/VOC_8_CLS/test/Annotations"
    # 获取目录下文件名清单
list = os.listdir(rootdir)
# print(files)
  # 目标路径
# 移动图片到指定文件夹
for i in range(0, len(list)):
    # 遍历目录下的所有文件夹
    for s in train:
        des_path = "/new_data/dataset/450/RGB/Detection/VOC_8_CLS/train/Annotations"
        print(os.path.splitext(list[i])[0])
        if os.path.splitext(list[i])[0] == s:
            path = os.path.join(rootdir, list[i])
            des_img_path = os.path.join(des_path, 'c'+list[i])
            shutil.copy(path, des_img_path)
            # 移动文件到目标路径
    for s in test:
        des_path = "/new_data/dataset/450/RGB/Detection/VOC_8_CLS/test/Annotations"
        print(os.path.splitext(list[i])[0])
        if os.path.splitext(list[i])[0] == s:
            path = os.path.join(rootdir, list[i])
            des_img_path = os.path.join(des_path, list[i])
            shutil.copy(path, des_img_path)
            # 移动文件到目标路径