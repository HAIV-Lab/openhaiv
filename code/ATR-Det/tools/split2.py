import os

import os
import xml.etree.ElementTree as ET
import math
import shutil
xml_path = '/new_data/dataset/Remote/RGB/plane/MAR20/Detection/Annotations/Horizontal Bounding Boxes/'
num=0
all=0
for xml_file in os.listdir(xml_path):
    all=all+1
    xml_name = xml_path + xml_file
    root = ET.parse(xml_name).getroot()  # 利用ET读取xml文件
    flag = 0
    for _object in root.findall("object"):
        category = _object.find("name").text
        if category == 'A6':
            flag = 1
        if category == 'A20':
            flag = 1
        if category == 'A4':
            flag = 1
    if flag ==1 :
        des_path = "/new_data/dx450/Project/xz/MAR20_detect/test/"
        des_img_path = os.path.join(des_path, xml_file)
        shutil.copy(xml_name, des_img_path)
    else:
        des_path = "/new_data/dx450/Project/xz/MAR20_detect/train/"
        des_img_path = os.path.join(des_path, xml_file)
        shutil.copy(xml_name, des_img_path)

print(num)
