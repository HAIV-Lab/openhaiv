import os
import xml.etree.ElementTree as ET
import math

def edit_xml(xml_file,file):

    tree = ET.parse(xml_file)
    # objs = tree.find('filename')
    uk=0
    for _object in tree.findall("object"):
        category = _object.find("name").text
        if category != 'A20':
            uk=1
    if uk == 0:
        tree.write(os.path.join("/data/datasets/450/MAR-20-tgt/onlyA20",file), method='xml', encoding='utf-8')
    # else:
    #     tree.write(os.path.join("/data/datasets/450/MAR-20-tgt/4620", file), method='xml', encoding='utf-8')
    # else:
    #     tree.write(os.path.join("/data/xz2002/450data/Annotations/Planenoa6/", file), method='xml', encoding='utf-8')
    # objs.text = 'c'+xml_file.split('/')[-1].split('.')[0]+'.jpg'
    # print(objs)
    # tree.write(xmlsave_file, method='xml', encoding='utf-8')  # 更新xml文件

if __name__ == '__main__':
    dir="/data/datasets/450/MAR20-raw/xz/Annoations/train/"
    savedit = "/"
    filelist = os.listdir(dir)
    # file = 'P4369.xml'
    for file in filelist:
        edit_xml(os.path.join(dir,file),file)