# *_* coding : UTF-8 *_*
# 开发人员   ：csu·pan-_-||
# 开发时间   ：2020/10/13 20:18
# 文件名称   ：roxml_to_dota.py
# 开发工具   ：PyCharm
# 功能描述   ：把rolabelimg标注的xml文件转换成dota能识别的xml文件
#             就是把旋转框 cx,cy,w,h,angle，转换成四点坐标x1,y1,x2,y2,x3,y3,x4,y4

import os
import xml.etree.ElementTree as ET
import math


def edit_xml(xml_file,xmlsave_file):
    """
    修改xml文件
    :param xml_file:xml文件的路径
    :return:
    """
    tree = ET.parse(xml_file)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        name = obj.find('name')
        name.text = 'ship'

    tree.write(xmlsave_file, method='xml', encoding='utf-8')  # 更新xml文件




# 转换成四点坐标
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc;
    yoff = yp - yc;
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(int(xc + pResx)), str(int(yc + pResy))

if __name__ == '__main__':
    dir  ="/data/datasets/450/ShipRSImageNet_V1/VOC_Format/Annotations5/val/"
    savedir= "/data/datasets/450/ShipRSImageNet_V1/VOC_Format/C-Annotations/val/"
    filelist = os.listdir(dir)
    # file = 'P4369.xml'
    for file in filelist:
        edit_xml(os.path.join(dir,file),os.path.join(savedir,file))
