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
        obj_bnd = obj.find('bndbox')
        if obj_bnd==None:
            continue
        # try:
        #     obj_bnd.tag = 'bndbox'
        # except:
        #     print(1)
        xmin = ET.Element("xmin")  # 创建节点
        xmax = ET.Element("xmax")
        ymin = ET.Element("ymin")
        ymax = ET.Element("ymax")

        obj_type = obj.find('type')
        obj_bnd = obj.find('bndbox')
        obj_x0 = obj_bnd.find('x0')
        obj_x1 = obj_bnd.find('x1')
        obj_x2 = obj_bnd.find('x2')
        obj_x3 = obj_bnd.find('x3')
        obj_y0= obj_bnd.find('y0')
        obj_y1 = obj_bnd.find('y1')
        obj_y2 = obj_bnd.find('y2')
        obj_y3 = obj_bnd.find('y3')

        x0 = float(obj_x0.text)
        x1 = float(obj_x1.text)
        x2 = float(obj_x2.text)
        x3 = float(obj_x3.text)
        y0 = float(obj_y0.text)
        y1 = float(obj_y1.text)
        y2 = float(obj_y2.text)
        y3 = float(obj_y3.text)

        xmin.text = str(int(min(x0,x1,x2,x3)))
        ymin.text = str(int(min(y0,y1,y2,y3)))
        xmax.text = str(int(max(x0,x1,x2,x3)))
        ymax.text = str(int(max(y0,y1,y2,y3)))

        obj_bnd.remove(obj_x0)  # 删除节点
        obj_bnd.remove(obj_x1)
        obj_bnd.remove(obj_x2)
        obj_bnd.remove(obj_x3)
        obj_bnd.remove(obj_y0)  # 删除节点
        obj_bnd.remove(obj_y1)
        obj_bnd.remove(obj_y2)
        obj_bnd.remove(obj_y3)

        obj_bnd.append(xmin)    # 新增节点
        obj_bnd.append(ymin)
        obj_bnd.append(xmax)
        obj_bnd.append(ymax)
        # type = obj_type.text
    print(xml_file)
    tree.write(xmlsave_file, method='xml', encoding='utf-8')  # 更新xml文件
        # if type == 'bndbox':
        #     obj_bnd = obj.find('bndbox')
        #     obj_xmin = obj_bnd.find('xmin')
        #     obj_ymin = obj_bnd.find('ymin')
        #     obj_xmax = obj_bnd.find('xmax')
        #     obj_ymax = obj_bnd.find('ymax')
        #     xmin = float(obj_xmin.text)
        #     ymin = float(obj_ymin.text)
        #     xmax = float(obj_xmax.text)
        #     ymax = float(obj_ymax.text)
        #     obj_bnd.remove(obj_xmin)  # 删除节点
        #     obj_bnd.remove(obj_ymin)
        #     obj_bnd.remove(obj_xmax)
        #     obj_bnd.remove(obj_ymax)
        #     x0.text = str(xmin)
        #     y0.text = str(ymin)
        #     x1.text = str(xmax)
        #     y1.text = str(ymin)
        #     x2.text = str(xmin)
        #     y2.text = str(ymax)
        #     x3.text = str(xmax)
        #     y3.text = str(ymax)
        # elif type == 'robndbox':
        #     obj_bnd = obj.find('robndbox')
        #     obj_bnd.tag = 'bndbox'   # 修改节点名
        #     obj_cx = obj_bnd.find('cx')
        #     obj_cy = obj_bnd.find('cy')
        #     obj_w = obj_bnd.find('w')
        #     obj_h = obj_bnd.find('h')
        #     obj_angle = obj_bnd.find('angle')
        #     cx = float(obj_cx.text)
        #     cy = float(obj_cy.text)
        #     w = float(obj_w.text)
        #     h = float(obj_h.text)
        #     angle = float(obj_angle.text)
        #     obj_bnd.remove(obj_cx)  # 删除节点
        #     obj_bnd.remove(obj_cy)
        #     obj_bnd.remove(obj_w)
        #     obj_bnd.remove(obj_h)
        #     obj_bnd.remove(obj_angle)
        #
        #     x0.text, y0.text = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
        #     x1.text, y1.text = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
        #     x2.text, y2.text = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
        #     x3.text, y3.text = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

        # obj.remove(obj_type)  # 删除节点
        # obj_bnd.append(x0)    # 新增节点
        # obj_bnd.append(y0)
        # obj_bnd.append(x1)
        # obj_bnd.append(y1)
        # obj_bnd.append(x2)
        # obj_bnd.append(y2)
        # obj_bnd.append(x3)
        # obj_bnd.append(y3)



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
    savedit="/new_data/dataset/450/RGB/Detection/VOC_8_CLS/Annoation_all/bndbox/003/"
    dir= "/new_data/dataset/450/RGB/Detection/VOC_8_CLS/Annoation_all/robndbox/003/"
    filelist = os.listdir(dir)
    # file = 'P4369.xml'
    for file in filelist:
        edit_xml(os.path.join(dir,file),os.path.join(savedit,file))
