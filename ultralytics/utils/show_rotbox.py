import cv2
import xml.etree.ElementTree as ET

def draw_bbox_on_image(image_path, xml_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取边界框信息
    cx = float(root.find('robndbox/cx').text)
    cy = float(root.find('robndbox/cy').text)
    w = float(root.find('robndbox/w').text)
    h = float(root.find('robndbox/h').text)
    angle = float(root.find('robndbox/angle').text)

    # 计算边界框的四个顶点坐标
    theta = angle * (3.141592653589793 / 180.0)
    cos_theta = abs(np.cos(theta))
    sin_theta = abs(np.sin(theta))
    box_w = w * cos_theta + h * sin_theta
    box_h = w * sin_theta + h * cos_theta
    x1 = int(cx - box_w / 2)
    y1 = int(cy - box_h / 2)
    x2 = int(cx + box_w / 2)
    y2 = int(cy + box_h / 2)

    # 在图像上绘制边界框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用函数，传入图像路径和XML文件路径
image_path = "/new_data/dx450/Project/xz/Att_ins/2660_2.jpg"
xml_path = "/new_data/dx450/Project/xz/Att_ins/Annotations_plane/A8_label/2660_2.xml"

draw_bbox_on_image(image_path, xml_path)