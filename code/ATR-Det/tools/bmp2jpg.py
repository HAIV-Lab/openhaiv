# coding:utf-8
import os
from PIL import Image


# bmp 转换为jpg
def bmpToJpg(file_path):
    for fileName in os.listdir(file_path):
        print('--fileName--', fileName)  # 看下当前文件夹内的文件名字
        # print(fileName)
        newFileName = fileName[0:fileName.find(".")] + ".jpg"  # 改后缀
        # print(newFileName)
        print('--newFileName--', newFileName)
        im = Image.open(file_path + "/" + fileName)
        im = im.convert('RGB')
        im.save("/data/datasets/450/ShipRSImageNet_V1/VOC_Format/JPEGImages/" + "/" + newFileName)  # 保存到当前文件夹内


# 删除原来的位图
def deleteImages(file_path, imageFormat):
    command = "del " + file_path + "/*." + imageFormat
    os.system(command)


def main():
    file_path = "/data/datasets/450/ShipRSImageNet_V1/VOC_Format/BMPImages/"
    bmpToJpg(file_path)
    # deleteImages(file_path, "bmp")


if __name__ == '__main__':
    main()