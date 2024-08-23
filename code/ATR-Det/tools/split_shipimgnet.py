import shutil
# with open("/data/datasets/450/ShipRSImageNet_V1/VOC_Format/ImageSets/val.txt", "r") as f:  # 打开文件
#     data = f.readlines()  # 读取文件
# print(data)
annroot = '/data/datasets/450/ShipRSImageNet_V1/VOC_Format/Annotations/'
desroot = '/data/datasets/450/ShipRSImageNet_V1/VOC_Format/Annotations5/val/'
for line in data:
    print(line.split('.')[0])
    shutil.copy(annroot+str(line.split('.')[0])+'.xml', desroot+str(line.split('.')[0])+'.xml')
