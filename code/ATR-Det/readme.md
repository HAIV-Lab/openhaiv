## 运行环境
安装命令见requirements.txt
```
python=3.9
pytorch==1.13
pytorch-lightning==1.9.0
opencv-python
pycocotools
tabulate
termcolor
future tensorboard
```

## 如何使用

本项目的代码入口为train.py，并且在其中提前配置了要运行项目的各项参数，执行代码前，配置数据集配置文件，使用python命令运行即可，下面给出详细步骤。


修改.\config\my_config\demo_rgb.yml文件中以下参数：
```
# 保存路径
line1 save_dir: xxxxx
#类别数量
line19 num_classes: X
#类别名称
line43 class_names: &class_names ['XX','XX']
#训练集图像路径
line48 img_path: XXXX  
#训练集标签路径
line49 ann_path: XXXX 
#训练集图像路径
line67 img_path: XXXX  
#训练集标签路径
line68 ann_path: XXXX 
```
数据集的存放格式:img_path文件夹下存放.jpg格式的图像，ann_path文件夹下存放标签，成对的图像和标签文件名保持一致。训练类无关检测器时，需要将xml文件中的标签改成粗类，例如飞机A1-A20均改成plane。

标签格式为VOC数据集的xml格式

配置完毕后，运行训练文件
```
python train.py
```

