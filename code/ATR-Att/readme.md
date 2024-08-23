## 运行环境
安装命令见requirements.txt
```
python=3.9
pytorch==1.13
opencv-python
psutil
matplotlib
pyyaml
tqdm
pandas
```

## 如何使用

本项目的代码入口为train.py，并且在其中提前配置了要运行项目的各项参数，执行代码前，配置数据集配置文件，使用python命令运行即可，下面给出详细步骤。


修改.\ultralytics\cfg\datasets\att.yaml文件中以下参数：
```
# 数据根路径
path: '/xxx/xx/xx/'
names:
  0: xxx
  1: xxx
  2: xxx
  ...
```
数据集的存放格式如下，images文件夹下存放图像，label文件夹下存放标签，成对的图像和标签文件名保持一致。
```
├── images
│   ├── train
│   │   └── 001.jpg
│   └── val
│       └── 002.jpg
└── labels
    ├── train
    │   └── 001.txt
    └── val
        └── 002.txt
```
标签格式为txt格式，每行表示一个属性标注，存储格式为 类别名称以及四点归一化坐标x1,y1,x2,y2,x3,y3,x4,y4，例如：
```
10 0.025 0.174 0.350 0.302 0.324 0.395 0.008 0.267
```
配置完毕后，运行训练文件
```
python train.py
```

