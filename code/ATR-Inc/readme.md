## 运行环境

```
python=3.8
pytorch==1.13
numpy
scikit-learn
tqdm
openpyxl
matplotlib
```

## 如何使用

本项目的代码入口为train.py，并且在其中提前配置了要运行项目的各项参数，执行代码前，给出数据集路径以及数据设定参数后使用python命令运行即可，下面给出详细步骤。


修改train.py文件中以下参数：
```
# 训练集路径
split_file_train = '/xxx/xx/xx/'
# 测试集路径
split_file_test = '/xxx/xx/xx/'
# 基础阶段类别数
base_class = xx
# 总类别数
num_classes = xx
# 每个增量阶段学习类别数
way = xx
# 每个类别提供的训练样本数
shot = xx
# 总阶段数
sessions = xx
```

此外，注意将测试样本txt文件放置在./data/index_list/remote文件夹下

执行train.py文件：
```
python train.py
```

