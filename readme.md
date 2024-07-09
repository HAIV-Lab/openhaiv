## 如何使用

本项目根据configs文件夹对代码中的相关参数进行配置，对configs文件夹中的各项配置正确后，可直接运行scripts中脚本文件执行代码，configs文件夹的结构如下

```
├─dataloader
├─increment
├─networks
└─pipelines
```
包括数据、增量、网络以及对整个流程的配置，需要根据可见光、红外和SAR三个模态分别配置，另外有是否选择使用属性的选项，研究人员已将三个模态的运行配置文件写好，下面分别给出运行三个模态代码的执行步骤。

### 可见光
修改dataloader/savc_att_remote.yml文件中以下参数：
```
# 测试样本txt文件路径
txt_path: /xxx/xxx/xxx/
# 属性文件路径
att_path: /xxx/xxx/xxx/
# 训练集文件路径
split_file_train: /xxx/xxx/xxx/
# 测试集文件路径
split_file_test: /xxx/xxx/xxx/
```
修改increment/savc_remote.yml文件中以下参数：

```
# 基础阶段类别数目
base_class: xx
# 总类别数目
num_classes: xx
# 每个增量阶段新学习类别
way: xx
# 每个类别提供的训练样本数目
shot: xx
# 总体阶段数目（基础阶段数+增量阶段数）
sessions: xx
```
修改pipelines/train_savc_att_base.yml文件中以下参数：
```
# 结果保存的路径
output_dir: ./results/
```

执行基础阶段训练代码等待训练完成：
```
bash ./scripts/train_savc_att_base.sh
```

训练完成后修改pipelines/train_savc_att_new.yml文件中以下参数：
```
# 结果保存路径
output_dir: ./results/
# base阶段模型文件参数路径（保存刚训练好的结果路径中，粘贴过来即可）
network:
    checkpoint: /xxx/xx/xx/
```

执行增量阶段脚本：

```
bash ./scripts/train_savc_att_new.sh
```

### SAR
修改dataloader/savc_remote_SAR.yml文件中以下参数：
```
# 测试样本txt文件路径
txt_path: /xxx/xxx/xxx/
# 属性文件路径
att_path: /xxx/xxx/xxx/
# 训练集文件路径
split_file_train: /xxx/xxx/xxx/
# 测试集文件路径
split_file_test: /xxx/xxx/xxx/
```
修改increment/savc_remote_SAR.yml文件中以下参数：

```
# 基础阶段类别数目
base_class: xx
# 总类别数目
num_classes: xx
# 每个增量阶段新学习类别
way: xx
# 每个类别提供的训练样本数目
shot: xx
# 总体阶段数目（基础阶段数+增量阶段数）
sessions: xx
```
修改pipelines/train_savc_base_SAR.yml文件中以下参数：
```
# 结果保存的路径
output_dir: ./results/
```

执行基础阶段训练代码等待训练完成：
```
bash ./scripts/train_savc_base_SAR.sh
```

训练完成后修改pipelines/train_savc_new_SAR.yml文件中以下参数：
```
# 结果保存路径
output_dir: ./results/
# base阶段模型文件参数路径（保存刚训练好的结果路径中，粘贴过来即可）
network:
    checkpoint: /xxx/xx/xx/
```

执行增量阶段脚本：

```
bash ./scripts/train_savc_new_SAR.sh
```


### 红外
修改dataloader/savc_remote_IR.yml文件中以下参数：
```
# 测试样本txt文件路径
txt_path: /xxx/xxx/xxx/
# 属性文件路径
att_path: /xxx/xxx/xxx/
# 训练集文件路径
split_file_train: /xxx/xxx/xxx/
# 测试集文件路径
split_file_test: /xxx/xxx/xxx/
```
修改increment/savc_remote_IR.yml文件中以下参数：

```
# 基础阶段类别数目
base_class: xx
# 总类别数目
num_classes: xx
# 每个增量阶段新学习类别
way: xx
# 每个类别提供的训练样本数目
shot: xx
# 总体阶段数目（基础阶段数+增量阶段数）
sessions: xx
```
修改pipelines/train_savc_base_IR.yml文件中以下参数：
```
# 结果保存的路径
output_dir: ./results/
```

执行基础阶段训练代码等待训练完成：
```
bash ./scripts/train_savc_base_IR.sh
```

训练完成后修改pipelines/train_savc_new_IR.yml文件中以下参数：
```
# 结果保存路径
output_dir: ./results/
# base阶段模型文件参数路径（保存刚训练好的结果路径中，粘贴过来即可）
network:
    checkpoint: /xxx/xx/xx/
```

执行增量阶段脚本：

```
bash ./scripts/train_savc_new_IR.sh
```