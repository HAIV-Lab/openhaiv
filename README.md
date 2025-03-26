### Welcome to openhaiv

This is a framework for open-world object recognition, supporting out-of-distribution detection, novel class discovery, and incremental learning algorithms to enable robust and flexible object recognition in unconstrained environments.

### Install
#### install python3
#### in linux 
It is recommended to use anaconda3 to manage and maintain the python library environment.
1. Download the .sh file from the anaconda3 website
2. install anaconda3 with .sh file
```
bash Anaconda3-2023.03-Linux-x86_64.sh
```


#### create virtual environment
```
conda create -n ncdia python=3.10 -y
conda activate ncdia
pip install -r requirements.txt
python setup.py install
```

#### install package
* pytorch>=1.12.0 torchvision>=0.13.0 (recommand offical torch command)
* numpy>=1.26.4
* scipy>=1.14.0
* scikit-learn>=1.5.1

### Support Methods for OpenHAIV

#### Class-incremental learning

- `Joint`: update models using all the data from all classes.
- `Finetune`: baseline method which simply update model using current data.
- `LwF`: Learning without Forgetting. ECCV2016 [[paper](https://arxiv.org/abs/1606.09282)]
-  `EWC`: Overcoming catastrophic forgetting in neural networks. PNAS2017 [[paper](https://arxiv.org/abs/1612.00796)]
-  `iCaRL`: Incremental Classifier and Representation Learning. CVPR2017 [[paper](https://arxiv.org/abs/1611.07725)]
-  `BiC`: Large Scale Incremental Learning. CVPR2019 [[paper](https://arxiv.org/abs/1905.13260)]
-  `WA`: Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR2020 [[paper](https://arxiv.org/abs/1911.07053)]
-  `DER`: DER: Dynamically Expandable Representation for Class Incremental Learning. CVPR2021 [[paper](https://arxiv.org/abs/2103.16788)]
-  `Coil`: Co-Transport for Class-Incremental Learning. ACM MM2021 [[paper](https://arxiv.org/abs/2107.12654)]
-  `MEMO`: A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning. ICLR 2023 Spotlight [[paper](https://openreview.net/forum?id=S07feAlQHgM)]

#### Out of Distribution

## 贡献者
<a href="https://github.com/eryajf/learn-github/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=eryajf/learn-github" />
</a>



#### Few-shot class-incremental learning
- `Alice`: Few-Shot Class-Incremental Learning from an Open-Set Perspective. ECCV2022 [[paper](https://arxiv.org/abs/2208.00147)]
- `FACT`: Forward Compatible Few-Shot Class-Incremental Learning. CVPR 2022 [[paper](https://arxiv.org/abs/2203.06953)]
- `SAVC`: Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning CVPR2023 [[paper](https://arxiv.org/abs/2304.00426)]


