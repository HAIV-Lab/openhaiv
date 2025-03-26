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
-  `EWC`: Overcoming catastrophic forgetting in neural networks. PNAS 2017 [[paper](https://arxiv.org/abs/1612.00796)]
-  `iCaRL`: Incremental Classifier and Representation Learning. CVPR 2017 [[paper](https://arxiv.org/abs/1611.07725)]
-  `BiC`: Large Scale Incremental Learning. CVPR 2019 [[paper](https://arxiv.org/abs/1905.13260)]
-  `WA`: Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR 2020 [[paper](https://arxiv.org/abs/1911.07053)]
-  `DER`: DER: Dynamically Expandable Representation for Class Incremental Learning. CVPR 2021 [[paper](https://arxiv.org/abs/2103.16788)]
-  `Coil`: Co-Transport for Class-Incremental Learning. ACM MM 2021 [[paper](https://arxiv.org/abs/2107.12654)]
-  `MEMO`: A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning. ICLR 2023 Spotlight [[paper](https://openreview.net/forum?id=S07feAlQHgM)]

#### Out of Distribution
- `MSP`: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017[[paper](https://arxiv.org/abs/1610.02136)]
- `MLS`: Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022[[paper](https://arxiv.org/abs/1911.11132)]
- `MDS`: A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018[[paper](https://arxiv.org/abs/1807.03888)]
- `ODIN`: Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. ICLR 2018[[paper](https://arxiv.org/abs/1706.02690)]
- `FDBD`: Fast Decision Boundary based Out-of-Distribution Detector. ICML 2024[[paper](https://arxiv.org/abs/2312.11536)]
- `vim`: ViM: Out-Of-Distribution with Virtual-logit Matching. CVPR 2022[[paper](https://arxiv.org/abs/2203.10807)]
- `DML`: Decoupling MaxLogit for Out-of-Distribution Detection. CVPR 2023[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)]

#### Few-shot class-incremental learning
- `Alice`: Few-Shot Class-Incremental Learning from an Open-Set Perspective. ECCV 2022 [[paper](https://arxiv.org/abs/2208.00147)]
- `FACT`: Forward Compatible Few-Shot Class-Incremental Learning. CVPR 2022 [[paper](https://arxiv.org/abs/2203.06953)]
- `SAVC`: Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning CVPR 2023 [[paper](https://arxiv.org/abs/2304.00426)]


