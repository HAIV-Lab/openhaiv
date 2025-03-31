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

### Supported Models for OpenHAIV
- `ResNet`: .
- `ViT`: .
- `Swin-T`: .
- `CLIP`: .

### Supported Augmentations for OpenHAIV

- `Mixup`: .
- `Cutmix`: .
- `Styleaugment`: .
- `Randaugment`: .
- `Augmix`: .
- `Deepaugment`: .
- `Pixmix`: .
- `Regmixup`: .

### Supported Methods for OpenHAIV

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

#### Out-of-Distribution Detection
##### Unimodal Methods
###### Post-hoc Methods
- `OpenMax`: . CVPR 2016[[paper]()]
- `MSP`: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017[[paper](https://arxiv.org/abs/1610.02136)]
- `ODIN`: Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. ICLR 2018[[paper](https://arxiv.org/abs/1706.02690)]
- `MDS`: A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018[[paper](https://arxiv.org/abs/1807.03888)]
- `GRAM` . ICML 2020[[paper]()]
- `EBO` . NeurIPS 2020[[paper]()]
- `RMDS` . Arxiv 2021[[paper]()]
- `GranNorm`: . NeurIPS 2021[[paper]()]
- `React`: . NeurIPS 2021[[paper]()]
- `SEM`: . Arxiv 2022[[paper]()]
- `MLS`: Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022[[paper](https://arxiv.org/abs/1911.11132)]
- `KLM`: Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022[[paper](https://arxiv.org/abs/1911.11132)]
- `KNN`: . ICML 2022[[paper]()]
- `vim`: ViM: Out-Of-Distribution with Virtual-logit Matching. CVPR 2022[[paper](https://arxiv.org/abs/2203.10807)]
- `Dice`: . ECCV 2022[[paper]()]
- `RankFeat`: . NeurIPS 2022[[paper]()]
- `ASH`: Extremely Simple Activation Shaping for Out-of-Distribution Detection. ICLR 2023[[paper](https://arxiv.org/abs/2209.09858)]
- `SHE`: . ICLR 2023[[paper]()]
- `GEN`: . CVPR 2023[[paper]()]
- `NNGuide`: . ICCV 2023[[paper]()]
- `Relation`: . NeurIPS 2023[[paper]()]
- `Scale`: . ICLR 2024[[paper]()]
- `FDBD`: Fast Decision Boundary based Out-of-Distribution Detector. ICML 2024[[paper](https://arxiv.org/abs/2312.11536)]
- `AdaScale A`: . Arxiv 2025[[paper]()]
- `AdaScale L`: . Arxiv 2025[[paper]()]
- `IODIN`: . Arxiv 2025[[paper]()]
- `NCI`: . CVPR 2025[[paper]()]
###### Training Methods
- `ConfBranch`: .Arxiv 2018 [[paper]()]
- `RotPred`: .NeurIPS 2018 [[paper]()]
- `GODIN`: . CVPR 2020[[paper]()]
- `CSI`: . NeurIPS 2020[[paper]()]
- `SSD`: . ICLR 2021[[paper]()]
- `MOS`: . CVPR 2021[[paper]()]
- `VOS`: . ICLR 2022[[paper]()]
- `LogitNorm`: . ICML 2022[[paper]()]
- `CIDER`: . ICLR 2023[[paper]()]
- `NPOS`: . ICLR 2023[[paper]()]
- `DML`: Decoupling MaxLogit for Out-of-Distribution Detection. CVPR 2023[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)]
- `ISH`: . ICLR 2024[[paper]()]
- `PALM`: . ICLR 2024[[paper]()]
- `T2FNorm`: . CVPRW 2024[[paper]()]
- `RewightOOD`: . CVPRW 2024[[paper]()]
- `RewightOOD`: . CVPRW 2024[[paper]()]
- `ASCOOD`: . Arxiv 2025[[paper]()]
###### Method Uncertainty
- `MC-Dropout`: . ICML 2016[[paper]()]
- `Deep-ensemble`: . NeurIPS 2017[[paper]()]
- `Temp-scaling`: . ICML 2017[[paper]()]
- `RTS`: . AAAI 2023[[paper]()]

##### CLIP-based Methods
- `MCM`: Delving into Out-of-Distribution Detection with Vision-Language Representations. NeurIPS 2022[[paper](https://arxiv.org/abs/2211.13445)]
- `GL-MCM`: GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot Out-of-Distribution Detection. IJCV 2025[[paper](https://arxiv.org/abs/2304.04521)]
- `NegLabel`: Negative Label Guided OOD Detection with Pretrained Vision-Language Models. ICLR 2024[[paper](https://arxiv.org/abs/2403.20078)]
- `CoOp`: Learning to Prompt for Vision-Language Models. IJCV 2022[[paper](https://arxiv.org/abs/2109.01134)]
- `LoCoOp`: LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning. NeurIPS 2023[[paper](https://arxiv.org/abs/2306.01293)]
- `SCT`: Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection. NeurIPS 2024[[paper](https://arxiv.org/abs/2411.03359)]
- `Maple`: MaPLe: Multi-modal Prompt Learning. CVPR 2023[[paper](https://arxiv.org/abs/2210.03117)]
- `DPM`: Vision-Language Dual-Pattern Matching for Out-of-Distribution Detection. ECCV 2024[[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11399.pdf)]
- `CALIP`: . [[paper]()]
- `Tip-Adapter`: Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling. ECCV 2022[[paper](https://arxiv.org/abs/2111.03930)]
- `NegPrompt`: Learning Transferable Negative Prompts for Out-of-Distribution Detection. CVPR 2024[[paper](https://arxiv.org/abs/2404.03248)]

#### Few-shot class-incremental learning
- `Alice`: Few-Shot Class-Incremental Learning from an Open-Set Perspective. ECCV 2022 [[paper](https://arxiv.org/abs/2208.00147)]
- `FACT`: Forward Compatible Few-Shot Class-Incremental Learning. CVPR 2022 [[paper](https://arxiv.org/abs/2203.06953)]
- `SAVC`: Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning. CVPR 2023 [[paper](https://arxiv.org/abs/2304.00426)]

### Contributors

### Citation
If you find our repository useful for your research, please consider citing these papers:
```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Author Name and Co-author Name},
  journal={Journal Name},
  year={2023},
  volume={XX},
  number={YY},
  pages={ZZZ},
  doi={10.xxxx/your-doi},
  url={https://arxiv.org/abs/your-arxiv-id}
}