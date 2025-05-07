### Welcome to openhaiv

This is a framework for open-world object recognition, supporting out-of-distribution detection, novel class discovery, and incremental learning algorithms to enable robust and flexible object recognition in unconstrained environments.

## News

[04/2024] 🌟 The code repository is created.

### Install

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
- `GEM`: Gradient Episodic Memory for Continual Learning. NIPS2017 NIPS2017 [[paper](https://arxiv.org/abs/1706.08840)]
- `SSRE`: Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning. CVPR2022 [[paper](https://arxiv.org/abs/2203.06359)]




#### Out of Distribution
##### Unimodal Methods
- `MSP`: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017[[paper](https://arxiv.org/abs/1610.02136)]
- `MLS`: Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022[[paper](https://arxiv.org/abs/1911.11132)]
- `MDS`: A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018[[paper](https://arxiv.org/abs/1807.03888)]
- `ODIN`: Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. ICLR 2018[[paper](https://arxiv.org/abs/1706.02690)]
- `FDBD`: Fast Decision Boundary based Out-of-Distribution Detector. ICML 2024[[paper](https://arxiv.org/abs/2312.11536)]
- `vim`: ViM: Out-Of-Distribution with Virtual-logit Matching. CVPR 2022[[paper](https://arxiv.org/abs/2203.10807)]
- `DML`: Decoupling MaxLogit for Out-of-Distribution Detection. CVPR 2023[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)]
- `ASH`: . [[paper]()]
- `CIDER`: . [[paper]()]
- `CSI`: . [[paper]()]
- `GODIN`: . [[paper]()]
- `MCP`: . [[paper]()]
- `OpenMax`: . [[paper]()]
- `MCD`: . [[paper]()]
- `NPOS`: . [[paper]()]
- `React`: . [[paper]()]
##### CLIP-based Methods
- `MCM`: Delving into Out-of-Distribution Detection with Vision-Language Representations. NeurIPS 2022[[paper](https://arxiv.org/abs/2211.13445)]
- `GL-MCM`: GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot Out-of-Distribution Detection. IJCV 2025[[paper](https://arxiv.org/abs/2304.04521)]
- `NegLabel`: Negative Label Guided OOD Detection with Pretrained Vision-Language Models. ICLR 2024[[paper](https://arxiv.org/abs/2403.20078)]
- `CoOp`: Learning to Prompt for Vision-Language Models. IJCV 2022[[paper](https://arxiv.org/abs/2109.01134)]
- `LoCoOp`: LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning. NeurIPS 2023[[paper](https://arxiv.org/abs/2306.01293)]
- `SCT`: Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection. NeurIPS 2024[[paper](https://arxiv.org/abs/2411.03359)]
- `Maple`: MaPLe: Multi-modal Prompt Learning. CVPR 2023[[paper](https://arxiv.org/abs/2210.03117)]
- `DPM`: Vision-Language Dual-Pattern Matching for Out-of-Distribution Detection. ECCV 2024[[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11399.pdf)]
- `Tip-Adapter`: Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling. ECCV 2022[[paper](https://arxiv.org/abs/2111.03930)]
- `NegPrompt`: Learning Transferable Negative Prompts for Out-of-Distribution Detection. CVPR 2024[[paper](https://arxiv.org/abs/2404.03248)]

### 🤝Contributors

#### Few-shot class-incremental learning
- `Alice`: Few-Shot Class-Incremental Learning from an Open-Set Perspective. ECCV2022 [[paper](https://arxiv.org/abs/2208.00147)]
- `FACT`: Forward Compatible Few-Shot Class-Incremental Learning. CVPR 2022 [[paper](https://arxiv.org/abs/2203.06953)]
- `SAVC`: Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning CVPR2023 [[paper](https://arxiv.org/abs/2304.00426)]


### 🙏Acknowledgement
- [OpenOOD](https://github.com/Jingkang50/OpenOOD), an extensible codebase for out-of-distribution detection with Vision Models only.
- [OpenOOD-VLM](https://github.com/YBZh/OpenOOD-VLM), an extensible codebase for out-of-distribution detection with both Vision Models and Vision-Language Models.
- [PyCIL](https://github.com/G-U-N/PyCIL), an extensible codebase for incremental learning.

### ✉️Contact
If there are any questions, please feel free to propose new features by opening an issue or contact with the author: Xiang Xiang. Enjoy the code.