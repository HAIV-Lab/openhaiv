### Welcome to openhaiv

This is a framework for open-world object recognition, supporting out-of-distribution detection, novel class discovery, and incremental learning algorithms to enable robust and flexible object recognition in unconstrained environments.



### 🎉News

[09/2024] 🌟 The code repository is created.


### ⚙️Install
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




### 🤖Supported Models for OpenHAIV
  `ResNet`、`ViT`、`CLIP`

### 🎨Supported Augmentations for OpenHAIV

<!--`Mixup`、`Cutmix`、`Styleaugment`、`Randaugment`、`Augmix`、`Deepaugment`、`Pixmix`、`Regmixup`-->

### 📚Supported Methods for OpenHAIV

#### Class-incremental learning

- `Joint`: update models using all the data from all classes.
- `Finetune`: baseline method which simply update model using current data.
- `LwF`: Learning without Forgetting. ECCV 2016 [[paper](https://arxiv.org/abs/1606.09282)]
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
- `GRAM`: Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices. ICML 2020[[paper](https://arxiv.org/abs/1912.12510)]
- `EBO`: Energy-based Out-of-distribution Detection. NeurIPS 2020[[paper](https://arxiv.org/abs/2010.03759)]
- `RMDS`: A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection. Arxiv 2021[[paper](https://arxiv.org/abs/2106.09022)]
- `GranNorm`: On the Importance of Gradients for Detecting Distributional Shifts in the Wild. NeurIPS 2021[[paper](https://arxiv.org/abs/2110.00218)]
- `React`: ReAct: Out-of-distribution Detection With Rectified Activations. NeurIPS 2021[[paper](https://arxiv.org/abs/2111.12797)]
- `SEM`: . Arxiv 2022[[paper]()]
- `MLS`: Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022[[paper](https://arxiv.org/abs/1911.11132)]
- `KLM`: Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022[[paper](https://arxiv.org/abs/1911.11132)]
- `KNN`: Out-of-Distribution Detection with Deep Nearest Neighbors. ICML 2022[[paper](https://arxiv.org/abs/2204.06507)]
- `vim`: ViM: Out-Of-Distribution with Virtual-logit Matching. CVPR 2022[[paper](https://arxiv.org/abs/2203.10807)]
- `Dice`: DICE: Leveraging Sparsification for Out-of-Distribution Detection. ECCV 2022[[paper](https://arxiv.org/abs/2111.09805)]
- `RankFeat`: RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection. NeurIPS 2022[[paper](https://arxiv.org/abs/2209.08590)]
- `ASH`: Extremely Simple Activation Shaping for Out-of-Distribution Detection. ICLR 2023[[paper](https://arxiv.org/abs/2209.09858)]
- `SHE`: . ICLR 2023[[paper]()]
- `GEN`:  GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection. CVPR 2023[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)]
- `NNGuide`: Nearest Neighbor Guidance for Out-of-Distribution Detection. ICCV 2023[[paper](https://arxiv.org/abs/2309.14888)]
- `Relation`: Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data. NeurIPS 2023[[paper](https://arxiv.org/abs/2301.12321)]
- `Scale`: Scaling for Training Time and Post-hoc Out-of-distribution Detection Enhancement. ICLR 2024[[paper](https://arxiv.org/abs/2310.00227)]
- `FDBD`: Fast Decision Boundary based Out-of-Distribution Detector. ICML 2024[[paper](https://arxiv.org/abs/2312.11536)]
<!--- `AdaScale A`: AdaSCALE: Adaptive Scaling for OOD Detection. Arxiv 2025[[paper](https://arxiv.org/abs/2503.08023)]-->
<!--- `AdaScale L`: AdaSCALE: Adaptive Scaling for OOD Detection. Arxiv 2025[[paper](https://arxiv.org/abs/2503.08023)]-->
<!--- `IODIN`: Going Beyond Conventional OOD Detection. Arxiv 2025[[paper](https://arxiv.org/abs/2411.10794)]-->
<!--- `NCI`: Detecting Out-of-Distribution Through the Lens of Neural Collapse. CVPR 2025[[paper](https://arxiv.org/abs/2311.01479)]-->
###### Training Methods
- `ConfBranch`: Learning Confidence for Out-of-Distribution Detection in Neural Networks. Arxiv 2018 [[paper](https://arxiv.org/abs/1802.04865)]
- `RotPred`: Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty. NeurIPS 2018 [[paper](https://arxiv.org/abs/1906.12340)]
- `GODIN`: Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data. CVPR 2020[[paper](https://arxiv.org/abs/2002.11297)]
<!--- `CSI`: CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances. NeurIPS 2020[[paper](https://arxiv.org/abs/2002.11297)]-->
- `SSD`: SSD: A Unified Framework for Self-Supervised Outlier Detection. ICLR 2021[[paper](https://arxiv.org/abs/2103.12051)]
- `MOS`: MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space. CVPR 2021[[paper](https://arxiv.org/abs/2105.01879)]
- `VOS`: VOS: Learning What You Don't Know by Virtual Outlier Synthesis. ICLR 2022[[paper](https://arxiv.org/abs/2202.01197)]
- `LogitNorm`: Mitigating Neural Network Overconfidence with Logit Normalization. ICML 2022[[paper](https://arxiv.org/abs/2205.09310)]
- `CIDER`: How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?. ICLR 2023[[paper](https://arxiv.org/abs/2203.04450)]
<!--- `NPOS`: Non-Parametric Outlier Synthesis. ICLR 2023[[paper](https://arxiv.org/abs/2303.02966)]-->
- `DML`: Decoupling MaxLogit for Out-of-Distribution Detection. CVPR 2023[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)]
<!--- `ISH`: Scaling for Training Time and Post-hoc Out-of-distribution Detection Enhancement. ICLR 2024[[paper](https://arxiv.org/abs/2310.00227)]-->
<!--- `PALM`: Learning with Mixture of Prototypes for Out-of-Distribution Detection. ICLR 2024[[paper](https://arxiv.org/abs/2402.02653)]-->
<!--- `T2FNorm`: T2FNorm: Train-time Feature Normalization for OOD Detection
 in Image Classification. CVPRW 2024[[paper](https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/papers/Regmi_T2FNorm_Train-time_Feature_Normalization_for_OOD_Detection_in_Image_Classification_CVPRW_2024_paper.pdf)]-->
<!--- `RewightOOD`: ReweightOOD: Loss Reweighting for Distance-based OOD Detection. CVPRW 2024[[paper](https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/papers/Regmi_ReweightOOD_Loss_Reweighting_for_Distance-based_OOD_Detection_CVPRW_2024_paper.pdf)]-->
<!--- `ASCOOD`: Going Beyond Conventional OOD Detection. Arxiv 2025[[paper](https://arxiv.org/abs/2411.10794)]-->
<!--###### Method Uncertainty
- `MC-Dropout`: . ICML 2016[[paper]()]
- `Deep-ensemble`: . NeurIPS 2017[[paper]()]
- `Temp-scaling`: . ICML 2017[[paper]()]
- `RTS`: . AAAI 2023[[paper]()]-->

##### CLIP-based Methods
- `MCM`: Delving into Out-of-Distribution Detection with Vision-Language Representations. NeurIPS 2022[[paper](https://arxiv.org/abs/2211.13445)]
- `GL-MCM`: GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot Out-of-Distribution Detection. IJCV 2025[[paper](https://arxiv.org/abs/2304.04521)]
- `NegLabel`: Negative Label Guided OOD Detection with Pretrained Vision-Language Models. ICLR 2024[[paper](https://arxiv.org/abs/2403.20078)]
- `CoOp`: Learning to Prompt for Vision-Language Models. IJCV 2022[[paper](https://arxiv.org/abs/2109.01134)]
- `LoCoOp`: LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning. NeurIPS 2023[[paper](https://arxiv.org/abs/2306.01293)]
- `SCT`: Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection. NeurIPS 2024[[paper](https://arxiv.org/abs/2411.03359)]
<!--- `Maple`: MaPLe: Multi-modal Prompt Learning. CVPR 2023[[paper](https://arxiv.org/abs/2210.03117)]-->
- `DPM`: Vision-Language Dual-Pattern Matching for Out-of-Distribution Detection. ECCV 2024[[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11399.pdf)]
<!--- `CALIP`: CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention. AAAI 2023[[paper](https://arxiv.org/abs/2209.14169)]-->
<!--- `Tip-Adapter`: Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling. ECCV 2022[[paper](https://arxiv.org/abs/2111.03930)]-->
<!--- `NegPrompt`: Learning Transferable Negative Prompts for Out-of-Distribution Detection. CVPR 2024[[paper](https://arxiv.org/abs/2404.03248)]-->

#### Few-shot class-incremental learning
- `Alice`: Few-Shot Class-Incremental Learning from an Open-Set Perspective. ECCV 2022 [[paper](https://arxiv.org/abs/2208.00147)]
- `FACT`: Forward Compatible Few-Shot Class-Incremental Learning. CVPR 2022 [[paper](https://arxiv.org/abs/2203.06953)]
- `SAVC`: Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning. CVPR 2023 [[paper](https://arxiv.org/abs/2304.00426)]

### 🤝Contributors

### 📖Citation
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
```
### 🙏Acknowledgement
- [OpenOOD](https://github.com/Jingkang50/OpenOOD), an extensible codebase for out-of-distribution detection with Vision Models only.
- [OpenOOD-VLM](https://github.com/YBZh/OpenOOD-VLM), an extensible codebase for out-of-distribution detection with both Vision Models and Vision-Language Models.
- [PyCIL](https://github.com/G-U-N/PyCIL), an extensible codebase for incremental learning.

### ✉️Contact
If there are any questions, please feel free to propose new features by opening an issue or contact with the author: Xiang Xiang. Enjoy the code.