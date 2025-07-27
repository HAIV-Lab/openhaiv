from .clip import clip
from .clip_dpm import clip_dpm
from .clip_locoop import clip_locoop

# from .clip_maple import clip_maple
import torch.nn as nn
from .customclip import (
    CustomCLIP,
    CustomCLIP_LoCoOp,
    CustomCLIP_DPM,
)  # , CustomCLIP_Maple, CustomCLIP_NegPrompt,
from .clip_utils import *
from ncdia.utils import MODELS, Configs


@MODELS.register
class CoOp(nn.Module):
    def __init__(
        self,
        backbone,
        local_path,
        dataset,
        N_CTX,
        CTX_INIT,
        image_size,
        CSC,
        CLASS_TOKEN_POSITION,
        # checkpoint=None,
    ):
        super().__init__()
        classnames = get_class_names(dataset)  # imagenet
        self.n_cls = len(classnames)
        self.n_output = self.n_cls
        assert backbone in clip.available_models()
        # clip_model, self.preprocess = clip.load(backbone, device='cuda')
        if local_path == "":
            local_path = None
            print(f"Loading CLIP (backbone: {backbone})")
        else:
            print(f"Loading CLIP (backbone from {local_path})")
        clip_model = load_clip_to_cpu(backbone, local_path)

        self.logit_scale = clip_model.logit_scale.data
        # self.zeroshot_weights = zeroshot_classifier(clip_model, classnames,
        #                                             templates)
        print("Building custom CLIP for CoOp")
        self.model = CustomCLIP(
            classnames,
            clip_model,
            N_CTX,
            CTX_INIT,
            image_size,
            CSC,
            CLASS_TOKEN_POSITION,
        )

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def forward(self, x, return_feat=False):
        return self.model(x, return_feat)

    def get_features(self, x, return_feat=True):
        return self.model(x, return_feat)


"""  
LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning 
NeurIPS 2023 https://arxiv.org/abs/2306.01293
SCT: Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection 
NeurIPS 2024 https://arxiv.org/abs/2411.03359
"""


@MODELS.register
class LoCoOp(nn.Module):
    def __init__(
        self,
        backbone,
        local_path,
        dataset,
        N_CTX,
        CTX_INIT,
        image_size,
        CSC,
        CLASS_TOKEN_POSITION,
    ):
        super().__init__()
        classnames = get_class_names(dataset)  # imagenet
        self.n_cls = len(classnames)
        self.n_output = self.n_cls
        assert backbone in clip_locoop.available_models()
        # clip_model, self.preprocess = clip.load(backbone, device='cuda')
        if local_path == "":
            local_path = None
            print(f"Loading CLIP (backbone: {backbone})")
        else:
            print(f"Loading CLIP (backbone from {local_path})")
        clip_model = load_clip_to_cpu_locoop(backbone, local_path)

        self.logit_scale = clip_model.logit_scale.data
        # self.zeroshot_weights = zeroshot_classifier(clip_model, classnames,
        #                                             templates)
        print("Building custom CLIP for LoCoOp")
        self.model = CustomCLIP_LoCoOp(
            classnames,
            clip_model,
            N_CTX,
            CTX_INIT,
            image_size,
            CSC,
            CLASS_TOKEN_POSITION,
        )

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def forward(self, x, return_feat=False):
        return self.model(x, return_feat)

    def get_features(self, x, return_feat=True):
        return self.model(x, return_feat)


@MODELS.register
class DPM(nn.Module):
    def __init__(
        self,
        backbone,
        local_path,
        dataset,
        N_CTX,
        CTX_INIT,
        image_size,
        CSC,
        CLASS_TOKEN_POSITION,
        # checkpoint=None,
    ):
        super().__init__()
        classnames = get_class_names(dataset)  # imagenet
        self.n_cls = len(classnames)
        self.n_output = self.n_cls
        assert backbone in clip.available_models()

        if local_path == "":
            local_path = None
            print(f"Loading CLIP (backbone: {backbone})")
        else:
            print(f"Loading CLIP (backbone from {local_path})")
        clip_model = load_clip_to_cpu_dpm(backbone, local_path)

        self.logit_scale = clip_model.logit_scale.data
        # self.zeroshot_weights = zeroshot_classifier(clip_model, classnames,
        #                                             templates)
        print("Building custom CLIP for DPM")
        self.model = CustomCLIP_DPM(
            classnames,
            clip_model,
            N_CTX,
            CTX_INIT,
            image_size,
            CSC,
            CLASS_TOKEN_POSITION,
        )

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "dpmt" not in name:
                if "prompt_learner" not in name:
                    param.requires_grad_(False)

    def forward(self, x, label, return_feat=False):
        return self.model(x, label, return_feat)

    def evaluate(self, x, return_feat=False):
        return self.model.evaluate(x, return_feat)

    def get_features(self, x, return_feat=True):
        return self.model.evaluate(x, return_feat)


# '''
# MaPLe: Multi-modal Prompt Learning
# CVPR 2023 https://arxiv.org/abs/2210.03117
# '''
# @MODELS.register
# class Maple(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         classnames = get_class_names(cfg.backbone.dataset) # imagenet
#         self.n_cls = len(classnames)
#         self.n_output = self.n_cls
#         # templates = get_templates(cfg.backbone.text_prompt) # simple
#         backbone = cfg.backbone.name # 'ViT-B/16'
#         assert backbone in clip_maple.available_models()
#         print(f"Loading CLIP (backbone: {backbone})")
#         clip_model = load_clip_to_cpu_maple(backbone, cfg)
#         clip_model_official = load_clip_to_cpu(backbone)

#         self.logit_scale = clip_model.logit_scale.data
#         print("Building custom CLIP for Maple")
#         self.model = CustomCLIP_Maple(cfg, classnames, clip_model)

#         print("Turning off gradients in both the image and the text encoder")
#         name_to_update = "prompt_learner"

#         for name, param in self.model.named_parameters():
#             if name_to_update not in name:
#                 # Make sure that VPT prompts are updated
#                 if "VPT" in name:
#                     param.requires_grad_(True)
#                 else:
#                     param.requires_grad_(False)
#         # Double check
#         enabled = set()
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 enabled.add(name)
#         print(f"Parameters to be updated: {enabled}")
#         self.text_features = get_text_features(clip_model_official.cuda(), cfg.backbone.dataset, cfg.backbone.text_prompt)
#         print('shape of pre-computed text features:', self.text_features.shape)


#     def forward(self, x, return_feat=False):
#         return self.model(x, return_feat)


# '''
# Learning Transferable Negative Prompts for Out-of-Distribution Detection
# CVPR 2024 https://arxiv.org/abs/2404.03248
# '''
# @MODELS.register
# class NegPrompt(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         classnames = get_class_names(cfg.backbone.dataset) # imagenet
#         self.n_cls = len(classnames)
#         self.n_output = self.n_cls + int(cfg.backbone.OOD_NUM)
#         backbone = cfg.backbone.name # 'ViT-B/16'
#         assert backbone in clip.available_models()
#         print(f"Loading CLIP (backbone: {backbone})")
#         clip_model = load_clip_to_cpu(backbone)

#         self.logit_scale = clip_model.logit_scale.data
#         print("Building custom CLIP for NegPrompt")
#         self.model = CustomCLIP_NegPrompt(cfg, classnames, clip_model)

#         print("Turning off gradients in both the image and the text encoder")
#         for name, param in self.model.named_parameters():
#             if "prompt_learner" not in name:
#                 param.requires_grad_(False)

#         text_center = cfg.backbone.text_center
#         self.text_features, self.text_features_unselected = get_text_features_neg(self.model, cfg.backbone.dataset, cfg.backbone.text_prompt, text_center, cfg.backbone.OOD_NUM)
#         print('shape of pre-computed text features:', self.text_features.shape)


#     def forward(self, x, return_feat=False):
#         return self.model(x, return_feat)
