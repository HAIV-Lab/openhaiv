import clip
import torch.nn as nn
from .customclip_negprompt import CustomCLIP_NegPrompt
from ncdia.utils import get_class_names, load_clip_to_cpu, get_text_features_neg
from ncdia.utils import MODELS, Configs

@MODELS.register
class CoOp_NegOODPrompt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        classnames = get_class_names(cfg.backbone.dataset) # imagenet 
        self.n_cls = len(classnames) 
        self.n_output = self.n_cls + int(cfg.backbone.OOD_NUM)
        backbone = cfg.backbone.name # 'ViT-B/16'
        assert backbone in clip.available_models()
        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(backbone)
        
        self.logit_scale = clip_model.logit_scale.data
        print("Building custom CLIP")
        self.model = CustomCLIP_NegPrompt(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        text_center = cfg.backbone.text_center
        self.text_features, self.text_features_unselected = get_text_features_neg(self.model, cfg.backbone.dataset, cfg.backbone.text_prompt, text_center, cfg.backbone.OOD_NUM)
        print('shape of pre-computed text features:', self.text_features.shape)
                

    def forward(self, x, return_feat=False):
        return self.model(x, return_feat)
