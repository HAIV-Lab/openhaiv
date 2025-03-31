import clip
import torch.nn as nn
from .customclip_maple import CustomCLIP_Maple
from .clip_utils import load_clip_to_cpu, load_clip_to_cpu_maple, get_class_names, get_text_features
from ncdia.utils import MODELS, Configs

@MODELS.register
class Maple(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        classnames = get_class_names(cfg.backbone.dataset) # imagenet 
        self.n_cls = len(classnames)
        self.n_output = self.n_cls
        # templates = get_templates(cfg.backbone.text_prompt) # simple
        backbone = cfg.backbone.name # 'ViT-B/16'
        assert backbone in clip.available_models()
        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu_maple(backbone, cfg)
        clip_model_official = load_clip_to_cpu(backbone)
        
        self.logit_scale = clip_model.logit_scale.data
        print("Building custom CLIP")
        self.model = CustomCLIP_Maple(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        self.text_features = get_text_features(clip_model_official.cuda(), cfg.backbone.dataset, cfg.backbone.text_prompt) 
        print('shape of pre-computed text features:', self.text_features.shape)
                

    def forward(self, x, return_feat=False):
        return self.model(x, return_feat)