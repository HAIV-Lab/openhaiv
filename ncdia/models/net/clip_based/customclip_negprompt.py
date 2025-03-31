import torch.nn as nn
from .negpromptlearner import NegPromptLearner
from .clip_utils import TextEncoder
from ncdia.utils import MODELS, Configs

@MODELS.register
class CustomCLIP_NegPrompt(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = NegPromptLearner(cfg.backbone, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vanilla_clip = clip_model.cuda()
        self.text_features = None

    def forward(self, image, return_feat):
        image_features = self.image_encoder(image.type(self.dtype)) ##128*512
        if not self.training and self.text_features is not None:
                text_features = self.text_features  ## accrelating testing. 
        else:
            print('re-calculate the text feature with learned prompts.')
            prompts = self.prompt_learner() # torch.Size([1000, 77, 512])
            tokenized_prompts = self.tokenized_prompts  ## 1000*77
            text_features = self.text_encoder(prompts, tokenized_prompts) # 1000*512
            self.text_features = text_features
            
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if return_feat:
            return image_features, text_features, logit_scale
        else:
            logits = logit_scale * image_features @ text_features.t()
            return logits