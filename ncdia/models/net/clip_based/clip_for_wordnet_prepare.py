###################### calculate the cosine similarity matrix between wordnet text and imagenet classes.

import torch
import torch.nn as nn
from clip_utils import TextEncoder
from ncdia.utils import MODELS, Configs

@MODELS.register
class CLIP_scoring(nn.Module):
    def __init__(self, clip_model, train_preprocess, val_preprocess, tokenized_prompts):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess
        self.tokenized_prompts = tokenized_prompts
    
    def prepare_id(self, tokenized_prompts_id):
        with torch.no_grad():
            text_features = self.text_encoder(tokenized_prompts_id)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        cos_sim = text_features @ self.text_features.t()
        return cos_sim

    def forward(self, tokenized_prompts_wordnet):
        with torch.no_grad():
            text_features_wordnet = self.text_encoder(tokenized_prompts_wordnet)

            text_features_wordnet = text_features_wordnet / text_features_wordnet.norm(dim=-1, keepdim=True)

            cos_sim = text_features_wordnet @ self.text_features.t()

        return cos_sim








