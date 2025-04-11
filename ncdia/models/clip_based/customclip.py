import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .clip import clip
from .clip_locoop import clip_locoop
from .clip_maple import clip_maple
from .clip_dpm import clip_dpm
from .promptlearner import PromptLearner, NegPromptLearner, MultiModalPromptLearner, MLCPromptLearner
from .clip_utils import load_clip_to_cpu, get_text_features
from ncdia.utils import MODELS, Configs

@MODELS.register
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding 
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection 
        self.dtype = clip_model.dtype
        
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

@MODELS.register
class TextEncoder_LoCoOp(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding 
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection 
        self.dtype = clip_model.dtype
        
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

@MODELS.register
class TextEncoder_Maple(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

# custom CLIP for zero-shot classification
@MODELS.register
class CustomCLIP_ZeroShot(nn.Module):
    def __init__(
        self, 
        backbone,
        dataset,
        text_prompt,
        **kwargs
    ) -> None:
        super().__init__()
        backbone = backbone # 'ViT-B/16'
        assert backbone in clip.available_models()
        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(backbone)
        clip_model = clip_model.cuda()
        self.model = clip_model
        self.logit_scale = clip_model.logit_scale.data
        print("Turning off gradients in both the image and the text encoder")
        for name, param in clip_model.named_parameters():
            param.requires_grad_(False) 
        self.text_features = get_text_features(self.model, dataset, text_prompt)


    def forward(self, x):
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        text_features = self.text_features.squeeze(1) 

        logits = logit_scale * image_features @ text_features.T 
        return logits

    def get_features(self, x):
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
'''
custom CLIP for prepross NegLabel text features
Negative Label Guided OOD Detection with Pretrained Vision-Language Models
ICLR 2024 spotlight, https://arxiv.org/abs/2403.20078
'''

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
        print("Turning off gradients in both the image and the text encoder")
        for name, param in clip_model.named_parameters():
            param.requires_grad_(False)
    
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

# custom CLIP for vanilla prompt learning
@MODELS.register
class CustomCLIP(nn.Module):
    def __init__(
        self, 
        # backbone, 
        classnames, 
        clip_model,
        N_CTX,
        CTX_INIT,
        image_size,
        CSC,
        CLASS_TOKEN_POSITION
    ) -> None:
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, N_CTX, CTX_INIT, image_size, CSC, CLASS_TOKEN_POSITION)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, return_feat):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if return_feat:
            return image_features
        else:
            logits = logit_scale * image_features @ text_features.t()
            return logits

'''
custom CLIP with global_image_feature and local_image_features, which can be used for LoCoOp and SCT etc.   
LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning 
NeurIPS 2023 https://arxiv.org/abs/2306.01293
SCT: Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection 
NeurIPS 2024 https://arxiv.org/abs/2411.03359
'''
@MODELS.register
class CustomCLIP_LoCoOp(nn.Module):
    def __init__(
        self, 
        # backbone, 
        classnames, 
        clip_model,
        N_CTX,
        CTX_INIT,
        image_size,
        CSC,
        CLASS_TOKEN_POSITION
    ):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, N_CTX, CTX_INIT, image_size, CSC, CLASS_TOKEN_POSITION)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_LoCoOp(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, return_feat):
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_image_features @ text_features.T

        # return logits, logits_local
        if return_feat:
            return image_features, local_image_features
        else:
            return logits, logits_local


        
'''
custom CLIP for Dual-pattern Matching
Vision-Language Dual-Pattern Matching for Out-of-Distribution Detection
ECCV 2024 https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11399.pdf
'''
@MODELS.register
class CustomCLIP_DPM(nn.Module):
    def __init__(
        self, 
        # backbone, 
        classnames, 
        clip_model,
        N_CTX,
        CTX_INIT,
        image_size,
        CSC,
        CLASS_TOKEN_POSITION
    ):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        # self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(classnames, clip_model, N_CTX, CTX_INIT, image_size, CSC, CLASS_TOKEN_POSITION)
        for _, param in clip_model.named_parameters():
            param.requires_grad = False
        with torch.no_grad():
            temp = "a photo of a {}."
            prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.dpmt = DPM_Block(text_features, 512)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def forward(self, image, label, cls_id=None):
        image_features, local_features = self.image_encoder(
            image.type(self.dtype))  # image_features, [B, C], local [B, 49, C]
        # prompts, tokenized_prompts = self.prompt_learner(cls_id)  # prompts [2*L, 77, D]  tokenized_prompts  [2*L, 77]
        prompts = self.prompt_learner()  
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)  # text_features [2*L, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [2*L, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # image_f [B, C]
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)  # local [B, 49, C]
        logits1, logits2, logits3 = self.dpmt(Fs=local_features, Ft=text_features, Fv=image_features,
                                              label=label)  # .squeeze()
        return logits1, logits2, logits3

    def evaluate(self, image, cls_id=None):
        image_features, local_features = self.image_encoder(
            image.type(self.dtype))  # image_features, [B, C], local [B, 49, C]
        # prompts, tokenized_prompts = self.prompt_learner(
            # cls_id)  # prompts [2*L, 77, D]  tokenized_prompts  [2*L, 77]
        prompts = self.prompt_learner()  
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)  # text_features [2*L, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [2*L, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # image_f [B, C]
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)  # local [B, 49, C]
        logits1, logits2, logits3 = self.dpmt.evaluate(Fs=local_features, Ft=text_features,
                                                       Fv=image_features)  # .squeeze()
        return logits1, logits2, logits3
    
@MODELS.register
class DPM_Block(nn.Module):
    def __init__(self, text_features,
                 input_dim):  # input_dim=512
        super().__init__()
        self.softmax = nn.Softmax(-1)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pre_project_s = DPM_Proj2()  # (B, D)
        self.pre_project_t = DPM_Proj1()
        self.pre_project_vv = DPM_Proj1()
        self.scale = input_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.ones([]) * 30., requires_grad=False)
        self.vis_gamma_p = nn.Parameter(torch.ones([]) * 0.99)  # 1e-3)  # for updating visual embedding diff
        self.vis_gamma_n = nn.Parameter(torch.ones([]) * 0.99)  # 1e-3)  # for updating visual embedding diff
        self.visual_prototype = nn.Parameter(text_features.clone().detach())  # , requires_grad=False)

    def forward(self, Fs, Ft, Fv, label):
        L, D = Ft.shape
        B, _, C = Fs.shape
        Fs = self.pre_project_s(Fs)
        Fs = Fs.half()
        A_weight = F.conv1d(Fs.permute(0, 2, 1), Ft[:, :, None])  # [B, 2L, 49]
        A_weight1 = F.softmax(A_weight, dim=-1)  # [B, 2L, 49]
        feat_v_a = A_weight1 @ Fs  # [B, L, C]
        Fv = self.pre_project_vv(Fv)
        Fv = Fv.half()
        Fv = Fv.unsqueeze(1)
        Fv = Fv.expand(-1, L, -1)
        feat_v = self.vis_gamma_p * feat_v_a + Fv  # [B, L, C] + Fv
        A_weightv = F.conv1d(Fs.permute(0, 2, 1), self.visual_prototype[:, :, None])  # [B, 2L, 49]
        A_weight1v = F.softmax(A_weightv, dim=-1)  # [B, 2L, 49]
        feat_v_av = A_weight1v @ Fs  # [B, L, C]
        feat_vv = self.vis_gamma_n * feat_v_av + Fv  # [B, L, C] + Fv
        Ft = F.normalize(Ft, dim=-1, p=2)  # [L, C]
        Fv = F.normalize(Fv, dim=-1, p=2)  # [B, L, C]
        feat_v = F.normalize(feat_v, dim=-1, p=2)  # [B, L, C]
        feat_vv = F.normalize(feat_vv, dim=-1, p=2)  # [B, L, C]
        logits1 = torch.mul(Fv, Ft).sum(-1)
        logits2 = torch.mul(feat_v, Ft).sum(-1)
        logits3 = torch.mul(feat_vv, self.visual_prototype).sum(-1)
        with torch.no_grad():
            class_count = torch.bincount(label, minlength=L)
            class_sum = Fv[:, 0, :].new_zeros(L, C)
            class_sum.index_add_(0, label, Fv[:, 0, :])
            # safe_class_count = class_count.float().unsqueeze(1).clamp_min(1e-8)
            safe_class_count = class_count.unsqueeze(1).clamp_min(1e-8)
            class_mean = class_sum / safe_class_count
            class_mean = class_mean.half()
            mask = class_count > 0
            new_visual_prototype = 0.99 * self.visual_prototype + 0.01 * class_mean
            updated_visual_prototype = self.visual_prototype.clone()
            updated_visual_prototype[mask] = new_visual_prototype[mask]
            self.visual_prototype.data = updated_visual_prototype
        return logits1, logits2, logits3

    def evaluate(self, Fs, Ft, Fv):
        L, D = Ft.shape
        B, _, C = Fs.shape
        Fs = self.pre_project_s(Fs)
        Fs = Fs.half()
        A_weight = F.conv1d(Fs.permute(0, 2, 1), Ft[:, :, None])  # [B, 2L, 49]
        A_weight1 = F.softmax(A_weight, dim=-1)  # [B, 2L, 49]
        feat_v_a = A_weight1 @ Fs  # [B, L, C]
        Fv = self.pre_project_vv(Fv)
        Fv = Fv.half()
        Fv = Fv.unsqueeze(1)
        Fv = Fv.expand(-1, L, -1)
        feat_v = self.vis_gamma_p * feat_v_a + Fv  # [B, L, C] + Fv
        A_weightv = F.conv1d(Fs.permute(0, 2, 1), self.visual_prototype[:, :, None])  # [B, 2L, 49]
        A_weight1v = F.softmax(A_weightv, dim=-1)  # [B, 2L, 49]
        feat_v_av = A_weight1v @ Fs  # [B, L, C]
        feat_vv = self.vis_gamma_n * feat_v_av + Fv  # [B, L, C] + Fv
        Ft = F.normalize(Ft, dim=-1, p=2)  # [L, C]
        Fv = F.normalize(Fv, dim=-1, p=2)  # [B, L, C]
        feat_v = F.normalize(feat_v, dim=-1, p=2)  # [B, L, C]
        feat_vv = F.normalize(feat_vv, dim=-1, p=2)  # [B, L, C]
        logits1 = self.logit_scale * torch.mul(Fv, Ft).sum(-1)
        logits2 = self.logit_scale * torch.mul(feat_v, Ft).sum(-1)
        logits3 = self.logit_scale * torch.mul(feat_vv, self.visual_prototype).sum(-1)
        return logits1, logits2, logits3
    
@MODELS.register   
class DPM_Proj1(nn.Module):
    def __init__(self,
                 visual_dim=512,
                 token_embed_dim=512,
                 **kwargs
                 ):
        super(DPM_Proj1, self).__init__()

        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, token_embed_dim)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.prompt_proj(x.float())
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02).half()
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0).half()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0).half()
            nn.init.constant_(m.weight, 1.0).half()

@MODELS.register
class DPM_Proj2(nn.Module):
    def __init__(self,
                 visual_dim=512,
                 token_embed_dim=512,
                 **kwargs
                 ):
        super(DPM_Proj2, self).__init__()

        self.prompt_proj = nn.Sequential(
            nn.GroupNorm(1, visual_dim),  # Use GroupNorm instead of LayerNorm
            nn.Conv1d(visual_dim, visual_dim, 1),
            nn.ReLU(),
            nn.GroupNorm(1, visual_dim),  # Use GroupNorm instead of LayerNorm
            nn.Conv1d(visual_dim, token_embed_dim, 1)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change the order of dimensions to (B, D, 49)
        x = self.prompt_proj(x.float())
        x = x.permute(0, 2, 1)  # Change the order of dimensions to (B, 49, D)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02).half()
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0).half()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0).half()
            nn.init.constant_(m.weight, 1.0).half()

'''
custom CLIP for NegPrompt 
Learning Transferable Negative Prompts for Out-of-Distribution Detection
CVPR 2024 https://arxiv.org/abs/2404.03248
'''   
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

'''
custom CLIP for CALIP
CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention
AAAI 2023 https://arxiv.org/abs/2209.14169
'''
# 以下代码实现有问题
@MODELS.register
class CustomCLIP_CALIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = cfg.backbone.name # 'ViT-B/16'
        assert backbone in clip.available_models()
        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(backbone)
        clip_model = clip_model.cuda()
        self.model = clip_model
        self.logit_scale = clip_model.logit_scale.data
        print("Turning off gradients in both the image and the text encoder")
        for name, param in clip_model.named_parameters():
            param.requires_grad_(False) 
        self.text_features = get_text_features(self.model, cfg.backbone.dataset, cfg.backbone.text_prompt)


    def forward(self, x):
        '''
        image_features: [B, D]
        image_spatial_features: [B, P, D]
        self.text_features: [N, D]
        '''
        image_global_features, image_spatial_features = self.model.encode_image(x)
        image_global_features /= image_global_features.norm(dim=-1, keepdim=True)
        image_spatial_features /= image_spatial_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits1 = logit_scale * image_global_features @ self.text_features 
        
        A_weight = torch.matmul(image_spatial_features, self.text_features.t())  # [B,P,D]x[D,N] -> [B,P,N]
        A_weight1 = F.softmax(A_weight, dim=0) # softmax along the spatial dimension
        A_weight2 = F.softmax(A_weight, dim=1) # softmax along the text dimension
        
        feat_t_a = torch.matmul(image_global_features, A_weight1) # [B,1,D][B,P,N]
        feat_v_a = torch.matmul(A_weight2, self.text_features.permute(1, 0)) 
        feat_v_a = feat_v_a.mean(0)+feat_v_a.max(0)[0]
        logits2 = image_global_features @ feat_t_a
        logits3 = feat_v_a @ self.text_features
        logits = logits1 + logits2 + logits3
        return logits
'''
custom CLIP for multi-modal prompt learning, eg. Maple
MaPLe: Multi-modal Prompt Learning
CVPR 2023 https://arxiv.org/abs/2210.03117
'''
@MODELS.register
class CustomCLIP_Maple(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_Maple(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, return_feat):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if return_feat:
            return image_features, text_features, logit_scale
        else:
            logits = logit_scale * image_features @ text_features.t()
            return logits
