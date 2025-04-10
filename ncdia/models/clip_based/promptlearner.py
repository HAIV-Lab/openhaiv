import clip
import torch
import torch.nn as nn
import copy
from copy import deepcopy
from ncdia.utils import MODELS, Configs
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# Textual Prompt Learner
@MODELS.register
class PromptLearner(nn.Module):
    def __init__(
        self, 
        # cfg, 
        classnames, 
        clip_model,
        N_CTX,
        CTX_INIT,
        image_size,
        CSC,
        CLASS_TOKEN_POSITION
    ):
        super().__init__()
        n_cls = len(classnames)
        # n_ctx = cfg.N_CTX 
        # ctx_init = cfg.CTX_INIT  
        n_ctx = N_CTX
        ctx_init = CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.image_size # 224
        cfg_imsize = image_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
    
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            # if cfg.CSC:
            print("Random Initialization")
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            # print(f"init ctx_vectors: {ctx_vectors}")
            prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # print('ctx:', self.ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.CLASS_TOKEN_POSITION
        self.class_token_position = CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            # print(f"ctx:{ctx}")
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
# Textual and Visual Prompt Learner
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
@MODELS.register
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.backbone.N_CTX 
        ctx_init = cfg.backbone.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.backbone.image_size # 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        assert cfg.backbone.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.backbone.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
    
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.backbone.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768) # to be optimized
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.backbone.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required
    
# Textual Prompt Learner for NegPrompt    
@MODELS.register
class NegPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX 
        OOD_NUM = cfg.OOD_NUM # number of ood prompts
        self.OOD_NUM = OOD_NUM
        ctx_init = cfg.CTX_INIT  # ''
        prompttype = cfg.prompttype
        dtype = clip_model.dtype # torch.float16
        ctx_dim = clip_model.ln_final.weight.shape[0] # 512
        clip_imsize = clip_model.visual.input_resolution ## 224
        cfg_imsize = cfg.image_size # 224

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding_temp = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding_temp[0, 1 : 1 + n_ctx, :]
            ood_ctx_vectors = embedding_temp[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
            ood_prompt_prefix = ctx_init
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
            self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512
        else:
            # random initialization
            if cfg.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ood_ctx_vectors = torch.empty(OOD_NUM, n_ctx, ctx_dim, dtype=dtype) # OOD_NUM*77*512
                nn.init.normal_(ctx_vectors, std=0.02)
                nn.init.normal_(ood_ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
                ood_prompt_prefix = " ".join(["X"] * (n_ctx+1))
                self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512
            else:
                if prompttype == 'dis_aware':
                    print("Initializing a distribution aware context")
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # 16*512
                    ood_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # 16*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    nn.init.normal_(ood_ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx+1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512
                elif prompttype == 'unified':
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # 16*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    ood_ctx_vectors = ctx_vectors
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx+1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = self.ctx
                elif prompttype == 'class_specific':
                    ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                    ood_ctx_vectors = torch.empty(OOD_NUM, n_ctx, ctx_dim, dtype=dtype) # OOD_NUM*77*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    nn.init.normal_(ood_ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx+1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512
                else:
                    raise NotImplementedError


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
        prompts = [prompt_prefix + " " + name + "." for name in classnames]  # 'X X X X X X X X X X X X X X X X toilet paper.'
        selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text = get_selected_ood_text_list(self.OOD_NUM)
        selected_ood_text = selected_adj_text + selected_noun_text
        ood_prompts = [prompt_prefix + " " + name + "." for name in selected_ood_text]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 1000*77
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 1000*77*512
        
        ood_tokenized_prompts = torch.cat([clip.tokenize(p) for p in ood_prompts]) # ood number *77
        with torch.no_grad():
            ood_embedding = clip_model.token_embedding(ood_tokenized_prompts).type(dtype) # 1000*77*512
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS, 1000*1*512
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS, 1000*60*512

        self.register_buffer("ood_token_prefix", ood_embedding[:, :1, :])  # SOS, 
        self.register_buffer("ood_token_suffix", ood_embedding[:, 1 + n_ctx :, :])  # CLS, EOS, 
        self.n_cls = n_cls # 1000
        self.n_ctx = n_ctx # 16
        
        self.tokenized_prompts = torch.cat((tokenized_prompts, ood_tokenized_prompts), dim=0)  # torch.Tensor, 1001*77
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION # end

    def forward(self):
        ctx_vanilla = self.ctx # 16*512, parameters to learn.
        ood_ctx_vanilla = self.ood_ctx
        if ctx_vanilla.dim() == 2:
            ctx = ctx_vanilla.unsqueeze(0).expand(self.n_cls, -1, -1) # 100*16*512
            ctx_ood = ood_ctx_vanilla.unsqueeze(0).expand(self.OOD_NUM, -1, -1) # 100*16*512
        else:
            ctx = ctx_vanilla
            ctx_ood = ood_ctx_vanilla

        prefix = self.token_prefix ## 1000*1*512
        suffix = self.token_suffix ## 1000*60*512

        ood_prefix = self.ood_token_prefix ## 1000*1*512
        ood_suffix = self.ood_token_suffix ## 1000*60*512

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            ood_prompts = torch.cat(
                [
                    ood_prefix,  # (n_cls, 1, dim)
                    ctx_ood,     # (n_cls, n_ctx+1, dim)
                    ood_suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts = torch.cat((prompts, ood_prompts), dim=0)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        
        return prompts

@MODELS.register
class MLCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pos = cfg.TRAINER.COOP_MLC.N_CTX_POS  # 64
        ctx_init_pos = cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT  # .strip()  #template
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init_pos:
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            prompt_prefix_pos = ctx_init_pos
            if cfg.TRAINER.COOP_MLC.CSC:
                ctx_vectors_pos_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)

        else:
            # Random Initialization
            if cfg.TRAINER.COOP_MLC.CSC:  # default
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print("Number of Class:", len(classnames))

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]

        tokenized_prompts_pos = []
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])

        self.n_cls = n_cls  
        self.n_ctx_pos = n_ctx_pos
        tokenized_prompts = tokenized_prompts_pos
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):  
        ctx_pos = self.ctx_pos
        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[cls_id]


        if cls_id is None:
            prefix_pos = self.token_prefix_pos
            suffix_pos = self.token_suffix_pos
        else:  # suffix [10, 12, 512]  prefix [10,1,512]
            prefix_pos = self.token_prefix_pos[cls_id]  # [1,2,1,512]
            suffix_pos = self.token_suffix_pos[cls_id]  # [1,2,12,512]

        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim) # [2,1,64,512]
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts = prompts_pos  # torch.cat([prompts_neg, prompts_pos], dim=0)

        if cls_id is not None:
            tokenized_prompts_pos = self.tokenized_prompts[cls_id]
            tokenized_prompts = tokenized_prompts_pos  # torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts

        return prompts, tokenized_prompts