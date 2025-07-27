import clip
import torch
import torch.nn as nn
from ncdia.utils import MODELS, Configs
from .clip_maple import simple_tokenizer as _Tokenizer

_tokenizer = _Tokenizer()


@MODELS.register
class NegPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX  # 16 or 4
        OOD_NUM = cfg.OOD_NUM  # number of ood prompts
        self.OOD_NUM = OOD_NUM
        ctx_init = cfg.CTX_INIT  # ''
        prompttype = cfg.prompttype
        dtype = clip_model.dtype  # torch.float16
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        clip_imsize = clip_model.visual.input_resolution  ## 224
        cfg_imsize = cfg.image_size  # 224

        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

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
                ood_ctx_vectors = torch.empty(
                    OOD_NUM, n_ctx, ctx_dim, dtype=dtype
                )  # OOD_NUM*77*512
                nn.init.normal_(ctx_vectors, std=0.02)
                nn.init.normal_(ood_ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
                ood_prompt_prefix = " ".join(["X"] * (n_ctx + 1))
                self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                self.ood_ctx = nn.Parameter(
                    ood_ctx_vectors
                )  # to be optimized, 1*77*512
            else:
                if prompttype == "dis_aware":
                    print("Initializing a distribution aware context")
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # 16*512
                    ood_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # 16*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    nn.init.normal_(ood_ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx + 1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = nn.Parameter(
                        ood_ctx_vectors
                    )  # to be optimized, 1*77*512
                elif prompttype == "unified":
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # 16*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    ood_ctx_vectors = ctx_vectors
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx + 1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = self.ctx
                elif prompttype == "class_specific":
                    ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                    ood_ctx_vectors = torch.empty(
                        OOD_NUM, n_ctx, ctx_dim, dtype=dtype
                    )  # OOD_NUM*77*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    nn.init.normal_(ood_ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx + 1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = nn.Parameter(
                        ood_ctx_vectors
                    )  # to be optimized, 1*77*512
                else:
                    raise NotImplementedError

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [
            prompt_prefix + " " + name + "." for name in classnames
        ]  # 'X X X X X X X X X X X X X X X X toilet paper.'
        (
            selected_adj_text,
            selected_noun_text,
            unselected_adj_text,
            unselected_noun_text,
        ) = get_selected_ood_text_list(self.OOD_NUM)
        selected_ood_text = selected_adj_text + selected_noun_text
        ood_prompts = [prompt_prefix + " " + name + "." for name in selected_ood_text]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # 1000*77
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype
            )  # 1000*77*512

        ood_tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in ood_prompts]
        )  # ood number *77
        with torch.no_grad():
            ood_embedding = clip_model.token_embedding(ood_tokenized_prompts).type(
                dtype
            )  # 1000*77*512
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS, 1000*1*512
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :]
        )  # CLS, EOS, 1000*60*512

        self.register_buffer("ood_token_prefix", ood_embedding[:, :1, :])  # SOS,
        self.register_buffer(
            "ood_token_suffix", ood_embedding[:, 1 + n_ctx :, :]
        )  # CLS, EOS,
        self.n_cls = n_cls  # 1000
        self.n_ctx = n_ctx  # 16

        self.tokenized_prompts = torch.cat(
            (tokenized_prompts, ood_tokenized_prompts), dim=0
        )  # torch.Tensor, 1001*77
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION  # end

    def forward(self):
        ctx_vanilla = self.ctx  # 16*512, parameters to learn.
        ood_ctx_vanilla = self.ood_ctx
        if ctx_vanilla.dim() == 2:
            ctx = ctx_vanilla.unsqueeze(0).expand(self.n_cls, -1, -1)  # 100*16*512
            ctx_ood = ood_ctx_vanilla.unsqueeze(0).expand(
                self.OOD_NUM, -1, -1
            )  # 100*16*512
        else:
            ctx = ctx_vanilla
            ctx_ood = ood_ctx_vanilla

        prefix = self.token_prefix  ## 1000*1*512
        suffix = self.token_suffix  ## 1000*60*512

        ood_prefix = self.ood_token_prefix  ## 1000*1*512
        ood_suffix = self.ood_token_suffix  ## 1000*60*512

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            # pdb.set_trace()
            ood_prompts = torch.cat(
                [
                    ood_prefix,  # (n_cls, 1, dim)
                    ctx_ood,  # (n_cls, n_ctx+1, dim)
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
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
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
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
