from functools import partial
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._api import Weights, WeightsEnum
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from typing import Any, Dict

__all__ = [
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
]


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}


class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 86567656,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.072,
                    "acc@5": 95.318,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        transforms=partial(
            ImageClassification,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 86859496,
            "min_size": (384, 384),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                }
            },
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 86567656,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                }
            },
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_B_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88224232,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.912,
                    "acc@5": 92.466,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=242),
        meta={
            **_COMMON_META,
            "num_params": 304326632,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.662,
                    "acc@5": 94.638,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
        transforms=partial(
            ImageClassification,
            crop_size=512,
            resize_size=512,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 305174504,
            "min_size": (512, 512),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.064,
                    "acc@5": 98.512,
                }
            },
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 304326632,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.146,
                    "acc@5": 97.422,
                }
            },
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_32-c7638314.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 306535400,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.972,
                    "acc@5": 93.07,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
        transforms=partial(
            ImageClassification,
            crop_size=518,
            resize_size=518,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 633470440,
            "min_size": (518, 518),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.552,
                    "acc@5": 98.694,
                }
            },
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 632045800,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.708,
                    "acc@5": 97.730,
                }
            },
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_SWAG_E2E_V1
