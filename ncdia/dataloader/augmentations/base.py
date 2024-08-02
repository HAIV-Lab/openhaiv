from torchvision import transforms

from ncdia.utils import AUGMENTATIONS
from ncdia.dataloader.tools import _interpolation_modes_from_str
from .constrained_cropping import CustomMultiCropping
from .augexpand import CUSTOMFUNCS


AUGMENTATIONS.register_dict({
    'random_resized_crop': transforms.RandomResizedCrop,
    'random_horizontal_flip': transforms.RandomHorizontalFlip,
    'random_vertical_flip': transforms.RandomVerticalFlip,
    'random_rotation': transforms.RandomRotation,
    'random_affine': transforms.RandomAffine,
    'color_jitter': transforms.ColorJitter,
    'to_tensor': transforms.ToTensor,
    'normalize': transforms.Normalize,
    'resize': transforms.Resize,
    'center_crop': transforms.CenterCrop,
    'pad': transforms.Pad,
    'lambda': transforms.Lambda,
    'random_apply': transforms.RandomApply,
    'random_choice': transforms.RandomChoice,
    'random_crop': transforms.RandomCrop,
    'random_order': transforms.RandomOrder,
    'random_grayscale': transforms.RandomGrayscale,
    'random_perspective': transforms.RandomPerspective,
    'random_erasing': transforms.RandomErasing,
    'five_crop': transforms.FiveCrop,
    'ten_crop': transforms.TenCrop,
    'linear_transformation': transforms.LinearTransformation,
    'grayscale': transforms.Grayscale,
    'gaussian_blur': transforms.GaussianBlur,
    'multi_cropping': CustomMultiCropping,
})


def build_transform(trans: dict) -> transforms.Compose:
    """Build transform.

    Args:
        trans (dict): transform config
    
    Returns:
        torchvision.transforms.Compose: transform
    """
    if isinstance(trans, str):
        return trans
    
    transform = []
    
    for t in trans.keys():
        if t not in AUGMENTATIONS:
            raise ValueError(f'Unknown transform {t}.')

        transform.append(
            AUGMENTATIONS[t](**parse_parameters(trans[t]))
        )

    return transforms.Compose(transform)


def parse_parameters(params: dict) -> dict:
    """ Parse parameters for transforms.

    Args:
        params (dict): parameters for transforms

    Returns:
        dict: parsed parameters
    """
    def check_augments(params: dict, names: list):
        for name in names:
            if name in params:
                params[name] = build_transform(params[name])

    if not isinstance(params, dict):
        return {}
    else:
        if 'interpolation' in params:
            params['interpolation'] = _interpolation_modes_from_str(params['interpolation'])

        check_augments(params, ['baseaugment', 'preaugment', 'preprocess'])

        if 'custom_funcs' in params:
            custom_funcs = params['custom_funcs']
            if isinstance(custom_funcs, str):
                custom_funcs = [custom_funcs]
            params['custom_funcs'] = [CUSTOMFUNCS[func] for func in custom_funcs]
            
        return params
