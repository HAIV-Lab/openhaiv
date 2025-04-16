import torch
import numpy as np

from ncdia.utils import AUGMENTATIONS

__all__ = ['AugExpand']

# custom functions for AugExpand
CUSTOMFUNCS = {}


@AUGMENTATIONS.register
class AugExpand(torch.nn.Module):
    """ Expand the input tensor to multiple views.

               image
                 |
        ——————————————————
        |                |
        |            preaugment
    baseaugment          |
        |            aug_funcs
        |                |
        | (1 view)       | (n-1 views)
        ——————————————————
                 |
             preprocess
                 |
          augmented images

    Args:
        preaugment (callable): pre-augmentation function before applying AugExpand.
        preprocess (callable): pre-processing function after applying AugExpand.
        baseaugment (callable, optional): 
            augmentation to generate an original view without AugExpand,
            if None, original view is not included in the output.
        custom_funcs (list, optional): list of custom augmentation functions.
        n_views (int, optional): number of output views.

    Returns:
        torch.Tensor: expanded images, the size is (n_views, C, H, W),
            if baseaugment is not None, the first view is the original image.

    Examples:
        >>> from torchvision import transforms
        >>> from dataset.augexpand import AugExpand
        >>> mean = [0.485, 0.456, 0.406]
        >>> std = [0.229, 0.224, 0.225]
        >>> preprocess = transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     transforms.Normalize(mean, std)
        >>> ])
        >>> preaugment = transforms.Compose([
        >>>     transforms.RandomResizedCrop(224),
        >>>     transforms.RandomHorizontalFlip()
        >>> ])
        >>> augexpand = AugExpand(
        >>>     preaugment,
        >>>     preprocess,
        >>>     baseaugment=None,
        >>>     custom_funcs=[],
        >>>     n_views=2
        >>> )
        >>> image = Image.open('path/to/image.jpg')
        >>> augexpand(image)
        >>> # output: torch.Tensor (2, 3, 224, 224)

    """
    def __init__(
        self,
        preaugment,
        preprocess,
        baseaugment = None,
        custom_funcs: list = [],
        n_views: int = 2,
    ):
        super().__init__()
        self.preaugment = preaugment
        self.preprocess = preprocess
        self.baseaugment = baseaugment
        self.custom_funcs = custom_funcs or []
        self.n_views = n_views
    
    def augexpand(self, image, aug_funcs=[]):
        """ Apply augmentations to an image.
        Args:
            image (PIL.Image): input image.
            aug_funcs (list): list of augmentation functions.
        Returns:
            torch.Tensor: augmented image.
        """
        x_orig = self.preaugment(image)
        x_processed = self.preprocess(x_orig)
        if len(aug_funcs) == 0:
            return x_processed
        
        x_aug = x_orig.copy()
        x_aug = np.random.choice(aug_funcs)(x_aug)
        return self.preprocess(x_aug)

    def forward(self, img):
        """ Apply AugExpand to an image.
        Args:
            img (PIL.Image): input image.
        Returns:
            torch.Tensor: expanded image.
        """
        if self.baseaugment:
            base_img = [self.preprocess(self.baseaugment(img))]
        else:
            base_img = []
        
        views = [
            self.augexpand(img, self.custom_funcs)
            for _ in range(self.n_views - len(base_img))]
        
        return torch.stack(base_img + views)
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        format_string += f'preaugment={self.preaugment}, '
        format_string += f'preprocess={self.preprocess}, '
        format_string += f'baseaugment={self.baseaugment}, '
        format_string += f'custom_funcs={self.custom_funcs}, '
        format_string += f'n_views={self.n_views})'
        return format_string
