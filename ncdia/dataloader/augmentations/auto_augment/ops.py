# from PIL import Image, ImageEnhance, ImageOps
import random
import torch
import math
import numpy as np
from . import functional as F


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class ShearX(object):
    def __init__(self, interpolation, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.interpolation = interpolation

    def __call__(self, x, magnitude):
        # return x.transform(
        #     x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        #     Image.BICUBIC, fillcolor=self.fillcolor)
        return F.affine(x, angle=0.0, translate=[0, 0], scale=1.0, 
                        shear=[math.degrees(magnitude), 0.0], 
                        interpolation=self.interpolation, 
                        fill=self.fillcolor)


class ShearY(object):
    def __init__(self, interpolation, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.interpolation = interpolation

    def __call__(self, x, magnitude):
        # return x.transform(
        #     x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        #     Image.BICUBIC, fillcolor=self.fillcolor)
        return F.affine(x, angle=0.0, translate=[0, 0], scale=1.0, 
                        shear=[0.0, math.degrees(magnitude)], 
                        interpolation=self.interpolation, 
                        fill=self.fillcolor)


class TranslateX(object):
    def __init__(self, interpolation, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.interpolation = interpolation

    def __call__(self, x, magnitude):
        # return x.transform(
        #     x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
        #     fillcolor=self.fillcolor)
        return F.affine(x, angle=0.0, translate=[int(F._get_image_size(x)[0] * magnitude), 0], scale=1.0, 
                        interpolation=self.interpolation, shear=[0.0, 0.0], fill=self.fillcolor)


class TranslateY(object):
    def __init__(self, interpolation, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.interpolation = interpolation

    def __call__(self, x, magnitude):
        # return x.transform(
        #     x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
        #     fillcolor=self.fillcolor)
        return F.affine(x, angle=0.0, translate=[0, int(F._get_image_size(x)[1] * magnitude)], scale=1.0, 
                        interpolation=self.interpolation, shear=[0.0, 0.0], fill=self.fillcolor)


class Rotate(object):
    def __init__(self, interpolation, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.interpolation = interpolation

    def __call__(self, x, magnitude):
        # rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        # return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)
        return F.rotate(x, magnitude, interpolation=self.interpolation, fill=self.fillcolor)


class Posterize(object):
    def __call__(self, x, magnitude):
        # return ImageOps.posterize(x, magnitude)
        return F.posterize(x, int(magnitude))


class Solarize(object):
    def __call__(self, x, magnitude):
        # return ImageOps.solarize(x, magnitude)
        return F.solarize(x, magnitude)


class Color(object):
    def __call__(self, x, magnitude):
        # return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))
        return F.adjust_saturation(x, 1.0 + magnitude)


class Contrast(object):
    def __call__(self, x, magnitude):
        # return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))
        return F.adjust_contrast(x, 1.0 + magnitude)


class Sharpness(object):
    def __call__(self, x, magnitude):
        # return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))
        return F.adjust_sharpness(x, 1.0 + magnitude)


class Brightness(object):
    def __call__(self, x, magnitude):
        # return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))
        return F.adjust_brightness(x, 1.0 + magnitude)


class AutoContrast(object):
    def __call__(self, x, magnitude):
        # return ImageOps.autocontrast(x)
        return F.autocontrast(x)


class Equalize(object):
    def __call__(self, x, magnitude):
        # return ImageOps.equalize(x)
        return F.equalize(x)


class Invert(object):
    def __call__(self, x, magnitude):
        # return ImageOps.invert(x)
        return F.invert(x)
