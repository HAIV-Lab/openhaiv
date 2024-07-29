import numpy as np
from torch import Tensor
from ops import *
from typing import *
from functional.functional import InterpolationMode, _get_image_num_channels


def _get_transforms():
    # Transforms for ImageNet
    return [
        (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
        (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
        (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
        (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
        (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
        (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
        (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
        (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
        (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
        (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
        (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
        (("Color", 0.4, 0), ("Equalize", 0.6, None)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
    ]


def _get_magnitudes(_BINS = 10):
    # name: (magnitudes, signed)
    return {
        "ShearX": (np.linspace(0.0, 0.3, _BINS), True),
        "ShearY": (np.linspace(0.0, 0.3, _BINS), True),
        "TranslateX": (np.linspace(0.0, 150.0 / 331.0, _BINS), True),
        "TranslateY": (np.linspace(0.0, 150.0 / 331.0, _BINS), True),
        "Rotate": (np.linspace(0.0, 30.0, _BINS), True),
        "Brightness": (np.linspace(0.0, 0.9, _BINS), True),
        "Color": (np.linspace(0.0, 0.9, _BINS), True),
        "Contrast": (np.linspace(0.0, 0.9, _BINS), True),
        "Sharpness": (np.linspace(0.0, 0.9, _BINS), True),
        "Posterize": (np.array([8, 8, 7, 7, 6, 6, 5, 5, 4, 4]), False),
        "Solarize": (np.linspace(256.0, 0.0, _BINS), False),
        "AutoContrast": (None, None),
        "Equalize": (None, None),
        "Invert": (None, None),
    }


def _get_policy(param: tuple, interpolation, fillcolor=None):
    assert len(param) == 2
    group1, group2 = param
    op1, p1, magnitude_id1 = group1
    op2, p2, magnitude_id2 = group2
    if fillcolor is None:
        return SubPolicy(op1, p1, magnitude_id1, op2, p2, magnitude_id2, interpolation)
    else:
        return SubPolicy(op1, p1, magnitude_id1, op2, p2, magnitude_id2, interpolation, fillcolor)


def _get_policies(params: List[tuple], interpolation, fillcolor=None):
    return list(map(lambda p: _get_policy(p, interpolation, fillcolor), params))


# class ImageNetPolicy(object):
#     """ Randomly choose one of the best 24 Sub-policies on ImageNet.

#         Example:
#         >>> policy = ImageNetPolicy()
#         >>> transformed = policy(image)

#         Example as a PyTorch Transform:
#         >>> transform = transforms.Compose([
#         >>>     transforms.Resize(256),
#         >>>     ImageNetPolicy(),
#         >>>     transforms.ToTensor()])
#     """
#     def __init__(self, fillcolor=(128, 128, 128)):
#         self.policies = [
#             SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
#             SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
#             SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
#             SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
#             SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

#             SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
#             SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
#             SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
#             SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
#             SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

#             SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
#             SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
#             SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

#             SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
#             SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
#             SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
#             SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
#             SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

#             SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
#             SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
#             SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
#             SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
#             SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
#         ]

#     def __call__(self, img):
#         policy_idx = random.randint(0, len(self.policies) - 1)
#         return self.policies[policy_idx](img)

#     def __repr__(self):
#         return "AutoAugment ImageNet Policy"


class SubPolicy(object):
    def __init__(self, operation1, p1, magnitude_id1, operation2, p2, magnitude_id2, interpolation, fillcolor=(128, 128, 128)):
        self.ranges = _get_magnitudes(10)
        self.func = {
            "ShearX": ShearX(interpolation=interpolation, fillcolor=fillcolor),
            "ShearY": ShearY(interpolation=interpolation, fillcolor=fillcolor),
            "TranslateX": TranslateX(interpolation=interpolation, fillcolor=fillcolor),
            "TranslateY": TranslateY(interpolation=interpolation, fillcolor=fillcolor),
            "Rotate": Rotate(interpolation=interpolation),
            "Color": Color(),
            "Posterize": Posterize(),
            "Solarize": Solarize(),
            "Contrast": Contrast(),
            "Sharpness": Sharpness(),
            "Brightness": Brightness(),
            "AutoContrast": AutoContrast(),
            "Equalize": Equalize(),
            "Invert": Invert()
        }

        self.p1 = p1
        self.operation1 = self.func[operation1]
        magnitude1 = self._get_op_meta(operation1)
        self.magnitude1 = float(magnitude1[magnitude_id1]) if magnitude_id1 is not None else 0.0
        self.p2 = p2
        self.operation2 = self.func[operation2]
        magnitude2 = self._get_op_meta(operation2)
        self.magnitude2 = float(magnitude2[magnitude_id2]) if magnitude_id2 is not None else 0.0

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img
    
    def _get_op_meta(self, name):
        magnitude, signed = self.ranges[name]
        if signed is not None and signed and random.random() < 0.5:
            return magnitude * -1.0
        else:
            return magnitude
    
    def __repr__(self) -> str:
        op1 = self.operation1.__class__.__name__
        op2 = self.operation2.__class__.__name__
        return "{}(p={:.2f}, m={:.2f}) + {}(p={:.2f}, m={:.2f})".format(op1, self.p1, self.magnitude1, op2, self.p2, self.magnitude2)


class AutoAugment(torch.nn.Module):
    def __init__(self, interpolation = InterpolationMode.NEAREST, fill=None):
        super().__init__()
        self.transforms = _get_transforms()
        self.interpolation = interpolation
        self.fill = fill

    def forward(self, img):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * _get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id = random.randint(0, len(self.transforms) - 1)
        policy = _get_policy(self.transforms[transform_id], self.interpolation, fill)
        # print(policy)
        return policy(img)

    def __repr__(self):
        return self.__class__.__name__ 


if __name__ == "__main__":
    for _ in range(1000):
        ag = AutoAugment()
        ag(torch.zeros((3, 100, 100), dtype=torch.uint8))
