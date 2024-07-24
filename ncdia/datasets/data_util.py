import numpy as np
import torch
import torchvision.transforms as transforms
from ncdia.augmentations.constrained_cropping import CustomMultiCropDataset, CustomMultiCropping
# from .constrained_cropping import CustomMultiCropDataset, CustomMultiCropping

def get_transform(args):

    if args.dataset == 'remote':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    assert (len(args.size_crops) == 2)
    #这里是crop_transform
    crop_transform = CustomMultiCropping(size_large=args.size_crops[0],
                                         scale_large=(args.min_scale_crops[0], args.max_scale_crops[0]),
                                         size_small=args.size_crops[1],
                                         scale_small=(args.min_scale_crops[1], args.max_scale_crops[1]),
                                         N_large=args.num_crops[0], N_small=args.num_crops[1],
                                         condition_small_crops_on_key=args.constrained_cropping)

    if len(args.auto_augment) == 0:
        print('No auto augment - Apply regular moco v2 as secondary transform')
        secondary_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),    
            transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize])

    else:
        from ncdia.augmentations.auto_augment.auto_augment import AutoAugment
        from ncdia.augmentations.auto_augment.random_choice import RandomChoice
        print('Auto augment - Apply custom auto-augment strategy')
        counter = 0
        secondary_transform = []

        for i in range(len(args.size_crops)):
            for j in range(args.num_crops[i]):
                if not counter in set(args.auto_augment):
                    print('Crop {} - Apply regular secondary transform'.format(counter))
                    secondary_transform.extend([transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
#                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.ToTensor(),
                        normalize])])

                else:
                    print('Crop {} - Apply auto-augment/regular secondary transform'.format(counter))
                    trans1 = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        AutoAugment(),
                        transforms.ToTensor(),
                        normalize])

                    trans2 = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
#                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.ToTensor(),
                        normalize])

                    secondary_transform.extend([RandomChoice([trans1, trans2])])

                counter += 1

    return crop_transform, secondary_transform
