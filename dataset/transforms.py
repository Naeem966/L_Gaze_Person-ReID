import torch
import torchvision.transforms as T
from PIL import Image
import random

class RandomErasing(object):
    """Random Erasing augmentation for Re-ID."""
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

import math

def build_transforms(img_size=(256, 128), is_train=True):
    """
    Standard TransReID data augmentation pipeline.
    Section 4.3 in L-Gaze paper:
    - Resize 256x128
    - Random Horizontal Flip
    - 10-pixel padding followed by Random Crop
    - Random Erasing
    - ImageNet normalization
    """
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if is_train:
        transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(img_size),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5)
        ])
    else:
        transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            normalize_transform
        ])
    return transform
