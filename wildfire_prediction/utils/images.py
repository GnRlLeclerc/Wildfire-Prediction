"""Image utilities"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
import random


def normalize_tensor_image(x: Tensor) -> Tensor:
    """Normalize a tensor / batched tensor's values for image display depending on the type.
    - int: [0, 255] -> [0, 1]
    - float: [0, 1]
    """
    min = x.min()
    max = x.max()

    match x.dtype:
        case torch.float:
            return (x - min) / (max - min)
        case torch.int | torch.uint8:
            return ((x.float() - min) / (max - min) * 255).to(torch.uint8)
        case _:
            raise ValueError(f"Unsupported tensor type {x.dtype}")


def from_torch(x: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor representing images to a numpy array.
    Accepts a batched or unbatched single image."""

    dim = len(x.shape)
    x = normalize_tensor_image(x)

    if dim == 3:
        return x.permute(1, 2, 0).detach().cpu().numpy()
    elif dim == 4:
        return x.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()

    raise ValueError(f"Unsupported shape for image tensor: {dim}")


def imshow(x: Tensor, title: str):
    """Show a PyTorch tensor representing an image.
    Args:
        x (1, width, height, 3): The image tensor to show, batched or unbatched.
    """

    image = from_torch(x)

    # Remove the batch dimension if it exists
    if len(image.shape) == 4:
        assert (
            image.shape[0] == 1
        ), f"Batch size of {image.shape[0]} invalid, expected 1"
        image = image[0]

    plt.title(title)
    plt.imshow(image)
    plt.show()

class RandAugment:
    def __init__(self, N, M):
        self.N = N  # Number of augmentations to apply
        self.M = M  # Magnitude for each augmentation
        self.augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.ColorJitter(contrast=random.uniform(0.05, 0.95)),
            transforms.ColorJitter(brightness=random.uniform(0.05, 0.95)),
            transforms.RandomAutocontrast(),
            transforms.ColorJitter(contrast=random.uniform(0.05, 0.95)),
            #transforms.RandomEqualize(),
            #transforms.RandomPosterize(bits=random.randrange(4, 8)),
            transforms.RandomRotation(degrees=(-30, 30),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomAdjustSharpness(sharpness_factor=random.uniform(0.05, 0.95)),
            transforms.RandomAffine(degrees=0,shear=(-0.3, 0.3),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomAffine(degrees=0,translate=(0.3, 0.3),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomAffine(degrees=0,translate=(0, 0), scale=(0.7, 1.3),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomSolarize(threshold=random.random())
        ]

    def __call__(self, img):
        ops = random.sample(self.augmentations, self.N)
        for op in ops:
          if random.random() > 0.5:
                img = op(img)
        return img


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img