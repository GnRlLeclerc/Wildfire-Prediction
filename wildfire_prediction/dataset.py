"""Wildfire dataset.

42850 images of 350x350 pixels, with 3 channels (RGB).

means: [75.25387867 88.2539294  64.06851075]
stds: [49.79832831 42.03286918 42.90422537]
"""

from typing import Literal

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32),
        # Computed over the whole train x test x valid datasets
        transforms.Normalize(
            [75.25387867, 88.2539294, 64.06851075],
            [49.79832831, 42.03286918, 42.90422537],
        ),
        transforms.Resize((224, 224)),
    ]
)


class WildfireDataset(Dataset):
    """Wildfire dataset loader."""

    def __init__(self, split: Literal["train", "test", "train_unlabeled"]) -> None:
        super().__init__()

        # Load the paths
        self.split = np.loadtxt(f"{split}.csv", dtype=str, delimiter=";")

    def __len__(self):
        """Returns the length of the dataset."""

        return len(self.split)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get an image and its label by index. The index is over training or testing data.

        Returns:
            image: The image tensor.
            label: A boolean label tensor.
        """

        is_fire = self.split[index, 1] == "1"
        path = self.split[index, 0]

        img = Image.open(path)

        # Avoid corrupted images
        try:
            img = _transforms(img)
        except:
            # Default to the first image
            return self.__getitem__(0)

        return img, torch.tensor(is_fire, dtype=torch.bool)
