"""Image utilities"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


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
