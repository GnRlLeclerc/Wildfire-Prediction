"""Abstract base class for classifier models in order to enable easy model switching"""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Classifier(nn.Module, ABC):
    """Abstract base classifier class for wildfire prediction binary classification."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the classifier

        Args:
            x (torch.Tensor): Batched input tensor of images

        Returns:
            torch.Tensor (batch_size, 1): Output logits (apply sigmoid for probabilities)
        """
