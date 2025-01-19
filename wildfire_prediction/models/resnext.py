"""Resnext-50 classifier"""

import torch
from torch import Tensor, nn
from torchvision.models.resnet import ResNeXt50_32X4D_Weights

from wildfire_prediction.models.base import Classifier


class ResnextClassifier(Classifier):
    """Resnext-50 - based classifier for wildfire prediction"""

    def __init__(self) -> None:
        """Initialize the classifier"""
        super().__init__()

        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnext50_32x4d",
            weights=ResNeXt50_32X4D_Weights.DEFAULT,
        )

        # Replace the final layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # type: ignore

    def forward(self, x: Tensor):
        return self.resnet(x)  # type: ignore
