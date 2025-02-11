"""Alexnet-50 classifier"""

from torch import Tensor
from torchvision import models

from wildfire_prediction.models.base import Classifier


class AlexnetClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

        self.alexnet = models.alexnet(pretrained=True, num_classes=1)

    def forward(self, x: Tensor):
        return self.alexnet(x)  # type: ignore
