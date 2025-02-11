"""Alexnet-50 classifier"""

from torch import Tensor
from torchvision import models
from torch import Tensor, nn

from wildfire_prediction.models.base import Classifier


class AlexnetClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

        self.alexnet = models.alexnet(pretrained=True)
        n_alexnet_out_features = self.alexnet.classifier[-1].out_features
        self.classifier = nn.Linear(n_alexnet_out_features, 1)

    def forward(self, x: Tensor):
        return self.classifier(self.alexnet(x))  # type: ignore
