"""Vision Transformer classifier"""

from typing import Literal

from torch import Tensor, nn
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32

from wildfire_prediction.models.base import Classifier


def _load_model(variant: Literal["vit_b_16", "vit_b_32"]):

    match variant:
        case "vit_b_16":
            return vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        case "vit_b_32":
            return vit_b_32(weights=ViT_B_32_Weights.DEFAULT)


class VitClassifier(Classifier):
    """Vision Transformer - based classifier for wildfire prediction"""

    def __init__(self, variant: Literal["vit_b_16", "vit_b_32"] = "vit_b_16") -> None:
        """Initialize the classifier"""
        super().__init__()

        self.vit = _load_model(variant)

        # Replace the final layer for binary classification
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, 1)  # type: ignore

    def forward(self, x: Tensor):
        return self.vit(x)  # type: ignore
