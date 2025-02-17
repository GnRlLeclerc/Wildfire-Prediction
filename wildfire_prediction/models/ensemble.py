import torch
from typing import List, Optional
from torch import Tensor
from wildfire_prediction.models.alexnet import AlexnetClassifier
from wildfire_prediction.models.base import Classifier
import os
from wildfire_prediction.models.resnext import ResnextClassifier
from wildfire_prediction.models.vit import VitClassifier


def _get_classifier(classifier: str):
        match classifier:
            case "resnext":
                return ResnextClassifier()
            case "vit_b_16":
                return VitClassifier("vit_b_16")
            case "vit_b_32":
                return VitClassifier("vit_b_32")
            case "alexnet":
                return AlexnetClassifier()
            case _:
                raise ValueError(f"Unknown classifier variant: {classifier}")
            

class Ensemble():
    def __init__(self, models: List[Classifier], weight_paths: Optional[List[str]] = None):        
        self.models = models

        if weight_paths is not None:
            for model, weight_file in zip(self.models, weight_paths):
                model.load_state_dict(torch.load(weight_file, weights_only=True))
    
    @staticmethod  
    def from_checkpoint_folder(classifier: str, checkpoint_folder: str):
        models = []
        weight_paths = []

        for filename in os.listdir(checkpoint_folder):
            file_path = os.path.join(checkpoint_folder, filename)
            weight_paths.append(file_path)
            models.append(_get_classifier(classifier))

        return Ensemble(models, weight_paths)
    
    def predict(self, x: Tensor):
        predictions = []

        for model in self.models:
            prediction = model(x)
            predictions.append(prediction)

        return torch.mean(torch.sigmoid(torch.cat(predictions, 1)), 1)
    
    def predict_with_uncertainty(self, x: Tensor):
        predictions = []

        for model in self.models:
            prediction = model(x)
            predictions.append(prediction)

        gathered_predictions = torch.sigmoid(torch.cat(predictions, 1))

        return torch.mean(gathered_predictions, 1), torch.var(gathered_predictions, 1)
    