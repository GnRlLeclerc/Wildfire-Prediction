import torch
from typing import List, Optional
from torch import Tensor, nn
from wildfire_prediction.models.base import Classifier


class Ensemble():
    def __init__(self, models: List[Classifier], weight_paths: Optional[List[str]] = None):
        super(Ensemble, self).__init__()
        
        self.models = nn.ModuleList(models)

        if weight_paths is not None:
            for model, weight_file in zip(self.models, weight_paths):
                model.load_state_dict(torch.load(weight_file, weights_only=True))

    
    def forward(self, x: Tensor):
        predictions = []

        for model in self.models:
            prediction = model(x)
            predictions.append(prediction)

        return torch.mean(torch.cat(predictions, 1), 1)