"""
Example showcase.
"""

import torch.nn.functional as F
from torch.utils.data import DataLoader

from wildfire_prediction.dataset import WildfireDataset
from wildfire_prediction.models.resnext import ResnextClassifier
from wildfire_prediction.utils.images import imshow

dataset = WildfireDataset("train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

image, label = next(iter(dataloader))

imshow(image, title=f"Fire: {label.item()}")

classifier = ResnextClassifier()

result = classifier(image)

print(F.sigmoid(result))
