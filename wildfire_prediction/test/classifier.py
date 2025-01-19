"""Test classifiers"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildfire_prediction.models.base import Classifier
from wildfire_prediction.utils.results import Results


def test_classifier(
    classifier: Classifier, dataloader: DataLoader, device: str, verbose=True
) -> Results:
    """Test a classifier model on the given dataset loader"""

    classifier.eval()
    classifier.to(device)

    results = Results()

    with torch.no_grad():
        for images, labels in tqdm(
            dataloader, desc="Testing classifier", disable=not verbose
        ):
            images, labels = images.to(device), labels.to(device)

            outputs = classifier(images).squeeze()
            results.add_predictions(outputs, labels)

    results.compute_metrics()

    return results
