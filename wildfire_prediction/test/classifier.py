"""Test classifiers"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildfire_prediction.models.base import Classifier


@dataclass
class TestResults:
    """Classifier test results"""

    total: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


def test_classifier(
    classifier: Classifier, dataloader: DataLoader, device: str, verbose=True
) -> TestResults:
    """Test a classifier model on the given dataset loader"""

    classifier.eval()
    classifier.to(device)

    results = TestResults()

    with torch.no_grad():
        for images, labels in tqdm(
            dataloader, desc="Testing classifier", disable=not verbose
        ):
            images, labels = images.to(device), labels.to(device)

            outputs = classifier(images).squeeze()
            predicted = torch.sigmoid(outputs) > 0.5

            results.total += labels.size(0)
            results.true_positives += (predicted & labels).sum().item()
            results.true_negatives += ((~predicted) & (~labels)).sum().item()
            results.false_positives += (predicted & (~labels)).sum().item()
            results.false_negatives += ((~predicted) & labels).sum().item()

    results.accuracy = (results.true_positives + results.true_negatives) / results.total
    results.precision = results.true_positives / (
        results.true_positives + results.false_positives
    )
    results.recall = results.true_positives / (
        results.true_positives + results.false_negatives
    )
    results.f1 = (
        2 * (results.precision * results.recall) / (results.precision + results.recall)
    )

    return results
