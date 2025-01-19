"""Training and testing results utils"""

import json
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Results:
    """Classification results"""

    total: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def add_predictions(self, outputs: Tensor, labels: Tensor):
        """Add predictions to the results.

        Args:
            outputs: Raw logits from a classifier
            labels: Ground truth labels
        """
        predicted = torch.sigmoid(outputs) > 0.5

        self.total += labels.size(0)
        self.true_positives += int((predicted & labels).sum().item())
        self.true_negatives += int(((~predicted) & (~labels)).sum().item())
        self.false_positives += int((predicted & (~labels)).sum().item())
        self.false_negatives += int(((~predicted) & labels).sum().item())

    def compute_metrics(self):
        """Compute the metrics from the results"""
        self.accuracy = (self.true_positives + self.true_negatives) / self.total
        self.precision = self.true_positives / (
            self.true_positives + self.false_positives
        )
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def __str__(self) -> str:
        return (
            f"Total: {self.total}\n"
            f"True positives: {self.true_positives}\n"
            f"True negatives: {self.true_negatives}\n"
            f"False positives: {self.false_positives}\n"
            f"False negatives: {self.false_negatives}\n"
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1: {self.f1:.4f}"
        )

    @classmethod
    def from_json(cls, data: str) -> "Results":
        """Load results from a JSON string"""
        return cls(**json.loads(data))

    def to_json(self) -> str:
        """Dump results to a JSON string"""
        return json.dumps(self.__dict__)
