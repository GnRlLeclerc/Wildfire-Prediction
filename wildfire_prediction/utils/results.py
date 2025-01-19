"""Training and testing results utils"""

import json
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass
class Results:
    """Classification results"""

    # Training metadata
    iteration: int | None = None
    loss: float | None = None
    mode: Literal["train", "test"] | None = None

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
        parts = []

        if self.iteration is not None:
            parts.append(f"Iteration: {self.iteration}\n")
        if self.mode is not None:
            parts.append(f"Mode: {self.mode}\n")
        if self.loss is not None:
            parts.append(f"Loss: {self.loss:.4f}\n")

        parts.append(f"Total: {self.total}\n")
        parts.append(f"True positives: {self.true_positives}\n")
        parts.append(f"True negatives: {self.true_negatives}\n")
        parts.append(f"False positives: {self.false_positives}\n")
        parts.append(f"False negatives: {self.false_negatives}\n")
        parts.append(f"Accuracy: {self.accuracy:.4f}\n")
        parts.append(f"Precision: {self.precision:.4f}\n")
        parts.append(f"Recall: {self.recall:.4f}\n")
        parts.append(f"F1: {self.f1:.4f}")

        return "".join(parts)

    @classmethod
    def from_json(cls, data: str) -> "Results":
        """Load results from a JSON string"""
        return cls(**json.loads(data))

    def to_json(self) -> str:
        """Dump results to a JSON string"""
        return json.dumps(filter_null_attrs(self.__dict__), indent=2)

    def append_log(self, path: str):
        """Append the results to a log file"""
        with open(path, "a") as f:
            f.write(json.dumps(filter_null_attrs(self.__dict__)) + "\n")


def filter_null_attrs(d: dict) -> dict:
    """Filter out null attributes from a dictionary"""
    return {k: v for k, v in d.items() if v is not None}
