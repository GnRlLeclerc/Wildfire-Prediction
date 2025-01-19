"""Training and testing results utils"""

import json
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
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

    @staticmethod
    def plot_loss(results: list["Results"]):
        """Plot the loss from a list of training results"""

        losses = [result.loss for result in results if result.loss is not None]

        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid()
        plt.show()

    @staticmethod
    def plot_metrics(training: list["Results"], testing: list["Results"]):
        """Plot the metrics from training and testing results"""

        train_accuracy = [result.accuracy for result in training]
        test_accuracy = [result.accuracy for result in testing]

        train_precision = [result.precision for result in training]
        test_precision = [result.precision for result in testing]

        train_recall = [result.recall for result in training]
        test_recall = [result.recall for result in testing]

        train_f1 = [result.f1 for result in training]
        test_f1 = [result.f1 for result in testing]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        fig.suptitle("Training Metrics")

        axes[0, 0].plot(train_accuracy, label="Train")
        axes[0, 0].plot(test_accuracy, label="Test")
        axes[0, 0].set_title("Accuracy")

        axes[0, 1].plot(train_precision, label="Train")
        axes[0, 1].plot(test_precision, label="Test")
        axes[0, 1].set_title("Precision")

        axes[1, 0].plot(train_recall, label="Train")
        axes[1, 0].plot(test_recall, label="Test")
        axes[1, 0].set_title("Recall")

        axes[1, 1].plot(train_f1, label="Train")
        axes[1, 1].plot(test_f1, label="Test")
        axes[1, 1].set_title("F1")

        for row in axes:
            for cell in row:
                cell.legend()
                cell.grid()

        plt.show()


def filter_null_attrs(d: dict) -> dict:
    """Filter out null attributes from a dictionary"""
    return {k: v for k, v in d.items() if v is not None}
