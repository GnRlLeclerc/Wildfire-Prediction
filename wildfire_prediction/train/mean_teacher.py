"""Train mean teacher"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildfire_prediction.models.mean_teacher import MeanTeacherClassifier
from wildfire_prediction.test.classifier import test_classifier
from wildfire_prediction.utils.results import Results


def train_mean_teacher(
    model: MeanTeacherClassifier,
    train_loader_labeled: DataLoader,
    train_loader_unlabeled: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
):
    """Train mean teacher"""
    model.to(device)
    optimizer = Adam(model.student.parameters(), lr=learning_rate)
    log_file = "training_logs.jsonl"

    for i in tqdm(range(epochs), desc="Training mean teacher"):

        model.train()
        results = Results(iteration=i, mode="train", loss=0)

        for (labeled_data, labels), unlabeled_data in zip(
            train_loader_labeled, train_loader_unlabeled
        ):
            labeled_data, labels = labeled_data.to(device), labels.to(device)
            unlabeled_data = unlabeled_data.to(device)

            # Forward pass for labeled data
            student_outputs = model(labeled_data).squeeze()

            # Compute loss for labeled data
            loss_labeled = F.cross_entropy(student_outputs, labels.float())

            # Forward pass for unlabeled data
            with torch.no_grad():
                teacher_outputs = model.teacher(unlabeled_data).squeeze()
            student_outputs_unlabeled = model(unlabeled_data).squeeze()

            # Compute loss for unlabeled data
            loss_unlabeled = F.mse_loss(student_outputs_unlabeled, teacher_outputs)

            total_loss = loss_labeled + loss_unlabeled
            results.add_predictions(student_outputs, labels)

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            results.loss += total_loss.item()  # type: ignore

            # Update the weights
            optimizer.step()

            # Update teacher
            model.update_teacher()

        # Append training logs
        results.compute_metrics()
        results.append_log(log_file)

        results = test_classifier(model, test_loader, device, verbose=False)
        results.mode = "test"
        results.iteration = i
        results.append_log(log_file)

        # Save the model checkpoints every 20% of the epochs
        checkpoint_frequency = max(epochs // 5, 1)
        if i % checkpoint_frequency == 0:
            torch.save(model.state_dict(), f"mean-teacher-{i}-{epochs}.pth")

    # Save the final model
    torch.save(model.state_dict(), "mean-teacher-final.pth")
