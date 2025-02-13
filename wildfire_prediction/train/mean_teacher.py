"""Train mean teacher"""

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildfire_prediction.models.mean_teacher import MeanTeacherClassifier
from wildfire_prediction.test.classifier import test_classifier
from wildfire_prediction.utils.results import Results
from wildfire_prediction.train.losses import mse_scaled_loss, kl_divergence_loss


def get_teacher_student_loss(
    student_outputs: Tensor,
    teacher_outputs: Tensor,
    teacher_student_loss: str,
):
    """
    Compute loss between student and teacher models using either:
    - classical mse loss ("mse_standard"),
    - mse loss with temperature scaling ("mse_scaled"),
    - knowledge distillation loss (kl divergence with temperature scaling - "kl_divergence")

    The temperature scaling (suggested by Guo et al. in "On Calibration of Modern Neural Networks")
    is applied to both the student and teacher outputs to smooth the logits and improve model calibration.

    Reference:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks.
        https://arxiv.org/abs/1706.04599
    """
    match teacher_student_loss:
        case "mse_standard":
            return F.mse_loss(student_outputs, teacher_outputs, reduction="mean")
        case "mse_scaled":
            return mse_scaled_loss(student_outputs, teacher_outputs)
        case "kl_divergence":
            return kl_divergence_loss(student_outputs, teacher_outputs)
        case _:
            raise ValueError(f"Unknown loss type: {teacher_student_loss}")


def train_mean_teacher_classifier(
    model: MeanTeacherClassifier,
    train_loader_labeled: DataLoader,
    train_loader_unlabeled: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    teacher_student_loss: str,
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
            loss_labeled = F.binary_cross_entropy_with_logits(
                student_outputs, labels.float()
            )

            results.add_predictions(student_outputs, labels)

            # Forward pass for unlabeled data
            with torch.no_grad():
                teacher_outputs = model.teacher(unlabeled_data).squeeze()
            student_outputs_unlabeled = model(unlabeled_data).squeeze()

            # Compute loss for unlabeled data
            loss_unlabeled = get_teacher_student_loss(
                student_outputs_unlabeled, teacher_outputs, teacher_student_loss
            )
            total_loss = loss_labeled + loss_unlabeled

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
