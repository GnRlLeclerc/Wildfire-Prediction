"""Train mean teacher"""

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildfire_prediction.models.mean_teacher import MeanTeacherClassifier
from wildfire_prediction.test.classifier import test_classifier
from wildfire_prediction.utils.results import Results
from wildfire_prediction.train.losses import mse_scaled_loss, kl_divergence_loss


def linear_rampup(current_epoch: int, rampup_epochs: int):
    """Compute the linear ramp-up weight (between 0 and 1) for the current epoch"""
    if rampup_epochs == 0:
        return 1.0
    else:
        return min(current_epoch / rampup_epochs, 1.0)


def get_teacher_student_loss(
    student_outputs: Tensor,
    teacher_outputs: Tensor,
    teacher_student_loss: str,
    temperature: float | None,
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
    if temperature is None:
        temperature = 1.0

    match teacher_student_loss:
        case "mse_standard":
            return F.mse_loss(student_outputs, teacher_outputs, reduction="mean")
        case "mse_scaled":
            return mse_scaled_loss(student_outputs, teacher_outputs, temperature)
        case "kl_divergence":
            return kl_divergence_loss(student_outputs, teacher_outputs, temperature)
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
    temperature: float | None,
    rampup_epochs: int = 5,
):
    """Train mean teacher with linear ramp-up for consistency weight."""
    model.to(device)
    optimizer = Adam(model.student.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    log_file = f"training_logs_mt_{teacher_student_loss}_t-{str(temperature)}.jsonl"

    for i in tqdm(range(epochs), desc="Training mean teacher"):
        model.train()
        results = Results(iteration=i, mode="train", loss=0)

        consistency_weight = linear_rampup(i, rampup_epochs)

        for (labeled_data, labels), unlabeled_data in zip(
            train_loader_labeled, train_loader_unlabeled
        ):
            labeled_data, labels = labeled_data.to(device), labels.to(device)
            unlabeled_data = unlabeled_data.to(device)

            # forward pass for labeled data
            student_outputs = model(labeled_data).squeeze()

            # loss for labeled data
            loss_labeled = F.binary_cross_entropy_with_logits(
                student_outputs, labels.float()
            )

            results.add_predictions(student_outputs, labels)

            # forward pass for unlabeled data
            with torch.no_grad():
                teacher_outputs = model.teacher(unlabeled_data).squeeze()
            student_outputs_unlabeled = model(unlabeled_data).squeeze()

            # consistency loss for unlabeled data
            loss_unlabeled = get_teacher_student_loss(
                student_outputs_unlabeled,
                teacher_outputs,
                teacher_student_loss,
                temperature,
            )

            total_loss = loss_labeled + consistency_weight * loss_unlabeled

            # backward pass
            optimizer.zero_grad()
            total_loss.backward()

            results.loss += total_loss.item()  # type: ignore

            # update the weights
            optimizer.step()

            model.update_teacher()

        # append logs
        results.compute_metrics()
        results.append_log(log_file)

        # evaluate on test set
        results = test_classifier(model, test_loader, device, verbose=False)
        results.mode = "test"
        results.iteration = i
        results.append_log(log_file)

        scheduler.step()

    # save the final model
    torch.save(
        model.state_dict(),
        f"mt-final_loss-{teacher_student_loss}_t-{str(temperature)}.pth",
    )
