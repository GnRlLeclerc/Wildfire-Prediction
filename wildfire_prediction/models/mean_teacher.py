"""Mean-Teacher SSL method"""

import torch
from torch import Tensor, nn
from torchvision.models.resnet import ResNeXt50_32X4D_Weights

from wildfire_prediction.models.base import Classifier


class MeanTeacherClassifier(Classifier):
    def __init__(self, num_classes: int = 1) -> None:
        """Initialize the classifier"""
        super().__init__()

        self.student = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnext50_32x4d",
            weights=ResNeXt50_32X4D_Weights.DEFAULT,
        )
        self.teacher = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnext50_32x4d",
            weights=ResNeXt50_32X4D_Weights.DEFAULT,
        )

        self.student.fc = nn.Linear(self.student.fc.in_features, num_classes)  # type: ignore
        self.teacher.fc = nn.Linear(self.teacher.fc.in_features, num_classes)  # type: ignore

        # Initialise teacher model to match student model
        self.teacher.load_state_dict(self.student.state_dict())

    def forward(self, x: Tensor):
        return self.student(x)  # type: ignore

    def update_teacher(self, alpha: float = 0.99):
        """Update teacher weights using exponential moving average (EMA) of student weights"""

        for student_param, teacher_param in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            teacher_param.data = (
                alpha * teacher_param.data + (1.0 - alpha) * student_param.data
            )
