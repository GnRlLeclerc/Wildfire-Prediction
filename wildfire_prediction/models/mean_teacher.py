"""Mean-Teacher SSL method"""

from torch import nn, Tensor
from torchvision.models import resnet18, ResNet18_Weights

from wildfire_prediction.models.base import Classifier


class MeanTeacherClassifier(Classifier):
    def __init__(self, num_classes: int = 2) -> None:
        """Initialize the classifier"""
        super().__init__()

        self.student = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.teacher = resnet18(weights=ResNet18_Weights.DEFAULT)

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
