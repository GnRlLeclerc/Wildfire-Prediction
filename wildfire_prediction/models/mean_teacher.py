"""Mean-Teacher SSL method"""

from torch import Tensor

from wildfire_prediction.models.alexnet import AlexnetClassifier
from wildfire_prediction.models.base import Classifier
from wildfire_prediction.models.resnext import ResnextClassifier
from wildfire_prediction.models.vit import VitClassifier


def _get_classifier(classifier: str):
    match classifier:
        case "resnext":
            return ResnextClassifier()
        case "vit_b_16":
            return VitClassifier("vit_b_16")
        case "vit_b_32":
            return VitClassifier("vit_b_32")
        case "alexnet":
            return AlexnetClassifier()
        case _:
            raise ValueError(f"Unknown classifier variant: {classifier}")


class MeanTeacherClassifier(Classifier):
    def __init__(self, classifier: str = "alexnet") -> None:
        """Initialize the classifier"""
        super().__init__()

        self.student = _get_classifier(classifier)
        self.teacher = _get_classifier(classifier)

        # Initialise teacher model to match student model
        self.teacher.load_state_dict(self.student.state_dict())

    def forward(self, x: Tensor):
        return self.student(x)  # type: ignore

    def update_teacher(self, alpha: float = 0.9):
        """Update teacher weights using exponential moving average (EMA) of student weights"""

        for student_param, teacher_param in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            teacher_param.data = (
                alpha * teacher_param.data + (1.0 - alpha) * student_param.data
            )
