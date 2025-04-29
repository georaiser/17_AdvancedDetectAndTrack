from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models import ResNet50_Weights


class FasterRCNNHandler:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        model = None

        if self.model_name.startswith("fasterrcnn"):
            model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                progress=True,
                weights_backbone=ResNet50_Weights.DEFAULT,
            )
            model = model.to(self.device)
            model.eval()

            return model

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model
