import torch

from src.models.handlers.efficientdet_handler import EfficientdetHandler
from src.models.handlers.faster_rcnn_handler import FasterRCNNHandler
from src.models.handlers.yolox_handler import YOLOXHandler
from src.models.handlers.yolo_handler import YOLOHandler


class ModelManager:
    def __init__(self, model_name, config=None):
        self.model_name = model_name
        self.device = self.setup_device()
        self.config = config
        self.model = self._load_model()
        self.image_size = self.get_image_size()
        self.param = self.get_model_parameters(self.model)

    def _load_model(self):
        if self.model_name.startswith("tf_efficientdet"):
            model = EfficientdetHandler(self.model_name, self.device).model

        elif self.model_name.startswith("fasterrcnn"):
            model = FasterRCNNHandler(self.model_name, self.device).model

        elif self.model_name.startswith("yolo"):
            if "yolox" in self.model_name:
                # If the model name is YOLOX, use YOLOXHandler
                yolox_config = self.config.get("yolox", {})
                model = YOLOXHandler(self.model_name, self.device, **yolox_config)
            else:
                # Otherwise, use YOLOHandler
                yolo_config = self.config.get("yolo", {})
                model = YOLOHandler(self.model_name, self.device, **yolo_config)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model

    def get_model_parameters(self, model):
        """Verify and print model parameter information"""

        try:
            model_parameters = self.model.parameters()
        except AttributeError:
            model_parameters = self.model.model.parameters()

        total_params = sum(p.numel() for p in model_parameters)
        trained_params = sum(p.numel() for p in model_parameters if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trained_params:,}")

        return total_params, trained_params

    def get_image_size(self):
        size_map = {
            "tf_efficientdet_d0": 512,
            "tf_efficientdet_d1": 640,
            "tf_efficientdet_d2": 768,
            "tf_efficientdet_d3": 896,
            "tf_efficientdet_d4": 1024,
            "tf_efficientdet_d5": 1280,
            "tf_efficientdet_d6": 1280,
            "tf_efficientdet_d7": 1536,
            "fasterrcnn_resnet50_fpn": 1024,
        }

        # If model name starts with "yolo", return 640
        if self.model_name.startswith("yolo"):
            return 640

        # Return size from size_map or default to None
        return size_map.get(self.model_name, None)

    def get_model(self):
        """Return the loaded model"""
        return self.model

    def get_model_name(self):
        """Return the loaded model name"""
        return self.model_name

    def get_device(self):
        """Return the current device"""
        return self.device
    
    def setup_device(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            # Enable cudnn benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            return torch.device("cuda")
        return torch.device("cpu")
