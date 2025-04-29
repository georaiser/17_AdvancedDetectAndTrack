from ultralytics import YOLO  # YOLO11 uses the Ultralytics framework


class YOLOHandler:
    """Handles YOLO11 model implementation using Ultralytics framework"""

    def __init__(self, model_name, device, **kwargs):
        self.model_name = model_name
        self.device = device

        # Extract config parameters with defaults
        self.model_path = kwargs.get("model_path") + f"/{self.model_name}.pt"
        self.conf_thres = kwargs.get("conf_thres", 0.5)
        self.iou_thres = kwargs.get("iou_thres", 0.5)

        # Load model
        self.model = self._load_model()
        # Device
        self.model.to(self.device)
        # Set to evaluation mode
        self.model.eval()

    def _load_model(self):
        """Load YOLO model using Ultralytics"""
        model = YOLO(self.model_path)
        model.to(self.device)
        return model
