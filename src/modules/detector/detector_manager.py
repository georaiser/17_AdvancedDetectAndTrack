from src.modules.detector.detectors.efficientdet_detector import EfficientDetDetector
from src.modules.detector.detectors.fasterrcnn_detector import FasterRCNNDetector
from src.modules.detector.detectors.yolo_detector import YOLODetector
from src.modules.detector.detectors.yolox_detector import YOLOXDetector


class DetectorManager:
    def __init__(self, model_handler, detector_config, allowed_classes):
        self.detector_config = detector_config
        self.model_handler = model_handler
        self.allowed_classes = allowed_classes

        try:
            # yolo
            self.model          = model_handler.model.model
            self.model_name     = model_handler.model.model_name
            self.device         = model_handler.model.device
        except AttributeError:
            # fasterrcnn / efficiendet
            self.model          = model_handler.get_model()
            self.model_name     = model_handler.get_model_name()
            self.device         = model_handler.get_device()

    def get_detector(self):
        detector_map = {
            "tf_efficientdet": EfficientDetDetector,
            "fasterrcnn": FasterRCNNDetector,
            "yolox": YOLOXDetector,
            "yolov5": YOLODetector,
            "yolov8": YOLODetector,
            "yolo11": YOLODetector,
            "yolo12": YOLODetector,
        }

        for key, detector_ in detector_map.items():
            if self.model_name.startswith(key):
                args = (self.model_handler, self.device, self.detector_config, self.allowed_classes)
                return detector_(*args)

        raise ValueError(f"Unsupported model: {self.model_name}")

