import torch
import torchvision.transforms.functional as F
from box import Box

from src.modules.detector.utils.detection_filter import DetectionFilter
from src.modules.detector.utils.nms_filter import NMSFilter
from src.modules.engine.utils.image_square import ImageSquare


class FasterRCNNDetector:
    def __init__(self, model_handler, device, detector_config, allowed_classes):
        self.model = model_handler.get_model()
        self.device = device
        self.detector_config = Box(detector_config)

        self.detection_filter = DetectionFilter(
            score_threshold=self.detector_config.threshold,
            allowed_classes=allowed_classes,
        )
        self.nms_filter = NMSFilter(
            self.detector_config.iou_threshold, self.detector_config.nms_type
        )

    def detection_pipeline(self, frame, padding_info):
        # Preprocess image
        img_tensor, resized_image = self.preprocess(frame, padding_info)
        # Inference
        predictions = self.inference(img_tensor)
        # Parse detections
        boxes, scores, labels = self.parse_detections(predictions)
        # Filter detections
        boxes, scores, labels = self.filter_detections(boxes, scores, labels)
        # Adjust boxes coordinates
        boxes = self.adjust_boxes(boxes, padding_info)

        return boxes, scores, labels

    def preprocess(self, frame, padding_info):
        # Resize and pad image
        resized_image, padded_image = ImageSquare.pad_to_square(frame, padding_info)

        # Convert to tensor
        img_tensor = F.to_tensor(padded_image).unsqueeze(0)

        return img_tensor, resized_image

    def inference(self, img_tensor):
        # Ensure input tensor is on correct device
        img_tensor = img_tensor.to(self.device)
        # Ensure model is on correct device
        self.model.to(self.device)

        # model inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(img_tensor)
        return predictions

    def parse_detections(self, predictions):
        boxes = predictions[0]["boxes"]
        scores = predictions[0]["scores"]
        labels = predictions[0]["labels"]
        return boxes, scores, labels

    def filter_detections(self, boxes, scores, labels):
        # filter threshold and classes
        boxes, scores, labels = self.detection_filter.filter_detections(
            boxes, scores, labels)
        # Apply NMS
        boxes, scores = self.nms_filter.apply(boxes, scores)
        return boxes, scores, labels

    def adjust_boxes(self, boxes, padding_info):
        # Adjust boxes coordinates
        boxes = ImageSquare.unpad_coordinates(boxes, padding_info)
        return boxes

