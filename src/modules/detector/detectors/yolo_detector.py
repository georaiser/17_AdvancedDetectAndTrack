from src.modules.detector.utils.detection_filter import DetectionFilter
from src.modules.detector.utils.nms_filter import NMSFilter
from src.modules.engine.utils.image_square import ImageSquare
from box import Box

class YOLODetector:
    def __init__(self, model_handler, device, detector_config, allowed_classes):
        self.model_handler = model_handler
        self.model = model_handler.model.model
        print(self.model)
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
        image, resized_image = self.preprocess_pad(frame, padding_info)
         # Inference
        predictions = self.inference(image)
        # Parse detections
        boxes, scores, labels = self.parse_detections(predictions)
        # Filter detections
        boxes, scores, labels = self.filter_detections(boxes, scores, labels)
        # Adjust boxes coordinates
        boxes = self.adjust_boxes(boxes, padding_info)

        return boxes, scores, labels

    def preprocess_pad(self, frame, padding_info):
        # Resize and pad image
        resized_image, padded_image = ImageSquare.pad_to_square(frame, padding_info)
        return padded_image, resized_image

    def inference(self, img):
        # Run YOLO11 inference
        results = self.model(
            img,
            conf=self.detector_config["threshold"],
            iou=self.detector_config["iou_threshold"],
            verbose=False,
        )
        return results[0]

    def parse_detections(self, results):
        # Extract boxes, confidence scores, and class IDs from YOLO11 results

        boxes = results.boxes.xyxy
        scores = results.boxes.conf
        labels = results.boxes.cls
        return boxes, scores, labels

    def filter_detections(self, boxes, scores, labels):
        # Filter based on confidence and allowed classes
        boxes, scores, labels = self.detection_filter.filter_detections(
            boxes, scores, labels
        )
        # Apply NMS
        # boxes, scores = self.nms_filter.apply(boxes, scores)
        return boxes, scores, labels

    def adjust_boxes(self, boxes, padding_info):
        # Adjust boxes coordinates
        boxes = ImageSquare.unpad_coordinates(boxes, padding_info)
        return boxes

