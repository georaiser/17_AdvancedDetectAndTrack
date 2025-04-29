import torch
from box import Box
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

from src.modules.detector.utils.detection_filter import DetectionFilter
from src.modules.detector.utils.nms_filter import NMSFilter
from src.modules.engine.utils.image_square import ImageSquare


class YOLOXDetector:
    def __init__(self, model_handler, device, detector_config, allowed_classes):
        self.model_handler = model_handler
        self.model = model_handler.model.model
        self.device = device
        self.detector_config = Box(detector_config)

        self.val_preproc = ValTransform(False)

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
        # Preprocess image
        img, img_info = self.preprocess(image)
        # Inference
        boxes, scores, labels = self.inference(img, img_info)
        # Filter detections
        boxes, scores, labels = self.filter_detections(boxes, scores, labels)
        # Adjust boxes coordinates
        boxes = self.adjust_boxes(boxes, padding_info)

        return boxes, scores, labels

    def preprocess_pad(self, frame, padding_info):
        # Resize and pad image
        resized_image, padded_image = ImageSquare.pad_to_square(frame, padding_info)

        # Convert to tensor
        #img_tensor = F.to_tensor(padded_image).unsqueeze(0)

        return padded_image, resized_image

    def preprocess(self, img):
        """Preprocess image for inference"""
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        test_size = self.model_handler.model.test_size

        ratio = min(test_size[0] / height, test_size[1] / width)
        img_info["ratio"] = ratio

        img, _ = self.val_preproc(img, None, test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float().to(self.device)

        return img, img_info

    def inference(self, img_tensor, img_info):
        """Run inference with YOLOX model"""
        with torch.no_grad():
            outputs = self.model(img_tensor)

            outputs = postprocess(
                outputs,
                self.model_handler.model.num_classes,
                self.model_handler.model.confthre,
                self.model_handler.model.nmsthre,
                class_agnostic=True
            )

        # #print(f"Min confidence: {outputs[0][:, 4].min().item()}")
        # #print(f"Max confidence: {outputs[0][:, 4].max().item()}")

        # Process output format to match the expected format in the system
        if outputs[0] is not None:
            output = outputs[0].cpu()
            bboxes = output[:, 0:4]
            # Scale back to original image dimensions
            bboxes /= img_info["ratio"]
            scores = output[:, 4] * output[:, 5]
            cls_ids = output[:, 6]

            # Format for the tracking system
            return bboxes, scores, cls_ids
        else:
            return torch.empty((0, 4)), torch.empty(0), torch.empty(0)

    def filter_detections(self, boxes, scores, labels):
        # filter threshold and classes
        boxes, scores, labels = self.detection_filter.filter_detections(
            boxes, scores, labels
        )
        # Apply NMS
        #boxes, scores = self.nms_filter.apply(boxes, scores)
        return boxes, scores, labels

    def adjust_boxes(self, boxes, padding_info):
        # Adjust boxes coordinates
        boxes = ImageSquare.unpad_coordinates(boxes, padding_info)
        return boxes
