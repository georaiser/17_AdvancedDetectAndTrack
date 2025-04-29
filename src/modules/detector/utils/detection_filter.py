import torch

class DetectionFilter:
    """Handles filtering of detections based on scores and classes"""

    def __init__(self, score_threshold, allowed_classes):
        self.score_threshold = score_threshold
        self.allowed_classes = allowed_classes

    def filter_detections(self, boxes, scores, labels):
        """Filter detections based on score threshold and allowed classes"""
        # Score threshold filtering
        keep = scores > self.score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Class filtering if specified
        if self.allowed_classes:
            class_mask = torch.tensor(
                [label.item() in self.allowed_classes for label in labels],
                dtype=torch.bool,
                device=labels.device,
            )

            boxes = boxes[class_mask]
            scores = scores[class_mask]
            labels = labels[class_mask]

        return boxes, scores, labels
