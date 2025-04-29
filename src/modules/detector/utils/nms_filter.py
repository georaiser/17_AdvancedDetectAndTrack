import torchvision
import numpy as np
import cv2


class NMSFilter:
    def __init__(self, iou_threshold, nms_type="torchvision"):
        self.iou_threshold = iou_threshold
        self.nms_type = nms_type

    def apply(self, boxes, scores):
        if self.nms_type == "torchvision":
            return self._nms_torchvision(boxes, scores)
        if self.nms_type == "opencv":
            return self._nms_opencv(boxes, scores)

    def _nms_torchvision(self, boxes, scores):
        keep = torchvision.ops.nms(boxes, scores, self.iou_threshold)
        return boxes[keep], scores[keep]

    def _nms_opencv(self, boxes, scores):
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.3,
            nms_threshold=self.iou_threshold,
        )

        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = boxes[indices]
            filtered_scores = scores[indices]
            return filtered_boxes, filtered_scores
        return np.array([]), np.array([])
