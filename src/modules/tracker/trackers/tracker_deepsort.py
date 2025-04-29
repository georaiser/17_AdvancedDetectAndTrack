# -*- coding: utf-8 -*-
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker_DeepSort:
    def __init__(self, config_tracker):
        """Handles object tracking using DeepSORT algorithm"""
        self.config_tracker = config_tracker
        self.tracker = DeepSort(**self.config_tracker)

    def update(self, frame, boxes, scores, labels):
        # Convert detections to DeepSORT format - [x1,y1,w,h]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], score.item(), label.item()))

        # Update tracks
        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks