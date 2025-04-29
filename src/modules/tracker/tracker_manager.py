# -*- coding: utf-8 -*-
from src.modules.tracker.trackers.tracker_deepsort import Tracker_DeepSort
from src.modules.tracker.trackers.tracker_bytetrack import Tracker_ByteTrack

class TrackerManager:
    def __init__(self, config_processor, config_tracker):

        self.config_processor = config_processor
        self.config_tracker = config_tracker
        self.tracker_name = config_processor["tracker"]

        if "deepsort" in self.tracker_name:
            self.tracker_processor = Tracker_DeepSort(self.config_tracker)
        elif "bytetrack" in self.tracker_name:
            self.frame_skip = config_processor["frame_skip"]
            self.tracker_processor = Tracker_ByteTrack(self.frame_skip, self.config_tracker)
        else:
            raise ValueError(f"Unsupported tracking algorithm: {self.tracker_name}")

    def update(self, frame, boxes, scores, labels):
        return self.tracker_processor.update(frame, boxes, scores, labels)
