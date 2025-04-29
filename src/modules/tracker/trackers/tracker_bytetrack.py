import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from box import Box

class Tracker_ByteTrack:
    def __init__(self, frame_skip, config):
        """Handles object tracking using BYTETrack algorithm"""
        frame_rate = 30/frame_skip  # Assuming 30 FPS original for the video
        self.tracker = BYTETracker(Box(config), frame_rate)  
        self.track_labels = {}

    def update(self, frame, boxes, scores, labels):
        if len(boxes) == 0:
            return []

        height, width = frame.shape[:2]
        img_info = [height, width]
        img_size = [height, width]

        detections = []
        temp_labels = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            detections.append([x1, y1, x2, y2, score.item()])
            temp_labels.append(label.item())

        detections = np.array(detections)

        tracks = self.tracker.update(detections, img_info, img_size)

        # Get matching indices using scores by JRE
        temp_labels_matched = []

        scores_cpu = []
        for score in scores:
            score = score.cpu().item()
            scores_cpu.append(score)

        for track in tracks:
            track_score = track.score  # Get track score
            matched_idx = np.where(np.equal(track_score, scores_cpu))[0]

            if len(matched_idx) > 0:
                idx = matched_idx[0]
                temp_labels_matched.append(temp_labels[idx])

        for track, label in zip(tracks, temp_labels_matched):
            if track.track_id in self.track_labels:
                if track.score > self.track_labels[track.track_id][1]:
                    self.track_labels[track.track_id] = [label, track.score]
                    track.label = label
            else:
                self.track_labels[track.track_id] = [label, track.score]
                track.label = label

        return tracks