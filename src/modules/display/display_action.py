# -*- coding: utf-8 -*-
import cv2
from src.modules.display.algorithms.display_base import DisplayBase
from src.modules.display.algorithms.display_config import DisplayConfig

class DisplayAction(DisplayBase):
    def __init__(self, class_names):
        self.config_display = DisplayConfig()
        self.class_names = class_names

    def display_frame(self, frame):
        self.window_name = self.config_display.window_name
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, frame)

        return cv2.waitKey(1) & 0xFF

    def draw_detections(self, frame, boxes, scores, labels, fps):
        """Draw detection boxes and labels"""

        frame_out = frame.copy()

        # Draw boxes and labels
        for box, score, label in zip(boxes, scores, labels):
            # Draw box
            x1, y1, _, _ = self._draw_box(frame_out, box)

            # Draw label
            if self.config_display.show_labels:
                label_text = f"{self.class_names[int(label.item())]} {score:.2f}"

                self._draw_label(
                    frame_out, label_text, (x1, y1 - self.config_display.text_padding)
                )

        # Draw FPS
        self._draw_fps(frame_out, fps)

        return frame_out

    def draw_tracks(self, frame, tracks, fps):
        """Draw tracking boxes and labels"""

        frame_out = frame.copy()

        # Draw tracks
        for track in tracks:
            # Get track info

            track_id = track.track_id

            if hasattr(track, "to_ltrb") and callable(track.to_ltrb):
                # for DeepSORT
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb()
                class_id = track.det_class

            elif hasattr(track, "tlwh"):
                # for ByteTrack (STrack), convert tlwh to ltrb
                if not track.is_activated:
                    continue
                l, t, w, h = track.tlwh  # noqa: E741
                ltrb = [l, t, l + w, t + h]
                class_id = track.label

            else:
                raise AttributeError(
                    "Track object does not have 'to_ltrb()' or 'tlwh' attributes."
                )

            # Draw box
            x1, y1, _, _ = self._draw_box(frame_out, ltrb)

            # Draw label
            if self.config_display.show_labels:
                label_text = (
                    f"{self.class_names[int(class_id)]}-{track_id}"
                    if class_id < len(self.class_names)
                    else f"ID-{track_id}"
                )
                self._draw_label(
                    frame_out, label_text, (x1, y1 - self.config_display.text_padding)
                )

        # Draw FPS
        self._draw_fps(frame_out, fps)
        
        return frame_out

    def cleanup(self):
        """Clean up display resources"""
        cv2.destroyAllWindows()
