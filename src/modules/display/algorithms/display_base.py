import cv2
from src.modules.display.algorithms.display_config import DisplayConfig

class DisplayBase:
    def __init__(self):
        self.config_display = DisplayConfig()
          
    def _draw_box(self, frame, box):
        """Draw bounding box on frame"""      
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.config_display.box_color,
            self.config_display.box_thickness,
            self.config_display.box_line_type,
        )
        return (x1, y1, x2, y2)

    def _draw_label(self, frame, text, position):
        """Draw text label on frame"""
        cv2.putText(
            frame,
            text,
            position,
            self.config_display.text_font,
            self.config_display.text_scale,
            self.config_display.text_color,
            self.config_display.text_thickness,
            self.config_display.text_line_type,
        )

    def _draw_background(self, frame, text):
        y_offset = self.config_display.base_offset
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.config_display.text_font,
            self.config_display.text_scale,
            self.config_display.text_thickness,
        )
        cv2.rectangle(
            frame,
            (10, y_offset - text_height - baseline),
            (10 + text_width, y_offset + baseline),
            self.config_display.background_color,
            cv2.FILLED,
        )

    def _draw_fps(self, frame, fps):
        """Draw FPS counter on frame"""
        if self.config_display.show_fps:
            fps_text = f"FPS: {fps:.2f}"
            self._draw_background(frame, fps_text)
            self._draw_label(frame, fps_text, self.config_display.fps_position)
