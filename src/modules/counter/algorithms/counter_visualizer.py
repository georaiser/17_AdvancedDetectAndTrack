import cv2
from src.modules.display.algorithms.display_config import DisplayConfig

class CounterVisualizer:
    """Handles visualization of counter lines and statistics"""

    def __init__(self, counter, class_names):
        self.counter = counter
        self.class_names = class_names
        
        self.config_display = DisplayConfig()

    def draw(self, frame, lines):
        """Draw all lines and statistics on the frame"""
        # Draw lines first
        for line in lines:
            cv2.line(
                frame,
                line["start_point"],
                line["end_point"],
                line["color"],
                self.config_display.counter_line_thickness,
            )
        # Draw statistics
        self._draw_statistics(frame, lines)
        return frame

    def _draw_statistics(self, frame, lines):
        """Draw crossing statistics for all lines"""
        # Collect all unique classes across all lines
        all_classes = set()
        for line in lines:
            for direction in ["up", "down"]:
                all_classes.update(line["counts"][direction].keys())

        # Sort classes for consistent ordering
        sorted_classes = sorted(list(all_classes))

        # Base vertical offset
        y_offset = self.config_display.counter_base_offset

        for line in lines:
            # Calculate totals
            total_up = sum(line["counts"]["up"].values())
            total_down = sum(line["counts"]["down"].values())
            line_summary = f"{line['name']}: Up: {total_up}, Down: {total_down}"

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                line_summary,
                self.config_display.counter_text_font,
                self.config_display.counter_text_scale,
                self.config_display.counter_text_thickness
            )

            # Draw background
            cv2.rectangle(
                frame,
                (10, y_offset - text_height - baseline),
                (10 + text_width, y_offset + baseline),
                self.config_display.counter_background_color,
                cv2.FILLED,
            )

            # Draw text
            cv2.putText(
                frame,
                line_summary,
                (10, y_offset),
                self.config_display.counter_text_font,
                self.config_display.counter_text_scale,
                line["color"],
                self.config_display.counter_text_thickness,
                self.config_display.counter_text_line_type,
            )

            # Move to next line
            y_offset += self.config_display.counter_line_spacing

            # Draw class-specific counts
            if sorted_classes:
                self._draw_class_counts(frame, line, sorted_classes, y_offset)

                # Move to next line section
                y_offset += self.config_display.counter_cell_height * 3 + 10

    def _draw_class_counts(self, frame, line, classes, y_offset):
        """Draw class-specific counts for a line"""
        x_start = 10
        cell_width = self.config_display.counter_cell_width
        cell_height = self.config_display.counter_cell_height

        # Draw column headers (classes)
        for class_idx, class_id in enumerate(classes):
            class_name = (
                self.class_names[int(class_id)]
                if class_id < len(self.class_names)
                else f"Class {int(class_id)}"
            )

            # Get text size for proper background sizing
            (text_width, text_height), baseline = cv2.getTextSize(
                class_name,
                self.config_display.counter_text_font,
                self.config_display.counter_text_scale,
                self.config_display.counter_text_thickness
            )

            # Draw background
            cv2.rectangle(
                frame,
                (x_start + (class_idx + 1) * cell_width, y_offset - 15),
                (x_start + (class_idx + 2) * cell_width, y_offset + baseline),
                self.config_display.counter_background_color,
                cv2.FILLED,
            )

            # Draw class name
            cv2.putText(
                frame,
                class_name,
                (x_start + (class_idx + 1) * cell_width + 5, y_offset - 2),
                self.config_display.counter_text_font,
                self.config_display.counter_text_scale,
                (0, 0, 0),
                self.config_display.counter_text_thickness,
                self.config_display.counter_text_line_type,
            )

        y_offset += cell_height

        # Draw direction rows
        directions = [("Up", (0, 100, 200)), ("Down", (0, 100, 200))]

        for direction, color in directions:
            # Draw direction label up/down
            cv2.putText(
                frame,
                direction,
                (x_start, y_offset - 2),
                self.config_display.counter_text_font,
                self.config_display.counter_text_scale,
                color,
                self.config_display.counter_text_thickness,
                self.config_display.counter_text_line_type
            )

            # Draw counts for each class
            for class_idx, class_id in enumerate(classes):
                count = line["counts"][direction.lower()].get(class_id, 0)
                count_text = str(count)

                # Get text size for centering
                (text_width, _), _ = cv2.getTextSize(
                    count_text,
                    self.config_display.counter_text_font,
                    self.config_display.counter_text_scale,
                    self.config_display.counter_text_thickness
                )

                # Draw background
                cv2.rectangle(
                    frame,
                    (x_start + (class_idx + 1) * cell_width, y_offset - cell_height),
                    (x_start + (class_idx + 2) * cell_width, y_offset),
                    self.config_display.counter_background_color,
                    cv2.FILLED,
                )

                # Draw count
                text_x = (
                    x_start
                    + (class_idx + 1) * cell_width
                    + (cell_width - text_width) // 2
                )
                cv2.putText(
                    frame,
                    count_text,
                    (text_x, y_offset - 2),
                    self.config_display.counter_text_font,
                    self.config_display.counter_text_scale,
                    self.config_display.counter_text_color,
                    self.config_display.counter_text_thickness,
                    self.config_display.counter_text_line_type,
                )

            # Move to next row
            y_offset += cell_height

