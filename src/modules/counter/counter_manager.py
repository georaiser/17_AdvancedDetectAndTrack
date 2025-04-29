import json
from src.modules.counter.algorithms.object_counter import ObjectCounter
from src.modules.counter.algorithms.counter_visualizer import CounterVisualizer
from src.modules.drawer.line_drawer import LineDrawer

class CounterManager:
    """Integrates object counting with the main video processing pipeline"""

    def __init__(self, class_names, allowed_classes, config_video, config_drawer, config_processor):
     
        self.counter = ObjectCounter(allowed_classes, class_names)
        self.visualizer = CounterVisualizer(self.counter, class_names)
        self.drawer = (
            LineDrawer(config_video, config_drawer)
            if config_processor["enable_drawer"]
            else None
        )
        self.get_lines_geometry()

    def get_lines_geometry(self):
        """Get current lines geometry"""
        if self.drawer:
            self.lines_geometry = self.drawer.run()
            # Load lines from file
            self.lines_geometry = self.load_lines_geometry()
        else:
            # Load lines from file
            self.lines_geometry = self.load_lines_geometry()


    def process_frame(self, frame, tracks):
        """Process tracks for a frame and draw visualization"""
        # Update counters with current tracks
        self.counter.update(tracks, self.lines_geometry)
        # Draw visualization on the frame
        frame = self.visualizer.draw(frame, self.lines_geometry)
        return frame

    def load_lines_geometry(self):
        """
        Load lines geometry from a JSON file.
        Returns a list of line configurations.
        """
        try:
            # Load lines from file
            with open("src/modules/counter/lines_geometry.json", "r") as file:
                lines = json.load(file)
                # Initialize tracking state for each line
                for line in lines:
                    line.setdefault("counts", {"up": {}, "down": {}})
                    line.setdefault("tracked_objects", {})
                return lines
        except FileNotFoundError:
            return []

    def get_counts(self):
        """Get current counts for all lines"""
        return self.counter.get_counts()

