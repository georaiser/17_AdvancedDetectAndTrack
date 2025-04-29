class Line:
    """Represents a counting line with associated tracking state"""
    def __init__(self, lines_geometry):
        self.start_point = lines_geometry["start_point"]
        self.end_point = lines_geometry["end_point"]
        self.name = lines_geometry.get("name") or f"Line-{id(self)}"
        self.color = lines_geometry.get("color")
        # Counter state
        self.counts = {"up": {}, "down": {}}
        self.tracked_objects = {}


