import numpy as np

class GeometryCalculator:
    """Utility class for geometric calculations related to line crossing"""

    @staticmethod
    def compute_line_side(point, line_start, line_end):
        """
        Compute which side of a line a point is on.
        Returns a normalized value: positive = one side, negative = other side
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if line_length == 0:
            return 0

        # Normalized cross product
        return ((x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)) / line_length

    @staticmethod
    def is_point_near_line(point, line_start, line_end, threshold):
        """
        Check if a point is within a threshold distance of a line segment
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Vector math for accurate distance calculation
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([x - x1, y - y1])

        line_length = np.linalg.norm(line_vec)

        if line_length == 0:
            return False

        line_unit_vec = line_vec / line_length
        projection_length = np.dot(point_vec, line_unit_vec)

        # Check if projection is on the line segment
        if 0 <= projection_length <= line_length:
            distance = abs(np.cross(line_unit_vec, point_vec))
            return distance < threshold
        return False

    @staticmethod
    def get_bbox_center(bbox):
        """Calculate center point of a bounding box [x1, y1, x2, y2]"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def get_adaptive_threshold(bbox):
        """Calculate an adaptive threshold based on object size"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return min(width, height) * 1.5  # Adjust multiplier as needed
