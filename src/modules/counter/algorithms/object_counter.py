from src.modules.counter.algorithms.geometry_calculator import GeometryCalculator

class ObjectCounter:
    """
    Tracks objects crossing lines and maintains counts by direction and object class
    """
    def __init__(self, allowed_classes=None, class_names=None):
        self.allowed_classes = set(allowed_classes or [])
        self.class_names = class_names or []
        self.geometry = GeometryCalculator()

    def update(self, tracks, lines):
        """
        Update object counts based on current tracked objects
        """
        for line in lines:
            self._process_line_crossings(line, tracks)

    def _process_line_crossings(self, line, tracks):
        """Process all tracks for a specific line"""
        for track in tracks:
            # Skip if not a valid track or not in allowed classes
            if not self._is_valid_track(track):
                continue

            # Get the bounding box and center point
            bbox = self._get_track_bbox(track)
            if bbox is None:
                continue

            center = self.geometry.get_bbox_center(bbox)
            threshold = self.geometry.get_adaptive_threshold(bbox)

            # Skip if not near the line
            if not self.geometry.is_point_near_line(
                center, line["start_point"], line["end_point"], threshold
            ):
                continue

            # Process the crossing
            self._update_crossing_count(line, track, center)

    def _is_valid_track(self, track):
        """Check if track is valid and belongs to allowed classes"""
        # Check if track is confirmed - different trackers use different methods
        if hasattr(track, "is_confirmed") and callable(getattr(track, "is_confirmed")):
            is_confirmed = track.is_confirmed()
        elif hasattr(track, "is_activated"):
            is_confirmed = track.is_activated
        else:
            return False

        # Check class - different trackers use different attributes
        if hasattr(track, "det_class"):
            class_id = track.det_class
        elif hasattr(track, "label"):
            class_id = track.label
        else:
            return False

        # Track must be confirmed and in allowed classes (if specified)
        return is_confirmed and (
            not self.allowed_classes or class_id in self.allowed_classes
        )

    def _get_track_bbox(self, track):
        """Get the bounding box from a track object, handling different formats"""
        if hasattr(track, "to_ltrb") and callable(getattr(track, "to_ltrb")):
            # DeepSORT format
            return track.to_ltrb()
        elif hasattr(track, "tlwh"):
            # ByteTrack format
            l, t, w, h = track.tlwh  # noqa: E741
            return [l, t, l + w, t + h]
        return None

    def _update_crossing_count(self, line, track, center):
        """Update crossing counts for a single track"""
        track_id = track.track_id

        # Get class ID according to tracker type
        class_id = track.det_class if hasattr(track, "det_class") else track.label

        # Get current side of the line
        current_side = self.geometry.compute_line_side(
            center, line["start_point"], line["end_point"]
        )

        # Initialize tracking state for new objects
        if track_id not in line["tracked_objects"]:
            line["tracked_objects"][track_id] = {
                "prev_side": current_side,
                "class": class_id,
                "counted": False,
            }
            return

        # Get the current state for this object
        current_state = line["tracked_objects"][track_id]

        # Check if the object has crossed the line
        if (
            not current_state["counted"]
            and current_state["prev_side"] * current_side <= 0
        ):
            # Determine direction of crossing (based on sign of current side)
            direction = "down" if current_side > 0 else "up"

            # Initialize counter if needed
            if class_id not in line["counts"][direction]:
                line["counts"][direction][class_id] = 0

            # Increment count
            line["counts"][direction][class_id] += 1

            # Mark as counted to prevent multiple counts for same crossing
            current_state["counted"] = True

        # Update previous side
        current_state["prev_side"] = current_side

    def get_counts(self):
        """Get counts for all lines"""
        return [
            {
                "name": line["name"],
                "up": line["counts"]["up"],
                "down": line["counts"]["down"],
                "total_up": sum(line["counts"]["up"].values()),
                "total_down": sum(line["counts"]["down"].values()),
            }
            for line in self.lines
        ]

