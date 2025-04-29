import cv2
import json

class LineDrawer:
    def __init__(self, config_video, config_drawer):
        """Initialize the LineDrawer class"""
        self.config_video = config_video
        self.config_drawer = config_drawer

        self.video_path = config_video["input_path"]
        self.color_palette = config_drawer["color_palette"]

        self.lines_config = []
        self.frame_with_lines = None
        self.temp_start_point = None
        self.line_counter = 1

        self._load_video_frame()

    def _load_video_frame(self, frame_number=100):
        cap = cv2.VideoCapture(self.video_path)

        # Skip frames
        for _ in range(frame_number):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                break
        cap.release()

        self.frame = frame

        if not ret:
            raise RuntimeError(f"Failed to read the video: {self.video_path}")

    def _draw_line(self, event, x, y, flags, param):
        """
        Callback function to handle mouse events and draw lines.

        :param event: Mouse event.
        :param x: X-coordinate of the event.
        :param y: Y-coordinate of the event.
        :param flags: Additional flags for the event.
        :param param: Additional parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # On left mouse button click
            if self.temp_start_point is None:
                # Store the first point of the line
                self.temp_start_point = (x, y)
            else:
                # Store the second point, draw the line, and reset
                temp_end_point = (x, y)

                # Use a color from the palette
                color = self.color_palette[(self.line_counter - 1) % len(self.color_palette)]

                # Define the line configuration
                line_config = {
                    'start_point': self.temp_start_point,
                    'end_point': temp_end_point,
                    'name': f'Line{self.line_counter}',
                    'color': color
                }

                self.lines_config.append(line_config)
                self.line_counter += 1

                # Draw the line on the frame
                cv2.line(self.frame, line_config['start_point'], line_config['end_point'], color,
                         self.config_drawer["line_thickness"])
                cv2.imshow("Draw Lines", self.frame)

                # Reset the start point
                self.temp_start_point = None

    def run(self):
        """
        Start the line drawing interaction.
        """
        cv2.namedWindow("Draw Lines")
        cv2.setMouseCallback("Draw Lines", self._draw_line)

        print("Click to define points for lines (two clicks per line). Press 'q' to quit.")
        while True:
            cv2.imshow("Draw Lines", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to quit
                break
            # Saving the frame
            cv2.imwrite('frame.jpg', self.frame)

        cv2.destroyAllWindows()
        # Save to JSON file
        with open("src/modules/counter/lines_geometry.json", "w") as file:
            json.dump(self.lines_config, file, indent=4)

        return self.lines_config
