import cv2

class DisplayConfig:  
    def __init__(self):
        # Window settings
        self.window_name = "Object Detection"

        # Box settings
        self.box_color = (0, 255, 0)  # BGR format
        self.box_thickness = 1
        self.box_line_type = cv2.LINE_AA

        # Text settings
        self.text_color = (0, 0, 0)  # BGR format
        self.text_scale = 0.4
        self.text_thickness = 1
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_line_type = cv2.LINE_AA
        self.text_padding = 5  # Pixels above box

        # Display options
        self.show_fps = True
        self.show_labels = True
        self.show_confidence = True
        self.show_tracking_id = True
        self.show_class_name = True

        # FPS display settings
        self.fps_position = (10, 20)  # (x,y) coordinates
        self.base_offset = 20
        self.fps_color = (0, 0, 0)
        self.fps_scale = 0.4

        # Background settings
        self.background_color = (200, 100, 200)  # BGR format
        
        ## Counter settings
        self.counter_background_color = (240, 240, 210)
        self.counter_text_color = (100, 10, 10)
        self.counter_text_scale = 0.4
        self.counter_text_thickness = 1
        self.counter_text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.counter_text_line_type = cv2.LINE_AA
        self.counter_text_padding = 5
        self.counter_base_offset = 40
        self.counter_cell_height = 15
        self.counter_cell_width = 40
        self.counter_line_thickness = 2
        self.counter_line_spacing = 25
        
        
