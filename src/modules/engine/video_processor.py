import os
import time
import cv2
from tqdm import tqdm
from src.modules.videoIO.video_io import VideoReader, VideoWriter
from src.modules.engine.utils.component_manager import ComponentManager
from src.modules.engine.utils.image_square import ImageSquare

class VideoProcessor:
    def __init__(self, model_handler, config, max_frame):
        os.makedirs("Output", exist_ok=True)
        os.makedirs("Summary", exist_ok=True)

        self.model_handler = model_handler
        self.device = model_handler.device
        self.config = config
        self.max_frame = max_frame

        try:
            self.model_name = self.model_handler.model.model_name
        except AttributeError:
            self.model_name = self.model_handler.get_model_name()

        # Initialize components
        self.components = ComponentManager.create(model_handler, config)
        self.detector = self.components["detector"]
        self.tracker = self.components["tracker"]
        self.counter = self.components["counter"]
        self.display = self.components["display"]
        self.memory = self.components["memory"]
        self.summary = self.components["summary"]

    def process_video(self, config):
        
        config_processor = config.sub_configs.get("processor")["processor"]
        config_video = config.sub_configs.get("video")["video"]

        video_reader = VideoReader(config_video["input_path"])
        target_size = self.model_handler.image_size
        frame_skip = config_processor["frame_skip"]

        # Start summary timing
        self.summary.start_processing()

        # Get video parameters
        fps_orig, width_orig, height_orig, total_frames = (
            video_reader.video_parameters()
        )
        fps_adjusted = int(fps_orig / frame_skip)

        # Calculate new dimensions
        padding_info = ImageSquare.calculate_dimensions(
            width_orig, height_orig, target_size
        )

        # Initialize video writer if needed
        writer = (
            VideoWriter(
                config_video["output_path"], fps_adjusted, padding_info["original_size"]
            )
            if config_processor["enable_save"]
            else None
        )

        frame_count = 0
        with tqdm(total=total_frames) as pbar:
            while True and frame_count <= self.max_frame:  # stop process by frame count
                ret, frame = video_reader.read_frame()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    # Process frame
                    start_time = time.time()

                    # Detect objects
                    if self.detector:
                        boxes, scores, labels = (
                            self.detector.detection_pipeline(frame, padding_info)
                        )
                        # Update detection count for summary
                        self.summary.update_frame_stats(0, len(boxes))
                       
                    # Track objects
                    if self.tracker:
                        tracks = self.tracker.update(frame, boxes, scores, labels)
                    
                    # Draw results
                    if self.tracker:
                        frame_processed = self.display.draw_tracks(
                            frame, tracks, 1.0 / (time.time() - start_time)
                        )
                    else:
                        frame_processed = self.display.draw_detections(
                            frame,
                            boxes,
                            scores,
                            labels,
                            1.0 / (time.time() - start_time),
                        )

                    # Calculate FPS and update summary
                    current_fps = 1.0 / (time.time() - start_time)
                    self.summary.update_frame_stats(current_fps)

                    # Counter:
                    if self.counter and self.tracker:
                        frame_processed = self.counter.process_frame(
                            frame_processed, tracks
                        )

                    # Display/save results
                    if config_processor["enable_display"]:
                        key = self.display.display_frame(frame_processed)
                        if key == ord("q"):  # Quit if 'q' is pressed
                            break

                    if writer and config_processor["enable_save"]:
                        writer.write(frame_processed)

                frame_count += 1

                # Call memory cleanup
                self.memory.cleanup(frame_count)

                pbar.update(1)

        # End summary timing
        self.summary.end_processing()

        # Generate and export summary
        if self.counter:
            self.summary.update_from_lines(self.counter.lines_geometry)
        # self.summary.print_summary()
        self.summary.export_to_file()
        self.summary.export_to_csv()

        # Cleanup
        video_reader.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
