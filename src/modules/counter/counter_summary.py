import torch
import datetime
import csv

class CounterSummary:
    def __init__(self, model_handler, class_names, config_processor, config_video):
        self.model_handler = model_handler
        self.class_names = class_names
        
        self.model_name = config_processor["model"]
        self.summary_data = {}
        self.output_file = f"Summary/counter_summary_{self.model_name}.txt"
        self.start_time = None
        self.end_time = None
        self.frame_count = 0
        self.fps_measurements = []
        self.detection_counts = 0
        self.processing_stats = {
            "input_video": config_video["input_path"],
            "output_video": config_video["output_path"],
            "tracking_algorithm": config_processor["tracker"]
            if config_processor["enable_tracking"]
            else "None",
            "frame_skip": config_processor["frame_skip"],
        }

    def start_processing(self):
        """Record the start time of processing"""
        self.start_time = datetime.datetime.now()

    def end_processing(self):
        """Record the end time of processing"""
        self.end_time = datetime.datetime.now()

    def update_frame_stats(self, fps, detection_count=0):
        """Update per-frame statistics"""
        self.frame_count += 1
        if fps > 0:  # Ignore zero FPS values
            self.fps_measurements.append(fps)
        self.detection_counts += detection_count

    def update_from_lines(self, lines_geometry):
        """Update summary data from the lines geometry data"""
        if self.model_name not in self.summary_data:
            self.summary_data[self.model_name] = {}

        for line in lines_geometry:
            line_name = line["name"]
            if line_name not in self.summary_data[self.model_name]:
                self.summary_data[self.model_name][line_name] = {
                    "up": line["counts"]["up"].copy(),
                    "down": line["counts"]["down"].copy(),
                    "total_up": sum(line["counts"]["up"].values()),
                    "total_down": sum(line["counts"]["down"].values()),
                }

    def get_processing_stats(self):
        """Calculate processing statistics"""
        stats = self.processing_stats.copy()

        # Get model parameters
        total_params, trained_params = self.model_handler.param
        stats["total_parameters"] = total_params
        stats["trainable_parameters"] = trained_params

        # Calculate timing information
        if self.start_time and self.end_time:
            processing_duration = self.end_time - self.start_time
            stats["processing_start"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
            stats["processing_end"] = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
            stats["processing_duration"] = str(processing_duration)
            stats["processing_seconds"] = processing_duration.total_seconds()

        # Calculate FPS statistics
        if self.fps_measurements:
            stats["fps_min"] = min(self.fps_measurements)
            stats["fps_max"] = max(self.fps_measurements)
            stats["fps_mean"] = sum(self.fps_measurements) / len(self.fps_measurements)
            stats["fps_median"] = sorted(self.fps_measurements)[
                len(self.fps_measurements) // 2
            ]

        # Other statistics
        stats["frames_processed"] = self.frame_count
        stats["detections_total"] = self.detection_counts
        if self.frame_count > 0:
            stats["detections_per_frame_avg"] = self.detection_counts / self.frame_count

        # Calculate total counts across all lines
        total_up = 0
        total_down = 0
        total_objects = 0

        for model, lines in self.summary_data.items():
            for line_name, counts in lines.items():
                total_up += counts["total_up"]
                total_down += counts["total_down"]
                total_objects += counts["total_up"] + counts["total_down"]

        stats["total_up"] = total_up
        stats["total_down"] = total_down
        stats["total_objects"] = total_objects

        # Device information
        if torch.cuda.is_available():
            stats["device"] = f"CUDA - {torch.cuda.get_device_name(0)}"
            stats["cuda_memory_allocated_peak"] = (
                f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
            )
            stats["cuda_memory_reserved_peak"] = (
                f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB"
            )
        else:
            stats["device"] = "CPU"

        return stats

    def print_summary(self):
        """Print the summary to console"""
        stats = self.get_processing_stats()

        print("\n" + "=" * 80)
        print(f"OBJECT COUNTING SUMMARY FOR MODEL: {self.model_name}")
        print("=" * 80)

        # Print processing statistics
        print("\nPROCESSING INFORMATION:")
        print(f"  Input Video: {stats['input_video']}")
        print(f"  Output Video: {stats['output_video']}")
        print(f"  Model: {self.model_name}")
        print(f"  Tracking Algorithm: {stats['tracking_algorithm']}")
        print(f"  Device: {stats['device']}")
        if "cuda_memory_allocated_peak" in stats:
            print(f"  Peak GPU Memory: {stats['cuda_memory_allocated_peak']}")

        print("\nTIMING:")
        if "processing_start" in stats:
            print(f"  Start: {stats['processing_start']}")
            print(f"  End: {stats['processing_end']}")
            print(f"  Duration: {stats['processing_duration']}")
            print(f"  Total Seconds: {stats['processing_seconds']:.2f}")

        print("\nPERFORMANCE:")
        if "fps_mean" in stats:
            print(f"  Average FPS: {stats['fps_mean']:.2f}")
            print(f"  Median FPS: {stats['fps_median']:.2f}")
            print(f"  Min FPS: {stats['fps_min']:.2f}")
            print(f"  Max FPS: {stats['fps_max']:.2f}")
        print(f"  Frames Processed: {stats['frames_processed']}")
        print(f"  Frame Skip Rate: {stats['frame_skip']}")
        if "detections_per_frame_avg" in stats:
            print(
                f"  Average Detections per Frame: {stats['detections_per_frame_avg']:.2f}"
            )

        print("\nCOUNTING STATISTICS:")
        print(f"  Total Up: {stats['total_up']}")
        print(f"  Total Down: {stats['total_down']}")
        print(f"  Total Objects: {stats['total_objects']}")

        # Print detailed counting data
        for model, lines in self.summary_data.items():
            for line_name, counts in lines.items():
                print(f"\nLine: {line_name}")
                print("-" * 40)

                # Print class-specific counts by direction
                for direction in ["up", "down"]:
                    print(f"\n{direction.upper()} Direction:")
                    if counts[direction]:
                        for class_id, count in counts[direction].items():
                            class_name = (
                                self.class_names[int(class_id)]
                                if int(class_id) < len(self.class_names)
                                else f"Class {int(class_id)}"
                            )
                            print(f"  {class_name}: {count}")
                    else:
                        print("  No objects counted")

                # Print totals
                print(f"\nTotals:")
                print(f"  Total Up: {counts['total_up']}")
                print(f"  Total Down: {counts['total_down']}")
                print(f"  Overall Total: {counts['total_up'] + counts['total_down']}")

    def export_to_file(self):
        """Export the summary to a text file"""
        stats = self.get_processing_stats()

        with open(self.output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"OBJECT COUNTING SUMMARY FOR MODEL: {self.model_name}\n")
            f.write(f"  Total Parameters: {stats['total_parameters']}\n")
            f.write(f"  Trainable Parameters: {stats['trainable_parameters']}\n")
            f.write("=" * 80 + "\n\n")

            # Write processing statistics
            f.write("PROCESSING INFORMATION:\n")
            f.write(f"  Input Video: {stats['input_video']}\n")
            f.write(f"  Output Video: {stats['output_video']}\n")
            f.write(f"  Model: {self.model_name}\n")
            f.write(f"  Tracking Algorithm: {stats['tracking_algorithm']}\n")
            f.write(f"  Device: {stats['device']}\n")
            if "cuda_memory_allocated_peak" in stats:
                f.write(f"  Peak GPU Memory: {stats['cuda_memory_allocated_peak']}\n")

            f.write("\nTIMING:\n")
            if "processing_start" in stats:
                f.write(f"  Start: {stats['processing_start']}\n")
                f.write(f"  End: {stats['processing_end']}\n")
                f.write(f"  Duration: {stats['processing_duration']}\n")
                f.write(f"  Total Seconds: {stats['processing_seconds']:.2f}\n")

            f.write("\nPERFORMANCE:\n")
            if "fps_mean" in stats:
                f.write(f"  Average FPS: {stats['fps_mean']:.2f}\n")
                f.write(f"  Median FPS: {stats['fps_median']:.2f}\n")
                f.write(f"  Min FPS: {stats['fps_min']:.2f}\n")
                f.write(f"  Max FPS: {stats['fps_max']:.2f}\n")
            f.write(f"  Frames Processed: {stats['frames_processed']}\n")
            f.write(f"  Frame Skip Rate: {stats['frame_skip']}\n")
            if "detections_per_frame_avg" in stats:
                f.write(
                    f"  Average Detections per Frame: {stats['detections_per_frame_avg']:.2f}\n"
                )

            f.write("\nCOUNTING STATISTICS:\n")
            f.write(f"  Total Up: {stats['total_up']}\n")
            f.write(f"  Total Down: {stats['total_down']}\n")
            f.write(f"  Total Objects: {stats['total_objects']}\n")

            # Write detailed counting data
            for model, lines in self.summary_data.items():
                for line_name, counts in lines.items():
                    f.write(f"\nLine: {line_name}\n")
                    f.write("-" * 40 + "\n")

                    # Write class-specific counts by direction
                    for direction in ["up", "down"]:
                        f.write(f"\n{direction.upper()} Direction:\n")
                        if counts[direction]:
                            for class_id, count in counts[direction].items():
                                class_name = (
                                    self.class_names[int(class_id)]
                                    if int(class_id) < len(self.class_names)
                                    else f"Class {int(class_id)}"
                                )
                                f.write(f"  {class_name}: {count}\n")
                        else:
                            f.write("  No objects counted\n")

                    # Write totals
                    f.write(f"\nTotals:\n")
                    f.write(f"  Total Up: {counts['total_up']}\n")
                    f.write(f"  Total Down: {counts['total_down']}\n")
                    f.write(
                        f"  Overall Total: {counts['total_up'] + counts['total_down']}\n\n"
                    )

        #print(f"\nSummary exported to {self.output_file}")

    def export_to_csv(self):
        """Export summary data to CSV for further analysis"""
        csv_file = f"Summary/counter_summary_{self.model_name}.csv"

        # Create data for CSV
        rows = []

        # Add header row
        header = ["Model", "Line", "Direction", "Class_ID", "Class_Name", "Count"]
        rows.append(header)

        # Add data rows
        for model, lines in self.summary_data.items():
            for line_name, counts in lines.items():
                for direction in ["up", "down"]:
                    for class_id, count in counts[direction].items():
                        class_name = (
                            self.class_names[int(class_id)]
                            if int(class_id) < len(self.class_names)
                            else f"Class {int(class_id)}"
                        )
                        row = [model, line_name, direction, class_id, class_name, count]
                        rows.append(row)

        # Write to CSV
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        #print(f"CSV data exported to {csv_file}")

