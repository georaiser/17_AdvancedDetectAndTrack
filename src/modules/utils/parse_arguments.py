import argparse

class ParseArguments:
    """
    This class is responsible for parsing command-line arguments
    and overriding YAML configurations.
    """

    @staticmethod
    def str_to_bool(value):
        """Convert a string to a boolean (case insensitive)."""
        if isinstance(value, bool):
            return value  # If it's already a boolean, return as-is
        value = value.strip().lower()
        if value in {"true", "yes", "1"}:
            return True
        elif value in {"false", "no", "0"}:
            return False
        else:
            raise argparse.ArgumentTypeError(f"Bool expected, got '{value}'")

    @staticmethod
    def parse_arguments():
        """Parse command-line arguments to override YAML configurations."""
        parser = argparse.ArgumentParser(
            description="Override YAML configurations dynamically."
        )

        # Input & Output
        parser.add_argument(
            "--video", type=str, required=True, help="Path to input video")
        parser.add_argument(
            "--output", type=str, default="output.mp4", help="Path to save output video")
        # Video Processor Arguments
        parser.add_argument(
            "--frame-skip", type=int, help="frames to skip for processing")
        parser.add_argument(
            "--tracker", type=str, default="bytetrack", help="Choose tracker")
        parser.add_argument(
            "--enable-tracking", type=ParseArguments.str_to_bool, help="Enable tracking"
        )
        parser.add_argument(
            "--enable-counter", type=ParseArguments.str_to_bool, help="Enable counter"
        )
        parser.add_argument(
            "--enable-display", type=ParseArguments.str_to_bool, help="Enable display"
        )
        parser.add_argument(
            "--enable-save", type=ParseArguments.str_to_bool, default=True, help="Enable saving",
        )
        parser.add_argument(
            "--enable-drawer", type=ParseArguments.str_to_bool, help="Enable drawer"
        )
        # Model Arguments
        parser.add_argument(  
            "--model", type=str, default="yolo11s", help="Choose detection model")

        return parser.parse_args()
    
    @staticmethod
    def override_config(config, args):
        """Override YAML configurations with command-line arguments."""

        config.set("video", "input_path", args.video)
        config.set("video", "output_path", args.output)
        config.set("processor", "frame_skip", args.frame_skip)
        config.set("processor", "tracker", args.tracker)
        config.set("processor", "enable_tracking", args.enable_tracking)
        config.set("processor", "enable_counter", args.enable_counter)
        config.set("processor", "enable_display", args.enable_display)
        config.set("processor", "enable_save", args.enable_save)
        config.set("processor", "enable_drawer", args.enable_drawer)
        config.set("processor", "model", args.model)    
        # Show updated processor configuration
        updated_processor_config = config.sub_configs.get("processor", {}).get(
            "processor", {})
        print("\n🔄Processor Configuration:")
        for key, value in updated_processor_config.items():
            print(f"  {key}: {value}")


