from src.modules.detector.detector_manager import DetectorManager
from src.modules.tracker.tracker_manager import TrackerManager
from src.modules.counter.counter_manager import CounterManager
from src.modules.counter.counter_summary import CounterSummary
from src.modules.display.display_action import DisplayAction
from src.modules.memory.memory_manager import MemoryManager
from src.modules.utils.allowed_classes import AllowedClasses

"""
ComponentManager is responsible for creating and initialize the components
of the video processing pipeline. 
The components include: 
- Detector: Handles object detection in video frames.
- Tracker: Manages object tracking across frames.
- Counter: Counts objects based on detection and tracking.
- Display: Visualizes the results on the video frames. 
- Memory: Manages memory for the components.
- Summary: Summarizes the counting results.     
"""

class ComponentManager:
    @staticmethod
    def create(model_handler, config):
        
        config_detector = config.sub_configs.get("detector")["detector"]
        config_processor = config.sub_configs.get("processor")["processor"] 
        config_video = config.sub_configs.get("video")["video"]
        config_drawer = config.sub_configs.get("drawer")["drawer"]
               
        class_names, allowed_classes = AllowedClasses(config).get_allowed_classes()
 
        return {
            "detector": DetectorManager(
                model_handler, config_detector, allowed_classes
            ).get_detector(),
            "tracker": TrackerManager(
                config_processor,
                config.sub_configs.get("tracker")[config_processor["tracker"]],
            )
            if config_processor["enable_tracking"]
            else None,
            "counter": CounterManager(
                class_names,
                allowed_classes,
                config_video,
                config_drawer,
                config_processor,
            )
            if config_processor["enable_counter"]
            else None,
            "display": DisplayAction(class_names),
            "memory": MemoryManager(cleanup_frequency=100),
            "summary": CounterSummary(
                model_handler, class_names, config_processor, config_video
            ),
        }
