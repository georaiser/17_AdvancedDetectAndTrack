# -*- coding: utf-8 -*-
"""
python3 -m src.main --video data/input/Video1a.mp4 --output data/output/Output.mp4
--frame-skip 2
--enable-display True
--enable-tracking True
--tracker bytetrack
--enable-counter True
--enable-drawer True
--enable-save True
--model yolo11s

## models available
tf_efficientdet_d0  tf_efficientdet_d2  tf_efficientdet_d3 ...
fasterrcnn_resnet50_fpn
yolo11l  yolo11s  yolo12m  yolov5mu  yolov7   yolov8m  yolox-m
yolo11m  yolo12l  yolo12s  yolov5su  yolov7x  yolov8s  yolox-s

## after install requirements
sudo mount -o remount,size=12G /tmp # maybe not necessary
pip3 install --upgrade pip setuptools wheel
git clone https://github.com/georaiser/YOLOX.git
cd YOLOX
pip3 install -v -e .



## tracker available
deepsort  bytetrack
"""

import logging
from src.config.config import ConfigManager
from src.models.model_manager import ModelManager
from src.modules.utils.parse_arguments import ParseArguments
from src.modules.engine.video_processor import VideoProcessor

def setup_logging():
    """Configura el sistema de logs."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    setup_logging()
    
    # Load configurations
    config = ConfigManager()

    # Parse arguments
    args = ParseArguments.parse_arguments()
    
    # Override configurations with arguments
    ParseArguments.override_config(config, args)
    config.save()
    
    # Get the model name and model config
    model_name = config.get("processor", "model")
    model_config = config.sub_configs.get("model", None)

    # Initialize the model manager with the specific config
    model_handler = ModelManager(model_name, model_config)

    # Initialize the video processor with the model handler and config
    video_processor = VideoProcessor(model_handler, config, max_frame=400)
    # Process the video
    video_processor.process_video(config)

if __name__ == "__main__":
    main()