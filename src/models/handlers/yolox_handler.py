import torch
from yolox.exp import get_exp
from yolox.utils import fuse_model


class YOLOXHandler:
    """Handles YOLOX model implementation based on official demo code"""

    def __init__(self, model_name, device, **kwargs):
        self.model_name = model_name
        self.device = device

        # Extract config parameters with defaults
        self.exp_file = kwargs.get("exp_file")
        self.ckpt_file = kwargs.get("ckpt_file") + f"/{self.model_name}.pth"
        self.confthre = kwargs.get("confthre", 0.5)
        self.nmsthre = kwargs.get("nmsthre", 0.5)
        self.legacy = kwargs.get("legacy", False)

        # Initialize model from experiment file or model name
        self.exp = get_exp(exp_file=self.exp_file, exp_name=self.model_name)
        self.exp.test_conf = self.confthre
        self.exp.nmsthre = self.nmsthre
        self.num_classes = self.exp.num_classes
        self.test_size = self.exp.test_size

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load YOLOX model based on experiment definition"""
        model = self.exp.get_model()

        # Load checkpoint
        ckpt = torch.load(self.ckpt_file, map_location=self.device)

        model.load_state_dict(ckpt["model"])

        model.to(self.device)
        # Set to evaluation mode
        model.eval()

        # Optionally for better performance
        model = fuse_model(model)

        return model
