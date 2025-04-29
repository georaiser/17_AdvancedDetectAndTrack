from effdet import create_model


class EfficientdetHandler:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        model = None

        if self.model_name.startswith("tf_efficientdet"):
            model = create_model(self.model_name, pretrained=True, bench_task="predict")
            model = model.to(self.device)
            model.eval()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model
