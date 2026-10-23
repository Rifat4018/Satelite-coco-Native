"""FAKE sahi for the CPU smoke test only."""
__version__ = "0.0-fake"


class _DM:
    def __init__(self, model):
        self.model = model


class AutoDetectionModel:
    @staticmethod
    def from_pretrained(model_type=None, model=None, model_path=None, **kw):
        if model is None:
            from ultralytics import YOLO
            model = YOLO(model_path)
        return _DM(model)
