
from ultralytics import YOLO

def load_model(config):
    model = YOLO(config.get("pretrained_weights", "yolov9.pt"))
    return model
