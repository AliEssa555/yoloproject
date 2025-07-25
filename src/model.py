from ultralytics import YOLO
import os
import requests
import torch


from ultralytics import YOLO

def load_model(config):
    weights_path = config.get("pretrained_weights", "yolov11n-seg.pt")
    model = YOLO(weights_path)  # This handles loading safely
    return model


