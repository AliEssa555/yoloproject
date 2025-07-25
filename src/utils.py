
import json

def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f)
