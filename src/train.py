import mlflow
import yaml
from src.model import load_model
from src.utils import save_metrics

mlflow.set_tracking_uri("https://dagshub.com/AliEssa555/yoloproject.mlflow")

def train():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(config)

        model = load_model(config)

        # Train directly using YOLO's .train() method
        results = model.train(
            data=config["data_config"],
            epochs=config["epochs"],
            imgsz=config["image_size"],
            batch=config["batch_size"],
            name=config["experiment_name"],
            project="experiments",
            augment=True,  # enable augmentations
            translate=config["augmentations"].get("translate", 0.0),
            shear=config["augmentations"].get("shear", 0.0),
            flipud=config["augmentations"].get("flipud", 0.0),
            fliplr=config["augmentations"].get("fliplr", 0.0),
            mosaic=config["augmentations"].get("mosaic", 1.0),
            cutmix=config["augmentations"].get("cutmix", 1.0),
        )


import numpy as np

# Convert NumPy arrays or tensors to scalars
def make_scalar(val):
    if isinstance(val, (np.ndarray, list)) and len(val) == 1:
        return float(val[0])
    elif hasattr(val, 'item'):
        return float(val.item())
    return float(val)  # fallback for int, float

    metrics = {
        "mAP_0.5": make_scalar(results.box.map50),
        "mAP_0.5_0.95": make_scalar(results.box.map),
        "Precision": make_scalar(results.box.p),
        "Recall": make_scalar(results.box.r),
        "Epochs": make_scalar(results.epoch + 1)
    }

    # Optional speed values (convert safely)
    speed = results.speed
    if speed:
        metrics.update({
            "Train_speed": make_scalar(speed.get("train", 0)),
            "Val_speed": make_scalar(speed.get("val", 0)),
            "Inference_speed": make_scalar(speed.get("inference", 0))
        })

    mlflow.log_metrics(metrics)


    # Save metrics locally
    save_metrics(metrics, "metrics.json")

    # Log artifacts (ensure correct paths)
    artifact_path = f"experiments/{config['experiment_name']}"
    mlflow.log_artifacts(artifact_path)

    # If specific images (e.g., confusion matrix) exist, log them:
    import os
    for img in ["confusion_matrix.png", "results.png"]:
        img_path = os.path.join(artifact_path, img)
        if os.path.exists(img_path):
            mlflow.log_artifact(img_path)

if __name__ == "__main__":
    train()
