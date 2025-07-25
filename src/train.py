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
            project="experiments"
        )

        # Extract metrics safely
        metrics = {
            "mAP_0.5": results.box.map50,
            "mAP_0.5_0.95": results.box.map,
            "Precision": results.box.p,
            "Recall": results.box.r
        }

        # Extract speed metrics (optional)
        speed = results.speed
        if speed:
            metrics.update({
                "Train_speed": speed.get("train", 0),
                "Val_speed": speed.get("val", 0),
                "Inference_speed": speed.get("inference", 0)
            })

        # Log metrics to MLflow
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
