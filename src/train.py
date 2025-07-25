
import mlflow
import yaml
from src.model import load_model
from src.utils import save_metrics

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

        # Log final model and metrics
        mlflow.log_artifacts("experiments/" + config["experiment_name"])
        save_metrics({"results": str(results)}, "metrics.json")

if __name__ == "__main__":
    train()
