
import mlflow
import yaml
from src.model import load_model
from src.data_loader import get_dataloaders
from src.utils import save_metrics

def train():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run():
        # Log config parameters
        mlflow.log_params(config)

        # Load data and model
        train_loader, val_loader = get_dataloaders(config)
        model = load_model(config)

        # Dummy training loop (replace with actual YOLOv9 training)
        for epoch in range(config["epochs"]):
            print(f"Epoch {epoch+1}/{config['epochs']} - Training...")
            mlflow.log_metric("train_loss", 0.01 * (config['epochs'] - epoch), step=epoch)

        # Save a dummy model file
        model_path = "models/best_model.pt"
        with open(model_path, "w") as f:
            f.write("dummy model content")
        mlflow.log_artifact(model_path)

        save_metrics({"final_loss": 0.01}, "metrics.json")

if __name__ == "__main__":
    train()
