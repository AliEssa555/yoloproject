
# YOLOv9 Modular Training Pipeline

This project sets up a modular YOLOv9 training pipeline with:

- 📦 **DVC** for data & pipeline versioning
- 📊 **MLflow** for experiment tracking
- 🚀 Modular Python scripts
- 🧪 Ready for integration with GitHub + DagsHub

## Project Structure

```
src/
  ├── train.py
  ├── data_loader.py
  ├── model.py
  ├── utils.py
  └── config.yaml
models/
  └── best_model.pt
metrics.json
params.yaml
dvc.yaml
```

## Setup Instructions

```bash
# Init Git, DVC, and install MLflow
git init
dvc init
pip install mlflow dvc

# Run training via DVC
dvc repro

# Track experiments
mlflow ui
```

## GitHub + DagsHub

1. Push repo to GitHub.
2. Import repo into DagsHub (auto-MLflow/DVC UI).
