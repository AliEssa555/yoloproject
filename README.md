
# YOLOv9-10-12 Training Pipeline for ElectroCom Dataset

This project sets up a modular YOLO training pipeline with:

- **DVC** for data & pipeline versioning
- **MLflow** for experiment tracking
- Modular Python scripts
- Data Augmentation
- Ready for integration with GitHub + DagsHub

## Project Structure

```
data/
  ├──train/
     ├──images
     ├──labels
  ├──val/
     ├──images
     ├──labels
src/
  ├── train.py
  ├── augmentations.py
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
pip install mlflow dvc albumentations ultralytics

# Run training via DVC
dvc repro

# Track experiments
mlflow ui
Through Dagshub
```

## GitHub + DagsHub

1. Push repo to GitHub.
2. Import repo into DagsHub (auto-MLflow/DVC UI).
