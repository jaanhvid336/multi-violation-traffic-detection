"""
Train the helmet / no-helmet detector.

The shipped helmet.pt was trained for only 5 epochs and on a class-imbalanced
set (≈3:1 helmet:no-helmet), so it rarely flags a bare head. This script
retrains YOLOv8n properly with early stopping and copies the best weights to
yolov8_models/helmet.pt (the previous file is backed up first).
"""

import os
import shutil
from datetime import datetime
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")
MODEL_DIR = os.path.join(BASE_DIR, "yolov8_models")
TARGET = os.path.join(MODEL_DIR, "helmet.pt")

# Train long; patience stops early once validation mAP plateaus (the dataset is
# small, so it usually converges well before the cap).
EPOCHS = 80
PATIENCE = 15
IMGSZ = 640
BATCH = 16


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device="cpu",
        workers=8,
        project=os.path.join(BASE_DIR, "runs", "detect"),
        name="helmet_train",
        exist_ok=True,
        plots=True,
        # light augmentation aimed at CCTV variation
        hsv_v=0.4, degrees=5, translate=0.1, scale=0.4, fliplr=0.5,
    )

    best = os.path.join(BASE_DIR, "runs", "detect", "helmet_train",
                        "weights", "best.pt")
    if not os.path.exists(best):
        raise FileNotFoundError(f"best.pt not found at {best}")

    # Back up the existing model before overwriting.
    if os.path.exists(TARGET):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(TARGET, os.path.join(MODEL_DIR, f"helmet_prev_{stamp}.pt"))

    shutil.copy(best, TARGET)
    print(f"Training complete. Best weights copied to {TARGET}")


if __name__ == "__main__":
    train()
