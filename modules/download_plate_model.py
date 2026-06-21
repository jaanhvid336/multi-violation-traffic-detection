"""Fetch the YOLOv8 license-plate detection model used by PlateDetector.

Run once after cloning:  python modules/download_plate_model.py
It downloads Koushim/yolov8-license-plate-detection from the Hugging Face Hub
and saves it to yolov8_models/plate.pt.
"""

import os
import shutil

HF_REPO = "Koushim/yolov8-license-plate-detection"
HF_FILE = "best.pt"
DEST = os.path.join("yolov8_models", "plate.pt")


def download_plate_model(dest=DEST):
    if os.path.exists(dest):
        print(f"Plate model already present at {dest}")
        return dest
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cached = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    shutil.copy(cached, dest)
    print(f"Saved plate model to {dest} ({os.path.getsize(dest)} bytes)")
    return dest


if __name__ == "__main__":
    download_plate_model()
