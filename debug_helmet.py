"""
Debug script - Run this to check if the helmet model works correctly.
Usage: python debug_helmet.py
"""
import cv2
import os
from ultralytics import YOLO

MODEL_PATH = r"c:\Users\Jaanhvi Dahibaokar\OneDrive\Desktop\traffic_violation_system\yolov8_models\helmet.pt"
INPUT_VIDEO = r"c:\Users\Jaanhvi Dahibaokar\OneDrive\Desktop\traffic_violation_system\input_videos"

print("=== Helmet Model Diagnostic ===\n")

# 1. Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file NOT found at: {MODEL_PATH}")
    exit()
else:
    print(f"[OK] Model file found: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)//1024} KB)")

# 2. Load model
try:
    model = YOLO(MODEL_PATH)
    print(f"[OK] Model loaded successfully!")
    print(f"[INFO] Class names: {model.names}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# 3. Find a video to test with
video_files = []
for f in os.listdir(INPUT_VIDEO):
    if f.endswith(('.mp4', '.avi', '.mov')):
        video_files.append(os.path.join(INPUT_VIDEO, f))

if not video_files:
    print("[WARNING] No video found in input_videos folder. Skipping frame test.")
    exit()

video_path = video_files[0]
print(f"\n[INFO] Testing on video: {video_path}")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("[ERROR] Could not read frame from video!")
    exit()

# 4. Run model directly on full frame
print("\n[INFO] Running helmet model on full frame...")
results = model.predict(frame, verbose=False, conf=0.1)  # low conf to see all detections

if len(results) > 0 and results[0].boxes is not None:
    boxes = results[0].boxes
    print(f"[INFO] Total detections: {len(boxes)}")
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"  - Class: {cls_name} | Confidence: {conf:.2f}")
else:
    print("[WARNING] No detections found on full frame!")

print("\n=== Diagnostic Complete ===")
print("If you see class names above (like 'helmet', 'no-helmet'), the model is working.")
print("If there are NO detections, the model needs more training epochs.")
