import cv2
import os
from ultralytics import YOLO

class DetectionEngine:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the base YOLOv8 model for vehicles and persons
        self.device = "cpu"
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            self.error = str(e)
            
    def process_frame(self, frame, persist=True):
        if not self.model_loaded:
            return frame, []
            
        # Run inference WITH tracking so every vehicle/person gets a stable id.
        # Stable ids are what the triple-riding cooldown keys on; without them
        # every object shares id -1 and the cooldown misbehaves.
        # Classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        #
        # imgsz=960: the single biggest lever for triple-riding accuracy. At the
        # default 640 the 3rd (rear, heavily-occluded) rider is routinely missed,
        # so a triple-rider bike shows up as "Normal (2)". Upsizing to 960 recovers
        # those riders (~50% more 3+ rider bikes detected in testing) and beat 1280
        # while staying faster. conf=0.15 catches the faint occluded rider; iou=0.7
        # keeps heavily-overlapping rider boxes from being merged by NMS.
        results = self.model.track(
            frame,
            persist=persist,
            verbose=False,
            conf=0.15,        # lower -> catches distant / partially hidden riders
            iou=0.70,         # high -> keep overlapping rider boxes (don't NMS-merge)
            imgsz=768,        # balance: recovers most occluded riders, ~30% faster than 960
            classes=[0, 3],
        )
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    # xyxy format
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    
                    track_id = -1
                    if box.id is not None:
                        track_id = int(box.id[0])
                    
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                        "track_id": track_id
                    })
                    
                    # Generic class labels intentionally suppressed.
                    # Violation-specific overlays (helmet, triple-riding,
                    # signal-jump) are drawn by their own modules.
                
        return frame, detections
