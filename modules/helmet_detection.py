import os
import cv2
from ultralytics import YOLO


class HelmetDetector:
    def __init__(self):
        self.model_loaded = False
        self.device = "cpu"

        possible_paths = [
            "yolov8_models/helmet.pt",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "yolov8_models",
                "helmet.pt",
            ),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.model = YOLO(path)
                    self.model.to(self.device)

                    self.model_loaded = True

                    print("=" * 60)
                    print("Helmet model loaded")
                    print("Path :", path)
                    print("Classes :", self.model.names)
                    print("=" * 60)

                    break

                except Exception as e:
                    print("Helmet model load error:", e)

        if not self.model_loaded:
            print("Helmet model NOT found")

    def check_helmet(self, frame, rider_bbox):
        """Classify the helmet status of a rider.

        Returns:
            True  -> helmet detected
            False -> no-helmet detected
            None  -> uncertain (model saw nothing); caller should NOT flag a
                     violation in this case.
        """

        if not self.model_loaded:
            return None

        x1, y1, x2, y2 = map(int, rider_bbox)

        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # ------------------------------------------------
        # USE THE FULL PERSON BOX
        # The model was trained on full rider images, so we feed it the whole
        # person crop. A tight head-only slice starves it of context and it
        # detects nothing. We rely on the highest-confidence detection below to
        # decide helmet vs no-helmet, rather than cropping aggressively.
        # ------------------------------------------------

        crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            return None

        # ------------------------------------------------
        # HELMET DETECTION
        # ------------------------------------------------

        try:
            # conf=0.10: the model is weak on rear-view / CCTV heads and returns
            # nothing for most riders at 0.20. Lowering the threshold roughly
            # halves the "detected nothing" rate and surfaces no-helmet cases.
            results = self.model.predict(
                crop,
                conf=0.10,
                verbose=False,
            )
        except Exception as e:
            print("Prediction error:", e)
            return None

        # Pick the single most confident detection and decide from its class.
        # This avoids the old "any no-helmet box wins" bias, which mislabelled
        # riders that had both a helmet and a (lower-confidence) no-helmet box.
        best_helmet = 0.0
        best_no_helmet = 0.0

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0:        # helmet
                    best_helmet = max(best_helmet, conf)
                elif cls == 1:      # no-helmet
                    best_no_helmet = max(best_no_helmet, conf)

        # Nothing detected -> uncertain, do not flag a violation.
        if best_helmet == 0.0 and best_no_helmet == 0.0:
            return None

        # Highest-confidence class wins.
        return best_helmet >= best_no_helmet