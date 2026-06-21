import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_MODEL_PATH = os.path.join("yolov8_models", "plate.pt")


class PlateDetector:
    """
    License-plate reader: a dedicated YOLO plate-detection model localises the
    plate, then EasyOCR reads the tight plate crop.

    The whole-vehicle box that the main detector produces is far too coarse for
    OCR — the plate is only a few dozen pixels wide inside it. So for each
    vehicle we upscale its crop, let the plate model find the plate rectangle,
    and OCR only that. This is the single biggest accuracy lever.
    """

    def __init__(self, plate_model_path=PLATE_MODEL_PATH, use_yolo_plate_model=True):
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        self.plate_model = None
        self.plate_model_loaded = False

        if not os.path.exists(plate_model_path):
            # Self-heal: fetch the model from the Hugging Face Hub on first run.
            try:
                from modules.download_plate_model import download_plate_model
                download_plate_model(plate_model_path)
            except Exception:
                pass

        try:
            if os.path.exists(plate_model_path):
                self.plate_model = YOLO(plate_model_path)
                self.plate_model_loaded = True
        except Exception:
            self.plate_model_loaded = False

    # ── public API ──────────────────────────────────────────────

    def read_vehicle_plate(self, frame, vehicle_bbox):
        """Read the number plate for a single vehicle box. Returns a cleaned
        plate string, or "" when nothing plausible is read."""
        crop = self._safe_crop(frame, vehicle_bbox)
        if crop is None:
            return ""

        # The plate model gives the tightest (best) crop when it fires. OCR that
        # first; if it yields a confident valid plate, return immediately — this
        # short-circuit is what keeps the live feed responsive on CPU (one OCR
        # pass instead of two). Only when the model misses or reads weakly do we
        # fall back to OCR-ing the lower region of the vehicle.
        located = self._locate_plate(crop)
        best, best_score = "", -1.0
        if located is not None:
            best, best_score = self._read_plate_crop(located)
            if best and best_score >= 0.50:
                return best

        h = crop.shape[0]
        text, score = self._read_plate_crop(crop[int(h * 0.45):, :])
        if text and score > best_score:
            best = text
        return best

    def detect_plates_in_frame(self, frame):
        """Detect every plate in a full frame.
        Returns [{'bbox': (x1,y1,x2,y2), 'plate_text': str, 'confidence': float}].
        """
        if not self.plate_model_loaded or frame is None or frame.size == 0:
            return []

        boxes = self._predict_plate_boxes(frame, imgsz=1280)
        detections = []
        for (x1, y1, x2, y2, conf) in boxes:
            plate_crop = frame[y1:y2, x1:x2]
            text, _ = self._read_plate_crop(plate_crop)
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'plate_text': text,
                'confidence': conf,
            })
        return detections

    def recognize_plate_for_vehicle(self, frame, vehicle_bbox):
        """Backwards-compatible alias for read_vehicle_plate."""
        return self.read_vehicle_plate(frame, vehicle_bbox)

    # ── plate localisation ──────────────────────────────────────

    def _locate_plate(self, vehicle_crop):
        """Find the plate rectangle inside a vehicle crop and return that tight
        sub-crop (at the upscaled resolution used for detection), or None."""
        if not self.plate_model_loaded:
            return None

        h, w = vehicle_crop.shape[:2]
        if w == 0 or h == 0:
            return None

        # Upscale small vehicle crops so the plate model sees a plate big enough
        # to localise. Target ~1280px on the long side, capped at 6x — recall on
        # low-res footage roughly doubles vs a 640px target.
        scale = max(1.0, min(6.0, 1280 / max(w, h)))
        if scale != 1.0:
            vehicle_crop = cv2.resize(
                vehicle_crop, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC)

        boxes = self._predict_plate_boxes(vehicle_crop, conf=0.10)
        if not boxes:
            return None

        # Most confident plate.
        x1, y1, x2, y2, _ = max(boxes, key=lambda b: b[4])
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.10)
        H, W = vehicle_crop.shape[:2]
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x); y2 = min(H, y2 + pad_y)
        plate = vehicle_crop[y1:y2, x1:x2]
        return plate if plate.size else None

    def _predict_plate_boxes(self, image, conf=0.25, imgsz=640):
        """Run the plate model and return [(x1,y1,x2,y2,conf), ...]."""
        try:
            results = self.plate_model.predict(
                image, verbose=False, conf=conf, imgsz=imgsz)
        except Exception:
            return []
        out = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                out.append((x1, y1, x2, y2, float(box.conf[0])))
        return out

    # ── OCR ─────────────────────────────────────────────────────

    def _read_plate_crop(self, plate_crop):
        """OCR a tight plate crop, keep only the main number row, and validate.
        Returns (plate_text, score); ("", 0.0) when nothing plausible is read.
        The score (mean token confidence of the chosen row) lets the caller pick
        the best result across several candidate regions."""
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        gray = self._preprocess(plate_crop)
        try:
            results = self.ocr_reader.readtext(
                gray, detail=1, allowlist=PLATE_CHARS, canvas_size=1280)
        except Exception:
            return "", 0.0
        if not results:
            return "", 0.0

        # Build cleaned tokens with geometry.
        tokens = []
        for (box, text, prob) in results:
            cleaned = "".join(c for c in text if c.isalnum()).upper()
            if prob < 0.30 or len(cleaned) < 2:
                continue
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            tokens.append({
                "text": cleaned, "prob": float(prob),
                "yc": (min(ys) + max(ys)) / 2,
                "x0": min(xs),
                "h": max(ys) - min(ys),
            })
        if not tokens:
            return "", 0.0

        row = self._dominant_row(tokens)
        row.sort(key=lambda t: t["x0"])
        plate = "".join(t["text"] for t in row)

        if not self._is_plausible_plate(plate):
            return "", 0.0
        score = float(np.mean([t["prob"] for t in row]))
        return plate, score

    @staticmethod
    def _is_plausible_plate(plate):
        """A real registration has both letters and digits and a sane length.
        Requiring a letter is what rejects the registration date line ("1124")
        that sits just below the plate and often reads more cleanly than it."""
        if not (4 <= len(plate) <= 12):
            return False
        return any(c.isalpha() for c in plate) and any(c.isdigit() for c in plate)

    @staticmethod
    def _dominant_row(tokens):
        """Cluster tokens into text rows by vertical position and return the
        plate row. The registration number is the tallest, highest-confidence
        row; the date line below ("11-24") is smaller and gets dropped."""
        tokens = sorted(tokens, key=lambda t: t["yc"])
        median_h = float(np.median([t["h"] for t in tokens])) or 1.0

        rows = []
        for t in tokens:
            placed = False
            for r in rows:
                if abs(t["yc"] - r["yc"]) <= 0.6 * median_h:
                    r["items"].append(t)
                    r["yc"] = float(np.mean([i["yc"] for i in r["items"]]))
                    placed = True
                    break
            if not placed:
                rows.append({"yc": t["yc"], "items": [t]})

        def score(r):
            joined = "".join(i["text"] for i in r["items"])
            chars = len(joined)
            avg_h = float(np.mean([i["h"] for i in r["items"]]))
            avg_p = float(np.mean([i["prob"] for i in r["items"]]))
            has_alpha = any(c.isalpha() for c in joined)
            has_digit = any(c.isdigit() for c in joined)
            # Strongly favour rows that look like a registration (letters AND
            # digits) over a pure-digit date line; then more chars, taller text,
            # higher confidence.
            bonus = 5.0 if (has_alpha and has_digit) else 0.0
            return bonus + chars * 1.0 + avg_h * 0.05 + avg_p * 2.0

        best = max(rows, key=score)
        return best["items"]

    @staticmethod
    def _preprocess(plate_crop):
        """Upscale + grayscale a plate crop for OCR. Upscaling is the dominant
        accuracy factor; grayscale reads as well as colour and is faster."""
        w = plate_crop.shape[1]
        scale = max(1.0, min(8.0, 320 / max(w, 1)))
        if scale != 1.0:
            plate_crop = cv2.resize(
                plate_crop, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        return gray

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _safe_crop(frame, bbox):
        if frame is None or bbox is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None
