"""
Shared backend for the TrafficVision UI.

Holds everything the individual pages need in common: the violations database,
the Mumbai location table, the per-run duplicate guard, and the video-processing
pipeline. Pages import from here so the UI files stay thin.
"""

import os
import cv2
import tempfile
import pandas as pd
from datetime import datetime

from modules.detection import DetectionEngine
from modules.helmet_detection import HelmetDetector
from modules.triple_riding import (
    get_motorcycle_associations,
    classify_associations,
    check_triple_riding,
    draw_triple_riding,
    draw_normal_motorcycles,
    reset_cooldown,
    _dedupe_persons,
)
from modules.signal_jump import check_signal_jump
from modules.plate_detection import PlateDetector
from modules.multi_frame_ocr import MultiFrameOCR
from modules.heatmap import MUMBAI_LOCATIONS

CSV_PATH = "database/violations.csv"

CSV_COLUMNS = [
    "Vehicle Number", "Violation Type", "Timestamp",
    "Location", "Time Slot", "Riders Count",
]

VIOLATION_TYPES = ["Triple Riding", "No Helmet", "Signal Jumping", "Plate Detection"]

# Ordered list of the predefined Mumbai areas the heatmap understands.
MUMBAI_AREAS = list(MUMBAI_LOCATIONS.keys())


# ─────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)
        return
    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)
        return
    changed = False
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            changed = True
    if changed:
        df.to_csv(CSV_PATH, index=False)


def load_violations() -> pd.DataFrame:
    init_db()
    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=CSV_COLUMNS)
    return df


def clear_violations():
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)


def time_slot(dt: datetime) -> str:
    h = dt.hour
    return f"{h:02d}:00-{(h+1)%24:02d}:00"


# ─────────────────────────────────────────────────────────────
# Logging with per-run duplicate guard
# ─────────────────────────────────────────────────────────────
# Keyed by (violation_type, vehicle track_id) for the CURRENT run so the same
# motorcycle does not produce a new row on every frame. Cleared at the start of
# each video via reset_run_state(). track_id == -1 (untracked) is allowed
# through once per call so plate reads / signal jumps without an id still log.

_seen_this_run: set = set()


def reset_run_state():
    _seen_this_run.clear()
    reset_cooldown()


def log_violation(v_type, v_num="UNKNOWN", vehicle_id=-1,
                  location="Andheri", riders_count=0):
    """Append one violation row. Returns the new row's index, or None if this
    (type, vehicle) was already logged this run."""
    if vehicle_id is not None and vehicle_id >= 0:
        key = (v_type, int(vehicle_id))
        if key in _seen_this_run:
            return None
        _seen_this_run.add(key)

    now = datetime.now()
    row = {
        "Vehicle Number": v_num,
        "Violation Type": v_type,
        "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Location": location,
        "Time Slot": time_slot(now),
        "Riders Count": riders_count,
    }

    df = load_violations()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    return len(df) - 1


def update_vehicle_number(row_index, plate):
    """Fill in the number plate for an already-logged violation row."""
    if not plate:
        return
    df = load_violations()
    if 0 <= row_index < len(df):
        df.at[row_index, "Vehicle Number"] = plate
        df.to_csv(CSV_PATH, index=False)


# ─────────────────────────────────────────────────────────────
# Models (cached once)
# ─────────────────────────────────────────────────────────────

_models = None


def load_models():
    global _models
    if _models is None:
        _models = (DetectionEngine(), HelmetDetector(),
                   PlateDetector(), MultiFrameOCR())
    return _models


# ─────────────────────────────────────────────────────────────
# Video processing pipeline
# ─────────────────────────────────────────────────────────────

def save_upload_to_tempfile(uploaded_file) -> str:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    tfile.close()
    return tfile.name


def process_video(video_path, video_placeholder, on_counts,
                  detect_triple, detect_helmet, detect_signal,
                  detect_plate, location):
    """
    Run detection over the video, drawing annotated frames into
    `video_placeholder` and reporting running counts via on_counts(dict).
    """
    detector, helmet_det, plate_detector, ocr_history = load_models()
    reset_run_state()
    ocr_history.plate_history.clear()

    cap = cv2.VideoCapture(video_path)
    counts = {"Triple Riding": 0, "No Helmet": 0,
              "Signal Jumping": 0, "Plate Detection": 0}
    frame_count = 0

    last_detections = None
    helmet_cache = []
    triple_cache = []
    detection_skip_frames = 8

    # Plate jobs: for each logged violation row we keep reading the vehicle's
    # number plate over the next few frames and back-fill that single row.
    # row_index -> {"vid", "kind" ("moto"/"person"), "attempts", "done"}
    plate_jobs = {}
    MAX_PLATE_ATTEMPTS = 10

    def register_plate_job(row_index, vid, kind):
        if row_index is not None and vid is not None and vid >= 0:
            plate_jobs[row_index] = {"vid": vid, "kind": kind,
                                     "attempts": 0, "done": False}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        detect_now = frame_count % detection_skip_frames == 0
        if detect_now:
            frame, detections = detector.process_frame(frame)
            last_detections = detections
        else:
            detections = last_detections if last_detections else []

        associations = get_motorcycle_associations(detections)

        # Clean (un-annotated) copy for OCR — boxes drawn below would otherwise
        # sit on top of the plate and corrupt the read.
        clean_frame = frame.copy() if detect_now else None

        # rider person track_id -> motorcycle dict (for plate lookup by bike)
        person_moto = {}
        for a in associations:
            mc_d = a["motorcycle"]
            for r in a.get("riders", []):
                rtid = r.get("track_id", -1)
                if rtid >= 0:
                    person_moto[rtid] = mc_d

        def do_plate_read(row_index, bbox):
            """Read the plate from the clean frame and back-fill the row."""
            if clean_frame is None or bbox is None:
                return
            text = plate_detector.read_vehicle_plate(clean_frame, bbox)
            if text:
                ocr_history.add_reading(row_index, text, 0.6)
                best = ocr_history.get_best_result(row_index)
                if best:
                    update_vehicle_number(row_index, best)

        # ── Triple Riding ───────────────────────────────────────
        if detect_triple:
            if detect_now:
                triple_cache = classify_associations(associations)
                for v in check_triple_riding(detections, frame_number=frame_count):
                    ridx = log_violation("Triple Riding",
                                         vehicle_id=v["vehicle_id"],
                                         location=location,
                                         riders_count=v["riders_count"])
                    if ridx is not None:
                        counts["Triple Riding"] += 1
                        register_plate_job(ridx, v["vehicle_id"], "moto")
                        do_plate_read(ridx, v["bbox"])  # immediate, vehicle in frame
            for kind, payload in triple_cache:
                if kind == "triple":
                    frame = draw_triple_riding(frame, [payload])
                else:
                    frame = draw_normal_motorcycles(frame, [payload], set())

        # ── Helmet ──────────────────────────────────────────────
        # Run on EVERY detected person, not only riders matched to a bike.
        # Bikes are frequently missed (rear views, occlusion), and gating the
        # helmet check behind motorcycle association meant whole videos got no
        # helmet detection at all. Dedupe persons so one rider isn't checked
        # (and flagged) several times from overlapping boxes.
        if detect_helmet:
            # Run helmet only on EVEN detection cycles (cache redraws between),
            # and only on the largest riders, to bound cost on busy frames.
            helmet_now = detect_now and (frame_count // detection_skip_frames) % 2 == 0
            if helmet_now:
                helmet_cache = []
                persons = _dedupe_persons(
                    [d for d in detections if d["class"] == "person"])
                persons = sorted(
                    persons, key=lambda p: p["bbox"][3] - p["bbox"][1],
                    reverse=True)[:6]
                for person in persons:
                    px1, py1, px2, py2 = map(int, person["bbox"])
                    if (py2 - py1) < 90:
                        continue
                    has_helmet = helmet_det.check_helmet(
                        frame, (px1, py1, px2, py2))
                    if has_helmet is None:
                        continue
                    bw, bh = px2 - px1, py2 - py1
                    cx = (px1 + px2) // 2
                    half_w = max(int(bw * 0.25), 10)
                    helmet_cache.append(
                        (cx - half_w, py1, cx + half_w,
                         py1 + int(bh * 0.28), has_helmet))
                    if not has_helmet:
                        pid = person.get("track_id", -1)
                        ridx = log_violation("No Helmet",
                                             vehicle_id=pid,
                                             location=location)
                        if ridx is not None:
                            counts["No Helmet"] += 1
                            # Prefer the rider's motorcycle (plate lives there).
                            mc_d = person_moto.get(pid)
                            if mc_d is not None and mc_d.get("track_id", -1) >= 0:
                                register_plate_job(ridx, mc_d["track_id"], "moto")
                                do_plate_read(ridx, mc_d["bbox"])
                            else:
                                register_plate_job(ridx, pid, "person")
                                ex = (px1, py1, px2, int(py2 + (py2 - py1) * 0.45))
                                do_plate_read(ridx, ex)
            for hx1, hy1, hx2, hy2, has_helmet in helmet_cache:
                color = (0, 200, 0) if has_helmet else (0, 0, 255)
                label = "Helmet" if has_helmet else "No Helmet"
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ly = max(hy1, th + 8)
                cv2.rectangle(frame, (hx1, ly - th - 8),
                              (hx1 + tw + 8, ly), color, -1)
                cv2.putText(frame, label, (hx1 + 4, ly - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Signal Jump ─────────────────────────────────────────
        if detect_signal:
            h_f, w_f = frame.shape[:2]
            red_line_y = int(h_f * 0.7)
            cv2.line(frame, (0, red_line_y), (w_f, red_line_y), (0, 0, 255), 2)
            if detect_now:
                for v in check_signal_jump(detections, red_line_y,
                                           is_red_signal=True):
                    vid = v.get("vehicle_id", -1)
                    ridx = log_violation("Signal Jumping",
                                         vehicle_id=vid,
                                         location=location)
                    if ridx is not None:
                        counts["Signal Jumping"] += 1
                        register_plate_job(ridx, vid, "moto")

        # ── Plate Detection ─────────────────────────────────────
        if detect_plate and frame_count % 100 == 0:
            for plate in plate_detector.detect_plates_in_frame(frame):
                px1, py1, px2, py2 = plate["bbox"]
                plate_text = plate["plate_text"]
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 255), 2)
                if plate_text:
                    cv2.putText(frame, f"Plate: {plate_text}", (px1, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    if log_violation("Plate Detection", v_num=plate_text,
                                     location=location) is not None:
                        counts["Plate Detection"] += 1

        # ── Plate OCR for logged violations ─────────────────────
        # For each violation we recorded, keep reading that vehicle's plate over
        # the next few detection frames and back-fill the SAME row (one entry
        # per violation, progressively filled). EasyOCR is slow, so attempts are
        # capped and we stop once a solid plate is read.
        # Plate OCR refinement: keep improving plates for recent violations.
        # One read per detection frame keeps cost bounded; the immediate read at
        # violation time already gives each violation a first plate.
        if detect_now and clean_frame is not None and plate_jobs:
            moto_by_id, person_by_id = {}, {}
            for d in detections:
                tid = d.get("track_id", -1)
                if tid < 0:
                    continue
                if d["class"] == "motorcycle":
                    moto_by_id[tid] = d["bbox"]
                elif d["class"] == "person":
                    person_by_id[tid] = d["bbox"]

            # OCR is expensive (~0.5 s), so do at most ONE plate read per
            # detection frame, picking the active job with the fewest attempts.
            # This spreads plate reading over time instead of freezing a frame.
            pending = [(ridx, job) for ridx, job in plate_jobs.items()
                       if not job["done"] and job["attempts"] < MAX_PLATE_ATTEMPTS]
            pending.sort(key=lambda x: x[1]["attempts"])
            for ridx, job in pending:
                if job["kind"] == "moto":
                    bbox = moto_by_id.get(job["vid"])
                else:
                    bbox = person_by_id.get(job["vid"])
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        bbox = (x1, y1, x2, int(y2 + (y2 - y1) * 0.45))
                if bbox is None:
                    continue
                job["attempts"] += 1
                text = plate_detector.read_vehicle_plate(clean_frame, bbox)
                if text:
                    ocr_history.add_reading(ridx, text, 0.6)
                    best = ocr_history.get_best_result(ridx)
                    if best:
                        update_vehicle_number(ridx, best)
                        if len(best) >= 6:
                            job["done"] = True
                break  # only one OCR call this frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB",
                                width="stretch")
        if on_counts is not None:
            on_counts(counts)

    cap.release()
    return counts


def process_image(image_bgr, detect_triple, detect_helmet, detect_signal,
                  detect_plate, location):
    """
    Run detection on a SINGLE image. Returns (annotated_rgb, results) where
    results is a list of {violation, plate, riders, timestamp} dicts. Each
    violation is logged once with its plate (same DB as the video pipeline).
    """
    detector, helmet_det, plate_detector, ocr_history = load_models()
    reset_run_state()
    ocr_history.plate_history.clear()

    frame = image_bgr.copy()
    clean = image_bgr.copy()
    # persist=False -> treat the image independently (no tracker carry-over)
    frame, detections = detector.process_frame(frame, persist=False)
    associations = get_motorcycle_associations(detections)

    person_moto = {}
    for a in associations:
        mc_d = a["motorcycle"]
        for r in a.get("riders", []):
            rtid = r.get("track_id", -1)
            if rtid >= 0:
                person_moto[rtid] = mc_d

    results = []

    def plate_for(bbox):
        return plate_detector.read_vehicle_plate(clean, bbox) or "UNKNOWN"

    # ── Triple Riding ──
    if detect_triple:
        for kind, payload in classify_associations(associations):
            if kind == "triple":
                frame = draw_triple_riding(frame, [payload])
            else:
                frame = draw_normal_motorcycles(frame, [payload], set())
        for v in check_triple_riding(detections, frame_number=0):
            ridx = log_violation("Triple Riding", vehicle_id=v["vehicle_id"],
                                 location=location, riders_count=v["riders_count"])
            if ridx is not None:
                plate = plate_for(v["bbox"])
                if plate != "UNKNOWN":
                    update_vehicle_number(ridx, plate)
                results.append({"violation": "Triple Riding", "plate": plate,
                                "riders": v["riders_count"]})

    # ── Helmet ──
    if detect_helmet:
        persons = _dedupe_persons(
            [d for d in detections if d["class"] == "person"])
        for person in persons:
            px1, py1, px2, py2 = map(int, person["bbox"])
            if (py2 - py1) < 90:
                continue
            has_helmet = helmet_det.check_helmet(clean, (px1, py1, px2, py2))
            if has_helmet is None:
                continue
            bw, bh = px2 - px1, py2 - py1
            cx = (px1 + px2) // 2
            half_w = max(int(bw * 0.25), 10)
            hx1, hy1, hx2, hy2 = cx - half_w, py1, cx + half_w, py1 + int(bh * 0.28)
            color = (0, 200, 0) if has_helmet else (0, 0, 255)
            label = "Helmet" if has_helmet else "No Helmet"
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3)
            cv2.putText(frame, label, (hx1, max(hy1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if not has_helmet:
                pid = person.get("track_id", -1)
                ridx = log_violation("No Helmet", vehicle_id=pid, location=location)
                if ridx is not None:
                    mc_d = person_moto.get(pid)
                    bbox = mc_d["bbox"] if mc_d else (
                        px1, py1, px2, int(py2 + bh * 0.45))
                    plate = plate_for(bbox)
                    if plate != "UNKNOWN":
                        update_vehicle_number(ridx, plate)
                    results.append({"violation": "No Helmet", "plate": plate,
                                    "riders": None})

    # ── Signal Jump ──
    if detect_signal:
        h_f, w_f = frame.shape[:2]
        red_line_y = int(h_f * 0.7)
        cv2.line(frame, (0, red_line_y), (w_f, red_line_y), (0, 0, 255), 2)
        for v in check_signal_jump(detections, red_line_y, is_red_signal=True):
            vid = v.get("vehicle_id", -1)
            ridx = log_violation("Signal Jumping", vehicle_id=vid, location=location)
            if ridx is not None:
                bbox = next((d["bbox"] for d in detections
                             if d.get("track_id", -1) == vid
                             and d["class"] == "motorcycle"), None)
                plate = plate_for(bbox) if bbox else "UNKNOWN"
                if plate != "UNKNOWN":
                    update_vehicle_number(ridx, plate)
                results.append({"violation": "Signal Jumping", "plate": plate,
                                "riders": None})

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), results
