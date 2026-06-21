"""
Triple Riding Detection Module
===============================
Pipeline:
  Video → Motorcycle Detection → Person Detection
       → Associate Persons with Motorcycle (center-point method)
       → Count Riders
       → If riders >= 3 → Triple Riding Violation
       → If riders <= 2 → Normal

Association strategy (accurate):
  A person is considered ON a motorcycle if:
    1. The person's bottom-center point falls inside the motorcycle
       bbox (expanded by a small horizontal pad only), AND
    2. The person bbox vertically overlaps the motorcycle bbox by > 10%
       of the person's height.
  This avoids picking up pedestrians beside the bike or cargo objects.
"""

import cv2
from datetime import datetime

# ─────────────────────────────────────────────
# Cooldown: avoid re-flagging the same motorcycle
# on every consecutive frame.
# ─────────────────────────────────────────────
_violation_cooldown: dict = {}
COOLDOWN_FRAMES = 15  # ~1.5 s at 30 fps

# Sticky triple-riding state keyed by motorcycle track_id. The per-frame rider
# count jitters (a rear rider drops out for a frame, count 3->2->3), which makes
# the box flip TRIPLE RIDING (red) -> Normal (green) -> red. Once a tracked bike
# is confirmed with 3+ riders we latch it as a violation so the label stays put.
_triple_sticky: dict = {}
STICKY_RIDER_COUNT = 3


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def _is_rider_on_motorcycle(mc_box, person_box):
    """
    A person is a RIDER if they sit on/over the motorcycle.

    Key geometry: riders sit ON TOP of the bike, so the person bbox starts
    well ABOVE the motorcycle box (head/torso) and reaches down into it. Its
    bottom edge is often near the seat — and when the legs are occluded by the
    bike or the rider in front, the bottom edge sits ABOVE the bike's top edge.
    The old "feet must be inside the bike box" rule rejected exactly these
    stacked riders, so a 3-rider bike read as "Normal (1/2)".

    New rule:
      1. Horizontal: the person overlaps the bike's width substantially (rejects
         pedestrians standing beside the bike).
      2. Vertical: the person's bottom edge lands in a band that spans from a
         bit above the seat (occluded-leg rider) to a bit below the bike
         (dangling feet). Background people further up the road fall above this
         band; people fully below it are not riders.
    """
    mx1, my1, mx2, my2 = mc_box
    px1, py1, px2, py2 = person_box

    mc_w = mx2 - mx1
    mc_h = my2 - my1
    p_w  = px2 - px1

    if mc_w <= 0 or mc_h <= 0 or p_w <= 0:
        return False

    # 1. Horizontal: person centre over the bike width (small pad) AND a
    #    meaningful share of the person's width sits over the bike. This is the
    #    main guard against pedestrians beside the bike / in adjacent lanes.
    person_cx = (px1 + px2) / 2
    h_pad = max(int(mc_w * 0.20), 20)
    if person_cx < (mx1 - h_pad) or person_cx > (mx2 + h_pad):
        return False

    x_overlap = min(px2, mx2) - max(px1, mx1)
    if x_overlap < 0.30 * p_w:
        return False

    # 2. Vertical band for the person's bottom edge. Riders straddle the bike,
    #    so allow the box to end up to ~0.6 bike-heights above the seat
    #    (occluded legs) and up to ~0.5 below the bike (dangling feet).
    lo = my1 - 0.6 * mc_h
    hi = my2 + 0.5 * mc_h
    if py2 < lo or py2 > hi:
        return False

    return True

def _get_time_slot(dt: datetime) -> str:
    h = dt.hour
    return f"{h:02d}:00-{(h+1)%24:02d}:00"


def _bbox_centre(box) -> str:
    x1, y1, x2, y2 = box
    return f"{(x1+x2)//2},{(y1+y2)//2}"


# ─────────────────────────────────────────────────────
# Core association (shared by triple-riding + helmet)
# ─────────────────────────────────────────────────────

def _centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(a, b):
    """Fraction of the SMALLER box that lies inside the other box."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _dedupe_persons(persons, iou_thresh=0.55):
    """
    Drop duplicate person boxes for the SAME human.

    The detector runs at a high NMS IoU (0.70) so that two real riders whose
    boxes overlap heavily both survive. The side effect is that a single person
    can also yield several near-identical boxes. Without removing them, one rider
    inflates into "3+ riders" and a lone scooter reads as TRIPLE RIDING.

    Keep the highest-confidence box in each cluster of mutually-overlapping
    boxes; discard the rest. This only merges boxes that overlap a LOT (same
    person), so genuinely distinct riders (lower mutual IoU) are preserved.
    """
    ordered = sorted(persons, key=lambda p: p.get("confidence", 0.0), reverse=True)
    kept = []
    for p in ordered:
        dup = any(
            _iou(p["bbox"], k["bbox"]) >= iou_thresh
            or _containment(p["bbox"], k["bbox"]) >= 0.65
            for k in kept
        )
        if not dup:
            kept.append(p)
    return kept


def _dedupe_motorcycles(motorcycles, iou_thresh=0.55):
    """
    Drop duplicate motorcycle boxes for the SAME bike.

    Two overlapping bike detections produce two labels on one physical bike,
    and they often split the riders between them -> one renders red
    (TRIPLE RIDING) while its twin renders green (Normal) on the very same
    motorcycle. Keep the highest-confidence box per overlapping cluster.
    """
    ordered = sorted(motorcycles, key=lambda m: m.get("confidence", 0.0), reverse=True)
    kept = []
    for m in ordered:
        dup = any(
            _iou(m["bbox"], k["bbox"]) >= iou_thresh
            or _containment(m["bbox"], k["bbox"]) >= 0.70
            for k in kept
        )
        if not dup:
            kept.append(m)
    return kept


def get_motorcycle_associations(detections):
    """
    Associate each person with ONLY ONE motorcycle: the NEAREST bike that the
    person qualifies as a rider for.

    Previously each bike grabbed its riders in turn, so with two adjacent bikes
    the first one could swallow a person who actually belonged to the second
    (detection order, not proximity). That split a 3-rider bike's riders across
    neighbours and produced "Normal (1/2)" on a clearly overloaded bike. Here we
    score every (person, bike) pair and let each person pick their closest bike.
    """

    motorcycles = _dedupe_motorcycles(
        [d for d in detections if d["class"] == "motorcycle"])
    persons = _dedupe_persons([d for d in detections if d["class"] == "person"])

    riders_by_mc = {id(mc): [] for mc in motorcycles}

    for p in persons:
        pcx, pcy = _centre(p["bbox"])

        best_mc = None
        best_dist = None

        for mc in motorcycles:
            if not _is_rider_on_motorcycle(mc["bbox"], p["bbox"]):
                continue

            mcx, mcy = _centre(mc["bbox"])
            dist = (pcx - mcx) ** 2 + (pcy - mcy) ** 2

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_mc = mc

        if best_mc is not None:
            riders_by_mc[id(best_mc)].append(p)

    return [
        {"motorcycle": mc, "riders": riders_by_mc[id(mc)]}
        for mc in motorcycles
    ]


def classify_associations(associations: list) -> list:
    """
    Decide, per bike, whether to render TRIPLE RIDING or Normal — with sticky
    state so the label doesn't flip when the rider count jitters frame-to-frame.

    A tracked bike (track_id >= 0) that has EVER been seen with 3+ riders stays
    classified as "triple" while it remains on screen. Returns a list of
    (kind, payload) tuples ready for drawing:
        kind == "triple": payload = {bbox, riders_count, rider_bboxes}
        kind == "normal": payload = the association dict
    """
    out = []
    for assoc in associations:
        mc = assoc["motorcycle"]
        riders = assoc.get("riders", [])
        count = len(riders)
        track_id = mc.get("track_id", -1)

        latched = track_id >= 0 and _triple_sticky.get(track_id, False)

        if count == 0 and not latched:
            # empty ghost box, never a confirmed triple -> don't draw
            continue

        is_triple = count >= STICKY_RIDER_COUNT or latched
        if is_triple:
            if track_id >= 0:
                _triple_sticky[track_id] = True
            shown_count = count if count >= STICKY_RIDER_COUNT else STICKY_RIDER_COUNT
            out.append(("triple", {
                "bbox": mc["bbox"],
                "riders_count": shown_count,
                "rider_bboxes": [r["bbox"] for r in riders],
            }))
        else:
            out.append(("normal", assoc))
    return out


# ─────────────────────────────────────────────────────
# Triple-riding violation check
# ─────────────────────────────────────────────────────

def check_triple_riding(detections: list, frame_number: int = 0) -> list:
    """
    Detect triple-riding violations.

    Returns list of violation dicts:
        {
            vehicle_id, riders_count, bbox, confidence,
            timestamp, frame_number, location, time_slot,
            rider_bboxes
        }
    """
    associations = get_motorcycle_associations(detections)
    violations   = []
    now          = datetime.now()

    for assoc in associations:
        mc       = assoc["motorcycle"]
        riders   = assoc["riders"]
        mc_box   = mc["bbox"]
        track_id = mc.get("track_id", -1)
        mc_conf  = mc.get("confidence", 0.0)

        # Cooldown check
        last_frame = _violation_cooldown.get(track_id, -9999)
        if frame_number - last_frame < COOLDOWN_FRAMES:
            continue

        if len(riders) >= 3:
            _violation_cooldown[track_id] = frame_number
            all_confs = [mc_conf] + [p.get("confidence", 0.5) for p in riders]
            avg_conf  = round(sum(all_confs) / len(all_confs), 4)

            violations.append({
                "vehicle_id"   : track_id,
                "riders_count" : len(riders),
                "bbox"         : mc_box,
                "confidence"   : avg_conf,
                "timestamp"    : now.strftime("%Y-%m-%d %H:%M:%S"),
                "frame_number" : frame_number,
                "location"     : _bbox_centre(mc_box),
                "time_slot"    : _get_time_slot(now),
                "rider_bboxes" : [p["bbox"] for p in riders],
            })

    return violations


# ─────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────

def draw_triple_riding(frame, violations: list):
    """
    Red box on motorcycle + orange boxes on each associated rider.
    """
    for v in violations:
        x1, y1, x2, y2 = v["bbox"]
        label = f"TRIPLE RIDING ({v['riders_count']} riders)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        for rb in v.get("rider_bboxes", []):
            rx1, ry1, rx2, ry2 = rb
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2)
            cv2.putText(frame, "Rider", (rx1, ry1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

    return frame


def draw_normal_motorcycles(frame, associations: list, suppress_ids: set):
    """
    Draw green 'Normal (N)' on motorcycles not in violation/cooldown.
    Takes the pre-computed associations list (motorcycle+riders) so we
    can show the correct rider count even for normal bikes.
    """
    for assoc in associations:
        mc  = assoc["motorcycle"]
        tid = mc.get("track_id", -1)
        if tid in suppress_ids:
            continue

        x1, y1, x2, y2 = mc["bbox"]
        rider_count = len(assoc["riders"])
        # CHANGED: Use dynamic rider_count instead of hardcoded "(1)"
        label = f"Normal ({rider_count})" 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

    return frame


def reset_cooldown():
    """Clear cooldown + sticky state (call between video files)."""
    _violation_cooldown.clear()
    _triple_sticky.clear()


def get_cooldown_vehicle_ids() -> set:
    """Return track_ids currently in violation cooldown."""
    return set(_violation_cooldown.keys())
