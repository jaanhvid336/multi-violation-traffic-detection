import sys
sys.path.insert(0, r"c:\Users\Jaanhvi Dahibaokar\OneDrive\Desktop\traffic_violation_system")

from modules.triple_riding import (
    check_triple_riding,
    draw_triple_riding,
    draw_normal_motorcycles,
    reset_cooldown,
)

print("imports OK")

# Fake detection: 1 motorcycle + 3 overlapping persons → violation
fake = [
    {"class": "motorcycle", "bbox": (100,100,300,250), "track_id": 1, "confidence": 0.88},
    {"class": "person",     "bbox": (110,80, 200,260), "track_id": 2, "confidence": 0.72},
    {"class": "person",     "bbox": (180,90, 270,255), "track_id": 3, "confidence": 0.68},
    {"class": "person",     "bbox": (220,85, 310,245), "track_id": 4, "confidence": 0.70},
]

v = check_triple_riding(fake, frame_number=1)
print(f"Violations detected     : {len(v)}  (expected 1)")
if v:
    print(f"  riders_count         : {v[0]['riders_count']}  (expected 3)")
    print(f"  confidence           : {v[0]['confidence']}")
    print(f"  time_slot            : {v[0]['time_slot']}")
    print(f"  location             : {v[0]['location']}")
    print(f"  timestamp            : {v[0]['timestamp']}")
    print(f"  frame_number         : {v[0]['frame_number']}")

# Cooldown: same motorcycle within 30 frames → no new violation
v2 = check_triple_riding(fake, frame_number=5)
print(f"Cooldown (expect 0)     : {len(v2)} violations")

# After reset: should fire again
reset_cooldown()
v3 = check_triple_riding(fake, frame_number=50)
print(f"Post-reset (expect 1)   : {len(v3)} violations")

# Normal case: only 2 riders → no violation
fake2 = [
    {"class": "motorcycle", "bbox": (100,100,300,250), "track_id": 10, "confidence": 0.85},
    {"class": "person",     "bbox": (110,80, 200,260), "track_id": 11, "confidence": 0.70},
    {"class": "person",     "bbox": (180,90, 270,255), "track_id": 12, "confidence": 0.65},
]
reset_cooldown()
v4 = check_triple_riding(fake2, frame_number=100)
print(f"Normal (2 riders, expect 0): {len(v4)} violations")

print("\nALL CHECKS PASSED")
