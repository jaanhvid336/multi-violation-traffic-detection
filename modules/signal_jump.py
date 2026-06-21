crossed_vehicles = set()

def check_signal_jump(detections, red_line_y, is_red_signal=True):
    """
    Checks if a vehicle has crossed the red line during a red signal.
    red_line_y is the y-coordinate of the stop line.
    is_red_signal is a boolean indicating if the traffic light is currently red.
    """
    global crossed_vehicles
    violations = []
    
    if not is_red_signal:
        return violations
        
    for d in detections:
        if d['class'] in ['car', 'motorcycle', 'truck', 'bus']:
            x1, y1, x2, y2 = d['bbox']
            track_id = d.get("track_id", -1)
            
            # If the bottom of the bounding box (y2) has crossed the red line (y-coordinate is greater than red_line_y)
            # A more robust check might track previous positions to ensure it *just* crossed,
            if y2 > red_line_y and y2 < red_line_y + 100:
                if track_id != -1 and track_id not in crossed_vehicles:
                    crossed_vehicles.add(track_id)
                    violations.append({
                        "vehicle_id": track_id,
                        "bbox": d['bbox'],
                        "class": d['class']
                    })
                elif track_id == -1 and (y2 > red_line_y and y2 < red_line_y + 10):
                    # For untracked vehicles, use a very narrow margin to avoid multiple counts
                    violations.append({
                        "vehicle_id": -1,
                        "bbox": d['bbox'],
                        "class": d['class']
                    })
                
    return violations
