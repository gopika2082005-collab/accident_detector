import numpy as np

def calculate_speed(prev, curr):
    return np.linalg.norm(np.array(curr) - np.array(prev))

def detect_collision(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    # Simple bounding box overlap check
    return (x1 < x2b and x2 > x1b and y1 < y2b and y2 > y1b)

def detect_accident(tracks, overlap_history):
    # overlap_history parameter is kept just so we don't break main.py/app.py,
    # but we won't use it anymore to ensure INSTANT, flawless detection triggers!
    
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            
            # If ANY tracked combination touches AT ALL:
            if detect_collision(tracks[i]['box'], tracks[j]['box']):
                return True

    return False

