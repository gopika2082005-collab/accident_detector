import cv2
import time
from model import load_model
from detector import detect_accident, calculate_speed
from utils import play_alarm, save_frame, send_email

model = load_model()
cap = cv2.VideoCapture(0)

prev_positions = {}
speed_history = {}
overlap_history = {}
last_email_time = 0
EMAIL_COOLDOWN = 60 # wait 60 seconds before sending another email

# COCO Classes: 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    # Use track() for ID assignment
    results = model.track(frame, persist=True, verbose=False)[0]

    tracks = []
    
    if results.boxes.id is not None:
        # Zip bounding boxes, class ids, and tracking ids together
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        
        for box, cls, track_id in zip(boxes, class_ids, track_ids):
            if cls in TARGET_CLASSES:
                x1, y1, x2, y2 = box
                center = ((x1+x2)//2, (y1+y2)//2)

                prev = prev_positions.get(track_id, center)
                speed = calculate_speed(prev, center)

                prev_positions[track_id] = center
                
                # Store the last 10 speed readings for sudden stop detection
                if track_id not in speed_history:
                    speed_history[track_id] = []
                speed_history[track_id].append(speed)
                if len(speed_history[track_id]) > 10:
                    speed_history[track_id].pop(0)

                tracks.append({
                    "box": (x1, y1, x2, y2), 
                    "speed": speed, 
                    "id": track_id, 
                    "cls": cls,
                    "speed_history": speed_history[track_id]
                })

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                # optionally draw id and speed
                cv2.putText(frame, f"ID:{track_id} Spd:{int(speed)}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    accident = detect_accident(tracks, overlap_history)

    if accident:
        cv2.putText(frame, "ACCIDENT DETECTED!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        current_time = time.time()
        if current_time - last_email_time > EMAIL_COOLDOWN:
            print("Triggering accident workflow...")
            img = save_frame(frame)
            play_alarm()
            send_email(img) # Now runs in a background thread
            
            with open("log.txt", "a") as f:
                f.write(f"Accident detected at {current_time}: {img}\n")
                
            last_email_time = current_time

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
