import torch
import streamlit as st
import cv2
import time
import tempfile
import os
from model import load_model
from detector import detect_accident, calculate_speed
from utils import play_alarm, save_frame, send_email

st.set_page_config(page_title="Live AI Accident Detector", layout="wide")
st.title("🚨 Live CCTV AI Accident Detection")
st.markdown("Real-time traffic anomaly and collision monitoring system.")

# COCO Classes: 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
EMAIL_COOLDOWN = 60

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Camera Controls")
source_option = st.sidebar.radio("Select Input Source:", ("Live CCTV Camera", "Upload Video Test"))

cap = None

if source_option == "Upload Video Test":
    st.sidebar.markdown("Upload a video to simulate traffic.")
    video = st.file_uploader("Upload Traffic Video", type=["mp4", "mov", "avi"])
    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)

elif source_option == "Live CCTV Camera":
    st.sidebar.markdown('Click "Start Feed" to begin real-time CCTV analysis from your local camera.')
    start_button = st.sidebar.button("🔴 Start Live Feed")
    stop_button = st.sidebar.button("⬛ Stop Feed")
    
    if start_button:
        st.session_state['run_camera'] = True
    if stop_button:
        st.session_state['run_camera'] = False
        
    if st.session_state.get('run_camera', False):
        cap = cv2.VideoCapture(0)

# --- TRACKING LOOP ---
if cap is not None and cap.isOpened():
    col1, col2 = st.columns([2, 1])
    with col1:
        stframe = st.empty()
    with col2:
        st.subheader("System Event Logs")
        log_container = st.empty()
        
    logs = []
    
    def log_message(msg):
        logs.insert(0, msg)
        log_container.code("\n".join(logs[:10]))

    log_message("Camera feed engaged. Starting AI inference...")

    prev_positions = {}
    speed_history = {}
    overlap_history = {}
    last_email_time = 0

    while cap.isOpened():
        # Stop button logic
        if source_option == "Live CCTV Camera" and not st.session_state.get('run_camera', True):
            log_message("Camera disconnected by user.")
            break
            
        ret, frame = cap.read()
        if not ret:
            if source_option == "Upload Video Test":
                log_message("End of video stream.")
            break

        frame = cv2.resize(frame, (640, 480))
        
        # Use track() for ID assignment
        results = model.track(frame, persist=True, verbose=False)[0]

        tracks = []
        
        if results.boxes.id is not None:
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
                    
                    # Store the last speed readings for sudden stop detection
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

                    # Draw rect
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id} Spd:{int(speed)}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
        # LIVE DEBUG: Print how many objects the model actually sees on the screen!
        detected_classes = [t['cls'] for t in tracks]
        cv2.putText(frame, f"Objects Seen: {len(tracks)} | Classes: {detected_classes}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        accident = detect_accident(tracks, overlap_history)

        if accident:
            cv2.putText(frame, "ACCIDENT DETECTED!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            current_time = time.time()
            if current_time - last_email_time > EMAIL_COOLDOWN:
                log_message(f"[{time.strftime('%H:%M:%S')}] ⚠️ CRASH DETECTED!")
                log_message("Triggering Alert Workflow, saving evidence...")
                img = save_frame(frame)
                play_alarm()
                send_email(img)
                last_email_time = current_time
                log_message(f"📧 Sent details to police/hospital!")

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()
    if source_option == "Upload Video Test":
        try:
            os.unlink(tfile.name)
        except:
            pass

