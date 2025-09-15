import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from detection_1 import initialize_models, detect_accidents
from notification_system_2 import send_alert2, camera_locations

if not os.path.exists("detected_frames"):
    os.makedirs("detected_frames")

accident_logs = []
last_notification_time = None
cooldown_period = timedelta(minutes=5)

st.set_page_config(page_title="Accident Detection System", layout="wide")
st.title("🚦 Accident Detection System")
st.sidebar.title("Options")

# Sidebar options
option = st.sidebar.selectbox("Choose an action", ["Upload a Video", "Use Webcam"])
selected_camera = st.sidebar.selectbox("Select Camera", list(camera_locations.keys()))
camera_coords = camera_locations[selected_camera]
frame_id, skip_rate = 0, 5
log_placeholder = st.sidebar.empty()

def process_frame(frame):
    # Detect accidents returns a list of (x1, y1, x2, y2, label, score)
    detections = detect_accidents(frame)
    return detections

with st.spinner("Loading models..."):
    yolo_path = "E:/Final_BE_Project/work/test/yolov5lu.pt"            # update if necessary
    cnn_path = "E:/Final_BE_Project/work/test/roiresnetv2.h5"  # update path to your CNN
    yolo_model, cnn_model = initialize_models(yolo_path, cnn_path)

def display_video(video_source):
    global last_notification_time
    stframe = st.empty()
    cap = cv2.VideoCapture(video_source)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # frame = cv2.resize(frame, (640, 480))
        if frame_count % skip_rate != 0:
            continue
        # Simulated accident detection (every 50 frames)
        accident_detected = process_frame(frame)

        if accident_detected:
            for x1, y1, x2, y2, label, score in accident_detected:
                color = (0, 255, 0) if label == "Non-Accident" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            any_accident = any(label == "Accident" for _, _, _, _, label, _ in accident_detected)
            if any_accident:
                if last_notification_time is None or current_time - last_notification_time >= cooldown_period:
                    current_time = datetime.now()
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"Accident detected at frame {frame_count}, time {timestamp}, location: {camera_coords}"
                    accident_logs.append(log_entry)

                    # Save detected frame
                    frame_path = f"detected_frames/frame_{frame_count}.jpg"
                    cv2.imwrite(frame_path, frame)

                    # Display real-time alert
                    st.warning(f"🚨 Accident detected! Frame: {frame_count}, Time: {timestamp}")
                    st.write(f"📍 Location: {camera_coords['location']} (Latitude {camera_coords['latitude']}, Longitude {camera_coords['longitude']})")


                    hospital, police = send_alert2(selected_camera, timestamp, camera_coords)

                    # Display emergency services info in Streamlit
                    if hospital:
                        st.write(f"🚑 Nearest Hospital: **{hospital['name']}** (Phone: {hospital['phone']}, {hospital['distance']}km away)")
                    if police:
                        st.write(f"🚔 Nearest Police Station: **{police['name']}** (Phone: {police['phone']}, {police['distance']}km away)")
                    last_notification_time = current_time

        # Display video frame
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Update logs in sidebar
        log_placeholder.write("### Accident Logs\n" + "\n".join(accident_logs[-5:]))

    cap.release()

# Upload a video
if option == "Upload a Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        video_path = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")
        display_video(video_path)

# Use webcam
elif option == "Use Webcam":
    st.write("Starting webcam...")
    display_video(0)