
import cv2
import numpy as np
from playsound import playsound
import threading
import time
from ultralytics import YOLO

print("Loading YOLOv5 model using ultralytics package...")
model = YOLO('yolov5s.pt')  # Load YOLOv5s model

print("Searching for available cameras (0, 1, 2)...")
camera_found = False
for cam_index in range(3):
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print(f"Camera found and opened at index {cam_index}.")
        camera_found = True
        break
    cap.release()
if not camera_found:
    print("Error: Could not open any camera (indices 0, 1, 2). Please check your camera connection.")
    exit()

print("Camera opened successfully. Starting detection loop...")

last_alert_time = 0
alert_cooldown = 2  # seconds
def play_alert_sound():
    global last_alert_time
    now = time.time()
    if now - last_alert_time > alert_cooldown:
        last_alert_time = now
        threading.Thread(target=playsound, args=('alert.mp3',), daemon=True).start()

while True:

    try:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Failed to grab frame from webcam. Exiting loop.")
            break

        # Optionally resize frame for faster inference
        small_frame = cv2.resize(frame, (640, 480))
        results = model(small_frame)
        boxes = results[0].boxes
        names = model.names if hasattr(model, 'names') else results[0].names
        detected = False
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = names[cls] if names and cls < len(names) else str(cls)
                # Scale coordinates back to original frame size if resized
                h_ratio = frame.shape[0] / 480
                w_ratio = frame.shape[1] / 640
                x1, y1, x2, y2 = int(x1 * w_ratio), int(y1 * h_ratio), int(x2 * w_ratio), int(y2 * h_ratio)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                detected = True
        if detected:
            play_alert_sound()

        cv2.imshow("Real-time Yolov5", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Exiting.")
            break
    except Exception as e:
        print(f"Exception occurred: {e}")
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam and windows closed. Program ended.")
