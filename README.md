Overview
This project implements real-time object detection using YOLOv5 in Python. It captures live video from a webcam or video source, detects objects in real-time, and plays an audio alert when a target object is detected.

Features:
1. Real-time object detection using YOLOv5
2. Plays an audio alert (alert.mp3) upon detecting an object
3. Supports webcam or video file input
4. Pre-trained YOLOv5 model (yolov5su.pt) for accurate detection

How to Run: 
1. Clone this repository (or download the zip).
2. Place all files (realtime_yolo.py, yolov5su.pt, alert.mp3) in the same folder.
3. Run the script:
python realtime_yolo.py
4. The cam window will open, and detection will start immediately.

How It Works:
1. Loads YOLOv5 model from yolov5su.pt.
2. Captures frames from your webcam.
3. Runs object detection on each frame, If a target object is detected, plays alert.mp3.

Notes:
1. If you want to detect only specific objects, modify the class filter in realtime_yolo.py.
2. Ensure your webcam is connected and accessible.
3. You can replace alert.mp3 with any sound file of your choice.
