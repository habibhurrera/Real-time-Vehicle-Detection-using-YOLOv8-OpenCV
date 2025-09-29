from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog

# ==============================
# Step 1: Ask user to select a video file
# ==============================
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Select a video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    print("❌ No file selected. Exiting...")
    exit()

print(f"✅ Selected file: {video_path}")

# ==============================
# Step 2: Load YOLO model (Nano version for CPU speed)
# ==============================
model = YOLO("yolov8n.pt")  # nano = fastest

# ==============================
# Step 3: Open video (no CAP_DSHOW for files)
# ==============================
cap = cv2.VideoCapture(video_path)
vehicle_classes = ["car", "truck", "bus", "motorbike", "bicycle"]

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % 2 != 0:   # process every 2nd frame for speed
        continue

    # Resize frame
    frame = cv2.resize(frame, (640, 360))

    # Run YOLO on CPU
    results = model(frame, imgsz=320, verbose=False, device="cpu")

    # Draw detections
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show video
    cv2.imshow("YOLOv8 Vehicle Detection (CPU Optimized)", frame)

    # Press "q" to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
