Real-time Vehicle Detection using YOLOv8 and OpenCV

This project is a Python application for detecting vehicles in real time from a video file. It uses YOLOv8 for object detection and OpenCV for video processing and display. The program allows the user to select a video file through a file picker dialog and displays the detections frame by frame.

Features

Detects vehicles such as cars, trucks, buses, motorbikes, and bicycles

Works on CPU-only systems with optimizations for better frame rate

Uses a file picker dialog (Tkinter) for easy video selection

Supports real-time bounding boxes and labels on video frames

Exit the program by pressing the "q" key

Tools and Libraries Used

Python 3.9+

ultralytics (YOLOv8)

opencv-python

tkinter (for file dialog)

venv (for isolated environment)

Project Structure
vehicle-detection-yolov8/
│── vehicle_dtc.py        # main detection script
│── requirements.txt      # dependencies
│── README.md             # documentation

Installation

Clone the repository

git clone https://github.com/<your-username>/vehicle-detection-yolov8.git
cd vehicle-detection-yolov8


Create a virtual environment

python -m venv venv


Activate the virtual environment

Windows (PowerShell)

venv\Scripts\activate


Linux / macOS

source venv/bin/activate


Install dependencies

pip install -r requirements.txt

Usage

Run the script:

python vehicle_dtc.py


A dialog box will open for selecting a video file (MP4).

The video will play with real-time vehicle detections.

Press "q" to quit the application.

How It Works

The script captures frames from a video file using OpenCV.

YOLOv8 runs inference on each frame to detect vehicles.

Detected vehicles are labeled and marked with bounding boxes.

To improve performance on CPU, the program resizes frames, uses smaller image size for inference, and can skip frames.

Future Improvements

Add vehicle tracking across frames using DeepSORT

Train YOLOv8 on a custom vehicle dataset for improved accuracy

Convert the model to ONNX/TensorRT for faster inference

Add analytics such as vehicle counts, speed estimation, and statistics
