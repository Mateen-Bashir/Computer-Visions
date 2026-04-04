## Face Detection & Recognition Systems (Video-Based)

### 📌 Overview
This project demonstrates multiple real-world applications of face detection and face recognition using OpenCV and LBPH algorithm.  
All systems are implemented using **videos and datasets (no webcam)**.

---

## 🚀 Implemented Case Studies

### 1. Smart Attendance System
- Detects faces from lecture video
- Recognizes students using trained dataset
- Logs attendance with timestamps

📄 Output:
- `attendance_log.csv`

---

### 2. Access Control System
- Identifies authorized personnel
- Grants or denies access
- Displays result on video frames

---

### 3. Celebrity Face Recognition (Media Tagging)
- Trained on celebrity dataset
- Automatically labels faces in images
- Useful for media archiving

---

### 4. Intruder Detection System
- Detects faces in surveillance video
- Flags unknown faces as intruders
- Displays alert in real-time

---

### 5. Criminal Identification System
- Matches faces with criminal database
- Generates alerts when match found
- Saves detection logs

📄 Output:
- `detected_criminals.csv`

---

## 🧠 Techniques Used
- Haar Cascade Face Detection
- LBPH Face Recognition
- Video Processing (OpenCV)
- Frame Skipping Optimization
- Image Resizing for Speed

---

## ⚡ Optimizations Applied
- Frame skipping (process every 5th frame)
- Resize frames to 640px width
- Grayscale conversion for faster processing

---

---

## ▶️ How to Run
1. Install dependencies:
   ```bash
   pip install opencv-contrib-python numpy pandas matplotlib
