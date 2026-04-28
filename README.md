
<div align="center">



#  ShootSeer-AI
### Real-Time Player Monitoring, Ball Tracking & Pose Estimation

*Precision sports analytics powered by computer vision*

</div>

---

##  Overview

**ShootSeer-AI** is a real-time sports analytics system that combines ball detection, Kalman-filtered trajectory tracking, and full-body pose estimation to monitor player performance during gameplay. Built for basketball (adaptable to any sport), it overlays rich visual analytics on both a live camera feed and a custom UI canvas.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏀 **Ball Detection** | Custom-trained YOLOv8 model (`best.pt`) with 0.5 confidence threshold |
| 📍 **Trajectory Tracking** | Kalman filter for smooth motion prediction, 30-frame trail |
| 🧍 **Pose Estimation** | YOLOv8s-Pose with 17-keypoint skeleton rendering |
| 📐 **Distance Estimation** | Real-world distance via pinhole camera model |
| 🎨 **Dual Canvas** | Live annotated camera view + custom background UI overlay |
| ⚡ **Real-Time** | Processes webcam feed at inference speed |

---

##  System Architecture
```bash
Webcam Feed
│
├──▶ YOLOv8 Ball Model ──▶ Kalman Filter ──▶ Trajectory + Distance
│
└──▶ YOLOv8 Pose Model ──▶ Skeleton Overlay (12 limb connections)
│
▼
┌──────────────────────┐     ┌──────────────────────┐
│   Camera View (Raw)  │     │  Custom UI (BG Image) │
└──────────────────────┘     └──────────────────────┘
```
---

## 📁 Project Structure
```bash
ShootSeer-AI/
│
├── main.py                         # Core detection & tracking loop
├── best.pt                         # Custom YOLOv8 ball detection model
├── yolov8s-pose.pt                 # YOLOv8 pose estimation model
├── image/
│   └── Screenshot 2026-02-06...png # Custom UI background
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- Webcam

### Install Dependencies

```bash
git clone https://github.com/Priyadharshinik15/ShootSeer-AI.git
cd ShootSeer-AI
pip install -r requirements.txt
```

### requirements.txt
ultralytics
opencv-python
numpy

### Run

```bash
python main.py
```

> Press **ESC** to exit.

---

## 🔧 Configuration
```bash
| Parameter | Default | Description |
|---|---|---|
| `REAL_BALL_DIAMETER` | `0.24 m` | Standard basketball diameter |
| `FOCAL_LENGTH` | `800 px` | Camera focal length (calibrate per device) |
| `MAX_TRAJECTORY` | `30 frames` | Trail length |
| `UI_WIDTH / UI_HEIGHT` | `800 × 600` | Output canvas resolution |
| Ball confidence | `0.5` | YOLOv8 detection threshold |
| Pose confidence | `0.4` | Pose estimation threshold |
```
---

## 🦴 Pose Skeleton Map
```bash
Nose
│
L.Shoulder ─── R.Shoulder
│                  │
L.Elbow         R.Elbow
│                  │
L.Wrist         R.Wrist
L.Hip ────────── R.Hip
│                  │
L.Knee          R.Knee
│                  │
L.Ankle         R.Ankle
12 limb connections rendered with distinct colors per joint pair.
```

---

## 📊 Metrics Computed

- **Ball Position** — pixel (cx, cy) smoothed via Kalman filter
- **Distance to Ball** — `D = (REAL_DIAMETER × FOCAL_LENGTH) / pixel_width`
- **Trajectory** — 30-frame motion trail (red polyline)
- **Keypoints** — 17 COCO pose landmarks per detected person

---

## 🚀 Roadmap

-  Speed & velocity calculation (px/frame → m/s)
-  Shot arc angle detection
-  Player ID tracking across frames
-  Action classification (dribble / pass / shoot)
-  Streamlit dashboard with live stats
-  Multi-camera support

---

## 👩‍💻 Author

**Priyadharshini K** 

[![GitHub](https://img.shields.io/badge/GitHub-Priyadharshinik15-181717?style=flat&logo=github)](https://github.com/Priyadharshinik15)

---

<div align="center">
Made with 🏀 + Computer Vision
</div>
