
<div align="center">

<img src="https://img.shields.io/badge/ShootSeer--AI-Computer%20Vision-00ff88?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-Realtime-5C3EE8?style=for-the-badge&logo=opencv"/>

# 🏀 ShootSeer-AI
### Real-Time Player Monitoring, Ball Tracking & Pose Estimation

*Precision sports analytics powered by computer vision*

</div>

---

## 📌 Overview

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

## 🧠 System Architecture
