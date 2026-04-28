import cv2
import numpy as np
from ultralytics import YOLO


ball_model = YOLO("best.pt")
pose_model = YOLO("yolov8s-pose.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opened")
    exit(1)


REAL_BALL_DIAMETER = 0.24  # meters
FOCAL_LENGTH = 800          # pixels
trajectory = []
MAX_TRAJECTORY = 30

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03


SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

COLORS = [
    (0,255,255), (255,0,0), (0,255,0), (255,0,255),
    (0,165,255), (255,255,0), (128,0,128), (0,128,255),
    (0,128,0), (128,128,0), (255,128,0), (128,0,0)
]

UI_WIDTH, UI_HEIGHT = 800, 600


BG_PATH = "Screenshot 2026-02-06 205556.png"  # <-- update this path
bg_image = cv2.imread(BG_PATH)
if bg_image is None:
    print("Background image not found!")
    exit(1)
bg_image = cv2.resize(bg_image, (UI_WIDTH, UI_HEIGHT))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

  
    canvas = bg_image.copy()          # Custom UI with background
    camera_view = frame.copy()        # Camera feed with annotations

    
    ball_results = ball_model.predict(frame, conf=0.5, verbose=False)
    ball_detected = False
    ball_pixel_size = 0

    for r in ball_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            ball_detected = True
            ball_pixel_size = x2 - x1

            # Kalman filter update
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            kalman.correct(measurement)
            pred = kalman.predict()
            px, py = int(pred[0]), int(pred[1])
            trajectory.append((px, py))
            if len(trajectory) > MAX_TRAJECTORY:
                trajectory.pop(0)

            # Draw ball on both canvases
            scale_x = UI_WIDTH / w
            scale_y = UI_HEIGHT / h

            # Custom UI
            px_ui, py_ui = int(px * scale_x), int(py * scale_y)
            ball_radius = int(ball_pixel_size/2 * scale_x)
            cv2.circle(canvas, (px_ui, py_ui), ball_radius, (0,255,0), -1)

            # Camera view
            cv2.circle(camera_view, (px, py), ball_pixel_size//2, (0,255,0), -1)
            break

    if not ball_detected:
        trajectory.clear()

    # Draw ball trajectory
    if len(trajectory) > 1:
        # Custom UI
        traj_points = [(int(p[0]*UI_WIDTH/w), int(p[1]*UI_HEIGHT/h)) for p in trajectory]
        cv2.polylines(canvas, [np.array(traj_points, dtype=np.int32)], False, (0,0,255), 3)

        # Camera view
        cv2.polylines(camera_view, [np.array(trajectory, dtype=np.int32)], False, (0,0,255), 3)

    pose_results = pose_model.predict(frame, conf=0.4, verbose=False)
    for pr in pose_results:
        if pr.keypoints is None:
            continue

        keypoints = pr.keypoints.xy.cpu().numpy()
        classes = pr.boxes.cls.cpu().numpy()

        for idx, cls in enumerate(classes):
            if int(cls) != 0:  # Only person class
                continue
            person = keypoints[idx]

            # Draw joints and skeleton on both canvases
            for x, y in person:
                # Custom UI
                x_ui, y_ui = int(x * UI_WIDTH / w), int(y * UI_HEIGHT / h)
                cv2.circle(canvas, (x_ui, y_ui), 4, (255, 0, 255), -1)

                # Camera view
                cv2.circle(camera_view, (int(x), int(y)), 4, (255, 0, 255), -1)

            for i, (j1, j2) in enumerate(SKELETON):
                x1, y1 = person[j1]
                x2, y2 = person[j2]

                # Custom UI
                x1_ui, y1_ui = int(x1 * UI_WIDTH / w), int(y1 * UI_HEIGHT / h)
                x2_ui, y2_ui = int(x2 * UI_WIDTH / w), int(y2 * UI_HEIGHT / h)
                color = COLORS[i % len(COLORS)]
                cv2.line(canvas, (x1_ui, y1_ui), (x2_ui, y2_ui), color, 3)

                # Camera view
                cv2.line(camera_view, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    
    cv2.imshow("Camera View with Annotations", camera_view)
    cv2.imshow("Custom UI - Ball & Pose", canvas)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
