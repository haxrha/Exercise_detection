import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import urllib.request

# ── Download model ─────────────────────────────────────────────────────────────
model_path = "pose_landmarker_full.task"
if not Path(model_path).exists():
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        model_path
    )
    print("Model downloaded.")

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)

# ── Form assessment ────────────────────────────────────────────────────────────
def assess_plank_form(frame):
    shoulder_y = (frame[11][1] + frame[12][1]) / 2
    hip_y      = (frame[23][1] + frame[24][1]) / 2
    knee_y     = (frame[25][1] + frame[26][1]) / 2
    ankle_y    = (frame[27][1] + frame[28][1]) / 2
    shoulder_x = (frame[11][0] + frame[12][0]) / 2
    hip_x      = (frame[23][0] + frame[24][0]) / 2
    ankle_x    = (frame[27][0] + frame[28][0]) / 2

    feedback = []
    score    = 1.0

    expected_hip_y = shoulder_y + (ankle_y - shoulder_y) * (
        (hip_x - shoulder_x) / max(abs(ankle_x - shoulder_x), 1e-5)
    )
    hip_deviation = hip_y - expected_hip_y

    if hip_deviation > 0.06:
        score -= 0.4
        feedback.append("Hip sag — raise hips")
    elif hip_deviation < -0.06:
        score -= 0.3
        feedback.append("Hips too high — lower them")

    spine_len  = abs(ankle_y - shoulder_y)
    knee_ideal = shoulder_y + (ankle_y - shoulder_y) * 0.6
    knee_error = abs(knee_y - knee_ideal) / max(spine_len, 1e-5)

    if knee_error > 0.12:
        score -= 0.2
        feedback.append("Knees bent — straighten legs")

    left_shoulder_y  = frame[11][1]
    right_shoulder_y = frame[12][1]
    if abs(left_shoulder_y - right_shoulder_y) > 0.05:
        score -= 0.1
        feedback.append("Uneven shoulders")

    score = max(0.0, score)
    label = "GOOD FORM" if score >= 0.7 else "BAD FORM"
    if not feedback:
        feedback.append("Great plank position!")

    return label, score, feedback

# ── Drawing ────────────────────────────────────────────────────────────────────
GOOD_COLOR = (0, 220, 100)
BAD_COLOR  = (0, 80, 255)

def draw_skeleton(img, landmarks, w, h):
    CONNECTIONS = [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),
        (23,25),(25,27),(24,26),(26,28),
    ]
    pts = {i: (int(landmarks[i][0]*w), int(landmarks[i][1]*h)) for i in range(33)}
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (180,180,180), 2, cv2.LINE_AA)
    for i in [11,12,23,24,25,26,27,28]:
        cv2.circle(img, pts[i], 5, (255,255,255), -1, cv2.LINE_AA)

def draw_overlay(img, label, score, feedback):
    h, w   = img.shape[:2]
    color  = GOOD_COLOR if label == "GOOD FORM" else BAD_COLOR
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (w, 70), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, label, (16, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"Score: {score:.0%}", (w-180, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    bar_x, bar_y, bar_w, bar_h = 16, 52, w-32, 8
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+int(bar_w*score), bar_y+bar_h), color, -1)
    for i, tip in enumerate(feedback[:3]):
        cv2.putText(img, tip, (12, h-20-i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 1, cv2.LINE_AA)

# ── Main ───────────────────────────────────────────────────────────────────────
def run_live_demo(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Try camera_index=1 if you have multiple cameras.")
        return

    score_buffer = []
    print("Camera open. Press Q to quit.")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            h, w      = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result    = landmarker.detect(mp_image)

            if result.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                                       for lm in result.pose_landmarks[0]])
                label, score, feedback = assess_plank_form(landmarks)
                score_buffer.append(score)
                if len(score_buffer) > 8:
                    score_buffer.pop(0)
                smooth_score = np.mean(score_buffer)
                smooth_label = "GOOD FORM" if smooth_score >= 0.7 else "BAD FORM"
                draw_skeleton(frame, landmarks, w, h)
                draw_overlay(frame, smooth_label, smooth_score, feedback)
            else:
                cv2.putText(frame, "No pose detected — step back or adjust camera",
                            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,100,255), 2, cv2.LINE_AA)

            cv2.imshow("Plank Form Analyzer | Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

run_live_demo(camera_index=0)