# Deep Learning / Pose Estimation
from ultralytics import YOLO

# Computer Vision
import cv2

# Data Handling
import pandas as pd
import numpy as np
import os
from glob import glob, iglob

# Machine Learning
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Folder Setup
base_dir = os.path.expanduser("~/Desktop/Projects/squat_form_detector")

paths = {
    "good_form_videos":    os.path.join(base_dir, "videos/good_form"),
    "bad_form_videos":     os.path.join(base_dir, "videos/bad_form"),
    "good_form_keypoints": os.path.join(base_dir, "keypoints/good_form"),
    "bad_form_keypoints":  os.path.join(base_dir, "keypoints/bad_form"),
    "good_form_labeled":   os.path.join(base_dir, "labeled_videos/good_form"),
    "bad_form_labeled":    os.path.join(base_dir, "labeled_videos/bad_form"),
    "model":               os.path.join(base_dir, "model"),
}

for path_name, path in paths.items():
    os.makedirs(path, exist_ok=True)

# --- Load YOLO Pose Model ---
yolo_model = YOLO("yolov8n-pose.pt")

# --- Keypoint Extraction from Videos ---
def extract_keypoints_from_videos():
    for label, (folder, keypoints_path, labeled_path) in enumerate([
        (paths["good_form_videos"], paths["good_form_keypoints"], paths["good_form_labeled"]),
        (paths["bad_form_videos"],  paths["bad_form_keypoints"],  paths["bad_form_labeled"]),
    ]):
        video_files = []
        for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV"]:
            video_files.extend(iglob(os.path.join(folder, ext)))

        for video_path in video_files:
            filename = os.path.splitext(os.path.basename(video_path))[0]  # Get filename without extension
            output_video_path = os.path.join(labeled_path, f"output_{filename}.mp4")
            output_csv = os.path.join(keypoints_path, f"{filename}_keypoints.csv")

            # Skip if already processed
            if os.path.exists(output_csv):
                print(f"Already processed, skipping: {filename}")
                continue

            cap = cv2.VideoCapture(video_path)
            frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            frame_num = 0
            data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = yolo_model(frame, verbose=False)
                annotated_frame = results[0].plot()

                keypoints = results[0].keypoints
                if keypoints is not None and len(keypoints.xy) > 0:
                    for i, kp in enumerate(keypoints.xy[0]):
                        x, y = kp[0].item(), kp[1].item()
                        conf = keypoints.conf[0][i].item()
                        data.append({
                            'frame': frame_num,
                            'keypoint_id': i,
                            'x': x,
                            'y': y,
                            'confidence': conf
                        })

                out.write(annotated_frame)
                frame_num += 1

            cap.release()
            out.release()

            df = pd.DataFrame(data)
            df.to_csv(output_csv, index=False)
            print(f"Processed: {filename}")

# --- Angle Calculation Helper ---
def calculate_angle(a, b, c):
    """
    Calculate the angle at point b, formed by points a, b, c.
    a, b, c are (x, y) coordinate pairs.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180:
        angle = 360 - angle

    return angle

# --- Squat-Specific Feature Extraction ---
def extract_features_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # YOLOv8 pose keypoint indices:
    # 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear
    # 5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow
    # 9: left wrist, 10: right wrist, 11: left hip, 12: right hip
    # 13: left knee, 14: right knee, 15: left ankle, 16: right ankle

    LEFT_HIP    = 11
    RIGHT_HIP   = 12
    LEFT_KNEE   = 13
    RIGHT_KNEE  = 14
    LEFT_ANKLE  = 15
    RIGHT_ANKLE = 16
    LEFT_SHOULDER  = 5
    RIGHT_SHOULDER = 6

    frames = df['frame'].unique()

    knee_angles = []
    hip_angles = []
    knee_cave_ratios = []

    for frame in frames:
        frame_data = df[df['frame'] == frame].set_index('keypoint_id')

        # Skip frame if any keypoint we need is missing
        needed = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
                  LEFT_ANKLE, RIGHT_ANKLE, LEFT_SHOULDER, RIGHT_SHOULDER]
        if not all(k in frame_data.index for k in needed):
            continue

        # Extract coordinates
        left_hip     = (frame_data.loc[LEFT_HIP,    'x'], frame_data.loc[LEFT_HIP,    'y'])
        right_hip    = (frame_data.loc[RIGHT_HIP,   'x'], frame_data.loc[RIGHT_HIP,   'y'])
        left_knee    = (frame_data.loc[LEFT_KNEE,   'x'], frame_data.loc[LEFT_KNEE,   'y'])
        right_knee   = (frame_data.loc[RIGHT_KNEE,  'x'], frame_data.loc[RIGHT_KNEE,  'y'])
        left_ankle   = (frame_data.loc[LEFT_ANKLE,  'x'], frame_data.loc[LEFT_ANKLE,  'y'])
        right_ankle  = (frame_data.loc[RIGHT_ANKLE, 'x'], frame_data.loc[RIGHT_ANKLE, 'y'])
        left_shoulder  = (frame_data.loc[LEFT_SHOULDER,  'x'], frame_data.loc[LEFT_SHOULDER,  'y'])
        right_shoulder = (frame_data.loc[RIGHT_SHOULDER, 'x'], frame_data.loc[RIGHT_SHOULDER, 'y'])

        # --- Knee Angle (hip -> knee -> ankle) ---
        left_knee_angle  = calculate_angle(left_hip,  left_knee,  left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angles.append((left_knee_angle + right_knee_angle) / 2)

        # --- Hip Angle (shoulder -> hip -> knee) ---
        left_hip_angle  = calculate_angle(left_shoulder,  left_hip,  left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        hip_angles.append((left_hip_angle + right_hip_angle) / 2)

        # --- Knee Cave Ratio (are knees caving inward?) ---
        knee_width   = abs(left_knee[0]  - right_knee[0])
        ankle_width  = abs(left_ankle[0] - right_ankle[0])
        if ankle_width > 0:
            knee_cave_ratios.append(knee_width / ankle_width)

    if not knee_angles:
        return None

    # --- Build Feature Vector ---
    features = [
        np.mean(knee_angles),   np.std(knee_angles),   np.min(knee_angles),
        np.mean(hip_angles),    np.std(hip_angles),     np.min(hip_angles),
        np.mean(knee_cave_ratios), np.std(knee_cave_ratios), np.min(knee_cave_ratios),
    ]

    return features

# --- Train the Model ---
def train_model():
    X = []
    y = []

    for label, folder in enumerate([paths["good_form_keypoints"], paths["bad_form_keypoints"]]):
        for csv_path in glob(os.path.join(folder, "*.csv")):
            features = extract_features_from_csv(csv_path)
            if features is not None:
                X.append(features)
                y.append(label)

    if len(X) == 0:
        print("No data found! Make sure you have videos in the good_form and bad_form folders.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"Training on {len(X)} videos ({sum(y == 0)} good form, {sum(y == 1)} bad form)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    classifier = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=["Good Form", "Bad Form"]))

    model_path = os.path.join(paths["model"], "squat_classifier.pkl")
    joblib.dump(classifier, model_path)
    print(f"Model saved to {model_path}")

    return classifier

# --- Live Webcam Inference ---
def run_webcam():
    model_path = os.path.join(paths["model"], "squat_classifier.pkl")
    if not os.path.exists(model_path):
        print("No trained model found! Run train_model() first.")
        return

    classifier = joblib.load(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    data = []
    frame_num = 0
    current_label = "Analyzing..."
    label_color = (255, 255, 255)

    print("Webcam running. Press 's' to score your form, 'r' to reset, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            for i, kp in enumerate(keypoints.xy[0]):
                x, y = kp[0].item(), kp[1].item()
                conf = keypoints.conf[0][i].item()
                data.append({
                    'frame': frame_num,
                    'keypoint_id': i,
                    'x': x,
                    'y': y,
                    'confidence': conf
                })

        # Display current label on frame
        cv2.putText(annotated_frame, current_label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
        cv2.putText(annotated_frame, f"Frames collected: {frame_num}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Squat Form Analyzer", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            data = []
            frame_num = 0
            current_label = "Analyzing..."
            label_color = (255, 255, 255)
            print("Reset!")
        elif key == ord('s') and len(data) > 0:
            df = pd.DataFrame(data)
            features = extract_features_from_csv_df(df)
            if features is not None:
                X = np.array([features])
                prediction = classifier.predict(X)
                if prediction[0] == 0:
                    current_label = "Good Form!"
                    label_color = (0, 255, 0)  # Green
                    print("Form Assessment: Good Form!")
                else:
                    current_label = "Bad Form!"
                    label_color = (0, 0, 255)  # Red
                    print("Form Assessment: Bad Form!")
            else:
                print("Not enough data yet, keep squatting!")

    cap.release()
    cv2.destroyAllWindows()

# --- Feature Extraction from DataFrame (for live webcam) ---
def extract_features_from_csv_df(df):
    # Same as extract_features_from_csv but takes a dataframe directly
    LEFT_HIP    = 11
    RIGHT_HIP   = 12
    LEFT_KNEE   = 13
    RIGHT_KNEE  = 14
    LEFT_ANKLE  = 15
    RIGHT_ANKLE = 16
    LEFT_SHOULDER  = 5
    RIGHT_SHOULDER = 6

    frames = df['frame'].unique()
    knee_angles = []
    hip_angles = []
    knee_cave_ratios = []

    for frame in frames:
        frame_data = df[df['frame'] == frame].set_index('keypoint_id')

        needed = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
                  LEFT_ANKLE, RIGHT_ANKLE, LEFT_SHOULDER, RIGHT_SHOULDER]
        if not all(k in frame_data.index for k in needed):
            continue

        left_hip     = (frame_data.loc[LEFT_HIP,    'x'], frame_data.loc[LEFT_HIP,    'y'])
        right_hip    = (frame_data.loc[RIGHT_HIP,   'x'], frame_data.loc[RIGHT_HIP,   'y'])
        left_knee    = (frame_data.loc[LEFT_KNEE,   'x'], frame_data.loc[LEFT_KNEE,   'y'])
        right_knee   = (frame_data.loc[RIGHT_KNEE,  'x'], frame_data.loc[RIGHT_KNEE,  'y'])
        left_ankle   = (frame_data.loc[LEFT_ANKLE,  'x'], frame_data.loc[LEFT_ANKLE,  'y'])
        right_ankle  = (frame_data.loc[RIGHT_ANKLE, 'x'], frame_data.loc[RIGHT_ANKLE, 'y'])
        left_shoulder  = (frame_data.loc[LEFT_SHOULDER,  'x'], frame_data.loc[LEFT_SHOULDER,  'y'])
        right_shoulder = (frame_data.loc[RIGHT_SHOULDER, 'x'], frame_data.loc[RIGHT_SHOULDER, 'y'])

        left_knee_angle  = calculate_angle(left_hip,  left_knee,  left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angles.append((left_knee_angle + right_knee_angle) / 2)

        left_hip_angle  = calculate_angle(left_shoulder,  left_hip,  left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        hip_angles.append((left_hip_angle + right_hip_angle) / 2)

        knee_width  = abs(left_knee[0]  - right_knee[0])
        ankle_width = abs(left_ankle[0] - right_ankle[0])
        if ankle_width > 0:
            knee_cave_ratios.append(knee_width / ankle_width)

    if not knee_angles:
        return None

    features = [
        np.mean(knee_angles),   np.std(knee_angles),   np.min(knee_angles),
        np.mean(hip_angles),    np.std(hip_angles),     np.min(hip_angles),
        np.mean(knee_cave_ratios), np.std(knee_cave_ratios), np.min(knee_cave_ratios),
    ]

    return features

# --- Main ---
if __name__ == "__main__":
    print("Starting squat form detector...")
    
    print("\nStep 1: Extracting keypoints from videos...")
    extract_keypoints_from_videos()
    
    print("\nStep 2: Training model...")
    train_model()
    
    print("\nStep 3: Starting webcam. Press 's' to score, 'r' to reset, 'q' to quit.")
    run_webcam()
