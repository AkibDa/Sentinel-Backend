import cv2
import numpy as np
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Updated to match the new model filename from the notebook
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "video_detect", "finetuned_video_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

# Fake=0, Real=1 based on alphabetical sorting of subdirectories in the training notebook
FAKE_CLASS_INDEX = 0
REAL_CLASS_INDEX = 1

# Minimum face-frames needed before we trust the result
MIN_FRAMES_FOR_DECISION = 5

# Decision threshold on the ratio of fake frames
FAKE_RATIO_THRESHOLD = 0.50

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_face(frame: np.ndarray):
    """Extracts the largest face crop from a frame using Haar cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    face = frame[y:y+h, x:x+w]

    if face.shape[0] < 50 or face.shape[1] < 50:
        return None

    return face

def predict_video(video_path: str) -> dict:
    """
    Deepfake detection via discrete frame voting (matching the notebook logic).
    """
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    frame_interval = max(1, fps // 2)

    frame_count = 0
    total_frames_analysed = 0
    fake_frame_count = 0
    per_frame_fake_probs = []

    success, frame = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            face = extract_face(frame)

            if face is not None:
                face_resized = cv2.resize(face, (224, 224))
                
                # Crucial Fix: Convert BGR (OpenCV) to RGB (Keras Training format)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                
                face_input = np.expand_dims(face_rgb, axis=0).astype(np.float32)

                pred = model.predict(face_input, verbose=0) 
                
                fake_prob = float(pred[0][FAKE_CLASS_INDEX])
                per_frame_fake_probs.append(fake_prob)

                # Discrete vote matching Anit's notebook test function
                pred_class = np.argmax(pred[0])
                if pred_class == FAKE_CLASS_INDEX:
                    fake_frame_count += 1

                total_frames_analysed += 1

        success, frame = cap.read()
        frame_count += 1

    cap.release()

    if total_frames_analysed == 0:
        return {
            "error": "No faces could be detected in this video. Ensure it contains clear, frontal faces."
        }

    # Frame ratio logic instead of continuous weighting
    fake_ratio = fake_frame_count / total_frames_analysed
    avg_fake_prob = float(np.mean(per_frame_fake_probs))

    is_fake = fake_ratio > FAKE_RATIO_THRESHOLD

    if is_fake:
        confidence = round(fake_ratio * 100, 2)
        label = "fake"
    else:
        # If it's real, confidence is the percentage of frames that voted real
        confidence = round((1.0 - fake_ratio) * 100, 2)
        label = "real"

    return {
        "prediction": label,
        "confidence": confidence,
        "low_confidence": total_frames_analysed < MIN_FRAMES_FOR_DECISION,
        "raw_score": round(fake_ratio, 4),
        "fake_probability": round(avg_fake_prob * 100, 2),
        "real_probability": round((1.0 - avg_fake_prob) * 100, 2),
        "frames_analysed": total_frames_analysed,
        "fake_frame_ratio": round(fake_ratio * 100, 2),
    }