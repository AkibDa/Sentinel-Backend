import tensorflow as tf
import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "video_detect", "best_tf_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

REAL_CLASS_INDEX = 1
FAKE_CLASS_INDEX = 0


def extract_frames(video_path: str, max_frames: int = 20) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return np.array([])

    step = max(1, total_frames // max_frames)

    count = 0
    while cap.isOpened() and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count * step)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

        frame = frame.astype(np.float32)

        frames.append(frame)
        count += 1

    cap.release()
    return np.array(frames)  # shape: (num_frames, 224, 224, 3)


def predict_video(video_path: str) -> dict:
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return {"error": "No frames could be extracted from this video"}

    preds = model.predict(frames, verbose=0)  
    avg_real_prob = float(preds[:, REAL_CLASS_INDEX].mean())
    avg_fake_prob = float(preds[:, FAKE_CLASS_INDEX].mean())

    if avg_real_prob >= 0.5:
        label = "real"
        confidence = round(avg_real_prob * 100, 2)
    else:
        label = "fake"
        confidence = round(avg_fake_prob * 100, 2)

    return {
        "prediction": label,
        "confidence": confidence,
        "raw_score": round(avg_real_prob, 4),      
        "real_probability": round(avg_real_prob * 100, 2),
        "fake_probability": round(avg_fake_prob * 100, 2),
        "frames_analysed": len(frames)
    }