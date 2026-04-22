import tensorflow as tf
import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "video_detect", "best_tf_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

REAL_CLASS_INDEX = 1
FAKE_CLASS_INDEX = 0

REAL_THRESHOLD = 0.55

FAKE_FRAME_RATIO_THRESHOLD = 0.30  

def extract_frames(video_path: str, max_frames: int = 40) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return np.array([])

    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

        frame = frame.astype(np.float32)
        frames.append(frame)

    cap.release()
    return np.array(frames) 


def aggregate_frame_predictions(real_probs: np.ndarray) -> tuple[str, float]:

    fake_frame_ratio = float((real_probs < 0.5).mean())
    if fake_frame_ratio >= FAKE_FRAME_RATIO_THRESHOLD:
        fake_probs = 1.0 - real_probs[real_probs < 0.5]
        confidence = round(float(fake_probs.mean() * 100), 2)
        return "fake", confidence

    weights = np.where(real_probs < 0.5, 3.0, 1.0)
    weighted_real_prob = float(np.average(real_probs, weights=weights))

    if weighted_real_prob >= REAL_THRESHOLD:
        confidence = round(weighted_real_prob * 100, 2)
        return "real", confidence
    else:
        confidence = round((1.0 - weighted_real_prob) * 100, 2)
        return "fake", confidence


def predict_video(video_path: str) -> dict:
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return {"error": "No frames could be extracted from this video"}

    preds = model.predict(frames, verbose=0)

    real_probs = preds[:, REAL_CLASS_INDEX]   
    fake_probs = preds[:, FAKE_CLASS_INDEX]   

    label, confidence = aggregate_frame_predictions(real_probs)

    avg_real_prob = float(real_probs.mean())
    fake_frame_ratio = float((real_probs < 0.5).mean())

    return {
        "prediction": label,
        "confidence": confidence,
        "raw_score": round(avg_real_prob, 4),
        "real_probability": round(avg_real_prob * 100, 2),
        "fake_probability": round((1.0 - avg_real_prob) * 100, 2),
        "frames_analysed": len(frames),
        "fake_frame_ratio": round(fake_frame_ratio * 100, 2), 
    }