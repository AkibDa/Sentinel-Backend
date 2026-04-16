import tensorflow as tf
import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "app", "video_detect", "best_tf_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames=[]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return np.array([])
    
    step = max(1, total_frames // max_frames)

    count = 0
    while cap.isOpened() and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count * step)
        
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224,224))
        frame = frame / 255.0
        frames.append(frame)
        count += 1

    cap.release()
    return np.array(frames)

def predict_video(video_path):
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return {"error": "No frames extracted"}
    
    preds = model.predict(frames)

    avg_pred = preds.mean()

    return{
        "prediction": "fake" if avg_pred > 0.5 else "real",
        "confidence": float(avg_pred)
    }