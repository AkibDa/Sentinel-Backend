import tensorflow as tf
import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "video_detect", "finetuned_video_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

FAKE_CLASS_INDEX = 0
REAL_CLASS_INDEX = 1

FAKE_RATIO_THRESHOLD = 0.40

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_frames(frame: np.ndarray):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5
    )
    if len(faces) == 0:
        return None
    
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face

    face = frame[y:y+h, x:x+w]

    #discarding tiny detections(noise/false positives)
    if face.shape[0] < 50 or face.shape[1] < 50:
        return None
    
    return face




def predict_video(video_path: str) -> dict:

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30


    frame_interval = max(1,fps // 2)

    frame_count = 0
    fake_count = 0
    total_count = 0
    per_frame_fake_probs = []

    success, frame = cap.read()

    while success:
        if frame_count % frame_interval == 0 :
            face = extract_frames(frame)

            if face is not None:
                face_resized = cv2.resize(face, (224,224))
                face_input = np.expand_dims(face_resized, axis = 0).astype(np.float32)

                pred = model.predict(face_input, verbose=0)
                pred_class = int(np.argmax(pred[0]))
                fake_prob = float(pred[0][FAKE_CLASS_INDEX])

                per_frame_fake_probs.append(fake_prob)

                if pred_class == FAKE_CLASS_INDEX:
                    fake_count +=1

                total_count +=1

        success, frame = cap.read()
        frame.count =+1

    cap.release()

    if total_count == 0:
        return{"error": "No faces could be found in this video. The video may not contian visible faces"}
    
    fake_ratio = fake_count / total_count
    avg_fake_prob = float(np.mean(per_frame_fake_probs)) if per_frame_fake_probs else 0.0 

    if fake_ratio > FAKE_RATIO_THRESHOLD:
        label= "fake"
        confidence = round(avg_fake_prob *100,2)

    else:
        label = "real"
        confidence = round((1.0 - avg_fake_prob)* 100, 2)

    return{
        "prediction": label, 
        "confidence": confidence,
        "raw_score": round(avg_fake_prob,4),
        "fake_probability": round(avg_fake_prob * 100, 2),
        "real_probability": round((1.0 - avg_fake_prob)* 100, 2),
        "frames_analyzed": total_count,
        "fake_frame_ratio": round(fake_ratio * 100, 2),
    }