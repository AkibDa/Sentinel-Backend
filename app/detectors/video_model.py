import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp
from app.detectors.image_model import predict_image_from_file


_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def _crop_face(frame: np.ndarray, padding: float = 0.2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    # Pick the largest face
    largest = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = largest

    # Apply padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def predict_video(video_path: str, frame_interval: int = 10, max_frames: int = 50):
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_results = []
    frame_index = 0
    analyzed = 0
    skipped_no_face = 0

    try:
        while cap.isOpened() and analyzed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                face_crop = _crop_face(frame)

                if face_crop is None:
                    skipped_no_face += 1
                    frame_index += 1
                    continue

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    cv2.imwrite(tmp_path, face_crop)
                    result = predict_image_from_file(tmp_path)
                finally:
                    os.remove(tmp_path)

                if "error" not in result:
                    frame_results.append({
                        "frame_index": frame_index,
                        "timestamp_sec": round(frame_index / fps, 2) if fps > 0 else None,
                        **result
                    })
                    analyzed += 1

            frame_index += 1

    finally:
        cap.release()

    if not frame_results:
        return {"error": "No frames could be analyzed — no faces detected in sampled frames"}

    raw_scores = [r["raw_score"] for r in frame_results]
    avg_raw_score = float(np.mean(raw_scores))

    fake_count = sum(1 for r in frame_results if r["label"] == "fake")
    real_count = len(frame_results) - fake_count
    fake_ratio = fake_count / len(frame_results)

    final_label = "fake" if fake_ratio >= 0.5 else "real"

    if final_label == "fake":
        final_confidence = round(avg_raw_score * 100, 2)
    else:
        final_confidence = round((1.0 - avg_raw_score) * 100, 2)

    return {
        "label": final_label,
        "confidence": final_confidence,
        "avg_raw_score": round(avg_raw_score, 4),
        "fake_frame_ratio": round(fake_ratio, 4),
        "frames_analyzed": len(frame_results),
        "total_frames": total_frames,
        "fake_frames": fake_count,
        "real_frames": real_count,
        "frames_skipped_no_face": skipped_no_face,
        "frame_results": frame_results  
    }


def predict_video_from_url(video_url: str, frame_interval: int = 10, max_frames: int = 50):
    """
    Downloads a video to a temp file and runs predict_video on it.
    """
    import requests

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(video_url, headers=headers, timeout=30, stream=True)

        if response.status_code != 200:
            return {"error": f"Failed to download video (HTTP {response.status_code})"}

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)

        try:
            result = predict_video(tmp_path, frame_interval=frame_interval, max_frames=max_frames)
        finally:
            os.remove(tmp_path)

        return result

    except Exception as e:
        return {"error": str(e)}