import os
import threading
import logging
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

logger = logging.getLogger(__name__)


BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get(
    "VIDEO_MODEL_PATH",
    os.path.join(BASE_DIR, "app", "models", "video_detect", "video_model.keras"),
)

FAKE_THRESHOLD: float = float(os.environ.get("VIDEO_FAKE_THRESHOLD", "0.50"))

MIN_FRAMES_FOR_DECISION: int = int(os.environ.get("VIDEO_MIN_FRAMES", "5"))

INFERENCE_BATCH_SIZE: int = int(os.environ.get("VIDEO_INFERENCE_BATCH", "32"))

FACE_MARGIN: float = 0.20

MODEL_GENERATION: int = int(os.environ.get("MODEL_GENERATION", "1"))

IMG_SIZE: tuple[int, int] = (224, 224) if MODEL_GENERATION == 1 else (299, 299)


_lock    = threading.Lock()
_model:    Optional[tf.keras.Model] = None
_detector: Optional[MTCNN]          = None


def _load_resources() -> tuple[tf.keras.Model, MTCNN]:
    global _model, _detector
    if _model is None or _detector is None:
        with _lock:
            if _model is None:
                logger.info("Loading Gen-%d model from %s …", MODEL_GENERATION, MODEL_PATH)
                # Gen 1 (EfficientNetV2B0) has no custom objects.
                # Gen 2 (Xception+FFT) needs FFTBranch and BinaryFocalLoss stubs.
                custom_objects = (
                    {"FFTBranch": _FFTBranchStub, "BinaryFocalLoss": _BinaryFocalLossStub}
                    if MODEL_GENERATION == 2
                    else {}
                )
                _model = tf.keras.models.load_model(
                    MODEL_PATH,
                    custom_objects=custom_objects,
                    compile=False,
                )
                logger.info("Model loaded (generation %d, input %s).", MODEL_GENERATION, IMG_SIZE)
            if _detector is None:
                logger.info("Initialising MTCNN detector …")
                _detector = MTCNN()
                logger.info("MTCNN ready.")
    return _model, _detector


class _FFTBranchStub(tf.keras.layers.Layer):
    def __init__(self, fft_size: int = 75, **kwargs):
        super().__init__(**kwargs)
        self.fft_size = fft_size
        self.conv1 = tf.keras.layers.Conv2D(32, 5, strides=2, activation="relu", padding="same")
        self.bn1   = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")
        self.bn2   = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")
        self.gap   = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(256, activation="relu")

    def call(self, x, training=False):
        x_small = tf.image.resize(x, [self.fft_size, self.fft_size])
        gray    = tf.image.rgb_to_grayscale(x_small)
        gray    = tf.squeeze(gray, axis=-1)
        gray_c  = tf.cast(gray, tf.complex64)
        fft     = tf.signal.fft2d(gray_c)
        shifted = tf.signal.fftshift(fft)
        mag     = tf.math.log1p(tf.abs(shifted))
        mag     = tf.expand_dims(mag, axis=-1)
        mn      = tf.reduce_min(mag,  axis=[1, 2, 3], keepdims=True)
        mx      = tf.reduce_max(mag,  axis=[1, 2, 3], keepdims=True)
        mag     = (mag - mn) / (mx - mn + 1e-8)
        x = self.conv1(mag, training=training)
        x = self.bn1(x,     training=training)
        x = self.conv2(x,   training=training)
        x = self.bn2(x,     training=training)
        x = self.conv3(x,   training=training)
        x = self.gap(x)
        return self.dense(x)

    def get_config(self):
        return {**super().get_config(), "fft_size": self.fft_size}


class _BinaryFocalLossStub(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true  = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred  = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t     = tf.where(y_true == 1, y_pred, 1.0 - y_pred)
        alpha_t = tf.where(y_true == 0,
                           tf.ones_like(y_true) * self.alpha,
                           tf.ones_like(y_true) * (1.0 - self.alpha))
        return tf.reduce_mean(-alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t))

    def get_config(self):
        return {**super().get_config(), "gamma": self.gamma, "alpha": self.alpha}


def _extract_face(frame: np.ndarray, detector: MTCNN) -> Optional[np.ndarray]:

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    if not detections:
        return None

    best = max(
        (d for d in detections if d.get("confidence", 0) >= 0.90),
        key=lambda d: d["box"][2] * d["box"][3],
        default=None,
    )
    if best is None:
        return None

    x, y, w, h = best["box"]
    x, y = max(0, x), max(0, y)

    margin_x = int(w * FACE_MARGIN)
    margin_y = int(h * FACE_MARGIN)
    ih, iw   = frame.shape[:2]

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(iw, x + w + margin_x)
    y2 = min(ih, y + h + margin_y)

    face = rgb[y1:y2, x1:x2]

    if face.shape[0] < 50 or face.shape[1] < 50:
        return None

    return face

def _preprocess_face(face_rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    arr     = resized.astype(np.float32)
    if MODEL_GENERATION == 2:
        arr = tf.keras.applications.xception.preprocess_input(arr)
    return arr


def _run_inference(faces: list[np.ndarray], model: tf.keras.Model) -> np.ndarray:
    
    all_preds: list[np.ndarray] = []
    for start in range(0, len(faces), INFERENCE_BATCH_SIZE):
        chunk = np.array(faces[start : start + INFERENCE_BATCH_SIZE], dtype=np.float32)
        preds = model.predict(chunk, verbose=0)
        if MODEL_GENERATION == 1:
            all_preds.append(preds[:, 1].flatten())
        else:
            all_preds.append(preds.flatten())
    return np.concatenate(all_preds)   

def _confidence_band(fake_prob: float, threshold: float) -> str:
   
    distance = abs(fake_prob - threshold)
    if distance >= 0.25:
        return "HIGH"
    if distance >= 0.10:
        return "MEDIUM"
    return "LOW"


def predict_video(video_path: str) -> dict:
    
    model, detector = _load_resources()

    # --- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Gen 1 was trained at 1 fps (frame_interval = fps).
    # Gen 2 was trained at 2 fps (frame_interval = fps // 2).
    frame_interval = max(1, fps if MODEL_GENERATION == 1 else fps // 2)

    faces_batch: list[np.ndarray] = []
    frame_count  = 0

    ok, frame = cap.read()
    while ok:
        if frame_count % frame_interval == 0:
            face_rgb = _extract_face(frame, detector)
            if face_rgb is not None:
                faces_batch.append(_preprocess_face(face_rgb))
        ok, frame = cap.read()
        frame_count += 1

    cap.release()

    # --- Guard: no faces ---------------------------------------------------
    if not faces_batch:
        return {
            "error": (
                "No faces could be detected in this video. "
                "Ensure it contains clear, frontal faces with confidence ≥ 0.90."
            )
        }

    # --- Inference ---------------------------------------------------------
    real_probs     = _run_inference(faces_batch, model)   # shape (N,) — P(real)
    fake_probs     = 1.0 - real_probs                     # P(fake)

    avg_fake_prob  = float(np.mean(fake_probs))
    avg_real_prob  = float(np.mean(real_probs))
    frames_analysed = len(faces_batch)

    # Per-frame classification using the same threshold
    per_frame_fake = (fake_probs >= FAKE_THRESHOLD)
    fake_frame_ratio = float(np.mean(per_frame_fake)) * 100.0

    # --- Decision ----------------------------------------------------------
    is_fake = avg_fake_prob >= FAKE_THRESHOLD

    if is_fake:
        label      = "fake"
        confidence = round(avg_fake_prob * 100, 2)
    else:
        label      = "real"
        confidence = round(avg_real_prob * 100, 2)

    return {
        "prediction":       label,
        "confidence":       confidence,
        "confidence_band":  _confidence_band(avg_fake_prob, FAKE_THRESHOLD),
        "low_confidence":   frames_analysed < MIN_FRAMES_FOR_DECISION,
        "raw_score":        round(avg_real_prob, 4),      # P(real) — same semantics as training
        "fake_probability": round(avg_fake_prob * 100, 2),
        "real_probability": round(avg_real_prob * 100, 2),
        "frames_analysed":  frames_analysed,
        "fake_frame_ratio": round(fake_frame_ratio, 2),
        "threshold_used":   FAKE_THRESHOLD,
        "error":            None,
    }