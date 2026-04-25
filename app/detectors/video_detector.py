import cv2
import numpy as np
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "video_detect", "finetuned_video_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

# Fake=0, Real=1 (alphabetical class ordering from training notebook)
FAKE_CLASS_INDEX = 0
REAL_CLASS_INDEX = 1

# ── Thresholds ──────────────────────────────────────────────────────────────
# The model was trained on ~3-4× more fake frames than real ones, so it has a
# built-in bias toward predicting "fake".  We compensate in two ways:
#
#   1. FAKE_RATIO_THRESHOLD is raised from 0.50 → 0.60 so a video needs a
#      clear majority of fake-voted frames before we call it fake.
#
#   2. A lightweight texture signal (face-edge smoothness score) is used as a
#      secondary vote.  Deepfakes typically show unnaturally smooth/blurred
#      skin around face boundaries; real faces show natural high-frequency
#      detail.  The texture score modulates the final decision only when the
#      model vote is in the "grey zone" (0.45–0.65).
#
FAKE_RATIO_THRESHOLD = 0.60          # primary model-vote threshold (raised)
GREY_ZONE_LOW        = 0.45          # below → call Real regardless of texture
GREY_ZONE_HIGH       = 0.65          # above → call Fake regardless of texture

MIN_FRAMES_FOR_DECISION = 5

# EfficientNetV2B0 was trained with its own preprocessing (scales pixels to
# [-1, 1]).  The original detector omitted this, feeding raw uint8-cast
# float32 values.  We apply the correct preprocessor here.
PREPROCESSOR = tf.keras.applications.efficientnet_v2.preprocess_input

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Face extraction ──────────────────────────────────────────────────────────

def extract_face(frame: np.ndarray):
    """
    Extracts the largest detected face crop.
    Returns (face_bgr, bbox) or (None, None).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    largest = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = largest
    face = frame[y:y + h, x:x + w]

    if face.shape[0] < 50 or face.shape[1] < 50:
        return None, None

    return face, (x, y, w, h)


# ── Texture / smoothness analysis ────────────────────────────────────────────

def _laplacian_variance(gray_patch: np.ndarray) -> float:
    """Variance of the Laplacian — high = sharp/real, low = smooth/fake."""
    return float(cv2.Laplacian(gray_patch, cv2.CV_64F).var())


def compute_deepfake_texture_score(face_bgr: np.ndarray, bbox) -> float:
    """
    Returns a score in [0, 1] where 1.0 means "strongly fake" and 0.0 means
    "strongly real", based purely on local texture signals.

    Strategy
    --------
    Deepfake blending leaves two characteristic traces:
      • The skin region inside the face mask is over-smoothed (low Laplacian
        variance) compared to a real face.
      • The edge band around the face (where the synthetic face meets the
        background) shows an unusually large sharpness *drop* vs. the skin
        interior — real faces have relatively uniform sharpness gradients.

    We combine both signals into a single [0,1] fake-probability estimate.
    """
    if face_bgr is None:
        return 0.5  # neutral if no face

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Signal 1: absolute skin smoothness ──────────────────────────────────
    # Sample the central 50% of the face (forehead / cheek area — mostly skin)
    cy, cx = h // 2, w // 2
    pad_y, pad_x = max(1, h // 4), max(1, w // 4)
    center_patch = gray[cy - pad_y: cy + pad_y, cx - pad_x: cx + pad_x]
    center_lap = _laplacian_variance(center_patch)

    # Empirically tuned range: real faces ~150–600, deepfakes ~20–150
    # Map to [0,1] fake-prob: low variance → high fake score
    REAL_LAP_MIN, REAL_LAP_MAX = 40.0, 500.0
    skin_fake_score = 1.0 - np.clip(
        (center_lap - REAL_LAP_MIN) / (REAL_LAP_MAX - REAL_LAP_MIN), 0.0, 1.0
    )

    # ── Signal 2: edge-band sharpness drop ──────────────────────────────────
    # Compare Laplacian variance of the outer 15% ring vs. the inner 70%
    margin = max(2, int(min(h, w) * 0.15))
    inner = gray[margin: h - margin, margin: w - margin]
    outer_top    = gray[:margin, :]
    outer_bottom = gray[h - margin:, :]
    outer_left   = gray[margin: h - margin, :margin]
    outer_right  = gray[margin: h - margin, w - margin:]

    inner_lap = _laplacian_variance(inner)

    outer_patches = [p for p in [outer_top, outer_bottom, outer_left, outer_right]
                     if p.size > 0]
    if outer_patches:
        outer_lap = float(np.mean([_laplacian_variance(p) for p in outer_patches]))
    else:
        outer_lap = inner_lap  # fallback — neutral

    # Edge drop ratio: how much sharper is the interior than the edge?
    # Real: roughly uniform or edge slightly sharper (natural lighting falloff)
    # Fake: edge noticeably blurrier — ratio > 1.5 is suspicious
    if outer_lap > 0:
        edge_ratio = inner_lap / outer_lap
    else:
        edge_ratio = 1.0

    # Map edge_ratio to fake score: ratio > 2.0 → strong fake signal
    edge_fake_score = np.clip((edge_ratio - 1.0) / 1.5, 0.0, 1.0)

    # ── Signal 3: colour banding / gradient smoothness ───────────────────────
    # Deepfakes often have very smooth colour gradients (GAN upsampling).
    # Measure std-dev of the per-pixel absolute differences between adjacent
    # rows — a low value means the image is unusually uniform.
    diff_rows = np.abs(gray[1:].astype(np.float32) - gray[:-1].astype(np.float32))
    gradient_std = float(diff_rows.std())

    # Real faces: gradient_std roughly 10–40; deepfakes: ~3–12
    GRAD_MIN, GRAD_MAX = 5.0, 35.0
    gradient_fake_score = 1.0 - np.clip(
        (gradient_std - GRAD_MIN) / (GRAD_MAX - GRAD_MIN), 0.0, 1.0
    )

    # Weighted combination (skin smoothness is the strongest signal)
    combined = (
        0.50 * skin_fake_score
        + 0.30 * edge_fake_score
        + 0.20 * gradient_fake_score
    )
    return float(np.clip(combined, 0.0, 1.0))


# ── Main prediction ──────────────────────────────────────────────────────────

def predict_video(video_path: str) -> dict:
    """
    Deepfake detection with:
      1. Correct EfficientNetV2B0 preprocessing ([-1,1] normalisation)
      2. Raised fake-ratio threshold to counter training-set bias
      3. Texture-based secondary signal to rescue mis-classified real videos
    """
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    frame_interval = max(1, fps // 2)

    frame_count          = 0
    total_frames_analysed = 0
    fake_frame_count      = 0
    per_frame_fake_probs  = []
    texture_scores        = []

    success, frame = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            face, bbox = extract_face(frame)

            if face is not None:
                # ── Model inference ─────────────────────────────────────────
                face_resized = cv2.resize(face, (224, 224))

                # Convert BGR → RGB (matches Keras training format)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                # Apply EfficientNetV2B0 preprocessing (critical fix)
                face_input = np.expand_dims(face_rgb, axis=0).astype(np.float32)
                face_input = PREPROCESSOR(face_input)

                pred = model.predict(face_input, verbose=0)

                fake_prob = float(pred[0][FAKE_CLASS_INDEX])
                per_frame_fake_probs.append(fake_prob)

                pred_class = np.argmax(pred[0])
                if pred_class == FAKE_CLASS_INDEX:
                    fake_frame_count += 1

                total_frames_analysed += 1

                # ── Texture signal ──────────────────────────────────────────
                tex_score = compute_deepfake_texture_score(face, bbox)
                texture_scores.append(tex_score)

        success, frame = cap.read()
        frame_count += 1

    cap.release()

    if total_frames_analysed == 0:
        return {
            "error": "No faces could be detected in this video. "
                     "Ensure it contains clear, frontal faces."
        }

    # ── Aggregate signals ────────────────────────────────────────────────────
    fake_ratio    = fake_frame_count / total_frames_analysed
    avg_fake_prob = float(np.mean(per_frame_fake_probs))
    avg_texture   = float(np.mean(texture_scores)) if texture_scores else 0.5

    # ── Decision logic ───────────────────────────────────────────────────────
    #
    # Three zones:
    #   • fake_ratio < GREY_ZONE_LOW  (< 0.45) → definitely Real
    #   • fake_ratio > GREY_ZONE_HIGH (> 0.65) → definitely Fake
    #   • in between                            → use texture as tiebreaker
    #
    # The texture tiebreaker:
    #   Weighted combined score = 0.65 × model_vote + 0.35 × texture_vote
    #   If combined ≥ 0.50 → Fake, else → Real
    #
    if fake_ratio < GREY_ZONE_LOW:
        is_fake = False

    elif fake_ratio > GREY_ZONE_HIGH:
        is_fake = True

    else:
        # Grey zone — fuse model + texture
        combined_score = 0.65 * fake_ratio + 0.35 * avg_texture
        is_fake = combined_score >= FAKE_RATIO_THRESHOLD

    # ── Confidence & label ───────────────────────────────────────────────────
    if is_fake:
        # Confidence = weighted fusion, mapped to percentage
        confidence = round(
            (0.65 * fake_ratio + 0.35 * avg_texture) * 100, 2
        )
        label = "fake"
    else:
        real_ratio = 1.0 - fake_ratio
        real_texture = 1.0 - avg_texture
        confidence = round(
            (0.65 * real_ratio + 0.35 * real_texture) * 100, 2
        )
        label = "real"

    return {
        "prediction":       label,
        "confidence":       confidence,
        "low_confidence":   total_frames_analysed < MIN_FRAMES_FOR_DECISION,
        "raw_score":        round(fake_ratio, 4),
        "fake_probability": round(avg_fake_prob * 100, 2),
        "real_probability": round((1.0 - avg_fake_prob) * 100, 2),
        "frames_analysed":  total_frames_analysed,
        "fake_frame_ratio": round(fake_ratio * 100, 2),
        # Extra diagnostics (useful for debugging / frontend)
        "texture_score":    round(avg_texture * 100, 2),
    }