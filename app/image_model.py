import numpy as np
import requests
from io import BytesIO
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# ✅ Load model once (VERY IMPORTANT)
model = load_model("models/xception_deepfake_base.keras")


def predict_image_from_url(url: str):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction_score = model.predict(img_array)[0][0]

        if prediction_score >= 0.5:
            label = "Real"
            confidence = float(prediction_score * 100)
        else:
            label = "Fake"
            confidence = float((1.0 - prediction_score) * 100)

        return {
            "label": label,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }