import numpy as np
import requests
from io import BytesIO
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input  

model = load_model("models/xception_deepfake_base.keras")


def predict_image_from_url(url: str):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return {"error": "Failed to download image"}

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            return {"error": f"URL is not an image (Content-Type: {content_type})"}

        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  

        prediction_score = float(model.predict(img_array)[0][0])

        if prediction_score >= 0.5:
            label = "real"
            confidence = float(prediction_score * 100)
        else:
            label = "fake"
            confidence = float((1.0 - prediction_score) * 100)

        print(f"Raw score: {prediction_score}")

        return {
            "label": label,
            "confidence": round(float(confidence), 2),
            "raw_score": round(float(prediction_score), 4),
            "real_probability": round(float(prediction_score * 100), 2),       
            "fake_probability": round(float((1 - prediction_score) * 100), 2)  
        }

    except Exception as e:
        return {"error": str(e)}


def predict_image_from_file(file_path: str):
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  

        prediction_score = float(model.predict(img_array)[0][0])

        if prediction_score >= 0.5:
            label = "real"
            confidence = float(prediction_score * 100)
        else:
            label = "fake"
            confidence = float((1.0 - prediction_score) * 100)

        return {
            "label": label,
            "confidence": round(float(confidence), 2),
            "raw_score": round(float(prediction_score), 4)
        }

    except Exception as e:
        return {"error": str(e)}