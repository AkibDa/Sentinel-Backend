from fastapi import FastAPI
from app.db import Base, engine
from app import models
from app.routes import auth, api
from app.image_model import predict_image_from_url


app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(auth.router,prefix="/auth", tags=["Auth"])
app.include_router(api.router, tags=["API"])


@app.get("/")
def root():
    return {"message": "Sentinel AI backend running"}

@app.post("/analyze_url")
async def analyze_url(data: dict):
    url = data.get("url")

    result = predict_image_from_url(url)

    return{
        "input_url": url,
        "prediction": result
    }

@app.post("/analyze_video")
async def analyze_video():
    return {"result": "video processed"}


