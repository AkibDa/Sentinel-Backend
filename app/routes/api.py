import os
import uuid
import shutil
from fastapi import APIRouter, Header, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app import models
from app.auth import get_current_user
from app.schemas import AnalyseRequest
from app.services.video_detector import predict_video
from app.services.downloader import download_file
from app.image_model import predict_image_from_url


router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/analyse/video-url")
def analyse_video_url(
    request: AnalyseRequest,
    x_api_key: str = Header(...),
    db: Session = Depends(get_db)
):
    key = db.query(models.APIKey).filter(models.APIKey.key == x_api_key).first()

    if not key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    try:
        temp_path = download_file(request.input)

        prediction = predict_video(temp_path)

        result = prediction["prediction"]
        confidence = str(prediction["confidence"])

        if "error" in prediction:
            raise HTTPException(status_code=400, detail= prediction["error"])

        scan = models.Scan(
            user_id=key.user_id,
            input_data=request.input,
            result=result,
            confidence=confidence
        )
        db.add(scan)
        db.commit()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return{
            "type": "video-url",
            "result": result,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@router.post("/analyse/analyse_upload")
async def analyse_upload(
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
    db: Session = Depends(get_db)
):
    key = db.query(models.APIKey).filter(models.APIKey.key == x_api_key).first()

    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    temp_path = f"temp_{uuid.uuid4()}.mp4"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction =  predict_video(temp_path)

        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])

        result = prediction["prediction"]
        confidence = str(prediction["confidence"])

        scan = models.Scan(
            user_id = key.user_id,
            input_data = file.filename,
            result = result,
            confidence = confidence
        )
        db.add(scan)
        db.commit()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return{
            "type": "video",
            "result": result,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/analyse/image-url")
def analyse_image_url(
    request: AnalyseRequest,
    x_api_key: str = Header(...),
    db: Session = Depends(get_db)
):
    key = db.query(models.APIKey).filter(models.APIKey.key == x_api_key).first()

    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    temp_path = f"temp_{uuid.uuid4()}.mp4"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction = predict_video(temp_path)

        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])
        
        result = prediction["prediction"]
        confidence = str(prediction["confidence"])

        scan = models.Scan(
            user_id=key.user_id,
            input_data=request.input,
            result=result,
            confidence=confidence
        )
        db.add(scan)
        db.commit()

        return {
            "type": "image-url",
            "result": result,
            "confidence": confidence,
            "raw_score": prediction["raw_score"]  
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
def get_history(
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    scans = db.query(models.Scan).filter(models.Scan.user_id == user_id)\
        .order_by(models.Scan.created_at.desc()).all()

    return scans