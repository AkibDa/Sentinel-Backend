import os
import uuid
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app import models
from typing import Optional
from app.auth import get_current_user
from app.schemas import AnalyseRequest
from app.services.video_detector import predict_video
from app.image_model import predict_image_from_url, predict_image_from_file
from app.services.ytdlp_service import download_media_ytdlp
from app.services.image_scraper import get_raw_image_url

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def finalize_scan_response(db: Session, user_id: int, input_data: str, media_type: str, prediction: dict, is_url: bool = False):
    """Handles DB saving and response formatting for both URL and File endpoints."""
    
    if "label" in prediction:
        result = prediction["label"].lower()
    elif "prediction" in prediction:
        result = prediction["prediction"].lower()
    else:
        result = "unknown"
        
    confidence = str(prediction["confidence"])

    scan = models.Scan(
        user_id=user_id,
        input_data=input_data,
        media_type=media_type,
        result=result,
        confidence=confidence
    )
    db.add(scan)
    db.commit()

    response = {
        "type": media_type,
        "result": result,
        "confidence": confidence,
        "raw_score": prediction.get("raw_score")
    }
    
    if is_url:
        response["original_url"] = input_data

    return response



@router.post("/analyse/url")
def analyse_url(
    request: AnalyseRequest,
    user_id: int = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    target_url = request.input
    media_type = request.type
    temp_path = None

    try:
        if media_type == "video":
            temp_path = download_media_ytdlp(target_url)
            prediction = predict_video(temp_path)

        elif media_type == "image":
            if any(domain in target_url for domain in ["instagram.com", "twitter.com", "reddit.com", "x.com"]):
                raw_url = get_raw_image_url(target_url)
            else:
                raw_url = target_url

            prediction = predict_image_from_url(raw_url)

        else:
            raise HTTPException(status_code=400, detail="Invalid media type specified.")
        
        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])

        return finalize_scan_response(db, user_id, target_url, media_type, prediction, is_url=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/analyse/analyse_upload")
async def analyse_upload(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    file_ext = ".mp4" if file.content_type.startswith("video") else ".jpg"
    temp_path = f"temp_{uuid.uuid4()}{file_ext}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.content_type.startswith("video/"):
            prediction = predict_video(temp_path)
            media_type = "video"

        elif file.content_type.startswith("image/"):
            prediction = predict_image_from_file(temp_path)
            media_type = "image"

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image or video.")
        
        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])

        return finalize_scan_response(db, user_id, file.filename, media_type, prediction, is_url=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/history")
def get_history(
    media_type: Optional[str] = None,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(models.Scan).filter(models.Scan.user_id == user_id)

    if media_type:
        query = query.filter(models.Scan.media_type == media_type.lower())

    scans = query.order_by(models.Scan.created_at.desc()).all()

    return scans