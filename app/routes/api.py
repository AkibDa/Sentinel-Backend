import os
import uuid
import shutil
import requests
from fastapi import APIRouter, Header, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app import models
from typing import Optional
from app.auth import get_current_user
from app.schemas import AnalyseRequest
from app.services.video_detector import predict_video
from app.services.downloader import download_file
from app.image_model import predict_image_from_url, predict_image_from_file


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
    
@router.post("/analyse/url")
def analyse_url(
    request: AnalyseRequest,
    x_api_key: str = Header(...),
    db: Session = Depends(get_db)
):
    key = db.query(models.APIKey).filter(models.APIKey.key == x_api_key).first()
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    target_url = request.input
    media_type = request.type

    if "instagram.com" in target_url:
        n8n_webhook = "http://localhost:5678/webhook/8b86b827-299b-4004-9242-4381cbcc3043" 
        
        # Ask n8n to fetch the raw media URL
        n8n_response = requests.post(n8n_webhook, json={"url": target_url})
        
        # Print the exact n8n response to your terminal for debugging!
        if n8n_response.status_code != 200:
            print(f"N8N ERROR: {n8n_response.status_code} - {n8n_response.text}")
            raise HTTPException(status_code=400, detail=f"n8n failed: {n8n_response.text}")
        
        try:
            n8n_data = n8n_response.json()
            target_url = n8n_data.get("raw_url")
            
            if not target_url:
                raise HTTPException(status_code=400, detail="n8n succeeded but didn't return a 'raw_url'")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse n8n response: {str(e)}")
        
    try:
        if media_type == "image":
            prediction = predict_image_from_url(target_url)
            if "error" in prediction:
                raise HTTPException(status_code=400, detail=prediction["error"])
            
            result = prediction["label"].lower()
            confidence = str(prediction["confidence"])


        elif media_type == "video":
            temp_path = download_file(target_url)
            prediction = predict_video(temp_path)

            if "error" in prediction:
                raise HTTPException(status_code=400, detail=prediction["error"])
                
            result = prediction["prediction"].lower()
            confidence = str(prediction["confidence"])
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

        else:
            raise HTTPException(status_code=400, detail="Invalid media type specified. Use 'image' or 'video'. ")
        
        scan = models.Scan(
            user_id = key.user_id,
            input_data = request.input,
            media_type=media_type,
            result=result,
            confidence=confidence
        )
        db.add(scan)
        db.commit()

        return{
            "type": media_type,
            "original_url": request.input,
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
    
    file_ext = ".mp4" if file.content_type.startswith("video") else ".jpg"

    temp_path = f"temp_{uuid.uuid4()}{file_ext}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.content_type.startswith("video/"):
            prediction = predict_video(temp_path)
            media_type = "video"
            if "error" in prediction:
                raise HTTPException(status_code=400, detail=prediction["error"])
            result = prediction["prediction"].lower()

        elif file.content_type.startswith("image/"):
            prediction = predict_image_from_file(temp_path)
            media_type = "image"
            if "error" in prediction:
                raise HTTPException(status_code=400, detail=prediction["error"])
            result = prediction["label"].lower()

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image or video.")
        
        confidence = str(prediction["confidence"])

        scan = models.Scan(
            user_id = key.user_id,
            input_data = file.filename,
            media_type = media_type,
            result = result,
            confidence = confidence
        )
        db.add(scan)
        db.commit()

        return{
            "type": "media_type",
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

    try:
        prediction = predict_image_from_url(request.input)

        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])

        result = prediction["label"].lower()
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
    media_type: Optional[str] = None,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(models.Scan).filter(models.Scan.user_id == user_id)

    if media_type:
        query = query.filter(models.Scan.media_type == media_type.lower())

    scans = query.order_by(models.Scan.created_at.desc()).all()

    return scans