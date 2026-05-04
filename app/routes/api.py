import os
import uuid
import shutil
import tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app import tables
from typing import Optional
from app.auth import get_current_user, get_db
from app.schemas import AnalyseRequest
from app.detectors.video_detector import predict_video
from app.detectors.image_model import predict_image_from_url, predict_image_from_file
from app.services.ytdlp_service import download_media_ytdlp
from app.services.image_scraper import get_raw_image_url

router = APIRouter()

def finalize_scan_response(
    db: Session,
    user_id: int,
    input_data: str,
    media_type: str,
    prediction: dict,
    is_url: bool = False,
):
    if "label" in prediction:
        result = prediction["label"].lower()
    elif "prediction" in prediction:
        result = prediction["prediction"].lower()
    else:
        result = "unknown"

    confidence = str(prediction["confidence"])

    scan = tables.Scan(
        user_id=user_id,
        input_data=input_data,
        media_type=media_type,
        result=result,
        confidence=confidence,
    )
    db.add(scan)
    db.commit()

    response = {
        "type": media_type,
        "result": result,
        "confidence": confidence,
        "raw_score": prediction.get("raw_score"),
        "low_confidence": prediction.get("low_confidence", False),
    }

    if media_type == "video":
        response["frames_analysed"]  = prediction.get("frames_analysed")
        response["fake_frame_ratio"] = prediction.get("fake_frame_ratio")
        response["fake_probability"] = prediction.get("fake_probability")
        response["real_probability"] = prediction.get("real_probability")
        # NEW: confidence band (HIGH / MEDIUM / LOW) and threshold used
        response["confidence_band"]  = prediction.get("confidence_band")
        response["threshold_used"]   = prediction.get("threshold_used")

    if is_url:
        response["original_url"] = input_data

    return response

@router.post("/analyse/url")
def analyse_url(
    request: AnalyseRequest,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    target_url = request.input
    media_type = request.type
    temp_path  = None

    try:
        if media_type == "video":
            temp_path  = download_media_ytdlp(target_url)
            prediction = predict_video(temp_path)

        elif media_type == "image":
            if any(
                domain in target_url
                for domain in ["instagram.com", "twitter.com", "reddit.com", "x.com"]
            ):
                raw_url = get_raw_image_url(target_url)
            else:
                raw_url = target_url

            prediction = predict_image_from_url(raw_url)

        else:
            raise HTTPException(status_code=400, detail="Invalid media type specified.")

        if prediction.get("error"):
            raise HTTPException(status_code=400, detail=prediction["error"])

        return finalize_scan_response(db, user_id, target_url, media_type, prediction, is_url=True)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                import time
                time.sleep(0.5)
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


@router.post("/analyse/analyse_upload")
async def analyse_upload(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        ext = ".mp4" if (file.content_type or "").startswith("video") else ".jpg"

    temp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4()}{ext.lower()}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if (file.content_type or "").startswith("video/"):
            prediction = predict_video(temp_path)
            media_type = "video"
        elif (file.content_type or "").startswith("image/"):
            prediction = predict_image_from_file(temp_path)
            media_type = "image"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        if prediction.get("error"):
            raise HTTPException(status_code=400, detail=prediction["error"])

        return finalize_scan_response(db, user_id, file.filename, media_type, prediction)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                import time
                time.sleep(0.5)
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


@router.get("/history")
def get_history(
    media_type: Optional[str] = None,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(tables.Scan).filter(tables.Scan.user_id == user_id)

    if media_type:
        query = query.filter(tables.Scan.media_type == media_type.lower())

    scans = query.order_by(tables.Scan.created_at.desc()).all()

    return scans
