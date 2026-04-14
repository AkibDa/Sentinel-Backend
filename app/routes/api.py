from fastapi import APIRouter, Header, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app import models

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/detect")
def detect(input_data: str, x_api_key: str = Header(...), db: Session = Depends(get_db)):

    key = db.query(models.APIKey).filter(models.APIKey.key == x_api_key).first()

    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    result = "fake"   #later replace with ai model
    confidence = "0.87"

    scan = models.Scan(
        user_id=key.user_id,
        input_data=input_data,
        result=result,
        confidence=confidence
    )
    db.add(scan)
    db.commit()

    return {
        "result": result,
        "confidence": confidence
    }