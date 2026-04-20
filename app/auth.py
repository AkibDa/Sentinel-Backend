from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.orm import Session
from typing import Optional
import os
from app.db import SessionLocal
from app import models

SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")  # move to .env before deploying
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# These two schemes register properly with Swagger UI:
# - jwt_scheme     → adds the "Authorize" button with Bearer token input
# - api_key_scheme → adds the "Authorize" button with x-api-key input
jwt_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="x-api-key", auto_error=False)


# --- DB Dependency ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Token creation ---

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# --- JWT verification ---

def verify_jwt_and_get_user_id(token: str) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token: missing user_id")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# --- Unified auth dependency (JWT or API key) ---

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(jwt_scheme),
    x_api_key: Optional[str] = Security(api_key_scheme),
    db: Session = Depends(get_db)
) -> int:
    # Try JWT first
    if credentials and credentials.credentials:
        return verify_jwt_and_get_user_id(credentials.credentials)

    # Fall back to API key
    if x_api_key:
        api_key = db.query(models.APIKey).filter(
            models.APIKey.key == x_api_key
        ).first()
        if not api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return api_key.user_id

    raise HTTPException(status_code=401, detail="Not authenticated")