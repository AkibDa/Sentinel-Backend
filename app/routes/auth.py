from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app import schemas, tables
from app.utils import hash_password, verify_password, generate_api_key
from app.auth import create_token
from app.auth import get_current_user
from app.utils import generate_api_key

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register")
def register(user: schemas.UserCreate, db:Session = Depends(get_db)):
    existing_user = db.query(tables.User).filter(tables.User.email == user.email).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed = hash_password(user.password)

    new_user = tables.User(email=user.email, password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    #generating api key
    key = generate_api_key()
    api_key = tables.APIKey(key=key, user_id=new_user.id)
    db.add(api_key)
    db.commit()


    return{
        "message": "User created",
        "api_key": key
    }

@router.post("/login")
def login(user: schemas.UserLogin, db:Session = Depends(get_db)):
    db_user = db.query(tables.User).filter(tables.User.email == user.email).first()

    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token({"user_id" : db_user.id})
    api_key = db.query(tables.APIKey).filter(tables.APIKey.user_id == db_user.id).first()

    return {
        "access_token": token,
        "api_key" : api_key
    }


@router.post("/regenerate-key")
def regenerate_api_key(
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    api_key = db.query(tables.APIKey).filter(tables.APIKey.user_id == user_id).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    new_key = generate_api_key()
    api_key.key = new_key

    db.commit()
    
    return{
        "message" : "API key regenerated",
        "api_key" : new_key
    }