from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "supersecretkey" #have to change later
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp" :expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)