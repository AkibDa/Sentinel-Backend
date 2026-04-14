import secrets
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password:str, hashed: str):
    return pwd_context.verify(password,hashed)

def generate_api_key():
    return secrets.token_hex(32)