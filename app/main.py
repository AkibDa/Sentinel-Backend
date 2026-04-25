from fastapi import FastAPI
from app.db import Base, engine
from app import tables
from app.routes import auth, api


app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(auth.router,prefix="/auth", tags=["Auth"])
app.include_router(api.router, tags=["API"])


@app.get("/")
def root():
    return {"message": "Sentinel AI backend running"}



