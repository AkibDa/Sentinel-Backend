from fastapi import FastAPI
from app.routes import auth, api, news
from app.db import Base, engine

app = FastAPI(title="Sentinel AI", version="1.0.2")

Base.metadata.create_all(bind=engine)

app.include_router(auth.router,prefix="/auth", tags=["Auth"])
app.include_router(api.router, tags=["API"])
app.include_router(news.router, tags=["News"])

@app.get("/")
def root():
    return {"message": "Sentinel AI backend running"}
