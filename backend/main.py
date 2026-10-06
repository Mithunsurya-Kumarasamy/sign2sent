from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Sign2Sent API",
    description="Backend API for Sign2Sent Real-Time Sign Language Translation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api.routes_health import router as health_router
from backend.api.routes_prediction import router as prediction_router
from backend.api.routes_sentence import router as sentence_router
from backend.api.routes_metadata import router as metadata_router

app.include_router(health_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")
app.include_router(sentence_router, prefix="/api")
app.include_router(metadata_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to Sign2Sent API"}
