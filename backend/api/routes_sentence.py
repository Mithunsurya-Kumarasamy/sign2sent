from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from backend.services.inference_service import inference_service

router = APIRouter()

class SequenceRequest(BaseModel):
    sequence: List[str]

class SpeakRequest(BaseModel):
    text: str

@router.post("/sentence")
def generate_sentence(req: SequenceRequest):
    sentence = inference_service.generate_sentence(req.sequence)
    return {"sentence": sentence}

@router.post("/speak")
def speak_sentence(req: SpeakRequest):
    inference_service.speak(req.text)
    return {"status": "Speaking initiated"}
