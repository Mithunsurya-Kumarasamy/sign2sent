from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.services.inference_service import inference_service

router = APIRouter()

@router.post("/predict")
async def predict_frame(file: UploadFile = File(...)):
    """
    Accepts an uploaded image frame, processes it, and returns the current sign state.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    image_bytes = await file.read()
    result = inference_service.process_image(image_bytes)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result
