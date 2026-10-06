import os
from fastapi import APIRouter
from ml.utils.vocabulary import VOCABULARY
from ml.utils.config import CNN_INPUT_SIZE, SEQUENCE_LENGTH, RESULTS_DIR

router = APIRouter()

@router.get("/vocabulary")
def get_vocabulary():
    return {"vocabulary": VOCABULARY, "count": len(VOCABULARY)}

@router.get("/model-info")
def get_model_info():
    return {
        "cnn_input_size": CNN_INPUT_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "status": "Ready"
    }

@router.get("/metrics")
def get_metrics():
    # Read metrics from results directory if available
    metrics = {"cnn": None, "lstm": None}
    
    cnn_report = RESULTS_DIR / "cnn_classification_report.txt"
    if cnn_report.exists():
        with open(cnn_report, "r") as f:
            metrics["cnn"] = f.read()
            
    lstm_report = RESULTS_DIR / "lstm_classification_report.txt"
    if lstm_report.exists():
        with open(lstm_report, "r") as f:
            metrics["lstm"] = f.read()
            
    return {"metrics": metrics}
