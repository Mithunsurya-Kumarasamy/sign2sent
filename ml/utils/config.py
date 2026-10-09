import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# MediaPipe Configuration
MP_HANDS_CONFIDENCE = 0.7
MP_MAX_HANDS = 1

# CNN Hyperparameters
CNN_INPUT_SIZE = (128, 128)
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
CNN_EPOCHS = 30

# LSTM Hyperparameters
SEQUENCE_LENGTH = 30
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 50
LSTM_DROPOUT = 0.5

# Inference Configuration
CONFIDENCE_THRESHOLD = 0.75
COOLDOWN_FRAMES = 15
SEQUENCE_TIMEOUT_FRAMES = 60
