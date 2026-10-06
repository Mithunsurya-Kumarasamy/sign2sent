import torch
import numpy as np
import collections
import cv2

from ml.models.sign2sent import Sign2SentModel
from ml.preprocessing.preprocessing import get_base_transform
from ml.preprocessing.hand_detection import HandDetector
from ml.utils.config import MODELS_DIR, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, COOLDOWN_FRAMES
from ml.utils.vocabulary import get_num_classes, IDX_MAP

class SignRecognizer:
    def __init__(self, demo_mode=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.demo_mode = demo_mode
        self.num_classes = get_num_classes()
        self.feature_dim = 256
        self.hidden_size = 128
        self.num_layers = 2
        
        # Buffers and State
        self.frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.last_prediction = None
        self.cooldown_counter = 0
        self.transform = get_base_transform()
        
        if not self.demo_mode:
            self._load_model()
            
    def _load_model(self):
        self.model = Sign2SentModel(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        cnn_path = MODELS_DIR / "best_cnn.pth"
        lstm_path = MODELS_DIR / "best_lstm.pth"
        
        if cnn_path.exists() and lstm_path.exists():
            # Load CNN
            self.model.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device, weights_only=True))
            # Load LSTM
            self.model.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
            self.model.eval()
            self.demo_mode = False
        else:
            print("Models not found. Falling back to DEMO MODE.")
            self.demo_mode = True

    def process_frame(self, frame_rgb: np.ndarray):
        """
        Processes a single RGB frame (ideally cropped hand), updates the temporal buffer, 
        and returns a sign prediction if confidence and cooldown conditions are met.
        """
        # Preprocess
        tensor_frame = self.transform(frame_rgb)
        self.frame_buffer.append(tensor_frame)
        
        # Update cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            
        # Only predict if buffer is full
        if len(self.frame_buffer) == SEQUENCE_LENGTH:
            return self._predict_sequence()
            
        return None, 0.0

    def _predict_sequence(self):
        if self.demo_mode:
            # Fake prediction for demo
            # In a real scenario, this would just be ignored, but required per prompt if models missing
            import random
            if random.random() > 0.95 and self.cooldown_counter == 0:
                fake_idx = random.randint(0, self.num_classes - 1)
                self.last_prediction = IDX_MAP[fake_idx]
                self.cooldown_counter = COOLDOWN_FRAMES
                self.frame_buffer.clear()
                return self.last_prediction, 0.99
            return None, 0.0

        # Real Inference
        seq_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
            
            confidence_val = confidence.item()
            pred_idx_val = predicted_idx.item()
            
        if confidence_val >= CONFIDENCE_THRESHOLD:
            recognized_sign = IDX_MAP[pred_idx_val]
            
            # Duplicate suppression & Cooldown
            if recognized_sign != self.last_prediction or self.cooldown_counter == 0:
                self.last_prediction = recognized_sign
                self.cooldown_counter = COOLDOWN_FRAMES
                # Clear buffer to require a completely new sequence for the next sign
                self.frame_buffer.clear() 
                return recognized_sign, confidence_val
                
        return None, confidence_val
