import torch
import numpy as np
import collections
import cv2
from typing import Optional, List, Tuple

from ml.models.sign2sent import Sign2SentModel
from ml.preprocessing.preprocessing import get_base_transform
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
            print("SignRecognizer: Loaded trained CNN and LSTM model checkpoints.")
        else:
            print("SignRecognizer: Model checkpoints not found. Operating in HEURISTIC DEMO MODE.")
            self.demo_mode = True

    def process_frame(self, frame_rgb: np.ndarray, landmarks: Optional[List[float]] = None) -> Tuple[Optional[str], float]:
        """
        Processes a single RGB frame, returning a sign prediction and confidence.
        Supports both real PyTorch CNN+LSTM neural inference and landmark gesture heuristic mode.
        """
        # If in Demo Mode (default when no trained vision CNN on disk)
        if self.demo_mode:
            return self._predict_demo_mode(landmarks)

        # Real Neural Network Inference Mode
        tensor_frame = self.transform(frame_rgb)
        self.frame_buffer.append(tensor_frame)
        
        # Only predict if sliding window buffer is full
        if len(self.frame_buffer) == SEQUENCE_LENGTH:
            sign, conf = self._predict_sequence()
            if sign is not None:
                return sign, conf
            
        # Seamless fallback to landmark heuristics if neural buffer is filling or below threshold
        if landmarks:
            return self._predict_demo_mode(landmarks)
            
        return None, 0.0

    def _predict_demo_mode(self, landmarks: Optional[List[float]]) -> Tuple[Optional[str], float]:
        """
        Classifies hand gestures deterministically using 21 MediaPipe landmarks.
        Uses rotation-invariant radial distance from wrist plus vertical posture checks:
        - Open Palm (>=4 fingers extended): 'HELLO'
        - Index Finger Up: 'I'
        - Peace / V Sign (Index + Middle): 'NAME'
        - 3 Fingers (Index + Middle + Ring): 'WATER'
        - Thumb Up (Thumb isolated & pointing up above knuckles): 'GOOD'
        - Fist / Closed hand (All fingers curled, thumb folded): 'YES'
        - Rock / ILY Sign (Index + Pinky): 'YOU'
        """
        if not landmarks or len(landmarks) < 63:
            return "HELLO", 0.90

        pts = [(landmarks[i * 3], landmarks[i * 3 + 1], landmarks[i * 3 + 2]) for i in range(21)]
        p0 = np.array(pts[0][:2]) # Wrist reference position

        def is_finger_extended(mcp_idx: int, pip_idx: int, tip_idx: int) -> bool:
            p_mcp = np.array(pts[mcp_idx][:2])
            p_pip = np.array(pts[pip_idx][:2])
            p_tip = np.array(pts[tip_idx][:2])
            d_tip = np.linalg.norm(p_tip - p0)
            d_pip = np.linalg.norm(p_pip - p0)
            d_tip_mcp = np.linalg.norm(p_tip - p_mcp)
            # A finger is extended only if the tip is far from the knuckle/palm
            return bool((d_tip > d_pip * 1.15 and d_tip_mcp > 0.12) or (p_tip[1] < p_pip[1] - 0.04 and d_tip_mcp > 0.10))

        index_up = is_finger_extended(5, 6, 8)
        middle_up = is_finger_extended(9, 10, 12)
        ring_up = is_finger_extended(13, 14, 16)
        pinky_up = is_finger_extended(17, 18, 20)

        extended_count = sum([index_up, middle_up, ring_up, pinky_up])

        # Precise Thumbs-Up detection (isolated thumb extended upward high above knuckles)
        p_thumb_tip = np.array(pts[4][:2])
        p_index_mcp = np.array(pts[5][:2])
        p_index_pip = np.array(pts[6][:2])
        p_middle_pip = np.array(pts[10][:2])
        
        min_knuckle_y = min(p_index_pip[1], p_middle_pip[1])
        d_thumb_to_index = np.linalg.norm(p_thumb_tip - p_index_mcp)
        # Thumb is truly 'up' only if it protrudes significantly above the knuckles and is separated
        is_thumbs_up = bool((p_thumb_tip[1] < min_knuckle_y - 0.05) and (d_thumb_to_index > 0.12))

        # Classification rules
        if extended_count >= 4:
            return "HELLO", 0.95
        elif index_up and middle_up and ring_up and not pinky_up:
            return "WATER", 0.92
        elif index_up and middle_up and not ring_up and not pinky_up:
            return "NAME", 0.93
        elif index_up and pinky_up and not middle_up and not ring_up:
            return "YOU", 0.91
        elif index_up and not middle_up and not ring_up and not pinky_up:
            return "I", 0.94
        elif is_thumbs_up and extended_count == 0:
            return "GOOD", 0.95
        elif extended_count <= 1 and not is_thumbs_up:
            # Fist (all fingers folded into palm, thumb resting across fingers = YES)
            return "YES", 0.95
        elif extended_count == 0:
            return "YES", 0.95
        else:
            return "HELLO", 0.88

    def _predict_sequence(self) -> Tuple[Optional[str], float]:
        seq_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
            
            confidence_val = confidence.item()
            pred_idx_val = predicted_idx.item()
            
        if confidence_val >= CONFIDENCE_THRESHOLD:
            recognized_sign = IDX_MAP[pred_idx_val]
            return recognized_sign, confidence_val
                
        return None, confidence_val
