import torch
import numpy as np
import collections
import cv2
from typing import Optional, List, Tuple

from ml.models.sign2sent import Sign2SentModel
from ml.preprocessing.preprocessing import get_base_transform
from ml.utils.config import MODELS_DIR, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, COOLDOWN_FRAMES, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS
from ml.utils.vocabulary import get_num_classes, IDX_MAP

class SignRecognizer:
    def __init__(self, demo_mode=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.demo_mode = demo_mode
        self.num_classes = get_num_classes()
        self.feature_dim = 256
        self.hidden_size = LSTM_HIDDEN_SIZE
        self.num_layers = LSTM_NUM_LAYERS
        
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

    def reset_buffer(self):
        """Clears the temporal frame buffer when a sign ends or hand leaves frame."""
        self.frame_buffer.clear()

    def process_frame(self, frame_rgb: np.ndarray, landmarks: Optional[List[float]] = None) -> Tuple[Optional[str], float]:
        """
        Processes a single RGB frame, returning a sign prediction and confidence.
        Uses hybrid ensemble of real PyTorch CNN+LSTM neural network and landmark geometric analysis.
        """
        # Always compute geometric landmark gesture if landmarks exist
        lm_sign, lm_conf = self._predict_demo_mode(landmarks) if landmarks else (None, 0.0)

        # If in Pure Demo Mode (no trained checkpoints on disk)
        if self.demo_mode:
            return lm_sign, lm_conf

        # Real Neural Network Inference Mode
        tensor_frame = self.transform(frame_rgb)
        self.frame_buffer.append(tensor_frame)
        
        # When we have at least 4 frames (~1 sec at 4 FPS), run neural inference with padding if needed
        neural_sign, neural_conf = None, 0.0
        if len(self.frame_buffer) >= 4:
            neural_sign, neural_conf = self._predict_sequence()

        # Hybrid Decision:
        # 1. If landmark detects a distinctive hand pose with high confidence (>0.90)
        # e.g. HELLO, YES, GOOD, I, YOU, WATER, NAME, NO
        if lm_sign and lm_conf >= 0.90:
            # If neural model strongly predicts something specific (>0.75), give neural precedence,
            # otherwise trust landmark geometry
            if neural_sign and neural_conf >= 0.75:
                return neural_sign, neural_conf
            return lm_sign, lm_conf

        # 2. If neural model has a valid prediction meeting confidence threshold
        if neural_sign and neural_conf >= CONFIDENCE_THRESHOLD:
            return neural_sign, neural_conf

        # 3. Fallback to landmark gesture if available
        if lm_sign:
            return lm_sign, lm_conf

        return None, 0.0

    def _predict_demo_mode(self, landmarks: Optional[List[float]]) -> Tuple[Optional[str], float]:
        """
        Classifies hand gestures deterministically using 21 MediaPipe landmarks.
        Provides stable, repeatable sign outputs ideal for viva demonstration and testing:
        - Open Palm (>=4 fingers extended): 'HELLO'
        - Index Finger Up: 'I'
        - Peace / V Sign (Index + Middle): 'NAME'
        - 3 Fingers (Index + Middle + Ring): 'WATER'
        - Thumb Up: 'GOOD'
        - Fist (Closed hand): 'YES'
        - Rock Sign (Index + Pinky): 'YOU'
        - Snapping Beak (Thumb close to Index + Middle): 'NO'
        """
        if not landmarks or len(landmarks) < 63:
            return None, 0.0

        pts = [(landmarks[i * 3], landmarks[i * 3 + 1], landmarks[i * 3 + 2]) for i in range(21)]

        # Vertical extension checks (y decreases upward in normalized screen coordinates)
        index_up = pts[8][1] < pts[6][1]
        middle_up = pts[12][1] < pts[10][1]
        ring_up = pts[16][1] < pts[14][1]
        pinky_up = pts[20][1] < pts[18][1]

        thumb_dist = np.linalg.norm(np.array(pts[4][:2]) - np.array(pts[5][:2]))
        thumb_up = pts[4][1] < pts[3][1] and thumb_dist > 0.07

        extended_count = sum([index_up, middle_up, ring_up, pinky_up])

        # Distance between thumb tip and index/middle tips
        thumb_tip = np.array(pts[4][:2])
        index_tip = np.array(pts[8][:2])
        middle_tip = np.array(pts[12][:2])
        d_index = np.linalg.norm(thumb_tip - index_tip)
        d_middle = np.linalg.norm(thumb_tip - middle_tip)
        is_pinching_no = (d_index < 0.08 and d_middle < 0.08 and not ring_up and not pinky_up)

        if is_pinching_no:
            return "NO", 0.94
        elif extended_count >= 4:
            return "HELLO", 0.95
        elif index_up and middle_up and not ring_up and not pinky_up:
            return "NAME", 0.93
        elif index_up and middle_up and ring_up and not pinky_up:
            return "WATER", 0.92
        elif index_up and not middle_up and not ring_up and not pinky_up:
            return "I", 0.94
        elif index_up and pinky_up and not middle_up and not ring_up:
            return "YOU", 0.91
        elif thumb_up and extended_count == 0:
            return "GOOD", 0.94
        elif extended_count == 0:
            return "YES", 0.91
        else:
            return "HELLO", 0.88

    def _predict_sequence(self) -> Tuple[Optional[str], float]:
        frames = list(self.frame_buffer)
        if not frames:
            return None, 0.0

        # If buffer has fewer than SEQUENCE_LENGTH frames, pad with the last frame
        if len(frames) < SEQUENCE_LENGTH:
            frames = frames + [frames[-1]] * (SEQUENCE_LENGTH - len(frames))

        seq_tensor = torch.stack(frames).unsqueeze(0).to(self.device)
        
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
