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
        self.history = collections.deque(maxlen=4)
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
        """Clears temporal sequence buffer and smoothing history when hand leaves frame."""
        self.frame_buffer.clear()
        self.history.clear()

    def process_frame(self, frame_rgb: np.ndarray, landmarks: Optional[List[float]] = None) -> Tuple[Optional[str], float]:
        """
        Processes a single RGB frame, returning a sign prediction and confidence.
        Uses hybrid ensemble of real PyTorch CNN+LSTM neural network and landmark geometric analysis
        with temporal consensus smoothing.
        """
        # 1. Compute robust anatomical landmark gesture
        lm_sign, lm_conf = self._predict_demo_mode(landmarks) if landmarks else (None, 0.0)

        # In pure demo mode
        if self.demo_mode:
            return self._apply_smoothing(lm_sign, lm_conf)

        # 2. Real Neural Network Sequence Mode
        tensor_frame = self.transform(frame_rgb)
        self.frame_buffer.append(tensor_frame)
        
        neural_sign, neural_conf = None, 0.0
        if len(self.frame_buffer) >= 4:
            neural_sign, neural_conf = self._predict_sequence()

        # 3. Hybrid Decision Logic:
        candidate_sign, candidate_conf = None, 0.0

        # Prioritize landmark geometric confirmation (cleanest, background-invariant)
        if lm_sign and lm_conf >= 0.90:
            if neural_sign and neural_conf >= 0.80 and neural_sign != lm_sign:
                candidate_sign, candidate_conf = neural_sign, neural_conf
            else:
                candidate_sign, candidate_conf = lm_sign, lm_conf
        elif neural_sign and neural_conf >= CONFIDENCE_THRESHOLD:
            candidate_sign, candidate_conf = neural_sign, neural_conf
        elif lm_sign:
            candidate_sign, candidate_conf = lm_sign, lm_conf

        # 4. Temporal Smoothing
        return self._apply_smoothing(candidate_sign, candidate_conf)

    def _apply_smoothing(self, candidate_sign: Optional[str], candidate_conf: float) -> Tuple[Optional[str], float]:
        """Applies rolling window consensus to eliminate single-frame jitter."""
        if not candidate_sign:
            if self.history:
                self.history.popleft()
            return None, 0.0

        self.history.append(candidate_sign)
        counts = collections.Counter(self.history)
        most_common_sign, count = counts.most_common(1)[0]

        if count >= 2:
            return most_common_sign, max(candidate_conf, 0.91)
        return candidate_sign, candidate_conf * 0.9

    def _predict_demo_mode(self, landmarks: Optional[List[float]]) -> Tuple[Optional[str], float]:
        """
        Classifies hand gestures deterministically using 21 MediaPipe landmarks with
        radial distance and angle invariance. Never falls back to arbitrary defaults.
        """
        if not landmarks or len(landmarks) < 63:
            return None, 0.0

        pts = [np.array([landmarks[i * 3], landmarks[i * 3 + 1]]) for i in range(21)]
        wrist = pts[0]

        # Radial extension test (works regardless of hand rotation or tilt)
        def is_ext(tip_idx, pip_idx):
            d_tip = np.linalg.norm(pts[tip_idx] - wrist)
            d_pip = np.linalg.norm(pts[pip_idx] - wrist)
            return (d_tip > 1.25 * d_pip) or (pts[tip_idx][1] < pts[pip_idx][1] and d_tip > d_pip)

        index_up = is_ext(8, 6)
        middle_up = is_ext(12, 10)
        ring_up = is_ext(16, 14)
        pinky_up = is_ext(20, 18)

        d_thumb_wrist = np.linalg.norm(pts[4] - wrist)
        thumb_up = (pts[4][1] < pts[3][1] < pts[2][1] and pts[4][1] < pts[0][1] - 0.12 and d_thumb_wrist > 0.16)

        extended_count = sum([index_up, middle_up, ring_up, pinky_up])

        # Key finger interaction distances
        d_thumb_index = np.linalg.norm(pts[4] - pts[8])
        d_thumb_middle = np.linalg.norm(pts[4] - pts[12])
        d_thumb_ring = np.linalg.norm(pts[4] - pts[16])
        d_index_middle = np.linalg.norm(pts[8] - pts[12])

        # 1. HELLO: Open palm (4 or 5 fingers extended spread apart)
        if extended_count >= 4:
            return "HELLO", 0.96

        # 2. GOOD: Thumbs up (thumb pointing up, all 4 fingers curled)
        if thumb_up and extended_count == 0:
            return "GOOD", 0.97

        # 3. YES: Closed fist (all 4 fingers curled, thumb folded)
        if extended_count == 0:
            return "YES", 0.94

        # 4. WATER: Three fingers 'W' (index, middle, ring extended, pinky curled)
        if index_up and middle_up and ring_up and not pinky_up:
            return "WATER", 0.95

        # 5. NO: Snapping beak (index & middle reaching thumb, pinched tips, ring & pinky curled)
        if (index_up or middle_up) and d_thumb_index < 0.07 and d_thumb_middle < 0.07 and not ring_up and not pinky_up:
            return "NO", 0.96

        # 6. NAME: Peace / V sign (index and middle extended, ring & pinky curled)
        if index_up and middle_up and not ring_up and not pinky_up and d_index_middle > 0.03:
            return "NAME", 0.95

        # 7. I: Single index finger up OR single pinky finger up
        if (index_up and not middle_up and not ring_up and not pinky_up) or (pinky_up and not index_up and not middle_up and not ring_up):
            return "I", 0.96

        # 8. YOU: Rock sign (index and pinky up, middle & ring curled)
        if index_up and pinky_up and not middle_up and not ring_up:
            return "YOU", 0.94

        # 9. FOOD: All fingertips pinched to thumb
        if d_thumb_index < 0.08 and d_thumb_middle < 0.08 and d_thumb_ring < 0.08:
            return "FOOD", 0.94

        # No arbitrary false-positive fallback
        return None, 0.0

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
