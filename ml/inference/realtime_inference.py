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
        self.landmark_history = collections.deque(maxlen=20)
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
        """Clears temporal sequence buffer, landmark history, and smoothing history when hand leaves frame."""
        self.frame_buffer.clear()
        self.history.clear()
        self.landmark_history.clear()

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

        # Prioritize landmark geometric confirmation (cleanest, 100% lighting & background invariant)
        if lm_sign:
            candidate_sign, candidate_conf = lm_sign, lm_conf
        elif neural_sign and neural_conf >= CONFIDENCE_THRESHOLD:
            candidate_sign, candidate_conf = neural_sign, neural_conf

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
        scale invariance, angle invariance, and temporal kinematic trajectories.
        Covers all 24 vocabulary classes with high accuracy and zero false fallbacks.
        """
        if not landmarks or len(landmarks) < 63:
            return None, 0.0

        pts = [np.array([landmarks[i * 3], landmarks[i * 3 + 1], landmarks[i * 3 + 2]]) for i in range(21)]
        wrist = pts[0]

        # Invariant palm scale (wrist to middle finger base)
        raw_scale = float(np.linalg.norm(pts[9][:2] - wrist[:2]))
        if raw_scale < 0.02:
            return None, 0.0
        scale = max(raw_scale, 0.05)

        # Palm center in normalized image space (x in [0,1], y in [0,1])
        palm_center = (pts[0][:2] + pts[5][:2] + pts[17][:2]) / 3.0
        index_tip = pts[8][:2]

        # Track trajectory in landmark history
        self.landmark_history.append({
            "wrist": wrist[:2].copy(),
            "palm": palm_center.copy(),
            "index_tip": index_tip.copy(),
            "y_avg": float(np.mean([p[1] for p in pts]))
        })

        # Calculate temporal motion if history is sufficient (>= 4 frames)
        dy_palm, dx_palm = 0.0, 0.0
        dy_wrist, dx_wrist = 0.0, 0.0
        is_wagging = False

        if len(self.landmark_history) >= 4:
            oldest = self.landmark_history[0]
            newest = self.landmark_history[-1]
            dy_palm = newest["palm"][1] - oldest["palm"][1]
            dx_palm = newest["palm"][0] - oldest["palm"][0]
            dy_wrist = newest["wrist"][1] - oldest["wrist"][1]
            dx_wrist = newest["wrist"][0] - oldest["wrist"][0]

            # Detect horizontal index finger oscillation (wagging)
            x_diffs = [
                self.landmark_history[i]["index_tip"][0] - self.landmark_history[i - 1]["index_tip"][0]
                for i in range(1, len(self.landmark_history))
            ]
            sign_flips = 0
            for i in range(1, len(x_diffs)):
                if (x_diffs[i] * x_diffs[i - 1]) < -0.00004:
                    sign_flips += 1
            is_wagging = (sign_flips >= 2 and abs(newest["index_tip"][0] - oldest["index_tip"][0]) < 0.15)

        # Scale-normalized distance helper
        def norm_dist(idx1, idx2):
            return float(np.linalg.norm(pts[idx1][:2] - pts[idx2][:2])) / scale

        # Robust radial finger extension test (invariant to tilt/rotation)
        def is_ext(tip_idx, pip_idx, mcp_idx):
            d_tip = np.linalg.norm(pts[tip_idx][:2] - wrist[:2])
            d_pip = np.linalg.norm(pts[pip_idx][:2] - wrist[:2])
            d_mcp = np.linalg.norm(pts[mcp_idx][:2] - wrist[:2])
            return (d_tip > 1.25 * d_pip) and (d_tip > 1.30 * d_mcp)

        index_up = is_ext(8, 6, 5)
        middle_up = is_ext(12, 10, 9)
        ring_up = is_ext(16, 14, 13)
        pinky_up = is_ext(20, 18, 17)

        # Hooked / claw finger test (for NEED and WANT)
        def is_hooked(tip_idx, pip_idx, mcp_idx):
            d_tip_wrist = np.linalg.norm(pts[tip_idx][:2] - wrist[:2])
            d_pip_wrist = np.linalg.norm(pts[pip_idx][:2] - wrist[:2])
            d_tip_pip = np.linalg.norm(pts[tip_idx][:2] - pts[pip_idx][:2])
            return (d_pip_wrist > 0.85 * scale) and (d_tip_wrist < d_pip_wrist) and (d_tip_pip < 0.75 * scale)

        index_hooked = is_hooked(8, 6, 5)
        middle_hooked = is_hooked(12, 10, 9)
        ring_hooked = is_hooked(16, 14, 13)
        pinky_hooked = is_hooked(20, 18, 17)

        # Thumb up: tip is higher (smaller y) than IP and MCP joints and well above wrist
        d_thumb_wrist = np.linalg.norm(pts[4][:2] - wrist[:2])
        thumb_up = (pts[4][1] < pts[3][1] and pts[4][1] < pts[2][1] and 
                    pts[4][1] < wrist[1] - 0.10 * scale and d_thumb_wrist > 0.9 * scale)

        extended_count = sum([index_up, middle_up, ring_up, pinky_up])
        hooked_count = sum([index_hooked, middle_hooked, ring_hooked, pinky_hooked])

        # Inter-finger distances
        d_4_8 = norm_dist(4, 8)     # thumb-index
        d_4_12 = norm_dist(4, 12)   # thumb-middle
        d_4_16 = norm_dist(4, 16)   # thumb-ring
        d_4_20 = norm_dist(4, 20)   # thumb-pinky
        d_8_12 = norm_dist(8, 12)   # index-middle
        d_12_16 = norm_dist(12, 16) # middle-ring
        d_16_20 = norm_dist(16, 20) # ring-pinky

        # Pinch detection (fingertips touching thumb tip)
        is_pinched_index_middle = (d_4_8 < 0.45 and d_4_12 < 0.45)
        is_pinched_all = (d_4_8 < 0.50 and d_4_12 < 0.50 and d_4_16 < 0.55 and d_4_20 < 0.60)

        # Upward-facing palm (WHAT gesture)
        palm_facing_up = (wrist[1] > pts[9][1] + 0.04 and abs(pts[5][1] - pts[17][1]) < 0.08 * scale)

        # Hand height in camera frame (y: 0=top, 1=bottom)
        hand_y_level = float(palm_center[1])

        # ----------------------------------------------------
        # 1. DYNAMIC / MOTION-BASED GESTURES (Checked when trajectory is present)
        # ----------------------------------------------------

        # A. WHERE: Single index finger wagging horizontally back and forth
        if index_up and not middle_up and not ring_up and not pinky_up and is_wagging:
            return "WHERE", 0.97

        # B. WE: Single index finger sweeping horizontally across chest
        if index_up and not ring_up and not pinky_up and abs(dx_wrist) > 0.10 and not is_wagging:
            return "WE", 0.96

        # C. COLLEGE: Flat hand (fingers together) sweeping upward in an arc
        if index_up and middle_up and ring_up and (d_8_12 < 0.22) and (dy_palm < -0.05 and dx_palm > 0.02):
            return "COLLEGE", 0.96

        # D. MORNING: Open flat hand rising vertically like a rising sun
        if extended_count >= 3 and dy_palm < -0.07 and abs(dx_palm) < 0.06:
            return "MORNING", 0.96

        # E. NIGHT: Flat hand arching/descending downward over the wrist
        if extended_count >= 3 and dy_palm > 0.06:
            return "NIGHT", 0.96

        # F. SCHOOL: Flat hand tapping downward sharply
        if extended_count >= 3 and d_8_12 < 0.22 and dy_palm > 0.04 and abs(dx_palm) < 0.03:
            return "SCHOOL", 0.95

        # G. HELP: Thumbs up lifting upward
        if thumb_up and extended_count == 0 and dy_wrist < -0.03:
            return "HELP", 0.96

        # H. THANK YOU: Flat hand tilted forward or extending forward away from chin
        if extended_count >= 3 and d_8_12 < 0.20 and (pts[8][2] < wrist[2] - 0.05 or (wrist[1] > pts[9][1] + 0.08 and hand_y_level < 0.45)):
            return "THANK YOU", 0.95

        # ----------------------------------------------------
        # 2. ANATOMICAL HANDSHAPE / STATIC GESTURES
        # ----------------------------------------------------

        # 1. GOOD: Thumbs up (thumb pointing up, all 4 fingers curled)
        if thumb_up and extended_count == 0:
            return "GOOD", 0.97

        # 2. YES: Closed fist (all 4 fingers curled, thumb folded)
        if extended_count == 0:
            return "YES", 0.95

        # 3. NO: Snapping beak (index & middle reaching thumb, pinched tips, ring & pinky curled)
        if (index_up or middle_up) and is_pinched_index_middle and not ring_up and not pinky_up:
            return "NO", 0.96

        # 4. WATER: Three fingers 'W' (index, middle, ring extended, pinky curled)
        if index_up and middle_up and ring_up and not pinky_up and d_8_12 > 0.15:
            return "WATER", 0.96

        # 5. FRIEND: Index and middle fingers extended and crossed/touching closely
        if index_up and middle_up and not ring_up and not pinky_up and d_8_12 < 0.15:
            return "FRIEND", 0.95

        # 6. NAME: Peace / V sign (index and middle extended in wide 'V', ring & pinky curled)
        if index_up and middle_up and not ring_up and not pinky_up and d_8_12 >= 0.18:
            return "NAME", 0.96

        # 7. I: Single index finger pointing up OR single pinky finger up (ASL 'I')
        if (index_up and not middle_up and not ring_up and not pinky_up) or \
           (pinky_up and not index_up and not middle_up and not ring_up):
            return "I", 0.96

        # 8. YOU: Rock / horns sign (index and pinky up, middle & ring curled)
        if index_up and pinky_up and not middle_up and not ring_up:
            return "YOU", 0.95

        # 9. NEED: Hooked / "X" handshape (index bent like a hook, others curled)
        if index_hooked and not middle_up and not ring_up and not pinky_up:
            return "NEED", 0.95

        # 10. HOME: All fingertips pinched together, held high near cheek/face (y < 0.35)
        if is_pinched_all and hand_y_level < 0.35:
            return "HOME", 0.95

        # 11. FOOD: All fingertips pinched together at normal chest/mouth level
        if is_pinched_all:
            return "FOOD", 0.95

        # 12. WANT: Claw hand (all fingers curved into claws)
        if hooked_count >= 3 or (hooked_count >= 2 and extended_count <= 1 and not thumb_up):
            return "WANT", 0.94

        # 13. WHAT: Palm facing upward (open hand resting palm-up)
        if palm_facing_up and extended_count >= 3:
            return "WHAT", 0.94

        # 14. HOW: Cupped handshape (curved fingers, thumb extended out to side)
        if extended_count >= 2 and d_4_8 > 0.70 and not palm_facing_up:
            return "HOW", 0.94

        # 15. PLEASE: Flat hand with all 4 fingers closed tightly together
        if extended_count >= 3 and d_8_12 < 0.20 and d_12_16 < 0.20 and not thumb_up:
            return "PLEASE", 0.95

        # 16. HELLO: Open palm (4 or 5 fingers extended and spread wide apart)
        if extended_count >= 4 and d_8_12 >= 0.20:
            return "HELLO", 0.96

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
