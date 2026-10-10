import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from ml.utils.config import MP_HANDS_CONFIDENCE, MP_MAX_HANDS, MODELS_DIR

# Standard MediaPipe 21 Hand Landmark Connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm Base
    (0, 17)
]

class HandDetector:
    """
    3-Tier Hand Detector for Sign2Sent:
    1. MediaPipe Tasks API (HandLandmarker) using models/hand_landmarker.task
    2. Legacy MediaPipe Solutions (mp.solutions.hands) if available
    3. OpenCV Skin-Color / Contour Fallback
    """
    def __init__(self, 
                 static_image_mode: bool = False, 
                 max_num_hands: int = MP_MAX_HANDS, 
                 min_detection_confidence: float = MP_HANDS_CONFIDENCE, 
                 min_tracking_confidence: float = MP_HANDS_CONFIDENCE):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.backend = None # 'mediapipe_tasks', 'mediapipe_solutions', 'opencv', or None
        self.landmarker = None
        self.hands = None
        self.is_valid = False
        self._cached_hand_region = None
        self._cached_bbox = None

        # 1. Attempt MediaPipe Tasks API (modern mediapipe >= 0.10.x)
        task_path = MODELS_DIR / "hand_landmarker.task"
        if task_path.exists():
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=str(task_path.resolve()))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.landmarker = vision.HandLandmarker.create_from_options(options)
                self.mp_module = mp
                self.backend = "mediapipe_tasks"
                self.is_valid = True
                print("HandDetector: Initialized with MediaPipe Tasks (hand_landmarker.task).")
            except Exception as e:
                print(f"HandDetector: Could not initialize MediaPipe Tasks: {e}")

        # 2. Attempt legacy mediapipe.solutions if tasks was not initialized
        if not self.is_valid:
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                    self.mp_hands = mp.solutions.hands
                    self.mp_drawing = mp.solutions.drawing_utils
                    self.mp_drawing_styles = mp.solutions.drawing_styles
                    self.hands = self.mp_hands.Hands(
                        static_image_mode=static_image_mode,
                        max_num_hands=max_num_hands,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )
                    self.backend = "mediapipe_solutions"
                    self.is_valid = True
                    print("HandDetector: Initialized with legacy mediapipe.solutions.")
            except Exception as e:
                print(f"HandDetector: Legacy mediapipe.solutions unavailable: {e}")

        # 3. OpenCV fallback if MediaPipe is unavailable
        if not self.is_valid:
            print("HandDetector: Falling back to OpenCV Skin-Color Detection.")
            self.backend = "opencv"
            self.is_valid = True

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List[float]], bool]:
        """
        Processes an OpenCV BGR frame, extracts hand landmarks, and draws them.
        
        Returns:
            processed_frame (np.ndarray): The frame with landmarks drawn.
            landmarks (List[float] or None): Flattened list of 21 (x, y, z) landmarks if detected, else None.
            is_detected (bool): True if at least one hand is detected.
        """
        if frame is None or frame.size == 0:
            return frame, None, False

        h, w, _ = frame.shape
        self._cached_hand_region = None
        self._cached_bbox = None

        if self.backend == "mediapipe_tasks":
            return self._process_mediapipe_tasks(frame, h, w)
        elif self.backend == "mediapipe_solutions":
            return self._process_mediapipe_solutions(frame, h, w)
        elif self.backend == "opencv":
            return self._process_opencv(frame, h, w)

        return frame, None, False

    def _process_mediapipe_tasks(self, frame: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, Optional[List[float]], bool]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp_module.Image(image_format=self.mp_module.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            results = self.landmarker.detect(mp_image)
        except Exception as e:
            return frame, None, False

        if not results or not results.hand_landmarks:
            return frame, None, False

        processed_frame = frame.copy()
        first_hand = results.hand_landmarks[0]
        
        # Calculate bounding box
        x_coords = [lm.x for lm in first_hand]
        y_coords = [lm.y for lm in first_hand]
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        self._cached_bbox = (x_min, y_min, x_max, y_max)

        # Draw hand connections (lines)
        for p1, p2 in HAND_CONNECTIONS:
            if p1 < len(first_hand) and p2 < len(first_hand):
                pt1 = (int(first_hand[p1].x * w), int(first_hand[p1].y * h))
                pt2 = (int(first_hand[p2].x * w), int(first_hand[p2].y * h))
                cv2.line(processed_frame, pt1, pt2, (0, 230, 0), 2, cv2.LINE_AA)

        # Draw landmark points
        for lm in first_hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(processed_frame, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(processed_frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Extract normalized 63 coordinates (x, y, z for 21 landmarks)
        landmarks_list = []
        for lm in first_hand:
            landmarks_list.extend([float(lm.x), float(lm.y), float(lm.z)])

        return processed_frame, landmarks_list, True

    def _process_mediapipe_solutions(self, frame: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, Optional[List[float]], bool]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return frame, None, False

        processed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        first_hand = results.multi_hand_landmarks[0]
        
        self.mp_drawing.draw_landmarks(
            processed_frame,
            first_hand,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

        x_coords = [lm.x for lm in first_hand.landmark]
        y_coords = [lm.y for lm in first_hand.landmark]
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        self._cached_bbox = (x_min, y_min, x_max, y_max)

        landmarks_list = []
        for lm in first_hand.landmark:
            landmarks_list.extend([float(lm.x), float(lm.y), float(lm.z)])

        return processed_frame, landmarks_list, True

    def _process_opencv(self, frame: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, Optional[List[float]], bool]:
        # Fallback YCrCb skin thresholding
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame, None, False

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 3000: # filter out tiny noise
            return frame, None, False

        x, y, bw, bh = cv2.boundingRect(largest)
        self._cached_bbox = (x, y, x + bw, y + bh)
        
        processed = frame.copy()
        cv2.rectangle(processed, (x, y), (x + bw, y + bh), (0, 230, 0), 2)
        cv2.putText(processed, "Hand (OpenCV)", (x, max(20, y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1)

        # Synthesize basic landmarks from bounding box center
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        landmarks_list = [cx, cy, 0.0] * 21

        return processed, landmarks_list, True

    def extract_hand_region(self, frame: np.ndarray, padding: int = 20) -> Optional[np.ndarray]:
        """
        Extracts the cropped bounding box of the hand region.
        Uses cached bounding box from the most recent process_frame for maximum performance.
        """
        if frame is None or frame.size == 0:
            return None

        h, w, _ = frame.shape

        if self._cached_bbox is not None:
            x_min, y_min, x_max, y_max = self._cached_bbox
        else:
            _, _, detected = self.process_frame(frame)
            if not detected or self._cached_bbox is None:
                return None
            x_min, y_min, x_max, y_max = self._cached_bbox

        # Add padding and clamp
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        if x_max <= x_min or y_max <= y_min:
            return None

        return frame[y_min:y_max, x_min:x_max]

    def release(self):
        """Releases underlying resources."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
