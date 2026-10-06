import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional
from ml.utils.config import MP_HANDS_CONFIDENCE, MP_MAX_HANDS

class HandDetector:
    def __init__(self, 
                 static_image_mode: bool = False, 
                 max_num_hands: int = MP_MAX_HANDS, 
                 min_detection_confidence: float = MP_HANDS_CONFIDENCE, 
                 min_tracking_confidence: float = MP_HANDS_CONFIDENCE):
        """
        Initializes the MediaPipe Hand detector.
        """
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.is_valid = True
        except AttributeError:
            print("Warning: mediapipe.solutions is not available in this Python environment. Hand detection will be bypassed.")
            self.is_valid = False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List[float]], bool]:
        """
        Processes an OpenCV BGR frame, extracts hand landmarks, and draws them.
        
        Returns:
            processed_frame (np.ndarray): The frame with landmarks drawn.
            landmarks (List[float] or None): A flattened list of 21 (x, y, z) landmarks if detected, else None.
            is_detected (bool): True if at least one hand is detected.
        """
        if not self.is_valid:
            return frame, None, False
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        processed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        landmarks_list = None
        is_detected = False
        
        if results.multi_hand_landmarks:
            is_detected = True
            # For simplicity, we grab the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                processed_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract normalized coordinates (x, y, z)
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x, lm.y, lm.z])
                
        return processed_frame, landmarks_list, is_detected

    def extract_hand_region(self, frame: np.ndarray, padding: int = 20) -> Optional[np.ndarray]:
        """
        Extracts the cropped bounding box of the hand region.
        """
        if not self.is_valid:
            return None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
            
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
            
        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return frame[y_min:y_max, x_min:x_max]

    def release(self):
        """Releases the underlying MediaPipe resources."""
        if self.is_valid:
            self.hands.close()
