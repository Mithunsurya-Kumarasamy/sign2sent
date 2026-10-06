import cv2
import numpy as np
from ml.inference.realtime_inference import SignRecognizer
from ml.preprocessing.hand_detection import HandDetector
from ml.nlp.sentence_generator import SentenceGenerator
from ml.nlp.tts import TextToSpeech

class InferenceService:
    def __init__(self):
        self.detector = HandDetector()
        self.recognizer = SignRecognizer()
        self.sentence_gen = SentenceGenerator()
        self.tts = TextToSpeech()
        
    def process_image(self, image_bytes: bytes):
        """Processes an image byte array to detect hands and predict the sign."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid image"}
            
        _, _, is_detected = self.detector.process_frame(frame)
        
        # If no hand detected, check if we are in demo mode and bypassed
        if not is_detected:
            if not self.detector.is_valid and self.recognizer.demo_mode:
                # We are in bypass mode. Simulate passing the full frame to the recognizer.
                # In demo mode, it just returns a fake sign randomly anyway.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sign, conf = self.recognizer.process_frame(frame_rgb)
                return {
                    "sign": sign,
                    "confidence": conf,
                    "hand_detected": True, # Fake it for UI
                    "demo_mode": True
                }
            return {"sign": None, "confidence": 0.0, "hand_detected": False}
            
        hand_region = self.detector.extract_hand_region(frame)
        if hand_region is None or hand_region.size == 0:
            return {"sign": None, "confidence": 0.0, "hand_detected": False}
            
        hand_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
        sign, conf = self.recognizer.process_frame(hand_rgb)
        
        return {
            "sign": sign,
            "confidence": conf,
            "hand_detected": True,
            "demo_mode": self.recognizer.demo_mode
        }
        
    def generate_sentence(self, sequence: list[str]) -> str:
        return self.sentence_gen.generate_sentence(sequence)
        
    def speak(self, text: str):
        self.tts.speak(text)

# Singleton instance for the FastAPI app
inference_service = InferenceService()
