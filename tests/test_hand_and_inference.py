import cv2
import numpy as np
from ml.preprocessing.hand_detection import HandDetector
from backend.services.inference_service import InferenceService

def test_hand_detector_initialization():
    detector = HandDetector()
    assert detector.is_valid is True
    assert detector.backend in ["mediapipe_tasks", "mediapipe_solutions", "opencv"]
    
    # Process black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    proc_frame, landmarks, detected = detector.process_frame(frame)
    assert proc_frame.shape == frame.shape
    assert landmarks is None
    assert detected is False
    
    detector.release()

def test_inference_service_process_image():
    service = InferenceService()
    
    # Create black frame and encode to JPEG bytes
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", frame)
    image_bytes = encoded.tobytes()
    
    result = service.process_image(image_bytes)
    assert "sign" in result
    assert "confidence" in result
    assert "hand_detected" in result
    assert result["hand_detected"] is False

def test_sentence_generation_integration():
    service = InferenceService()
    sentence = service.generate_sentence(["HELLO", "MY", "NAME", "IS", "JOHN"])
    assert sentence == "Hello, my name is John."
