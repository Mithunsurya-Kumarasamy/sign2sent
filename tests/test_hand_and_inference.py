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

def test_heuristic_gesture_fist_and_thumbsup():
    from ml.inference.realtime_inference import SignRecognizer
    rec = SignRecognizer()

    # Closed Fist (YES): wrist at (0.5, 0.8), all fingers curled, thumb resting on index
    fist_pts = []
    for _ in range(21):
        fist_pts.extend([0.5, 0.5, 0.0])
    fist_pts[0*3] = 0.5; fist_pts[0*3+1] = 0.8 # wrist
    for base, mcp_y in [(5, 0.55), (9, 0.54), (13, 0.55), (17, 0.57)]:
        fist_pts[base*3] = 0.5; fist_pts[base*3+1] = mcp_y
        fist_pts[(base+1)*3] = 0.5; fist_pts[(base+1)*3+1] = mcp_y - 0.05 # pip
        fist_pts[(base+2)*3] = 0.5; fist_pts[(base+2)*3+1] = mcp_y - 0.02 # dip
        fist_pts[(base+3)*3] = 0.5; fist_pts[(base+3)*3+1] = mcp_y + 0.03 # tip (curled)
    fist_pts[4*3] = 0.46; fist_pts[4*3+1] = 0.52 # thumb resting on index

    sign, conf = rec._predict_demo_mode(fist_pts)
    assert sign == "YES", f"Expected 'YES' for closed fist, got '{sign}'"
    assert conf >= 0.90

    # Thumbs Up (GOOD): thumb protruding high above knuckles
    thumbsup_pts = list(fist_pts)
    thumbsup_pts[4*3] = 0.40; thumbsup_pts[4*3+1] = 0.32 # thumb tip high above knuckles

    sign_good, conf_good = rec._predict_demo_mode(thumbsup_pts)
    assert sign_good == "GOOD", f"Expected 'GOOD' for thumbs up, got '{sign_good}'"
    assert conf_good >= 0.90

