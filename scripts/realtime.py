import cv2
from ml.preprocessing.hand_detection import HandDetector
from ml.inference.realtime_inference import SignRecognizer

def run_realtime():
    print("Initializing components...")
    detector = HandDetector()
    recognizer = SignRecognizer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Real-time Sign2Sent started. Press 'q' to quit.")
    
    current_sign = "None"
    current_confidence = 0.0
    recognized_sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detect Hand
        processed_frame, landmarks, is_detected = detector.process_frame(frame)
        
        mode_text = "DEMO MODE" if recognizer.demo_mode else "REAL MODEL"
        cv2.putText(processed_frame, mode_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        if is_detected:
            # 2. Extract Hand Region
            # Using the raw frame for extraction so we don't pass drawn landmarks to CNN
            hand_region = detector.extract_hand_region(frame)
            
            if hand_region is not None and hand_region.size > 0:
                hand_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
                
                # 3. Real-time Inference
                sign, conf = recognizer.process_frame(hand_rgb)
                
                if sign:
                    current_sign = sign
                    current_confidence = conf
                    recognized_sequence.append(sign)
                    
        # 4. Display Info
        status_color = (0, 255, 0) if is_detected else (0, 0, 255)
        cv2.putText(processed_frame, f"Hand: {'Detected' if is_detected else 'Not Detected'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(processed_frame, f"Current Sign: {current_sign}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if current_confidence > 0:
            cv2.putText(processed_frame, f"Confidence: {current_confidence*100:.1f}%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        # Show sequence of recognized signs (last 5)
        display_seq = " ".join(recognized_sequence[-5:])
        cv2.putText(processed_frame, f"Sequence: {display_seq}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        cv2.imshow("Sign2Sent Real-Time", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear sequence manually
            recognized_sequence.clear()
            current_sign = "None"
            current_confidence = 0.0

    cap.release()
    detector.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
