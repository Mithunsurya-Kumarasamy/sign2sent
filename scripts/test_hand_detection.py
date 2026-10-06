import cv2
from ml.preprocessing.hand_detection import HandDetector
import time

def test_hand_detection():
    print("Initializing MediaPipe Hand Detector...")
    detector = HandDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("Webcam opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Process the frame for hand detection
        processed_frame, landmarks, is_detected = detector.process_frame(frame)
        
        # Display status text
        if is_detected:
            # We assume it detected if is_detected is True. The config defines the confidence threshold.
            cv2.putText(processed_frame, "Hand: Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Landmarks: {len(landmarks)//3} pts", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Optional: Test hand region extraction
            hand_img = detector.extract_hand_region(frame)
            if hand_img is not None and hand_img.size > 0:
                cv2.imshow("Extracted Hand Region", hand_img)
        else:
            cv2.putText(processed_frame, "Hand: Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("Sign2Sent Hand Detection Test", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    print("Test completed.")

if __name__ == "__main__":
    test_hand_detection()
