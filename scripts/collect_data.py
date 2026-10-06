import cv2
import os
import argparse
import time
from pathlib import Path
from ml.utils.config import RAW_DATA_DIR, SEQUENCE_LENGTH
from ml.utils.vocabulary import VOCABULARY

def collect_data(label: str, num_sequences: int, frames_per_sequence: int = SEQUENCE_LENGTH):
    if label not in VOCABULARY:
        print(f"Warning: Label '{label}' is not in the predefined vocabulary.")
        print(f"Vocabulary: {VOCABULARY}")
        
    # Create directory for the label
    label_dir = RAW_DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine starting sequence number
    existing_seqs = [d for d in os.listdir(label_dir) if os.path.isdir(label_dir / d)]
    start_seq = len(existing_seqs)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Collecting data for label: {label}")
    print(f"Will collect {num_sequences} sequences of {frames_per_sequence} frames each.")
    print("Press 'r' to start recording a sequence, 'q' to quit.")

    seq_count = 0
    while seq_count < num_sequences:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Display instructions
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Sequences: {seq_count}/{num_sequences}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'r' to record, 'q' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Sign2Sent Data Collection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting data collection.")
            break
        elif key == ord('r'):
            seq_dir = label_dir / f"sequence_{start_seq + seq_count:03d}"
            seq_dir.mkdir(exist_ok=True)
            print(f"Recording sequence {start_seq + seq_count}...")
            
            # Brief pause before recording
            cv2.waitKey(500)
            
            for frame_idx in range(frames_per_sequence):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save raw frame
                frame_path = seq_dir / f"frame_{frame_idx:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Display recording status
                rec_frame = frame.copy()
                cv2.putText(rec_frame, f"Recording: {frame_idx + 1}/{frames_per_sequence}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Sign2Sent Data Collection", rec_frame)
                cv2.waitKey(1) # small delay for frame capture rate
                
            print(f"Saved sequence {start_seq + seq_count}.")
            seq_count += 1
            
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect webcam data for Sign2Sent")
    parser.add_argument("--label", type=str, required=True, help="Label for the sign being recorded")
    parser.add_argument("--sequences", type=int, default=30, help="Number of sequences to record")
    parser.add_argument("--frames", type=int, default=SEQUENCE_LENGTH, help="Number of frames per sequence")
    
    args = parser.parse_args()
    collect_data(args.label, args.sequences, args.frames)
