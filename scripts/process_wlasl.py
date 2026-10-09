import os
import json
import cv2
import sys
import kagglehub
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.utils.config import RAW_DATA_DIR, SEQUENCE_LENGTH
from ml.utils.vocabulary import VOCABULARY

def process_wlasl():
    print("="*50)
    print("1. Downloading WLASL Raw Video Dataset from Kaggle...")
    print("This may take a few minutes as the dataset is large (2GB+).")
    print("="*50)
    
    # Download the dataset using kagglehub
    dataset_path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
    base_dir = Path(dataset_path)
    
    print(f"Dataset downloaded to: {base_dir}")
    
    # Locate the JSON file and Videos directory
    # Depending on how the Kaggle repo is zipped, it might be in the root or a subfolder
    json_path = base_dir / "WLASL_v0.3.json"
    videos_dir = base_dir / "videos"
    
    if not json_path.exists():
        # Sometimes Kaggle extracts into a subfolder with the dataset name
        potential_jsons = list(base_dir.rglob("WLASL_v0.3.json"))
        if not potential_jsons:
            print("ERROR: Could not find WLASL_v0.3.json in the downloaded dataset.")
            return
        json_path = potential_jsons[0]
        videos_dir = json_path.parent / "videos"
        
    if not videos_dir.exists():
        print(f"ERROR: Could not find the 'videos' folder at {videos_dir}")
        return

    print("\n2. Loading WLASL metadata...")
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)

    target_words = [word.lower() for word in VOCABULARY]
    processed_count = 0
    total_extracted_sequences = 0

    print("\n3. Extracting frames for our specific 24-word vocabulary...")
    for entry in wlasl_data:
        gloss = entry['gloss'].lower()
        if gloss in target_words:
            label = gloss.upper()
            label_dir = RAW_DATA_DIR / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            for instance in entry['instances']:
                video_id = instance['video_id']
                # The video files might have different extensions, usually .mp4
                video_path = videos_dir / f"{video_id}.mp4"
                
                if not video_path.exists():
                    # Fallback check if it's named differently
                    continue
                    
                # We will extract evenly spaced frames from the video to match our SEQUENCE_LENGTH
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames < SEQUENCE_LENGTH:
                    cap.release()
                    continue # Skip videos that are too short
                    
                # Calculate frame indices to extract evenly
                indices = [int(i * (total_frames - 1) / (SEQUENCE_LENGTH - 1)) for i in range(SEQUENCE_LENGTH)]
                
                # Create a sequence folder
                seq_dir = label_dir / f"wlasl_{video_id}"
                seq_dir.mkdir(parents=True, exist_ok=True)
                
                frame_idx = 0
                saved_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_idx in indices:
                        save_path = seq_dir / f"frame_{saved_count}.jpg"
                        cv2.imwrite(str(save_path), frame)
                        saved_count += 1
                        
                        if saved_count >= SEQUENCE_LENGTH:
                            break
                            
                    frame_idx += 1
                    
                cap.release()
                
                if saved_count == SEQUENCE_LENGTH:
                    total_extracted_sequences += 1
            
            processed_count += 1
            print(f" - Extracted '{label}'")

    print("="*50)
    print(f"DONE! Found {processed_count}/{len(VOCABULARY)} matching words in the dataset.")
    print(f"Successfully generated {total_extracted_sequences} video sequences for training!")
    print(f"The frames have been saved to {RAW_DATA_DIR}")
    print("You can now run: python scripts/train_pipeline.py")
    print("="*50)

if __name__ == "__main__":
    process_wlasl()
