import os
from pathlib import Path
from ml.utils.vocabulary import VOCABULARY

def validate_dataset(data_dir: Path, expected_frames: int):
    """
    Validates that the dataset has consistent sequences and frames.
    """
    issues = []
    
    if not data_dir.exists():
        issues.append(f"Directory {data_dir} does not exist.")
        return issues
        
    for label in os.listdir(data_dir):
        label_dir = data_dir / label
        if not label_dir.is_dir():
            continue
            
        if label not in VOCABULARY:
            issues.append(f"Warning: Label '{label}' found in dataset but not in vocabulary.")
            
        seq_dirs = [d for d in os.listdir(label_dir) if (label_dir / d).is_dir()]
        for seq in seq_dirs:
            seq_dir = label_dir / seq
            frames = [f for f in os.listdir(seq_dir) if f.endswith('.jpg')]
            if len(frames) < expected_frames:
                issues.append(f"Sequence {label}/{seq} has {len(frames)} frames (expected {expected_frames}).")
                
    return issues

if __name__ == "__main__":
    from ml.utils.config import RAW_DATA_DIR, SEQUENCE_LENGTH
    
    print("Validating dataset...")
    errors = validate_dataset(RAW_DATA_DIR, SEQUENCE_LENGTH)
    if errors:
        print("Dataset validation issues found:")
        for e in errors:
            print(f"- {e}")
    else:
        print("Dataset is valid and ready.")
