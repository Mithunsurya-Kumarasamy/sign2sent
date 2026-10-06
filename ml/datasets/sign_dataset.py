import os
import glob
import torch
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from ml.utils.vocabulary import LABEL_MAP
from ml.utils.config import SEQUENCE_LENGTH

class SignSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading sequences of hand frames.
    """
    def __init__(self, data_dir: str, transform=None, frames_per_sequence: int = SEQUENCE_LENGTH):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.frames_per_sequence = frames_per_sequence
        
        self.sequences = []
        self.labels = []
        
        self._load_metadata()
        
    def _load_metadata(self):
        """Scan the directory and organize sequences."""
        if not self.data_dir.exists():
            return
            
        for label_name in os.listdir(self.data_dir):
            label_dir = self.data_dir / label_name
            if not label_dir.is_dir() or label_name not in LABEL_MAP:
                continue
                
            label_idx = LABEL_MAP[label_name]
            
            for seq_name in os.listdir(label_dir):
                seq_dir = label_dir / seq_name
                if not seq_dir.is_dir():
                    continue
                    
                # Collect frames
                frame_paths = sorted(glob.glob(str(seq_dir / "*.jpg")))
                if len(frame_paths) >= self.frames_per_sequence:
                    # If we have more frames, take the first N (or resample)
                    # For simplicity, we truncate or pad
                    self.sequences.append(frame_paths[:self.frames_per_sequence])
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        label = self.labels[idx]
        
        # Load images
        frames = []
        for path in frame_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        # Stack frames into (Seq, C, H, W)
        if len(frames) > 0 and isinstance(frames[0], torch.Tensor):
            sequence_tensor = torch.stack(frames)
        else:
            sequence_tensor = frames # List of numpy arrays if no ToTensor transform
            
        return sequence_tensor, label
