import os
import glob
import torch
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from ml.utils.vocabulary import LABEL_MAP

class SignFrameDataset(Dataset):
    """
    PyTorch Dataset for loading individual frames for CNN pretraining.
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        self._load_metadata()
        
    def _load_metadata(self):
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
                    
                # Collect all frames from all sequences as individual samples
                frames = glob.glob(str(seq_dir / "*.jpg"))
                for frame_path in frames:
                    self.image_paths.append(frame_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
