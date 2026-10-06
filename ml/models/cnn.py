import torch
import torch.nn as nn
from ml.utils.config import CNN_INPUT_SIZE

class SignCNN(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 256):
        super(SignCNN, self).__init__()
        
        # Input size: (3, 128, 128) if CNN_INPUT_SIZE is 128
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, 64, 64)
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (64, 32, 32)
            
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (128, 16, 16)
            
            # Block 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # Output: (256, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """Forward pass to get logits for classification."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
        
    def extract_features(self, x):
        """Extract spatial features to feed into LSTM later."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
