import torch
import torch.nn as nn
from ml.models.cnn import SignCNN
from ml.models.lstm import SignLSTM

class Sign2SentModel(nn.Module):
    """
    Combined CNN + LSTM model for end-to-end inference.
    Not typically trained end-to-end due to memory constraints, but useful for inference.
    """
    def __init__(self, num_classes: int, feature_dim: int, hidden_size: int, num_layers: int):
        super(Sign2SentModel, self).__init__()
        self.cnn = SignCNN(num_classes=num_classes, feature_dim=feature_dim)
        self.lstm = SignLSTM(feature_dim=feature_dim, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
        
    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, C, H, W)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # Flatten batch and sequence to pass through CNN
        x = x.view(batch_size * seq_len, C, H, W)
        
        # Extract features using CNN
        features = self.cnn.extract_features(x)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)
        
        # Pass sequence to LSTM
        logits = self.lstm(features)
        
        return logits
