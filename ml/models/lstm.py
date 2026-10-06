import torch
import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.3):
        super(SignLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, feature_dim)
        """
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        
        out = self.dropout(last_out)
        logits = self.classifier(out)
        
        return logits
