import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Computes learnable temporal attention weights across sequence time steps.
    Enables the model to dynamically focus on peak gesture frames (the stroke phase)
    and down-weight preparatory, transitional, or resting frames.
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(32, hidden_dim // 2), 1, bias=False)
        )

    def forward(self, lstm_out):
        """
        Args:
            lstm_out: Tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            context: Tensor of shape (batch_size, hidden_dim)
            weights: Tensor of shape (batch_size, seq_len)
        """
        # score: (batch_size, seq_len, 1)
        score = self.attn(lstm_out)
        weights = F.softmax(score, dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        return context, weights.squeeze(-1)

class SignLSTM(nn.Module):
    """
    Enhanced Sign Language LSTM model featuring:
    1. Bidirectional LSTM (Bi-LSTM) to capture both forward and backward temporal dynamics.
    2. Temporal Self-Attention to focus on key gesture frames across variable sequence lengths.
    3. Input and Post-LSTM Layer Normalization for gradient stability and feature scaling.
    4. Two-stage MLP classification head with non-linear GELU activation and dropout regularization.
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super(SignLSTM, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        
        # Input Layer Normalization to stabilize spatial feature embeddings
        self.input_norm = nn.LayerNorm(feature_dim)
        
        # Recurrent Core
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_out_dim = hidden_size * self.num_directions
        
        # Temporal Attention Mechanism
        if use_attention:
            self.attention = TemporalAttention(lstm_out_dim)
            
        # Post-LSTM Layer Normalization
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        
        # Two-stage MLP Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, return_attention: bool = False):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, feature_dim)
            return_attention: If True, returns (logits, attention_weights)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
            (optional) attention_weights: Tensor of shape (batch_size, sequence_length)
        """
        # Normalize input features across channels
        x = self.input_norm(x)
        
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)
        
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            # Fallback to the final time step
            context = lstm_out[:, -1, :]
            attn_weights = None
            
        normed_context = self.layer_norm(context)
        logits = self.classifier(normed_context)
        
        if return_attention:
            return logits, attn_weights
        return logits
