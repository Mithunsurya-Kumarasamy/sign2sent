import torch
from ml.models.cnn import SignCNN
from ml.models.lstm import SignLSTM
from ml.models.sign2sent import Sign2SentModel

def test_cnn_output_shape():
    """Test that CNN produces the expected feature dimension given a dummy image."""
    model = SignCNN(num_classes=5, feature_dim=256)
    
    # Batch of 2 dummy images (Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_input)
    
    # Output should be (Batch, Feature_Dim)
    assert features.shape == (2, 256), f"Expected (2, 256), got {features.shape}"

def test_lstm_output_shape():
    """Test that LSTM processes sequences correctly."""
    model = SignLSTM(feature_dim=256, hidden_size=128, num_layers=2, num_classes=5)
    
    # Batch of 2 sequences, each of length 10, with feature dim 256
    dummy_sequence = torch.randn(2, 10, 256)
    logits = model(dummy_sequence)
    
    # Output should be (Batch, Num_Classes)
    assert logits.shape == (2, 5), f"Expected (2, 5), got {logits.shape}"

def test_sign2sent_end_to_end_shape():
    """Test the combined wrapper model shape."""
    model = Sign2SentModel(num_classes=10, feature_dim=128, hidden_size=64, num_layers=1)
    
    # Batch=2, SeqLen=5, C=3, H=224, W=224
    dummy_input = torch.randn(2, 5, 3, 224, 224)
    logits = model(dummy_input)
    
    assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"
