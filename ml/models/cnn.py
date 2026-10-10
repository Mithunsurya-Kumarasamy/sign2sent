import torch
import torch.nn as nn
import torchvision.models as models
from ml.utils.config import CNN_INPUT_SIZE

class SignCNN(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 256):
        super(SignCNN, self).__init__()
        
        # 1. Load Google's pre-trained MobileNetV2!
        # It has already been trained on 1.2 million images to understand shapes and lighting.
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # 2. Extract its "Brain" (we don't want its ImageNet classifier, just the feature extractor)
        self.features = mobilenet.features
        
        # 3. Freeze the early layers so we don't accidentally ruin its pre-trained knowledge during training
        for idx, child in enumerate(self.features.children()):
            if idx < 10: # Freeze the first 10 blocks out of 18
                for param in child.parameters():
                    param.requires_grad = False
                    
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Map MobileNet's 1280-dimension output down to our LSTM's expected feature_dim
        self.fc = nn.Sequential(
            nn.Linear(1280, feature_dim),
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
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
