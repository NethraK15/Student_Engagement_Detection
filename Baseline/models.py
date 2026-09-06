import torch
import torch.nn as nn

class BaselineClassifier(nn.Module):
    """
    Baseline Fully Connected Neural Network classifier for student engagement.
    Takes as input a temporally-averaged feature vector of shape [1280 + openface_dim]
    representing a video and outputs logits for the three classes:
    - Engaged (0)
    - Partially Engaged (1)
    - Disengaged (2)
    """
    def __init__(self, input_dim, num_classes=3):
        super(BaselineClassifier, self).__init__()
        
        self.net = nn.Sequential(
            # First Dense/FC Block
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),  # LayerNorm is batch-size independent and robust
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Second Dense/FC Block
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output Layer (scores/logits for 3 classes)
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [Batch_Size, Input_Dim]
        Returns:
            logits: Output tensor of shape [Batch_Size, Num_Classes]
        """
        return self.net(x)
