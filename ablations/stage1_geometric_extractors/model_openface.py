import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenFaceAblationClassifier(nn.Module):
    """
    Ablation Architecture:
    1. EfficientNet REMOVED
    2. Transformer Encoder REMOVED
    3. Input: OpenFace Features Alone (Head Pose, Eye Gaze, Facial Action Units, Landmarks)
    4. Temporal Representation: Mean Pooling across frames
    5. Classification Head:
       - FC Layer (input_dim -> hidden_dim)
       - ReLU Activation
       - Dropout
       - FC Layer (hidden_dim -> 2 classes)
    """
    def __init__(self, input_dim=128, hidden_dim=128, dropout=0.3, num_classes=2):
        super(OpenFaceAblationClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Classification Head (as specified in Architecture Step 5)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x: Input tensor of shape [Batch_Size, Seq_Len, Feature_Dim] or [Batch_Size, Feature_Dim]
            seq_lens: Optional tensor of sequence lengths [Batch_Size]
        Returns:
            logits: Output tensor of shape [Batch_Size, 2]
        """
        if x.dim() == 2:
            # Already pooled/single frame: [Batch_Size, Feature_Dim]
            pooled = x
        elif x.dim() == 3:
            # Temporal Sequence: [Batch_Size, Seq_Len, Feature_Dim]
            if seq_lens is not None:
                # Masked Mean Pooling over valid frames
                batch_size, max_len, _ = x.shape
                device = x.device
                mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1) # [B, T]
                mask = mask.unsqueeze(-1).float() # [B, T, 1]
                masked_x = x * mask
                sum_x = masked_x.sum(dim=1) # [B, D]
                lens = seq_lens.clamp(min=1).unsqueeze(1).float() # [B, 1]
                pooled = sum_x / lens
            else:
                # Standard Mean Pooling across time dimension (Dim 1)
                pooled = x.mean(dim=1)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        # Classification Head
        logits = self.classifier(pooled)
        return logits
