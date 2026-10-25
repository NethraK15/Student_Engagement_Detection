import torch
import torch.nn as nn
import torch.nn.functional as F
import config_gru

class EngagementGRUClassifier(nn.Module):
    """
    Refined Lightweight Bidirectional GRU Classifier for Student Engagement Detection.
    Features:
    1. Linear Bottleneck Projection (1989 -> 128): reduces parameter count by 20x
    2. Lightweight 1-Layer BiGRU (64 hidden units per direction -> 128 dim)
    3. Masked Self-Attention Temporal Pooling
    4. Small FC classification head
    """
    def __init__(self, input_dim, proj_dim=config_gru.PROJECTION_DIM, hidden_dim=config_gru.HIDDEN_DIM, num_classes=3):
        super(EngagementGRUClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        
        # 1. Linear Bottleneck Projection Layer (1989 -> 128)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 2. 1-Layer Lightweight Bidirectional GRU (Prevents Overfitting)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Self-Attention Temporal Pooling Layer
        # Output of BiGRU is [Batch, Seq_Len, hidden_dim * 2] (64 x 2 = 128)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # 4. FC Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x: Input sequence tensor of shape [Batch, Seq_Len, Input_Dim]
            seq_lens: Optional 1D tensor/list containing unpadded lengths per sample
        Returns:
            logits: Output classification scores of shape [Batch, Num_Classes]
        """
        # 1. Pass through Linear Bottleneck Projection -> [Batch, Seq_Len, 128]
        x_proj = self.bottleneck(x)
        
        # 2. Pass through 1-Layer BiGRU -> Output shape: [Batch, Seq_Len, 128]
        gru_out, _ = self.gru(x_proj)
        
        # 3. Self-attention temporal pooling across frames
        attn_scores = self.attention(gru_out)                # [Batch, Seq_Len, 1]
        
        if seq_lens is not None:
            batch_size, max_len, _ = gru_out.shape
            device = x.device
            lengths_tensor = seq_lens.to(device) if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=device)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths_tensor.unsqueeze(1)
            mask = mask.unsqueeze(-1)                          # [Batch, Seq_Len, 1]
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)           # Normalize across Seq_Len
        
        # 4. Compute weighted sum across frames -> [Batch, 128]
        context_vector = torch.sum(attn_weights * gru_out, dim=1)
        
        # 5. Pass through classification head -> [Batch, Num_Classes]
        logits = self.classifier(context_vector)
        return logits
