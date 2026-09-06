import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config_lightweight

# =====================================================================
# OPTION 1: Attention Pooling Classifier
# Single learned query vector attending over frame vectors (e.g. 16-120 frames).
# Ultra-lightweight: weights distracted / key frames heavily without full Transformer parameters.
# =====================================================================
class AttentionPoolingClassifier(nn.Module):
    def __init__(self, input_dim, d_model=config_lightweight.D_MODEL, dropout=config_lightweight.DROPOUT, num_classes=3):
        super(AttentionPoolingClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 1. Linear Projection Layer: input_dim -> d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 2. Single Learned Query Vector: [1, d_model]
        self.query = nn.Parameter(torch.randn(1, d_model) / math.sqrt(d_model))
        self.dropout = nn.Dropout(dropout)
        
        # 3. FC Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x: [Batch, Seq_Len, Input_Dim]
            seq_lens: Optional sequence lengths per sample
        Returns:
            logits: [Batch, Num_Classes], attn_weights: [Batch, Seq_Len]
        """
        batch_size, max_len, _ = x.shape
        device = x.device
        
        # Project inputs -> [Batch, Seq_Len, d_model]
        x_proj = self.layer_norm(self.input_proj(x))
        
        # Calculate attention scores: dot product of query and frame embeddings
        # [Batch, Seq_Len]
        scores = torch.sum(x_proj * self.query.unsqueeze(0), dim=-1) / math.sqrt(self.d_model)
        
        # Mask padded frames if seq_lens is provided
        if seq_lens is not None:
            lengths_tensor = seq_lens.to(device) if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=device)
            mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths_tensor.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1) # [Batch, Seq_Len]
        attn_weights_dropout = self.dropout(attn_weights)
        
        # Weighted sum across time dimension -> [Batch, d_model]
        pooled = torch.sum(attn_weights_dropout.unsqueeze(-1) * x_proj, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        return logits


# =====================================================================
# OPTION 2: 1-2 Layer Temporal Convolutional Network (TCN)
# Good local pattern detector across consecutive frames, cheap to train.
# =====================================================================
class TCNClassifier(nn.Module):
    def __init__(self, input_dim, d_model=config_lightweight.D_MODEL, kernel_size=config_lightweight.TCN_KERNEL_SIZE,
                 num_layers=config_lightweight.TCN_NUM_LAYERS, dropout=config_lightweight.DROPOUT, num_classes=3):
        super(TCNClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input Projection Layer
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        
        # TCN Conv1D Blocks
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        self.tcn_block = nn.Sequential(*layers)
        
        # Global Adaptive Average Pooling across time
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x: [Batch, Seq_Len, Input_Dim]
        Returns:
            logits: [Batch, Num_Classes]
        """
        # Convert to [Batch, Channels (Input_Dim), Seq_Len]
        x_trans = x.transpose(1, 2)
        
        # Input 1x1 projection -> [Batch, d_model, Seq_Len]
        feat = self.input_proj(x_trans)
        
        # Temporal convolutions -> [Batch, d_model, Seq_Len]
        feat = self.tcn_block(feat)
        
        # Global Average Pooling -> [Batch, d_model, 1] -> [Batch, d_model]
        pooled = self.global_pool(feat).squeeze(-1)
        
        # Classification Head -> [Batch, Num_Classes]
        logits = self.classifier(pooled)
        return logits


# =====================================================================
# OPTION 3: Single-Layer Unidirectional GRU
# Lighter & faster than multi-layer BiLSTM, keeps recurrent memory.
# =====================================================================
class SingleLayerGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=config_lightweight.GRU_HIDDEN_DIM, 
                 dropout=config_lightweight.DROPOUT, num_classes=3):
        super(SingleLayerGRUClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Single-layer GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=config_lightweight.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x: [Batch, Seq_Len, Input_Dim]
            seq_lens: Optional sequence lengths per sample
        Returns:
            logits: [Batch, Num_Classes]
        """
        batch_size, max_len, _ = x.shape
        device = x.device
        
        x_proj = F.relu(self.input_proj(x))
        x_proj = self.dropout(x_proj)
        
        # Run GRU
        gru_out, h_n = self.gru(x_proj)  # gru_out: [Batch, Seq_Len, Hidden_Dim]
        
        # Masked Average Pooling or Last Valid Hidden State
        if seq_lens is not None:
            lengths_tensor = seq_lens.to(device) if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=device)
            mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths_tensor.unsqueeze(1)).unsqueeze(-1).float()
            pooled = (gru_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = gru_out.mean(dim=1)
            
        logits = self.classifier(pooled)
        return logits


# =====================================================================
# Factory Function to instantiate selected model architecture
# =====================================================================
def get_lightweight_model(model_type, input_dim, d_model=config_lightweight.D_MODEL, 
                          dropout=config_lightweight.DROPOUT, num_classes=3):
    """
    Instantiates one of the 3 lightweight temporal model architectures:
    - 'attention_pool': Single learned query attention over frames
    - 'tcn': 1-2 Layer Temporal Convolutional Network
    - 'gru': Single-layer GRU
    """
    model_type_clean = model_type.lower().strip()
    
    if model_type_clean in ["attention_pool", "attn_pool", "attention"]:
        return AttentionPoolingClassifier(input_dim=input_dim, d_model=d_model, dropout=dropout, num_classes=num_classes)
    elif model_type_clean in ["tcn", "temporal_conv", "conv1d"]:
        return TCNClassifier(input_dim=input_dim, d_model=d_model, dropout=dropout, num_classes=num_classes)
    elif model_type_clean in ["gru", "single_gru"]:
        return SingleLayerGRUClassifier(input_dim=input_dim, hidden_dim=d_model, dropout=dropout, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from ['attention_pool', 'tcn', 'gru'].")
