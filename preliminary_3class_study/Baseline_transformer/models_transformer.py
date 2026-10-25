import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config_transformer

class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding for Temporal Transformer.
    Injects frame sequence order into embeddings.
    """
    def __init__(self, d_model, max_len=config_transformer.MAX_SEQ_LEN, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EngagementTransformerClassifier(nn.Module):
    """
    Standard Temporal Transformer Classifier.
    Pure, textbook Transformer Encoder architecture for video sequence classification:
    1. Linear Frame Projection: input_dim (1989) -> d_model (128)
    2. Standard Sinusoidal Positional Encoding
    3. Standard PyTorch Transformer Encoder Stack (Multi-Head Self-Attention)
    4. Temporal Mean Pooling across frames
    5. Linear FC Classification Head (d_model -> 3)
    """
    def __init__(self, input_dim, d_model=config_transformer.D_MODEL, nhead=config_transformer.NHEAD,
                 num_layers=config_transformer.NUM_LAYERS, dim_feedforward=config_transformer.DIM_FEEDFORWARD,
                 dropout=config_transformer.DROPOUT, num_classes=3):
        super(EngagementTransformerClassifier, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # 1. Linear Projection Layer (Input_Dim -> d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Sinusoidal Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # 3. Standard Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. FC Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x: Input sequence tensor of shape [Batch, Seq_Len, Input_Dim]
            seq_lens: Optional sequence lengths per sample
        Returns:
            logits: Output classification scores [Batch, Num_Classes]
        """
        batch_size, max_len, _ = x.shape
        device = x.device

        # 1. Linear Frame Projection -> [Batch, Seq_Len, d_model]
        x_proj = self.input_proj(x)

        # 2. Add Positional Encoding
        x_pos = self.pos_encoder(x_proj)

        # 3. Create Padding Mask (True for padded positions)
        src_key_padding_mask = None
        if seq_lens is not None:
            lengths_tensor = seq_lens.to(device) if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=device)
            src_key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths_tensor.unsqueeze(1)

        # 4. Transformer Encoder -> [Batch, Seq_Len, d_model]
        enc_out = self.transformer_encoder(x_pos, src_key_padding_mask=src_key_padding_mask)

        # 5. Temporal Mean Pooling (Masked)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [Batch, Seq_Len, 1]
            pooled = (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = enc_out.mean(dim=1)

        # 6. Classification Head -> [Batch, Num_Classes]
        logits = self.classifier(pooled)
        return logits
