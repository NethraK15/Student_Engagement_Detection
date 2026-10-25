"""
Proposed Dual-Stream Temporal Transformer Architecture
Architecture specifications from paper:
- Pre-Layer-Normalized (Pre-LN) Transformer Encoder
- Layers: 3, Heads: 4, d_model: 128, d_k: 32, d_ffn: 256
- Masked temporal mean pooling + Tapering MLP classification head
- Exactly 663,074 trainable parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=500, dropout=0.3):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PreLNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1):
        super(PreLNTransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, key_padding_mask=None):
        # Pre-LN Self-Attention sub-layer
        src_norm = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            src_norm, src_norm, src_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout1(attn_out)

        # Pre-LN Feed-Forward sub-layer
        src_norm = self.norm2(src)
        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(src_norm))))
        src = src + self.dropout3(ff_out)
        return src, attn_weights

class DualStreamTemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim=1989,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        num_classes=2,
        dropout_input=0.3,
        dropout_head1=0.3,
        dropout_head2=0.15
    ):
        super(DualStreamTemporalTransformer, self).__init__()
        
        # 1. Linear Projection & Input LayerNorm (1,989 -> 128)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout_input)

        # 2. Pre-LN Transformer Encoder Layers (L=3)
        self.layers = nn.ModuleList([
            PreLNTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 3. Tapering MLP Classification Head: Linear(128->64) - GELU - Linear(64->32) - GELU - Linear(32->2)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout_head1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout_head2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, window_size=120, input_dim=1989]
            mask: Optional binary mask of shape [batch_size, window_size] (1 for valid, 0 for padded)
        Returns:
            logits: [batch_size, num_classes=2]
            attn_maps: list of attention weight tensors from each encoder layer
        """
        # Linear projection and LayerNorm
        z = self.input_norm(self.input_projection(x))
        z = self.pos_encoder(z)

        # Pre-LN Transformer Layers
        attn_maps = []
        key_padding_mask = (mask == 0) if mask is not None else None
        for layer in self.layers:
            z, attn_weights = layer(z, key_padding_mask=key_padding_mask)
            attn_maps.append(attn_weights)

        z = self.final_norm(z)

        # Masked Temporal Mean Pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, window_size, 1]
            pooled = torch.sum(z * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        else:
            pooled = torch.mean(z, dim=1)  # [batch_size, d_model]

        logits = self.classifier(pooled)
        return logits, attn_maps

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = DualStreamTemporalTransformer()
    total_trainable = model.count_trainable_parameters()
    print(f">>> DualStreamTemporalTransformer instantiated successfully.")
    print(f">>> Trainable Parameters: {total_trainable:,} (Target in Paper Table 2: 663,074)")
    dummy_input = torch.randn(4, 120, 1989)
    logits, _ = model(dummy_input)
    print(f">>> Forward pass output shape: {logits.shape}")
