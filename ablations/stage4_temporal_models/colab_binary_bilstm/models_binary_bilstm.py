import math
import torch
import torch.nn as nn
import torchvision.models as models
import config_binary_bilstm as config

class BinaryEngagementBiLSTM(nn.Module):
    """
    ABLATION ARCHITECTURE: Dual-Stream Fusion + BiLSTM Temporal Modeling.
    
    Ablation Context:
      In the full architecture, video frames are modeled using a Multi-Head
      Self-Attention Transformer Encoder.
      
      In this Ablation Experiment, the Transformer is replaced with a
      Bidirectional LSTM (BiLSTM) — the classic recurrent benchmark in
      video sequence and engagement modeling:
        1. EfficientNet-B0 (1280-dim) + OpenFace (709-dim) Dual Stream
        2. Fusion & Linear Frame Projection (1989 -> 128) + LayerNorm
        3. 2-Layer Bidirectional LSTM (BiLSTM) for recurrent temporal modeling
        4. Masked Temporal Mean Pooling
        5. MLP Classification Head (128 -> 64 -> 32 -> 2)
    """
    def __init__(
        self,
        input_dim=config.FUSED_INPUT_DIM,
        d_model=config.D_MODEL,
        lstm_hidden=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        bidirectional=config.BIDIRECTIONAL,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES
    ):
        super(BinaryEngagementBiLSTM, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # 1. Dual-Stream Fusion Projection Layer (1989 -> 128)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # 2. Bidirectional LSTM Stack (Temporal Modeling)
        # If bidirectional, hidden=64 * 2 directions = 128 output dim (matches d_model)
        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # 3. Binary MLP Classification Head
        # EXACTLY IDENTICAL to Binary Transformer Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, d_model // 2),        # 128 -> 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),       # 64 -> 32
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, num_classes)         # 32 -> 2 (Engaged / Disengaged)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x        : [Batch, Window_Size, Input_Dim]
            seq_lens : Optional sequence lengths (for padded sequences)
        Returns:
            logits   : [Batch, 2] (Engaged vs Disengaged)
        """
        batch_size, max_len, _ = x.shape
        device = x.device

        # 1. Linear Projection: [B, T, 1989] -> [B, T, d_model]
        x_proj = self.input_proj(x)

        # 2. BiLSTM Temporal Modeling
        lstm_out, _ = self.lstm(x_proj)  # [B, T, lstm_out_dim=128]
        lstm_out = self.lstm_norm(lstm_out)

        # 3. Masked Temporal Mean Pooling
        src_key_padding_mask = None
        if seq_lens is not None:
            lengths_tensor = (
                seq_lens.to(device)
                if isinstance(seq_lens, torch.Tensor)
                else torch.tensor(seq_lens, device=device)
            )
            src_key_padding_mask = (
                torch.arange(max_len, device=device).unsqueeze(0) >= lengths_tensor.unsqueeze(1)
            )

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = lstm_out.mean(dim=1)

        # 4. Binary Classification Head -> [B, 2]
        logits = self.classifier(pooled)
        return logits


def get_efficientnet_backbone_param_counts():
    try:
        effnet_full = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        try:
            effnet_full = models.efficientnet_b0(pretrained=True)
        except Exception:
            return 4007548, 5288548

    total_full = sum(p.numel() for p in effnet_full.parameters())
    effnet_full.classifier = nn.Identity()
    total_features_only = sum(p.numel() for p in effnet_full.parameters())
    return total_features_only, total_full


def display_parameter_summary(model, input_dim=config.FUSED_INPUT_DIM):
    """
    Calculates and displays a comprehensive parameter count breakdown
    of the BiLSTM ablation architecture.
    """
    input_proj_params = sum(p.numel() for p in model.input_proj.parameters())
    lstm_params       = sum(p.numel() for p in model.lstm.parameters()) + sum(p.numel() for p in model.lstm_norm.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    trainable_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_model_params = sum(p.numel() for p in model.parameters())

    effnet_feat_params, effnet_full_params = get_efficientnet_backbone_param_counts()

    total_feat_pipeline = effnet_feat_params + total_model_params
    total_full_pipeline = effnet_full_params + total_model_params

    # Reference parameter counts for 3-way ablation comparison
    no_temp_trainable    = 265378  # Direct Mean Pooling Ablation
    transformer_trainable = 663586  # 3-Layer Transformer Encoder

    print("\n" + "=" * 75)
    print("        ARCHITECTURE PARAMETER COUNT: DUAL-STREAM + BiLSTM")
    print("=" * 75)
    print(f"  Stream 1: EfficientNet-B0 Visual Extractor      : {effnet_feat_params:,} params (Features only)")
    print(f"            [Full Standard EfficientNet-B0 Model] : {effnet_full_params:,} params (~5.29M)")
    print(f"  Stream 2: OpenFace Facial Behavior Stream       : {config.OPENFACE_DIM} numerical features (0 PyTorch params)")
    print(f"  Dual-Stream Fused Input Dimension               : {input_dim} (1280 CNN + 709 OpenFace)")
    print("-" * 75)
    print("  BiLSTM ABLATION MODEL COMPONENTS:")
    print(f"    • Fusion & Projection (Linear + LayerNorm)    : {input_proj_params:,} params")
    print(f"    • 2-Layer BiLSTM Stack (Hidden=64, Bidirect)  : {lstm_params:,} params")
    print(f"    • Temporal Pooling Layer                      : 0 params (Masked Mean Pooling)")
    print(f"    • MLP Classification Head (Binary 2-class)    : {classifier_params:,} params")
    print(f"    -------------------------------------------------------")
    print(f"    Total Trainable Parameters in BiLSTM Model    : {trainable_params:,}")
    print("-" * 75)
    print("  ENTIRE END-TO-END PIPELINE (Backbone + BiLSTM Head):")
    print(f"    • Option A (Conv Features + BiLSTM Model)     : {total_feat_pipeline:,} params (~{total_feat_pipeline/1e6:.2f}M)")
    print(f"    • Option B (Full Standard B0 + BiLSTM Model)  : {total_full_pipeline:,} params (~{total_full_pipeline/1e6:.2f}M)")
    print("-" * 75)
    print("  TEMPORAL MODELING 3-WAY ABLATION COMPARISON:")
    print(f"    1. No Temporal Modeling (Direct Mean Pooling) : {no_temp_trainable:,} trainable params")
    print(f"    2. BiLSTM Recurrent Modeling (This Model)     : {trainable_params:,} trainable params")
    print(f"    3. Transformer Self-Attention (Full Proposed) : {transformer_trainable:,} trainable params")
    print("=" * 75 + "\n")

    return {
        "input_proj_params": input_proj_params,
        "lstm_params": lstm_params,
        "classifier_params": classifier_params,
        "trainable_params": trainable_params,
        "total_model_params": total_model_params,
        "effnet_feat_params": effnet_feat_params,
        "effnet_full_params": effnet_full_params,
        "total_feat_pipeline": total_feat_pipeline,
        "total_full_pipeline": total_full_pipeline,
    }
