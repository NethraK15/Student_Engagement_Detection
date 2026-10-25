import math
import torch
import torch.nn as nn
import torchvision.models as models
import config_binary_fusion as config

class BinaryEngagementFusionModel(nn.Module):
    """
    ABLATION ARCHITECTURE: Dual-Stream Fusion WITHOUT Transformer.
    
    Ablation Concept:
      In the full architecture, video features pass through:
        1. EfficientNet-B0 (1280-dim) + OpenFace (709-dim) Dual Stream
        2. Fusion (1989-dim)
        3. Positional Encoding + Multi-Head Self-Attention Transformer Stack
        4. Temporal Pooling
        5. MLP Classification Head
      
      In this Ablation Study, we remove the Transformer (and Positional Encoding):
        1. EfficientNet-B0 (1280-dim) + OpenFace (709-dim) Dual Stream
        2. Fusion & Linear Projection (1989 -> 128) with LayerNorm
        3. Masked Temporal Mean Pooling (aggregates frame features over time)
        4. MLP Classification Head (128 -> 64 -> 32 -> 2)
      
      This directly isolates and evaluates the contribution of the Transformer
      component versus standard Dual-Stream Feature Fusion.
    """
    def __init__(
        self,
        input_dim=config.FUSED_INPUT_DIM,
        d_model=config.D_MODEL,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES
    ):
        super(BinaryEngagementFusionModel, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # -------------------------------------------------------------
        # 1. Dual-Stream Fusion Projection Layer
        # Fuses 1280 (EfficientNet-B0) + 709 (OpenFace) = 1989-dim
        # into a unified d_model (128-dim) representation with LayerNorm
        # -------------------------------------------------------------
        self.fusion_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # -------------------------------------------------------------
        # 2. Binary MLP Classification Head
        # EXACTLY IDENTICAL to the Binary Transformer Classification Head
        # -------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),        # 128 -> 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),   # 64 -> 32
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, num_classes)     # 32 -> 2 (Engaged / Disengaged)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x        : [Batch, Window_Size, Input_Dim] or [Batch, Input_Dim]
                       where Input_Dim = 1989 (1280 CNN + 709 OpenFace)
            seq_lens : Optional actual sequence lengths (for padded sequences)
        Returns:
            logits   : [Batch, 2] (Engaged vs Disengaged)
        """
        # If a 2D tensor is passed [Batch, Input_Dim]
        if x.dim() == 2:
            x_proj = self.fusion_proj(x)
            logits = self.classifier(x_proj)
            return logits

        batch_size, max_len, _ = x.shape
        device = x.device

        # 1. Project fused dual-stream features: [B, T, 1989] -> [B, T, d_model]
        x_proj = self.fusion_proj(x)

        # 2. Padding mask (True = padded/ignored position)
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

        # 3. Masked Temporal Mean Pooling (Ablation replacement for Transformer)
        # Directly aggregates temporal information across all video frames
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, T, 1]
            pooled = (x_proj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x_proj.mean(dim=1)  # [B, d_model]

        # 4. Binary Classification Head -> [B, 2]
        logits = self.classifier(pooled)
        return logits


def get_efficientnet_backbone_param_count():
    """
    Calculate parameters of the EfficientNet-B0 visual feature extractor
    used in the preprocessing stream.
    """
    try:
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        try:
            effnet = models.efficientnet_b0(pretrained=True)
        except Exception:
            return 4007548  # Standard EfficientNet-B0 features parameter count
    effnet.classifier = nn.Identity()
    return sum(p.numel() for p in effnet.parameters())


def display_parameter_summary(model, input_dim=config.FUSED_INPUT_DIM):
    """
    Calculates and displays a comprehensive parameter count breakdown
    of the entire architecture for ablation analysis.
    """
    # 1. Ablation model trainable components
    fusion_proj_params = sum(p.numel() for p in model.fusion_proj.parameters())
    classifier_params  = sum(p.numel() for p in model.classifier.parameters())
    trainable_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_model_params = sum(p.numel() for p in model.parameters())

    # 2. Dual-stream backbones
    effnet_params = get_efficientnet_backbone_param_count()
    openface_params = 0  # Algorithmic facial extraction pipeline (AU, landmarks, gaze)

    # 3. Full end-to-end architecture total
    entire_architecture_params = effnet_params + total_model_params

    # 4. Comparison with Transformer baseline (3 layers, d_model=128)
    # 3 * [Self-Attention (66,048) + FeedForward (66,176) + LayerNorm (512)] = 398,208 params
    transformer_stack_params = 398208
    transformer_baseline_trainable = trainable_params + transformer_stack_params

    print("\n" + "=" * 70)
    print("        ARCHITECTURE PARAMETER COUNT & ABLATION SUMMARY")
    print("=" * 70)
    print(f"  Stream 1: EfficientNet-B0 Visual Extractor : {effnet_params:,} params (Pretrained/Frozen)")
    print(f"  Stream 2: OpenFace Facial Behavior Stream  : {config.OPENFACE_DIM} features (Algorithmic / 0 PyTorch params)")
    print(f"  Dual-Stream Fused Input Dimension          : {input_dim} (1280 CNN + 709 OpenFace)")
    print("-" * 70)
    print("  ABLATION MODEL (Without Transformer):")
    print(f"    • Fusion & Projection (Linear + LayerNorm): {fusion_proj_params:,} params")
    print(f"    • Temporal Pooling Layer                   : 0 params (Non-parametric Mean Pooling)")
    print(f"    • MLP Classification Head (Binary 2-class) : {classifier_params:,} params")
    print(f"    -------------------------------------------------------")
    print(f"    Total Trainable Parameters                 : {trainable_params:,}")
    print(f"    Total Ablation Model Parameters            : {total_model_params:,}")
    print("-" * 70)
    print("  ENTIRE END-TO-END PIPELINE (Backbone + Fusion Head):")
    print(f"    Grand Total Architecture Parameters        : {entire_architecture_params:,} (~{entire_architecture_params/1e6:.2f}M)")
    print("-" * 70)
    print("  ABLATION COMPARISON (Transformer vs Non-Transformer):")
    print(f"    • Full Model with Transformer Stack        : ~{transformer_baseline_trainable:,} trainable params")
    print(f"    • Ablation Model (No Transformer)          : {trainable_params:,} trainable params")
    print(f"    • Parameter Reduction                      : -{transformer_stack_params:,} params (-{(transformer_stack_params/transformer_baseline_trainable)*100:.1f}%)")
    print("=" * 70 + "\n")

    return {
        "fusion_proj_params": fusion_proj_params,
        "classifier_params": classifier_params,
        "trainable_params": trainable_params,
        "total_model_params": total_model_params,
        "effnet_backbone_params": effnet_params,
        "entire_architecture_params": entire_architecture_params,
        "transformer_stack_removed_params": transformer_stack_params,
    }
