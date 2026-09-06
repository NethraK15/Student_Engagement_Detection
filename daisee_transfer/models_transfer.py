import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import config_daisee

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding for Temporal Transformer.
    """
    def __init__(self, d_model, max_len=config_daisee.MAX_SEQ_LEN, dropout=0.1):
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
    Standard Temporal Transformer Classifier (from colab_transformer).
    1. Linear Frame Projection: input_dim -> d_model (128)
    2. Sinusoidal Positional Encoding
    3. PyTorch Transformer Encoder Stack (Multi-Head Self-Attention)
    4. Temporal Mean Pooling across frames
    5. Linear FC Classification Head (d_model -> num_classes)
    """
    def __init__(self, input_dim, d_model=config_daisee.D_MODEL, nhead=config_daisee.NHEAD,
                 num_layers=config_daisee.NUM_LAYERS, dim_feedforward=config_daisee.DIM_FEEDFORWARD,
                 dropout=config_daisee.DROPOUT, num_classes=config_daisee.DAISEE_NUM_CLASSES):
        super(EngagementTransformerClassifier, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, seq_lens=None):
        batch_size, max_len, _ = x.shape
        device = x.device

        x_proj = self.input_proj(x)
        x_pos = self.pos_encoder(x_proj)

        src_key_padding_mask = None
        if seq_lens is not None:
            lengths_tensor = seq_lens.to(device) if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=device)
            src_key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths_tensor.unsqueeze(1)

        enc_out = self.transformer_encoder(x_pos, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = enc_out.mean(dim=1)

        logits = self.classifier(pooled)
        return logits

class AttentionPoolingClassifier(nn.Module):
    def __init__(self, input_dim, d_model=config_daisee.D_MODEL, dropout=config_daisee.DROPOUT, num_classes=config_daisee.DAISEE_NUM_CLASSES):
        super(AttentionPoolingClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(1, d_model) / math.sqrt(d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, seq_lens=None):
        batch_size, max_len, _ = x.shape
        device = x.device
        
        x_proj = self.layer_norm(self.input_proj(x))
        scores = torch.sum(x_proj * self.query.unsqueeze(0), dim=-1) / math.sqrt(self.d_model)
        
        if seq_lens is not None:
            lengths_tensor = seq_lens.to(device) if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=device)
            mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths_tensor.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropout = self.dropout(attn_weights)
        
        pooled = torch.sum(attn_weights_dropout.unsqueeze(-1) * x_proj, dim=1)
        logits = self.classifier(pooled)
        return logits

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, d_model=config_daisee.D_MODEL, kernel_size=3, num_layers=2, dropout=config_daisee.DROPOUT, num_classes=config_daisee.DAISEE_NUM_CLASSES):
        super(TCNClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        self.tcn_block = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, seq_lens=None):
        x_trans = x.transpose(1, 2)
        feat = self.input_proj(x_trans)
        feat = self.tcn_block(feat)
        pooled = self.global_pool(feat).squeeze(-1)
        logits = self.classifier(pooled)
        return logits

class SingleLayerGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=config_daisee.D_MODEL, dropout=config_daisee.DROPOUT, num_classes=config_daisee.DAISEE_NUM_CLASSES):
        super(SingleLayerGRUClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, seq_lens=None):
        batch_size, max_len, _ = x.shape
        device = x.device
        
        x_proj = F.relu(self.input_proj(x))
        x_proj = self.dropout(x_proj)
        gru_out, h_n = self.gru(x_proj)
        
        if seq_lens is not None:
            lengths_tensor = seq_lens.to(device) if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=device)
            mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths_tensor.unsqueeze(1)).unsqueeze(-1).float()
            pooled = (gru_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = gru_out.mean(dim=1)
            
        logits = self.classifier(pooled)
        return logits

def get_transfer_model(model_type, input_dim, d_model=config_daisee.D_MODEL, dropout=config_daisee.DROPOUT, num_classes=config_daisee.NUM_TARGET_CLASSES):
    model_type_clean = model_type.lower().strip()
    if model_type_clean in ["transformer", "colab_transformer", "temporal_transformer"]:
        return EngagementTransformerClassifier(
            input_dim=input_dim,
            d_model=d_model,
            nhead=config_daisee.NHEAD,
            num_layers=config_daisee.NUM_LAYERS,
            dim_feedforward=config_daisee.DIM_FEEDFORWARD,
            dropout=dropout,
            num_classes=num_classes
        )
    elif model_type_clean in ["attention_pool", "attn_pool", "attention"]:
        return AttentionPoolingClassifier(input_dim=input_dim, d_model=d_model, dropout=dropout, num_classes=num_classes)
    elif model_type_clean in ["tcn", "temporal_conv", "conv1d"]:
        return TCNClassifier(input_dim=input_dim, d_model=d_model, dropout=dropout, num_classes=num_classes)
    elif model_type_clean in ["gru", "single_gru"]:
        return SingleLayerGRUClassifier(input_dim=input_dim, hidden_dim=d_model, dropout=dropout, num_classes=num_classes)
    else:
        return EngagementTransformerClassifier(
            input_dim=input_dim,
            d_model=d_model,
            nhead=config_daisee.NHEAD,
            num_layers=config_daisee.NUM_LAYERS,
            dim_feedforward=config_daisee.DIM_FEEDFORWARD,
            dropout=dropout,
            num_classes=num_classes
        )

def load_pretrained_daisee_weights(model, pretrained_weights_path, num_target_classes=config_daisee.NUM_TARGET_CLASSES, freeze_backbone=False):
    if not torch.cuda.is_available():
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
    else:
        checkpoint = torch.load(pretrained_weights_path)
        
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    model_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if "classifier.3" in k or "classifier.final" in k or model_dict[k].shape != v.shape:
                print(f"[Transfer Learning] Skipping top classification layer weight '{k}' (Target shape: {model_dict[k].shape}, Pretrained shape: {v.shape})")
            else:
                filtered_state_dict[k] = v
                
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)
    print(f"[Transfer Learning] Loaded {len(filtered_state_dict)} pretrained temporal parameter layers from DAiSEE Transformer checkpoint!")
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not ("classifier" in name):
                param.requires_grad = False
        print("[Transfer Learning] Frozen temporal Transformer backbone layers. Training classifier head ONLY.")
        
    return model
