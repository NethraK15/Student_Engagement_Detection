import torch
import torch.nn as nn
import torch.nn.functional as F

class EngagementLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM (BiLSTM) Classifier with Self-Attention Temporal Pooling
    for Student Engagement Detection in Experiment 4.
    
    Takes 3D sequence tensors of shape [Batch, Seq_Len, Feature_Dim]
    extracted from CLAHE+Sharpened illuminated frames and outputs logits
    for the 3 engagement classes:
    - Engaged (0)
    - Partially Engaged (1)
    - Disengaged (2)
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.3):
        super(EngagementLSTMClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. Input normalization layer
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 2. Bidirectional LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3. Self-Attention Temporal Pooling Layer across frames
        # Maps [Batch, Seq_Len, 2 * hidden_dim] -> [Batch, Seq_Len, 1] -> Softmax attention weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 4. Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x: Input sequence tensor of shape [Batch, Seq_Len, Input_Dim]
            seq_lens: Optional 1D tensor/list containing unpadded lengths per sample
        Returns:
            logits: Output classification scores of shape [Batch, Num_Classes]
        """
        # 1. Normalize input features channel-wise
        x = self.input_norm(x)
        
        # 2. Pass through Bidirectional LSTM with optional packed sequences
        if seq_lens is not None:
            lengths_cpu = seq_lens.cpu() if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, dtype=torch.int64)
            lengths_cpu = torch.clamp(lengths_cpu, min=1)
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed_x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            lstm_out, _ = self.lstm(x)
        
        # 3. Self-attention temporal pooling across frames
        attn_scores = self.attention(lstm_out)                # [Batch, Seq_Len, 1]
        
        if seq_lens is not None:
            batch_size, max_len, _ = lstm_out.shape
            device = x.device
            lengths_tensor = seq_lens.to(device) if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=device)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths_tensor.unsqueeze(1)
            mask = mask.unsqueeze(-1)                          # [Batch, Seq_Len, 1]
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)           # Normalize across Seq_Len
        
        # 4. Compute weighted sum across frames -> [Batch, 2 * hidden_dim]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        # 5. Pass through classification head -> [Batch, Num_Classes]
        logits = self.classifier(context_vector)
        return logits
