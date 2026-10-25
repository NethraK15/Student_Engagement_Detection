"""
Global Configuration for Dual-Stream Temporal Transformer Framework
Matches experimental settings from:
'A Dual-Stream Temporal Transformer Framework for Student Engagement Detection in Classroom Videos'
"""

import os

# --- Dataset and Windowing Configuration ---
TOTAL_VIDEOS = 133
VIDEO_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
WINDOW_SIZE = 120       # W = 120 frames (4 seconds at 30 fps)
STRIDE = 30             # S = 30 frames (1 second, 75% overlap)
NUM_CLASSES = 2         # Binary classification: 0 = Engaged, 1 = Disengaged

# --- Feature Dimensions ---
STREAM1_APPEARANCE_DIM = 1280    # EfficientNet-B0 global average pooled embedding
STREAM2_BEHAVIORAL_DIM = 709     # OpenFace 2.0 (17 AUs, 3D gaze, 6-DoF pose, 2D/3D landmarks)
FUSED_FEATURE_DIM = 1989         # Concatenation: 1280 + 709

# --- Transformer Architecture (Pre-LN) ---
D_MODEL = 128                    # Latent projection dimension
NUM_HEADS = 4                    # Multi-Head Attention heads
D_K = 32                         # Key/Query/Value dimension per head (4 * 32 = 128)
NUM_LAYERS = 3                   # L = 3 Pre-LayerNorm Transformer encoder layers
DIM_FEEDFORWARD = 256            # Inner dimension of FFN
DROPOUT_INPUT = 0.3              # Positional encoding / projection dropout
DROPOUT_HEAD_1 = 0.3             # First classification layer dropout
DROPOUT_HEAD_2 = 0.15            # Second classification layer dropout

# --- Training Hyperparameters ---
BATCH_SIZE = 16                  # Batch size in windows
LEARNING_RATE = 5e-4             # AdamW initial learning rate
WEIGHT_DECAY = 1e-3              # AdamW weight decay
MIN_LR = 1e-6                    # Cosine annealing minimum learning rate
MAX_EPOCHS = 60                  # Maximum training epochs
EARLY_STOPPING_PATIENCE = 10     # Early stopping on validation loss
GRADIENT_CLIP_NORM = 1.0         # Gradient norm clipping

# --- Paths ---
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "splits")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
