import os

# =====================================================================
# Google Drive Paths & Hyperparameters Config
# ABLATION STUDY: EfficientNet-B0 + OpenFace Dual Stream + BiLSTM
# Temporal Modeling Ablation (BiLSTM vs Transformer vs No Temporal)
# Binary Engagement Classification: Engaged vs Disengaged (2 Classes)
# =====================================================================

# Candidate Base directory paths on Google Colab and Local
POSSIBLE_DRIVE_ROOTS = [
    "/content/drive/MyDrive/Student_Engagement_DL",
    "/content/drive/MyDrive",
    "/content/drive/Student_Engagement_DL",
    "/content/Student_Engagement_DL",
]
LOCAL_WORKSPACE_DIR = "c:/FYP"

# Detect runtime environment
RUNNING_ON_COLAB = os.path.exists("/content")

def auto_detect_project_root():
    if RUNNING_ON_COLAB:
        for root in POSSIBLE_DRIVE_ROOTS:
            if os.path.exists(root):
                return root
        return "/content/drive/MyDrive/Student_Engagement_DL"
    return LOCAL_WORKSPACE_DIR

PROJECT_ROOT = auto_detect_project_root()

# ── Feature directory detection (Handles spelling variations) ─────────
def auto_detect_sequence_dir(project_root):
    candidates = [
        "sequence_fuse_features_binary",   # Exactly as user specified in Drive
        "sequence_fused_features_binary",  # Alternate spelling with 'd'
        "sequence_fuse_features",
        "sequence_fused_features",
    ]
    for name in candidates:
        p = os.path.join(project_root, name)
        if os.path.exists(p) and os.path.isdir(p):
            return p

    if RUNNING_ON_COLAB:
        for drive_base in POSSIBLE_DRIVE_ROOTS:
            for name in candidates:
                p = os.path.join(drive_base, name)
                if os.path.exists(p) and os.path.isdir(p):
                    return p
            try:
                if os.path.exists(drive_base):
                    for sub in os.listdir(drive_base):
                        sub_path = os.path.join(drive_base, sub)
                        if os.path.isdir(sub_path):
                            for name in candidates:
                                p = os.path.join(sub_path, name)
                                if os.path.exists(p) and os.path.isdir(p):
                                    return p
            except Exception:
                pass

    return os.path.join(project_root, "sequence_fuse_features_binary")

SEQUENCE_FUSED_DIR = auto_detect_sequence_dir(PROJECT_ROOT)

# Dedicated output directories for BiLSTM Ablation Model
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models_binary_bilstm")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results_binary_bilstm")

if not RUNNING_ON_COLAB:
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# =====================================================================
# BINARY CLASSIFICATION: 2 classes — Engaged & Disengaged
# =====================================================================
CLASSES = ["engaged", "disengaged"]
CLASS_NAMES = ["Engaged", "Disengaged"]
NUM_CLASSES = 2

# =====================================================================
# DUAL STREAM FEATURE DIMENSIONS
# =====================================================================
EFFICIENTNET_DIM = 1280  # Stream 1: EfficientNet-B0 visual features
OPENFACE_DIM     = 709   # Stream 2: OpenFace AU, landmarks, gaze, head pose
FUSED_INPUT_DIM  = 1989  # Dual-Stream Fusion: 1280 + 709

# =====================================================================
# SLIDING WINDOW CONFIG
# (Identical to colab_binary_transformer for strict scientific uniformity)
# =====================================================================
WINDOW_SIZE = 120       # Frames per window
WINDOW_STRIDE = 30      # Stride between windows (75% overlap)

# =====================================================================
# BiLSTM ARCHITECTURE HYPERPARAMETERS
# =====================================================================
D_MODEL = 128           # Linear frame projection dimension (same as Transformer)
LSTM_HIDDEN = 64        # Bidirectional: 64 * 2 directions = 128 output dimension
LSTM_LAYERS = 2         # 2-layer stacked BiLSTM for temporal modeling
BIDIRECTIONAL = True    # Bidirectional LSTM (processes forward & backward in time)
DROPOUT = 0.3           # Dropout rate (identical to team standard)
MAX_GRAD_NORM = 1.0     # Gradient clipping threshold

# =====================================================================
# TRAINING HYPERPARAMETERS (Identical to colab_binary_transformer)
# =====================================================================
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 60
PATIENCE = 10           # Early stopping patience (stops when validation peaks)
RANDOM_STATE = 42

# Output files
MODEL_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "binary_bilstm_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions_binary_bilstm.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix_binary_bilstm.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve_binary_bilstm.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics_binary_bilstm.json")
