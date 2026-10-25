import os

# =====================================================================
# Google Drive Paths & Hyperparameters Config
# Binary Engagement Transformer (Engaged vs Disengaged, 2 Classes)
# =====================================================================

# Base directory paths
DRIVE_BASE_DIR = "/content/drive/MyDrive/Student_Engagement_DL"
LOCAL_WORKSPACE_DIR = "d:/Amrita/Final_Year_Project/Sem_6 Work/Student_Engagement_DL"

# Detect runtime environment (Google Colab vs Local Windows)
if os.path.exists("/content"):
    PROJECT_ROOT = DRIVE_BASE_DIR
    RUNNING_ON_COLAB = True
else:
    PROJECT_ROOT = LOCAL_WORKSPACE_DIR
    RUNNING_ON_COLAB = False

# ── Dataset directory ────────────────────────────────────────────────
# Your dataset folder on Google Drive is: New_Dataset_Crafted
# (This is checked first in the candidates list below)
# ─────────────────────────────────────────────────────────────────────
def auto_detect_dataset_dir(project_root):
    if RUNNING_ON_COLAB and os.path.exists("/content/Dataset"):
        return "/content/Dataset"
    if os.path.exists(project_root):
        # ← New_Dataset_Crafted is listed FIRST so it is always found first on Drive
        candidates = [
            "New_Dataset_Crafted",       # ← your actual Drive folder
            "New_Dataset_crafted",       # alternate casing
            "New_Datset_Crafted",        # typo variant (original colab_transformer)
            "Customised_Classroom_Dataset",
            "Customised_Classroom_Dataset_Flipped",
            "Dataset",
            "dataset",
        ]
        for name in candidates:
            path = os.path.join(project_root, name)
            if os.path.exists(path) and os.path.isdir(path):
                # Verify it actually contains class subfolders
                try:
                    subdirs = [d for d in os.listdir(path)
                               if os.path.isdir(os.path.join(path, d))]
                    norm_subdirs = [s.lower().replace(" ", "").replace("_", "") for s in subdirs]
                    if any("engaged" in n or "disengaged" in n for n in norm_subdirs):
                        return path
                except Exception:
                    pass
        # Last resort: any subfolder with "dataset" in the name
        try:
            for item in os.listdir(project_root):
                item_path = os.path.join(project_root, item)
                if os.path.isdir(item_path) and "dataset" in item.lower():
                    return item_path
        except Exception:
            pass
    return os.path.join(project_root, "Dataset")

DATASET_DIR = auto_detect_dataset_dir(PROJECT_ROOT)

# =====================================================================
# BINARY CLASSIFICATION: Only 2 classes — Engaged & Disengaged
# Partially Engaged is completely IGNORED during dataset loading.
# =====================================================================
CLASSES = ["engaged", "disengaged"]
CLASS_NAMES = ["Engaged", "Disengaged"]
NUM_CLASSES = 2  # Binary classification

# NOTE: No separate openface_features/ or efficientnet_features/ dirs needed.
# The preprocessing script (fuse_sequence_features_binary.py) extracts features
# directly from raw videos and saves full fused sequences to SEQUENCE_FUSED_DIR.

# Dedicated output directories for Binary Transformer
def auto_detect_sequence_dir(project_root):
    candidates = [
        "sequence_fuse_features_binary",
        "sequence_fused_features_binary",
    ]
    for name in candidates:
        p = os.path.join(project_root, name)
        if os.path.exists(p) and os.path.isdir(p):
            return p
    if RUNNING_ON_COLAB:
        for drive_base in ["/content/drive/MyDrive/Student_Engagement_DL", "/content/drive/MyDrive", "/content/drive/Student_Engagement_DL"]:
            for name in candidates:
                p = os.path.join(drive_base, name)
                if os.path.exists(p) and os.path.isdir(p):
                    return p
    return os.path.join(project_root, "sequence_fuse_features_binary")

SEQUENCE_FUSED_DIR = auto_detect_sequence_dir(PROJECT_ROOT)

MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models_binary_transformer")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results_binary_transformer")

# Ensure output directories exist locally
if not RUNNING_ON_COLAB:
    os.makedirs(SEQUENCE_FUSED_DIR, exist_ok=True)
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# =====================================================================
# SLIDING WINDOW CONFIG — Learn entire video, not just first 120 frames
# For a 10-15s video at ~30fps = 300-450 frames total.
# We slide a fixed-size window across all frames with overlap.
# =====================================================================
WINDOW_SIZE = 120       # Frames per window (same as original MAX_SEQ_LEN)
WINDOW_STRIDE = 30      # Stride between windows (75% overlap → max coverage)
# e.g., a 300-frame video produces: floor((300 - 120) / 30) + 1 = 7 windows

# Transformer Model Architecture Parameters
D_MODEL = 128           # Transformer embedding dimension
NHEAD = 4               # Number of multi-head attention heads
NUM_LAYERS = 3          # Slightly deeper for binary precision
DIM_FEEDFORWARD = 256   # Feedforward expansion dimension
DROPOUT = 0.3           # Dropout probability
MAX_GRAD_NORM = 1.0     # Gradient clipping threshold

# Training Hyperparameters & Early Stopping
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 60
PATIENCE = 10           # Early stopping patience
RANDOM_STATE = 42

# Output files
MODEL_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "binary_transformer_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions_binary_transformer.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix_binary_transformer.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve_binary_transformer.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics_binary_transformer.json")
