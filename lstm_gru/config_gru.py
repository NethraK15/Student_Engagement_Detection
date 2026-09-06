import os

# =====================================================================
# Google Drive Paths & Hyperparameters Config for Refined GRU Model
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

# Dynamic Dataset Directory Auto-Detection
def auto_detect_dataset_dir(project_root):
    if RUNNING_ON_COLAB and os.path.exists("/content/Dataset"):
        return "/content/Dataset"
    if os.path.exists(project_root):
        candidates = ["Dataset", "New_Datset_Crafted", "New_Dataset_Crafted", "dataset"]
        for name in candidates:
            path = os.path.join(project_root, name)
            if os.path.exists(path) and os.path.isdir(path):
                return path
        try:
            for item in os.listdir(project_root):
                item_path = os.path.join(project_root, item)
                if os.path.isdir(item_path) and "dataset" in item.lower():
                    return item_path
        except Exception:
            pass
    return os.path.join(project_root, "Dataset")

DATASET_DIR = auto_detect_dataset_dir(PROJECT_ROOT)

# Dataset folder names & human-readable labels
CLASSES = ["engaged", "partially engaged", "disengaged"]
CLASS_NAMES = ["Engaged", "Partially Engaged", "Disengaged"]

# Input directories (reusing pre-extracted features)
if RUNNING_ON_COLAB:
    drive_of = os.path.join(PROJECT_ROOT, "openface_features")
    drive_eff = os.path.join(PROJECT_ROOT, "efficientnet_features")
    OPENFACE_FEATURES_DIR = drive_of if os.path.exists(drive_of) else "/content/openface_features"
    EFFICIENTNET_FEATURES_DIR = drive_eff if os.path.exists(drive_eff) else "/content/efficientnet_features"
else:
    OPENFACE_FEATURES_DIR = os.path.join(PROJECT_ROOT, "openface_features")
    EFFICIENTNET_FEATURES_DIR = os.path.join(PROJECT_ROOT, "efficientnet_features")

# Dedicated output directories for GRU experiment
SEQUENCE_FUSED_DIR = os.path.join(PROJECT_ROOT, "sequence_fused_features_gru")

# Fallback to existing sequence_fused_features if GRU specific directory is empty/missing
if not os.path.exists(SEQUENCE_FUSED_DIR) or len([f for f in os.listdir(SEQUENCE_FUSED_DIR) if f.endswith('.pt')]) == 0:
    fallback_seq_dir = os.path.join(PROJECT_ROOT, "sequence_fused_features")
    if os.path.exists(fallback_seq_dir) and len([f for f in os.listdir(fallback_seq_dir) if f.endswith('.pt')]) > 0:
        SEQUENCE_FUSED_DIR = fallback_seq_dir
        print(f"[Config GRU] Using existing sequence features from: {SEQUENCE_FUSED_DIR}")

MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models_gru")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results_gru")

# Ensure output directories exist locally if running locally
if not RUNNING_ON_COLAB:
    os.makedirs(SEQUENCE_FUSED_DIR, exist_ok=True)
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# Architecture & Sequence Parameters (Compact to prevent overfitting on small dataset)
PROJECTION_DIM = 128 # Linear Bottleneck: 1989 -> 128
HIDDEN_DIM = 64      # BiGRU: 64 hidden units per direction (128 dim output)
MAX_SEQ_LEN = 120    # Max frames per sequence
MAX_GRAD_NORM = 1.0  # Gradient clipping threshold

# Training Hyperparameters & Early Stopping
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3  # L2 Regularization to prevent weight memorization
EPOCHS = 40
PATIENCE = 8         # Early stopping patience
RANDOM_STATE = 42

# Output files
MODEL_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "gru_engagement_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions_gru.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix_gru.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve_gru.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics_gru.json")
