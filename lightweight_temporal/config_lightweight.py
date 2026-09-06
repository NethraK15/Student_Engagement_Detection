import os

# =====================================================================
# Google Drive Paths & Hyperparameters Config for Lightweight Temporal Experiment
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

# Output & Fused Feature directories for Lightweight Temporal Models
SEQUENCE_FUSED_DIR = os.path.join(PROJECT_ROOT, "sequence_fused_features_lightweight")

def detect_fused_features_dir(project_root):
    candidates = [
        "sequence_fused_features_lightweight",
        "sequence_fused_features",
        "sequence_fused_features_transformer",
        "exp5_sequence_fused_features",
        "exp6_sequence_fused_features",
        "fused_features",
        "exp4_sequence_fused_features"
    ]
    # 1. Check Colab root /content/
    if RUNNING_ON_COLAB:
        for c in candidates:
            path = os.path.join("/content", c)
            if os.path.exists(path) and len([f for f in os.listdir(path) if f.endswith('.pt')]) > 0:
                return path

    # 2. Check Drive / Project root
    for c in candidates:
        path = os.path.join(project_root, c)
        if os.path.exists(path) and len([f for f in os.listdir(path) if f.endswith('.pt')]) > 0:
            return path

    return os.path.join(project_root, "sequence_fused_features_lightweight")

SEQUENCE_FUSED_DIR = detect_fused_features_dir(PROJECT_ROOT)

MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models_lightweight")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results_lightweight")

# Ensure output directories exist locally if running locally
if not RUNNING_ON_COLAB:
    os.makedirs(SEQUENCE_FUSED_DIR, exist_ok=True)
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# Default Model Selection: "attention_pool", "tcn", or "gru"
DEFAULT_MODEL_TYPE = "attention_pool"

# Model Architecture Parameters
D_MODEL = 64         # Projection / Hidden dimension (lightweight for small datasets)
MAX_SEQ_LEN = 120    # Max frames per sequence clip
DROPOUT = 0.3        # Regularization dropout
MAX_GRAD_NORM = 1.0  # Gradient clipping threshold

# Specific Parameters per Architecture Variant
# 1. Attention Pooling Parameters
ATTN_QUERY_DIM = 64

# 2. TCN Parameters
TCN_KERNEL_SIZE = 3
TCN_NUM_LAYERS = 2

# 3. GRU Parameters
GRU_NUM_LAYERS = 1
GRU_HIDDEN_DIM = 64

# Training Hyperparameters & Optimization
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3  # L2 Regularization to prevent overfitting on tiny datasets (~100 clips)
EPOCHS = 50
PATIENCE = 8         # Early stopping patience
RANDOM_STATE = 42

# Dynamic Output File Paths Helper
def get_model_paths(model_type=DEFAULT_MODEL_TYPE):
    m_name = model_type.lower()
    weights_path = os.path.join(MODEL_OUT_DIR, f"lightweight_{m_name}_engagement_weights.pth")
    preds_path = os.path.join(EVAL_OUT_DIR, f"predictions_{m_name}.csv")
    cm_path = os.path.join(EVAL_OUT_DIR, f"confusion_matrix_{m_name}.png")
    roc_path = os.path.join(EVAL_OUT_DIR, f"roc_curve_{m_name}.png")
    metrics_path = os.path.join(EVAL_OUT_DIR, f"metrics_{m_name}.json")
    return {
        "weights": weights_path,
        "predictions_csv": preds_path,
        "confusion_matrix": cm_path,
        "roc_curve": roc_path,
        "metrics_json": metrics_path
    }

MODEL_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "lightweight_attention_pool_engagement_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions_attention_pool.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix_attention_pool.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve_attention_pool.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics_attention_pool.json")
