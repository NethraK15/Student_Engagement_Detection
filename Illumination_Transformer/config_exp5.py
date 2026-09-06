import os

# =====================================================================
# Experiment 5: Temporal Transformer Sequence Model Configuration
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

# Dataset class definitions
CLASSES = ["engaged", "partially engaged", "disengaged"]
CLASS_NAMES = ["Engaged", "Partially Engaged", "Disengaged"]

# Feature Source: REUSES PRE-EXTRACTED ILLUMINATED FEATURES FROM EXPERIMENT 4
SEQUENCE_FUSED_DIR = os.path.join(PROJECT_ROOT, "exp4_sequence_fused_features")

# Experiment 5 specific output directories in Google Drive
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models_exp5")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results_exp5")

# Ensure output directories exist locally if running locally
if not RUNNING_ON_COLAB:
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# Temporal Transformer Encoder Hyperparameters
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 50
RANDOM_STATE = 42

# Output file paths in Google Drive
MODEL_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "exp5_transformer_engagement_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions_exp5.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix_exp5.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve_exp5.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics_exp5.json")
