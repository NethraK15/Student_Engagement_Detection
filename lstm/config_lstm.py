import os

# =====================================================================
# Google Drive Paths & Hyperparameters Config for Experiment 2 (LSTM)
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

# Experiment 2 specific output directories
SEQUENCE_FUSED_DIR = os.path.join(PROJECT_ROOT, "sequence_fused_features")
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models_lstm")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results_lstm")

# Ensure output directories exist locally if running locally
if not RUNNING_ON_COLAB:
    os.makedirs(SEQUENCE_FUSED_DIR, exist_ok=True)
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# Sequence Modeling Parameters
MAX_SEQ_LEN = 120    # Max frames per sequence (pads shorter, truncates longer)
HIDDEN_DIM = 128     # BiLSTM hidden dim (128 x 2 = 256 output dimensions)
NUM_LAYERS = 2       # Number of stacked LSTM layers
DROPOUT = 0.3        # Dropout probability

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
EPOCHS = 60
RANDOM_STATE = 42

# Output files
MODEL_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "lstm_engagement_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions_lstm.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix_lstm.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve_lstm.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics_lstm.json")
