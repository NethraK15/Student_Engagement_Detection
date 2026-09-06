import os

# =====================================================================
# Experiment 4: Google Drive Paths & Hyperparameters Config
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

# Preprocessing directories - Explicitly saved to Google Drive for Experiment 4
# (Preventing loss on Colab session timeouts)
PREPROCESSED_FRAMES_DIR = os.path.join(PROJECT_ROOT, "exp4_preprocessed_frames")
OPENFACE_FEATURES_DIR = os.path.join(PROJECT_ROOT, "exp4_openface_features")
EFFICIENTNET_FEATURES_DIR = os.path.join(PROJECT_ROOT, "exp4_efficientnet_features")
SEQUENCE_FUSED_DIR = os.path.join(PROJECT_ROOT, "exp4_sequence_fused_features")

# Model checkpoints and evaluation results in Google Drive
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models_exp4")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results_exp4")

# Ensure output directories exist locally if running locally
if not RUNNING_ON_COLAB:
    os.makedirs(PREPROCESSED_FRAMES_DIR, exist_ok=True)
    os.makedirs(OPENFACE_FEATURES_DIR, exist_ok=True)
    os.makedirs(EFFICIENTNET_FEATURES_DIR, exist_ok=True)
    os.makedirs(SEQUENCE_FUSED_DIR, exist_ok=True)
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# Image size for EfficientNet-B0 input
IMAGE_SIZE = (224, 224)

# Feature dimensions
EFFICIENTNET_DIM = 1280
OPENFACE_FALLBACK_DIM = 128

# Training Hyperparameters for Bidirectional LSTM (BiLSTM)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 60
RANDOM_STATE = 42
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# Output file paths
MODEL_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "exp4_lstm_engagement_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions_exp4.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix_exp4.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve_exp4.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics_exp4.json")
