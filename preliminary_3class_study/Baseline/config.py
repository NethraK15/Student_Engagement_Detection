import os

# =====================================================================
# Google Drive Paths & Hyperparameters Config
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

# Dataset folder names (should match the folders inside the dataset directory)
CLASSES = ["engaged", "partially engaged", "disengaged"]

# Human-readable labels for print reports and evaluation charts
CLASS_NAMES = ["Engaged", "Partially Engaged", "Disengaged"]

# Preprocessing directories
# Resized frames can be stored locally on Colab's SSD (highly recommended for performance)
if RUNNING_ON_COLAB:
    PREPROCESSED_FRAMES_DIR = "/content/preprocessed_frames"
    OPENFACE_FEATURES_DIR = os.path.join(PROJECT_ROOT, "openface_features")
    EFFICIENTNET_FEATURES_DIR = os.path.join(PROJECT_ROOT, "efficientnet_features")
else:
    PREPROCESSED_FRAMES_DIR = os.path.join(PROJECT_ROOT, "preprocessed_frames")
    OPENFACE_FEATURES_DIR = os.path.join(PROJECT_ROOT, "openface_features")
    EFFICIENTNET_FEATURES_DIR = os.path.join(PROJECT_ROOT, "efficientnet_features")

# Final fused features, model checkpoints, and evaluation results go directly to Google Drive (persistent)
FUSED_FEATURES_DIR = os.path.join(PROJECT_ROOT, "fused_features")
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results")

# Ensure output directories exist locally if running locally
if not RUNNING_ON_COLAB:
    os.makedirs(PREPROCESSED_FRAMES_DIR, exist_ok=True)
    os.makedirs(OPENFACE_FEATURES_DIR, exist_ok=True)
    os.makedirs(EFFICIENTNET_FEATURES_DIR, exist_ok=True)
    os.makedirs(FUSED_FEATURES_DIR, exist_ok=True)
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# Image size for EfficientNet-B0 input
IMAGE_SIZE = (224, 224)

# Feature dimensions
EFFICIENTNET_DIM = 1280
OPENFACE_FALLBACK_DIM = 128 # Default AU/landmark vector size if OpenFace is not used

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 30
RANDOM_STATE = 42

# Output files
MODEL_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "baseline_engagement_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics.json")
