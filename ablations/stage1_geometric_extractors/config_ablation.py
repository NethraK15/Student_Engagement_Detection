import os
import torch

# =====================================================================
# Configuration for OpenFace Binary Engagement Ablation Study
# =====================================================================

# Environment detection
if os.path.exists("/content"):
    RUNNING_ON_COLAB = True
    DRIVE_BASE_DIR = "/content/drive/MyDrive/Student_Engagement_DL"
else:
    RUNNING_ON_COLAB = False
    DRIVE_BASE_DIR = "d:/FINAL YEAR PROJECT-1/SEM7/Student_Engagement_Detection"

PROJECT_ROOT = DRIVE_BASE_DIR if RUNNING_ON_COLAB else DRIVE_BASE_DIR

# Candidate paths for the binary feature dataset folder
DATASET_CANDIDATE_PATHS = [
    # Colab direct upload or unzipped
    "/content/sequence_fused_features_binary",
    "/content/dataset/sequence_fused_features_binary",
    # Google Drive mounted
    os.path.join(PROJECT_ROOT, "sequence_fused_features_binary"),
    os.path.join(PROJECT_ROOT, "dataset", "sequence_fused_features_binary"),
    os.path.join(PROJECT_ROOT, "Dataset", "sequence_fused_features_binary"),
    # Local Windows workspace
    "d:/FINAL YEAR PROJECT-1/SEM7/Student_Engagement_Detection/sequence_fused_features_binary",
    os.path.join(os.getcwd(), "sequence_fused_features_binary")
]

def auto_detect_dataset_dir():
    """Detects the folder containing the binary feature files."""
    for path in DATASET_CANDIDATE_PATHS:
        if os.path.exists(path) and os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith(".pt") or f.endswith(".npy")]
            if len(files) > 0:
                return path
    # Default fallback
    if RUNNING_ON_COLAB:
        return "/content/sequence_fused_features_binary"
    return os.path.join(PROJECT_ROOT, "sequence_fused_features_binary")

SEQUENCE_FUSED_DIR = auto_detect_dataset_dir()

# Output directories for checkpoints, metrics, and plots
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ablation_results") if RUNNING_ON_COLAB else os.path.join(DRIVE_BASE_DIR, "ablation_results")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_openface_ablation_model.pth")
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "ablation_metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_binary.png")
ROC_CURVE_PATH = os.path.join(OUTPUT_DIR, "roc_curve_binary.png")

# Binary Classification Definition
# Class 0: Engaged, Class 1: Disengaged
CLASSES = ["engaged", "disengaged"]
CLASS_NAMES = ["Engaged", "Disengaged"]
NUM_CLASSES = 2

# OpenFace Feature Configuration
OPENFACE_DIM = 128          # Nominal OpenFace feature dimension
USE_OPENFACE_ONLY = True    # Ablation flag: strictly ignore EfficientNet
MAX_SEQ_LEN = 120           # Maximum frames per sequence

# Classification Head Architecture Parameters
HIDDEN_DIM = 128            # FC intermediate projection dimension
DROPOUT_RATE = 0.3          # Regularization dropout

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 40
PATIENCE = 10               # Early stopping patience
RANDOM_STATE = 42

# Data split ratios: 70% Train, 15% Val, 15% Test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
