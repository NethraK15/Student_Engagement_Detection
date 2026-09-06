import os
import glob

# =====================================================================
# Google Drive Paths & Config for DAiSEE Transfer Learning Experiment
# =====================================================================

DRIVE_BASE_DIR = "/content/drive/MyDrive/Student_Engagement_DL"
LOCAL_WORKSPACE_DIR = "d:/Amrita/Final_Year_Project/Sem_6 Work/Student_Engagement_DL"

if os.path.exists("/content"):
    PROJECT_ROOT = DRIVE_BASE_DIR
    RUNNING_ON_COLAB = True
else:
    PROJECT_ROOT = LOCAL_WORKSPACE_DIR
    RUNNING_ON_COLAB = False

# Auto-detect DAiSEE zip file path in Drive or Local Workspace
def find_daisee_zip(project_root):
    search_dirs = ["/content", project_root, DRIVE_BASE_DIR, os.path.dirname(project_root)]
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            matches = glob.glob(os.path.join(s_dir, "*daisee*.zip"), recursive=False)
            matches += glob.glob(os.path.join(s_dir, "*DAiSEE*.zip"), recursive=False)
            if matches:
                return matches[0]
    return os.path.join(project_root, "DAiSEE.zip")

DAISEE_ZIP_PATH = find_daisee_zip(PROJECT_ROOT)
DAISEE_EXTRACT_DIR = "/content/DAiSEE" if RUNNING_ON_COLAB else os.path.join(PROJECT_ROOT, "DAiSEE_Extracted")

# Target custom dataset directory (~100 clips)
def auto_detect_dataset_dir(project_root):
    if RUNNING_ON_COLAB and os.path.exists("/content/Dataset"):
        return "/content/Dataset"
    if os.path.exists(project_root):
        candidates = ["Dataset", "New_Datset_Crafted", "New_Dataset_Crafted", "dataset"]
        for name in candidates:
            path = os.path.join(project_root, name)
            if os.path.exists(path) and os.path.isdir(path):
                return path
    return os.path.join(project_root, "Dataset")

TARGET_DATASET_DIR = auto_detect_dataset_dir(PROJECT_ROOT)

# Dataset Class definitions
TARGET_CLASSES = ["engaged", "partially engaged", "disengaged"]
TARGET_CLASS_NAMES = ["Engaged", "Partially Engaged", "Disengaged"]
NUM_TARGET_CLASSES = len(TARGET_CLASSES)

DAISEE_NUM_CLASSES = 4  # DAiSEE Engagement levels: 0 (Very Low), 1 (Low), 2 (High), 3 (Very High)

# Fused features directory for Target dataset
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
    if RUNNING_ON_COLAB:
        for c in candidates:
            path = os.path.join("/content", c)
            if os.path.exists(path) and len([f for f in os.listdir(path) if f.endswith('.pt')]) > 0:
                return path
    for c in candidates:
        path = os.path.join(project_root, c)
        if os.path.exists(path) and len([f for f in os.listdir(path) if f.endswith('.pt')]) > 0:
            return path
    return os.path.join(project_root, "sequence_fused_features_lightweight")

TARGET_FUSED_DIR = detect_fused_features_dir(PROJECT_ROOT)
DAISEE_FUSED_DIR = os.path.join(PROJECT_ROOT, "daisee_fused_features")

MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models_daisee_transfer")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results_daisee_transfer")

if not RUNNING_ON_COLAB:
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)
    os.makedirs(DAISEE_FUSED_DIR, exist_ok=True)

# Architecture Hyperparameters for Temporal Transformer Model
DEFAULT_MODEL_TYPE = "transformer"  # Primary default: "transformer" (from colab_transformer)
D_MODEL = 128         # Transformer embedding dimension
NHEAD = 4             # Number of multi-head attention heads
NUM_LAYERS = 2        # Number of stacked Transformer Encoder layers
DIM_FEEDFORWARD = 256 # Feedforward expansion dimension
MAX_SEQ_LEN = 120     # Max frames per sequence clip
DROPOUT = 0.3         # Regularization dropout
MAX_GRAD_NORM = 1.0   # Gradient clipping threshold

# Phase 1: DAiSEE Pretraining Hyperparameters
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LR = 1e-3
PRETRAIN_EPOCHS = 30
PRETRAIN_PATIENCE = 6

# Phase 2: Target Dataset Fine-Tuning Hyperparameters
FINETUNE_BATCH_SIZE = 16
FINETUNE_LR = 1e-4          # Smaller learning rate for fine-tuning pretrained weights
FINETUNE_EPOCHS = 40
FINETUNE_PATIENCE = 8
FREEZE_TEMPORAL_BACKBONE = False  # Set True to freeze pre-trained layers & train classifier only

RANDOM_STATE = 42

# Weights File Paths
PRETRAINED_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "daisee_pretrained_temporal_weights.pth")
FINETUNED_WEIGHTS_PATH = os.path.join(MODEL_OUT_DIR, "finetuned_engagement_weights.pth")
PREDICTIONS_CSV_PATH = os.path.join(EVAL_OUT_DIR, "predictions_daisee_transfer.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_OUT_DIR, "confusion_matrix_daisee_transfer.png")
ROC_CURVE_PATH = os.path.join(EVAL_OUT_DIR, "roc_curve_daisee_transfer.png")
METRICS_JSON_PATH = os.path.join(EVAL_OUT_DIR, "metrics_daisee_transfer.json")
