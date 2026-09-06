import os
import sys
import shutil
import numpy as np
import cv2
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config_transformer

def create_mock_video(path, width=224, height=224, fps=30, duration_sec=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(int(fps * duration_sec)):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

def setup_mock_environment():
    base_test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data_transformer")
    if os.path.exists(base_test_dir):
        shutil.rmtree(base_test_dir)
    os.makedirs(base_test_dir)

    dataset_dir = os.path.join(base_test_dir, "Dataset")
    for cls in config_transformer.CLASSES:
        cls_dir = os.path.join(dataset_dir, cls)
        create_mock_video(os.path.join(cls_dir, "video_1.mp4"), duration_sec=1)
        create_mock_video(os.path.join(cls_dir, "video_2.mp4"), duration_sec=1)

    # Patch config paths
    config_transformer.DATASET_DIR = dataset_dir
    config_transformer.OPENFACE_FEATURES_DIR = os.path.join(base_test_dir, "openface_features")
    config_transformer.EFFICIENTNET_FEATURES_DIR = os.path.join(base_test_dir, "efficientnet_features")
    config_transformer.SEQUENCE_FUSED_DIR = os.path.join(base_test_dir, "sequence_fused_features_transformer")
    config_transformer.MODEL_OUT_DIR = os.path.join(base_test_dir, "models_transformer")
    config_transformer.EVAL_OUT_DIR = os.path.join(base_test_dir, "evaluation_results_transformer")

    os.makedirs(config_transformer.OPENFACE_FEATURES_DIR, exist_ok=True)
    os.makedirs(config_transformer.EFFICIENTNET_FEATURES_DIR, exist_ok=True)
    os.makedirs(config_transformer.SEQUENCE_FUSED_DIR, exist_ok=True)
    os.makedirs(config_transformer.MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(config_transformer.EVAL_OUT_DIR, exist_ok=True)

    config_transformer.MODEL_WEIGHTS_PATH = os.path.join(config_transformer.MODEL_OUT_DIR, "transformer_engagement_weights.pth")
    config_transformer.PREDICTIONS_CSV_PATH = os.path.join(config_transformer.EVAL_OUT_DIR, "predictions_transformer.csv")
    config_transformer.CONFUSION_MATRIX_PATH = os.path.join(config_transformer.EVAL_OUT_DIR, "confusion_matrix_transformer.png")
    config_transformer.ROC_CURVE_PATH = os.path.join(config_transformer.EVAL_OUT_DIR, "roc_curve_transformer.png")
    config_transformer.METRICS_JSON_PATH = os.path.join(config_transformer.EVAL_OUT_DIR, "metrics_transformer.json")
    
    config_transformer.EPOCHS = 2
    config_transformer.BATCH_SIZE = 2

    for cls in config_transformer.CLASSES:
        for vname in ["video_1", "video_2"]:
            fname = f"{cls}__{vname}.npy"
            np.save(os.path.join(config_transformer.OPENFACE_FEATURES_DIR, fname), np.random.randn(30, 128).astype(np.float32))
            np.save(os.path.join(config_transformer.EFFICIENTNET_FEATURES_DIR, fname), np.random.randn(30, 1280).astype(np.float32))

    return base_test_dir

def main():
    print("==================================================")
    print("  EXPERIMENT 3: TRANSFORMER LOCAL TEST RUNNER    ")
    print("==================================================")
    
    base_test_dir = setup_mock_environment()
    
    try:
        print("\n--> Running Step 1: Fuse Sequence Features...")
        from preprocess import fuse_sequence_features
        fuse_sequence_features.main()
        
        print("\n--> Running Step 2: Train Temporal Transformer Model...")
        import train_transformer
        train_transformer.main()
        
        print("\n--> Running Step 3: Evaluate Temporal Transformer Model...")
        import evaluate_transformer
        evaluate_transformer.main()

        print("\n================ VERIFYING OUTPUT FILES ================")
        files_to_check = [
            config_transformer.MODEL_WEIGHTS_PATH,
            config_transformer.PREDICTIONS_CSV_PATH,
            config_transformer.CONFUSION_MATRIX_PATH,
            config_transformer.ROC_CURVE_PATH,
            config_transformer.METRICS_JSON_PATH
        ]
        
        all_passed = True
        for path in files_to_check:
            exists = os.path.exists(path)
            status = "PASSED" if exists else "FAILED"
            print(f"[{status}] File: {os.path.basename(path)}")
            if not exists:
                all_passed = False

        if all_passed:
            print("\n[SUCCESS] All Transformer pipeline steps completed and all files generated!")
        else:
            print("\n[FAILURE] Missing output files.")
            sys.exit(1)

    finally:
        print("\nCleaning up mock test directories...")
        if os.path.exists(base_test_dir):
            shutil.rmtree(base_test_dir)
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
