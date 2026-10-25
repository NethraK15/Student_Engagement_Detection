import os
import sys
import shutil
import numpy as np
import cv2
import torch

# Add current folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def create_mock_video(path, width=224, height=224, fps=30, duration_sec=1):
    """
    Creates a tiny mock video with noise frames.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    num_frames = int(fps * duration_sec)
    for i in range(num_frames):
        # Generate random noise frames (simulating face cropped video frames)
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # Draw some text so it has visual variations
        cv2.putText(frame, f"Frame {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
        
    out.release()

def setup_mock_environment():
    """
    Creates a temporary sandbox environment inside colab_baseline/test_data/
    and patches config paths dynamically.
    """
    base_test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    if os.path.exists(base_test_dir):
        shutil.rmtree(base_test_dir)
    os.makedirs(base_test_dir)

    # 1. Create Mock Dataset structure
    dataset_dir = os.path.join(base_test_dir, "Dataset")
    for cls in config.CLASSES:
        cls_dir = os.path.join(dataset_dir, cls)
        # Create 2 videos per class
        create_mock_video(os.path.join(cls_dir, "video_1.mp4"), duration_sec=1)
        create_mock_video(os.path.join(cls_dir, "video_2.mp4"), duration_sec=1)
        
    print(f"[Test Setup] Mock dataset generated at: {dataset_dir}")

    # 2. Patch config paths
    config.DATASET_DIR = dataset_dir
    config.PREPROCESSED_FRAMES_DIR = os.path.join(base_test_dir, "preprocessed_frames")
    config.OPENFACE_FEATURES_DIR = os.path.join(base_test_dir, "openface_features")
    config.EFFICIENTNET_FEATURES_DIR = os.path.join(base_test_dir, "efficientnet_features")
    config.FUSED_FEATURES_DIR = os.path.join(base_test_dir, "fused_features")
    config.MODEL_OUT_DIR = os.path.join(base_test_dir, "models")
    config.EVAL_OUT_DIR = os.path.join(base_test_dir, "evaluation_results")

    # Re-verify and create patched local folders
    os.makedirs(config.PREPROCESSED_FRAMES_DIR, exist_ok=True)
    os.makedirs(config.OPENFACE_FEATURES_DIR, exist_ok=True)
    os.makedirs(config.EFFICIENTNET_FEATURES_DIR, exist_ok=True)
    os.makedirs(config.FUSED_FEATURES_DIR, exist_ok=True)
    os.makedirs(config.MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(config.EVAL_OUT_DIR, exist_ok=True)

    config.MODEL_WEIGHTS_PATH = os.path.join(config.MODEL_OUT_DIR, "baseline_engagement_weights.pth")
    config.PREDICTIONS_CSV_PATH = os.path.join(config.EVAL_OUT_DIR, "predictions.csv")
    config.CONFUSION_MATRIX_PATH = os.path.join(config.EVAL_OUT_DIR, "confusion_matrix.png")
    config.ROC_CURVE_PATH = os.path.join(config.EVAL_OUT_DIR, "roc_curve.png")
    config.METRICS_JSON_PATH = os.path.join(config.EVAL_OUT_DIR, "metrics.json")
    
    # Run training for only 2 epochs for quick mock test
    config.EPOCHS = 2
    # Set a tiny batch size so it splits properly even with few mock videos
    config.BATCH_SIZE = 2

    return base_test_dir

def main():
    print("==================================================")
    print("      PIPELINE LOCAL VERIFICATION RUNNER          ")
    print("==================================================")
    
    base_test_dir = setup_mock_environment()
    
    try:
        # Step 1: Run frame extraction
        print("\n--> Running Step 1: Extract Frames...")
        from preprocess import extract_frames
        extract_frames.main()
        
        # Step 2: Run OpenFace feature extraction (will trigger zero fallback locally)
        print("\n--> Running Step 2: Extract OpenFace...")
        from preprocess import extract_openface
        extract_openface.main()
        
        # Step 3: Run EfficientNet feature extraction
        print("\n--> Running Step 3: Extract EfficientNet...")
        from preprocess import extract_efficientnet
        extract_efficientnet.main()
        
        # Step 4: Run Feature Fusion
        print("\n--> Running Step 4: Fuse Features...")
        from preprocess import fuse_features
        fuse_features.main()
        
        # Step 5: Run Train
        print("\n--> Running Step 5: Train Classifier...")
        import train
        train.main()
        
        # Step 6: Run Evaluate
        print("\n--> Running Step 6: Evaluate Model...")
        import evaluate
        evaluate.main()

        # Check outputs
        print("\n================ VERIFYING OUTPUT FILES ================")
        files_to_check = [
            config.MODEL_WEIGHTS_PATH,
            config.PREDICTIONS_CSV_PATH,
            config.CONFUSION_MATRIX_PATH,
            config.ROC_CURVE_PATH,
            config.METRICS_JSON_PATH
        ]
        
        all_passed = True
        for path in files_to_check:
            exists = os.path.exists(path)
            status = "PASSED" if exists else "FAILED"
            print(f"[{status}] File: {os.path.basename(path)}")
            if not exists:
                all_passed = False

        if all_passed:
            print("\n🎉 SUCCESS: All mock pipeline steps completed and all files generated correctly!")
        else:
            print("\n❌ FAILURE: Some output files were not generated.")
            sys.exit(1)

    finally:
        # Cleanup mock test environment
        print("\nCleaning up mock test directories...")
        if os.path.exists(base_test_dir):
            shutil.rmtree(base_test_dir)
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
