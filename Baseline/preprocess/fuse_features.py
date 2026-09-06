import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from extract_frames import find_video_files

def main():
    print("==================================================")
    print("      STEP 4: MODULAR FEATURE FUSION & POOLING    ")
    print("==================================================")
    print(f"OpenFace features root    : {config.OPENFACE_FEATURES_DIR}")
    print(f"EfficientNet features root: {config.EFFICIENTNET_FEATURES_DIR}")
    print(f"Fused output dir          : {config.FUSED_FEATURES_DIR}\n")

    os.makedirs(config.FUSED_FEATURES_DIR, exist_ok=True)

    videos = find_video_files(config.DATASET_DIR, config.CLASSES)
    if not videos:
        print("[Error] No videos found in the dataset. Run Step 1 first.")
        sys.exit(1)

    success_count = 0
    skip_count = 0
    error_count = 0

    for video in tqdm(videos, desc="Fusing features"):
        base_name = video["base_name"]
        class_name = video["class_name"]
        label = video["label"]
        video_filename = video["filename"]

        save_filename = f"{class_name}__{base_name}.pt"
        save_path = os.path.join(config.FUSED_FEATURES_DIR, save_filename)

        # Resume check
        if os.path.exists(save_path):
            skip_count += 1
            continue

        of_filename = f"{class_name}__{base_name}.npy"
        of_path = os.path.join(config.OPENFACE_FEATURES_DIR, of_filename)
        cnn_path = os.path.join(config.EFFICIENTNET_FEATURES_DIR, of_filename)

        if not os.path.exists(of_path) or not os.path.exists(cnn_path):
            print(f"\n[Warning] Skipped '{base_name}': Features not found. "
                  f"Ensure Step 2 and Step 3 ran successfully.")
            error_count += 1
            continue

        try:
            # Load features
            of_feats = np.load(of_path)  # Shape: [N_of, openface_dim]
            cnn_feats = np.load(cnn_path) # Shape: [N_cnn, 1280]

            # Frame alignment (take minimum frame length to handle edge-case capture/tracking differences)
            n_frames = min(len(of_feats), len(cnn_feats))
            if len(of_feats) != len(cnn_feats):
                of_feats = of_feats[:n_frames]
                cnn_feats = cnn_feats[:n_frames]

            # Concatenate frame-by-frame
            fused_frames = np.concatenate([cnn_feats, of_feats], axis=1) # Shape: [n_frames, 1280 + openface_dim]

            # Average pool across the temporal dimension
            fused_vector = fused_frames.mean(axis=0) # Shape: [1280 + openface_dim]

            # Convert to PyTorch tensor and save with metadata
            torch.save({
                "feature": torch.from_numpy(fused_vector).float(),
                "label": label,
                "video_name": video_filename,
                "class_name": class_name,
                "cnn_dim": cnn_feats.shape[1],
                "openface_dim": of_feats.shape[1]
            }, save_path)
            
            success_count += 1
        except Exception as e:
            print(f"\n[Warning] Failed to fuse features for '{base_name}': {e}")
            error_count += 1

    print("\n" + "=" * 50)
    print("  FEATURE FUSION COMPLETE")
    print("=" * 50)
    print(f"Successfully fused: {success_count}")
    print(f"Resumed/Skipped   : {skip_count}")
    print(f"Failed            : {error_count}")
    print(f"Total files in output: {len(os.listdir(config.FUSED_FEATURES_DIR))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
