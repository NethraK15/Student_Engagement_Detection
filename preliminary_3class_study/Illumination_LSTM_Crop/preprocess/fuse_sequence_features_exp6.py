import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to sys.path to import config_exp6
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_exp6
from preprocess.extract_frames_crop_illumination import find_video_files

def find_features_from_dir(openface_dir, classes):
    video_list = []
    if not os.path.exists(openface_dir):
        return video_list
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    for fname in sorted(os.listdir(openface_dir)):
        if fname.endswith(".npy") and "__" in fname:
            parts = fname[:-4].split("__", 1)
            cls_name = parts[0]
            base_name = parts[1]
            if cls_name in class_map:
                video_list.append({
                    "video_path": "",
                    "class_name": cls_name,
                    "label": class_map[cls_name],
                    "filename": f"{base_name}.mp4",
                    "base_name": base_name
                })
    return video_list

def sync_output_directory(active_filenames, output_dir, extension=".pt"):
    if not os.path.exists(output_dir):
        return
    active_set = set(active_filenames)
    removed_count = 0
    for file in os.listdir(output_dir):
        if file.endswith(extension) and os.path.isfile(os.path.join(output_dir, file)):
            if file not in active_set:
                try:
                    os.remove(os.path.join(output_dir, file))
                    removed_count += 1
                except Exception as e:
                    print(f"[Warning] Failed to delete stale file {file}: {e}")
    if removed_count > 0:
        print(f"[Sync] Removed {removed_count} stale '{extension}' files from {output_dir}")

def main():
    print("==================================================")
    print("  EXPERIMENT 6: SEQUENCE FEATURE FUSION (DRIVE)   ")
    print("==================================================")
    print(f"OpenFace features root    : {config_exp6.OPENFACE_FEATURES_DIR}")
    print(f"EfficientNet features root: {config_exp6.EFFICIENTNET_FEATURES_DIR}")
    print(f"Sequence Fused output dir : {config_exp6.SEQUENCE_FUSED_DIR}\n")

    os.makedirs(config_exp6.SEQUENCE_FUSED_DIR, exist_ok=True)

    videos = find_video_files(config_exp6.DATASET_DIR, config_exp6.CLASSES)
    if not videos:
        print("[Info] Dataset directory not found or empty. Discovering videos from pre-extracted features...")
        videos = find_features_from_dir(config_exp6.OPENFACE_FEATURES_DIR, config_exp6.CLASSES)

    if not videos:
        print("[Error] No videos or pre-extracted features found. Run feature extraction first.")
        sys.exit(1)

    active_pt_files = [f"{v['class_name']}__{v['base_name']}.pt" for v in videos]
    sync_output_directory(active_pt_files, config_exp6.SEQUENCE_FUSED_DIR, ".pt")

    success_count = 0
    skip_count = 0
    error_count = 0

    for video in tqdm(videos, desc="Fusing Experiment 6 sequence features"):
        base_name = video["base_name"]
        class_name = video["class_name"]
        label = video["label"]
        video_filename = video["filename"]

        save_filename = f"{class_name}__{base_name}.pt"
        save_path = os.path.join(config_exp6.SEQUENCE_FUSED_DIR, save_filename)

        if os.path.exists(save_path):
            skip_count += 1
            continue

        of_filename = f"{class_name}__{base_name}.npy"
        of_path = os.path.join(config_exp6.OPENFACE_FEATURES_DIR, of_filename)
        cnn_path = os.path.join(config_exp6.EFFICIENTNET_FEATURES_DIR, of_filename)

        if not os.path.exists(of_path) or not os.path.exists(cnn_path):
            print(f"\n[Warning] Skipped '{base_name}': Features not found. "
                  f"Ensure OpenFace and EfficientNet features exist.")
            error_count += 1
            continue

        try:
            of_feats = np.load(of_path)    # Shape: [N_of, openface_dim]
            cnn_feats = np.load(cnn_path)  # Shape: [N_cnn, 1280]

            n_frames = min(len(of_feats), len(cnn_feats))
            if len(of_feats) != len(cnn_feats):
                of_feats = of_feats[:n_frames]
                cnn_feats = cnn_feats[:n_frames]

            fused_sequence = np.concatenate([cnn_feats, of_feats], axis=1)

            # Save full 2D sequence tensor to Google Drive
            torch.save({
                "sequence": torch.from_numpy(fused_sequence).float(),
                "seq_len": n_frames,
                "label": label,
                "video_name": video_filename,
                "class_name": class_name,
                "feature_dim": fused_sequence.shape[1]
            }, save_path)
            
            success_count += 1
        except Exception as e:
            print(f"\n[Warning] Failed to fuse sequence features for '{base_name}': {e}")
            error_count += 1

    print("\n" + "=" * 50)
    print("  SEQUENCE FEATURE FUSION COMPLETE")
    print("=" * 50)
    print(f"Successfully fused: {success_count}")
    print(f"Resumed/Skipped   : {skip_count}")
    print(f"Failed            : {error_count}")
    print(f"Total files in output: {len(os.listdir(config_exp6.SEQUENCE_FUSED_DIR))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
