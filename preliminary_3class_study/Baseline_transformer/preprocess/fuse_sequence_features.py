import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to sys.path to import config_transformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_transformer

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.3gp', '.flv', '.webm')

def find_video_files(dataset_dir, classes):
    video_list = []
    if not os.path.exists(dataset_dir):
        print(f"[Error] Dataset directory not found: {dataset_dir}")
        return video_list

    dir_contents = os.listdir(dataset_dir)
    matched_folders = {0: [], 1: [], 2: []}
    
    for item in dir_contents:
        item_path = os.path.join(dataset_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        norm_name = item.lower().replace(" ", "").replace("_", "").replace("-", "")
        
        if "disengaged" in norm_name:
            matched_folders[2].append(item)
        elif "partially" in norm_name or "partial" in norm_name or "enaged" in norm_name:
            matched_folders[1].append(item)
        elif "engaged" in norm_name:
            matched_folders[0].append(item)
            
    for label_idx in range(3):
        folders = matched_folders[label_idx]
        cls = classes[label_idx]
        if not folders:
            print(f"[Warning] No directory matched for class '{cls}' in {dataset_dir}")
            continue
            
        for match_folder in folders:
            class_dir = os.path.join(dataset_dir, match_folder)
            for root, _, files in os.walk(class_dir):
                for file in sorted(files):
                    if file.lower().endswith(VIDEO_EXTENSIONS):
                        video_list.append({
                            "video_path": os.path.join(root, file),
                            "class_name": cls,
                            "label": label_idx,
                            "filename": file,
                            "base_name": os.path.splitext(file)[0]
                        })
    return video_list

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
    print("   EXPERIMENT 3: TRANSFORMER FEATURE FUSION       ")
    print("==================================================")
    print(f"OpenFace features root    : {config_transformer.OPENFACE_FEATURES_DIR}")
    print(f"EfficientNet features root: {config_transformer.EFFICIENTNET_FEATURES_DIR}")
    print(f"Sequence Fused output dir : {config_transformer.SEQUENCE_FUSED_DIR}\n")

    os.makedirs(config_transformer.SEQUENCE_FUSED_DIR, exist_ok=True)

    videos = find_video_files(config_transformer.DATASET_DIR, config_transformer.CLASSES)
    if not videos:
        print("[Info] Dataset directory not found or empty. Discovering videos from pre-extracted features...")
        videos = find_features_from_dir(config_transformer.OPENFACE_FEATURES_DIR, config_transformer.CLASSES)

    if not videos:
        print("[Error] No videos or pre-extracted features found. Run feature extraction first.")
        sys.exit(1)

    active_pt_files = [f"{v['class_name']}__{v['base_name']}.pt" for v in videos]
    sync_output_directory(active_pt_files, config_transformer.SEQUENCE_FUSED_DIR, ".pt")

    success_count = 0
    skip_count = 0
    error_count = 0

    for video in tqdm(videos, desc="Fusing Transformer sequence features"):
        base_name = video["base_name"]
        class_name = video["class_name"]
        label = video["label"]
        video_filename = video["filename"]

        save_filename = f"{class_name}__{base_name}.pt"
        save_path = os.path.join(config_transformer.SEQUENCE_FUSED_DIR, save_filename)

        if os.path.exists(save_path):
            skip_count += 1
            continue

        of_filename = f"{class_name}__{base_name}.npy"
        of_path = os.path.join(config_transformer.OPENFACE_FEATURES_DIR, of_filename)
        cnn_path = os.path.join(config_transformer.EFFICIENTNET_FEATURES_DIR, of_filename)

        if not os.path.exists(of_path) or not os.path.exists(cnn_path):
            print(f"\n[Warning] Skipped '{base_name}': Features not found.")
            error_count += 1
            continue

        try:
            of_feats = np.load(of_path)
            cnn_feats = np.load(cnn_path)

            n_frames = min(len(of_feats), len(cnn_feats))
            if len(of_feats) != len(cnn_feats):
                of_feats = of_feats[:n_frames]
                cnn_feats = cnn_feats[:n_frames]

            fused_sequence = np.concatenate([cnn_feats, of_feats], axis=1)

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
    print("  SEQUENCE FUSION COMPLETE")
    print("=" * 50)
    print(f"Successfully fused: {success_count}")
    print(f"Resumed/Skipped   : {skip_count}")
    print(f"Failed            : {error_count}")
    print(f"Total files in output: {len(os.listdir(config_transformer.SEQUENCE_FUSED_DIR))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
