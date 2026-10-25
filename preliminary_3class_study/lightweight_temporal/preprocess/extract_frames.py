import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import config_lightweight
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config_lightweight

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.3gp', '.flv', '.webm')

def find_video_files(dataset_dir, classes):
    """
    Scans dataset directory and lists all video files.
    Matches folder names to classes robustly using normalization:
    - contains 'disengaged' -> Disengaged (label 2)
    - contains 'partially', 'partial', or 'enaged' -> Partially Engaged (label 1)
    - contains 'engaged' -> Engaged (label 0)
    """
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
            print(f"[Info] Mapping folder '{match_folder}' to class '{cls}' (label {label_idx})")
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

def extract_and_save_frames(video_info, output_dir, target_size=(224, 224)):
    """
    Extracts video frames, resizes to target_size (224x224), and saves as .npy array.
    """
    video_path = video_info["video_path"]
    class_name = video_info["class_name"]
    base_name = video_info["base_name"]
    
    save_filename = f"{class_name}__{base_name}.npy"
    save_path = os.path.join(output_dir, save_filename)
    
    if os.path.exists(save_path):
        return True, "Already processed (skipped)"
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file"
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, target_size)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    
    cap.release()
    
    if len(frames) == 0:
        return False, "0 frames extracted"
        
    frames_np = np.array(frames, dtype=np.uint8)  # Shape: [N, H, W, 3]
    np.save(save_path, frames_np)
    return True, f"Success ({len(frames)} frames)"

def main():
    print("==================================================")
    print("      STEP 1: MODULAR FRAME EXTRACTION           ")
    print("==================================================")
    
    out_dir = getattr(config_lightweight, "PREPROCESSED_FRAMES_DIR", os.path.join(config_lightweight.PROJECT_ROOT, "preprocessed_frames"))
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Dataset root : {config_lightweight.DATASET_DIR}")
    print(f"Output dir   : {out_dir}\n")

    videos = find_video_files(config_lightweight.DATASET_DIR, config_lightweight.CLASSES)
    if not videos:
        print("[Error] No videos found in dataset directory.")
        sys.exit(1)
        
    print(f"Found {len(videos)} videos to process.")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for video in tqdm(videos, desc="Extracting frames"):
        success, message = extract_and_save_frames(video, out_dir)
        if success:
            if "skipped" in message:
                skip_count += 1
            else:
                success_count += 1
        else:
            print(f"\n[Warning] Failed to process '{video['filename']}': {message}")
            error_count += 1
            
    print("\n" + "=" * 50)
    print("  FRAME EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Newly processed: {success_count}")
    print(f"Resumed/Skipped: {skip_count}")
    print(f"Failed         : {error_count}")
    print(f"Total output files: {len(os.listdir(out_dir))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
