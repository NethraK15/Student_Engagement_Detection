import os
import sys
import subprocess
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from extract_frames import find_video_files

def sync_output_directory(active_filenames, output_dir, extension=".npy"):
    """
    Deletes any files in output_dir with the given extension that do not correspond
    to any active filename.
    """
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

class OpenFaceWrapper:
    def __init__(self):
        """
        Initializes the OpenFace wrapper.
        Attempts to locate the FeatureExtraction executable.
        """
        self.executable_path = self._find_executable()

    def _find_executable(self):
        """
        Searches for the FeatureExtraction executable in common paths.
        """
        candidates = []
        
        # 1. Standard Google Colab installation paths
        candidates.append("/content/OpenFace/exe/FeatureExtraction/FeatureExtraction")
        candidates.append("/content/OpenFace/build/bin/FeatureExtraction")
        candidates.append("/content/OpenFace/exe/FeatureExtraction")
        
        # 2. Local Windows default installation path
        candidates.append("d:/Amrita/Final_Year_Project/Sem_6 Work/OpenFace_2.2.0_win_x64/FeatureExtraction.exe")
        
        # 3. Global system path check
        candidates.append("FeatureExtraction")

        for path in candidates:
            if path == "FeatureExtraction":
                try:
                    devnull = open(os.devnull, 'w')
                    subprocess.call([path, "--help"], stdout=devnull, stderr=devnull)
                    print(f"[OpenFace] Found executable in global PATH: {path}")
                    return path
                except Exception:
                    continue
            
            if os.path.isfile(path):
                try:
                    os.chmod(path, 0o755)
                except Exception:
                    pass
                print(f"[OpenFace] Found executable at: {path}")
                return path

        print("[OpenFace] WARNING: OpenFace FeatureExtraction executable NOT found.")
        print("           Fallback mode will be used (generating zero features).")
        return None

    def run_extraction(self, video_path, out_dir):
        """
        Runs OpenFace FeatureExtraction on the video.
        """
        if self.executable_path is None:
            return None

        os.makedirs(out_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # OpenFace command: FeatureExtraction -f <video> -out_dir <out_dir>
        cmd = [self.executable_path, "-f", video_path, "-out_dir", out_dir]
        
        try:
            # Run without printing full stdout/stderr
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Find the generated CSV
            csv_path = os.path.join(out_dir, f"{video_name}.csv")
            if os.path.exists(csv_path):
                return csv_path
            
            # Sometimes casing or spaces change the CSV filename
            for file in os.listdir(out_dir):
                if file.lower().endswith(".csv") and video_name.lower() in file.lower():
                    return os.path.join(out_dir, file)
                    
            return None
        except Exception as e:
            print(f"[OpenFace] Run failed for video {video_name}: {e}")
            return None

    def load_clean_features(self, csv_path, num_frames):
        """
        Loads the features from the OpenFace CSV file.
        Reindexes to match exact frame indices and drops metadata.
        """
        if not csv_path or not os.path.exists(csv_path):
            return np.zeros((num_frames, config.OPENFACE_FALLBACK_DIM), dtype=np.float32)

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            # Filter for main face detection (face_id == 0)
            if "face_id" in df.columns:
                df = df[df["face_id"] == 0]

            # Reindex to ensure frame-by-frame mapping from 1 to num_frames
            df = df.set_index("frame")
            full_index = pd.Index(range(1, num_frames + 1), name="frame")
            df = df.reindex(full_index)

            # Columns to drop
            meta_cols = ["face_id", "timestamp", "confidence", "success"]
            cols_to_drop = [c for c in meta_cols if c in df.columns]
            df = df.drop(columns=cols_to_drop)

            # Fill NaN (missing frames) with zeros
            df = df.fillna(0.0)
            features = df.values.astype(np.float32)

            if features.shape[1] == 0:
                return np.zeros((num_frames, config.OPENFACE_FALLBACK_DIM), dtype=np.float32)

            return features
        except Exception as e:
            print(f"[OpenFace] Error reading CSV {csv_path}: {e}")
            return np.zeros((num_frames, config.OPENFACE_FALLBACK_DIM), dtype=np.float32)

def get_num_frames(video_info, preprocessed_dir):
    """
    Finds the frame count by loading the preprocessed .npy array shape (fast),
    or opening the video file directly as a fallback.
    """
    npy_path = os.path.join(preprocessed_dir, f"{video_info['class_name']}__{video_info['base_name']}.npy")
    if os.path.exists(npy_path):
        try:
            # Use mmap_mode to read only the header/shape, avoiding memory load
            arr = np.load(npy_path, mmap_mode='r')
            return arr.shape[0]
        except Exception:
            pass

    # Video file fallback
    cap = cv2.VideoCapture(video_info["video_path"])
    count = 0
    if cap.isOpened():
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            # Manually count if API returns invalid count
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
        cap.release()
    return count if count > 0 else 30

def main():
    print("==================================================")
    print("      STEP 2: MODULAR OPENFACE FEATURE EXTRACTION ")
    print("==================================================")
    print(f"Dataset root : {config.DATASET_DIR}")
    print(f"Output dir   : {config.OPENFACE_FEATURES_DIR}\n")

    os.makedirs(config.OPENFACE_FEATURES_DIR, exist_ok=True)

    videos = find_video_files(config.DATASET_DIR, config.CLASSES)
    if not videos:
        print("[Error] No videos found in the dataset. Run Step 1 first.")
        sys.exit(1)

    wrapper = OpenFaceWrapper()
    
    # We will run OpenFace execution in a temp folder inside outputs to keep CSVs organized
    csv_temp_dir = os.path.join(config.OPENFACE_FEATURES_DIR, "raw_csvs")
    if wrapper.executable_path:
        os.makedirs(csv_temp_dir, exist_ok=True)

    success_count = 0
    skip_count = 0
    fallback_count = 0

    for video in tqdm(videos, desc="Extracting OpenFace features"):
        save_filename = f"{video['class_name']}__{video['base_name']}.npy"
        save_path = os.path.join(config.OPENFACE_FEATURES_DIR, save_filename)

        # Resume support
        if os.path.exists(save_path):
            skip_count += 1
            continue

        num_frames = get_num_frames(video, config.PREPROCESSED_FRAMES_DIR)

        if wrapper.executable_path:
            # Run actual OpenFace Feature Extraction
            csv_path = wrapper.run_extraction(video["video_path"], csv_temp_dir)
            if csv_path:
                features = wrapper.load_clean_features(csv_path, num_frames)
                np.save(save_path, features)
                success_count += 1
            else:
                # Fallback to zeros on error
                features = np.zeros((num_frames, config.OPENFACE_FALLBACK_DIM), dtype=np.float32)
                np.save(save_path, features)
                fallback_count += 1
        else:
            # Zero feature fallback because executable is not found
            features = np.zeros((num_frames, config.OPENFACE_FALLBACK_DIM), dtype=np.float32)
            np.save(save_path, features)
            fallback_count += 1

    print("\n" + "=" * 50)
    print("  OPENFACE EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Successfully extracted: {success_count}")
    print(f"Fallback (Zero Features): {fallback_count}")
    print(f"Resumed/Skipped        : {skip_count}")
    print(f"Total files in output   : {len(os.listdir(config.OPENFACE_FEATURES_DIR))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
