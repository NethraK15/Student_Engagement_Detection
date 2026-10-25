import os
import sys
import shutil
import subprocess
import tempfile
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to sys.path to import config_exp6
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_exp6
from preprocess.extract_frames_crop_illumination import find_video_files

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
        # 1. System PATH lookup
        path_in_env = shutil.which("FeatureExtraction")
        if path_in_env:
            print(f"[OpenFace] Found executable in system PATH: {path_in_env}")
            return path_in_env

        # 2. Known static installation candidate locations
        candidates = [
            "/content/OpenFace/build/bin/FeatureExtraction",
            "/content/OpenFace/bin/FeatureExtraction",
            "/content/OpenFace/exe/FeatureExtraction/FeatureExtraction",
            "/content/OpenFace/exe/FeatureExtraction",
            "/usr/local/bin/FeatureExtraction",
            "/usr/bin/FeatureExtraction",
            "d:/Amrita/Final_Year_Project/Sem_6 Work/OpenFace_2.2.0_win_x64/FeatureExtraction.exe"
        ]

        for path in candidates:
            if os.path.isfile(path):
                try:
                    os.chmod(path, 0o755)
                except Exception:
                    pass
                print(f"[OpenFace] Found executable at: {path}")
                return path

        # 3. Dynamic filesystem walk search across /content if on Colab
        if os.path.exists("/content"):
            print("[OpenFace] Searching /content directory for FeatureExtraction binary...")
            for root, _, files in os.walk("/content"):
                for file in files:
                    if file == "FeatureExtraction" or file == "FeatureExtraction.exe":
                        full_p = os.path.join(root, file)
                        if os.path.isfile(full_p) and not os.path.islink(full_p):
                            try:
                                os.chmod(full_p, 0o755)
                            except Exception:
                                pass
                            print(f"[OpenFace] Dynamically discovered executable at: {full_p}")
                            return full_p

        print("[OpenFace] WARNING: OpenFace FeatureExtraction executable NOT found.")
        print("           If you want OpenFace features, please run the OpenFace Installation cell in Colab.")
        print("           Otherwise, fallback mode will be used (generating 0-filled vectors so EfficientNet features can train).")
        return None

    def run_extraction_on_illuminated_frames(self, frames_np, base_name, out_dir):
        """
        Creates a temporary illuminated video from the CLAHE+Sharpened face crop array,
        runs OpenFace FeatureExtraction on it, and cleans up temporary files.
        """
        if self.executable_path is None:
            return None

        os.makedirs(out_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, f"{base_name}_illuminated.avi")
        
        try:
            num_frames, h, w, _ = frames_np.shape
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (w, h))
            
            for i in range(num_frames):
                bgr_frame = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
            writer.release()

            # OpenFace command: FeatureExtraction -f <temp_video> -out_dir <out_dir>
            cmd = [self.executable_path, "-f", temp_video_path, "-out_dir", out_dir]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            csv_path = os.path.join(out_dir, f"{base_name}_illuminated.csv")
            if not os.path.exists(csv_path):
                for file in os.listdir(out_dir):
                    if file.lower().endswith(".csv") and base_name.lower() in file.lower():
                        csv_path = os.path.join(out_dir, file)
                        break

            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass

            if os.path.exists(csv_path):
                return csv_path
            return None
        except Exception as e:
            print(f"[OpenFace] Run failed for illuminated video '{base_name}': {e}")
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass
            return None

    def load_clean_features(self, csv_path, num_frames):
        if not csv_path or not os.path.exists(csv_path):
            return np.zeros((num_frames, config_exp6.OPENFACE_FALLBACK_DIM), dtype=np.float32)

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            if "face_id" in df.columns:
                df = df[df["face_id"] == 0]

            df = df.set_index("frame")
            full_index = pd.Index(range(1, num_frames + 1), name="frame")
            df = df.reindex(full_index)

            meta_cols = ["face_id", "timestamp", "confidence", "success"]
            cols_to_drop = [c for c in meta_cols if c in df.columns]
            df = df.drop(columns=cols_to_drop)

            df = df.fillna(0.0)
            features = df.values.astype(np.float32)

            if features.shape[1] == 0:
                return np.zeros((num_frames, config_exp6.OPENFACE_FALLBACK_DIM), dtype=np.float32)

            return features
        except Exception as e:
            print(f"[OpenFace] Error reading CSV {csv_path}: {e}")
            return np.zeros((num_frames, config_exp6.OPENFACE_FALLBACK_DIM), dtype=np.float32)

def main():
    print("==================================================")
    print("  EXPERIMENT 6: OPENFACE FEATURE EXTRACTION      ")
    print("   (FROM CROPPED ILLUMINATED FRAMES ON DRIVE)     ")
    print("==================================================")
    print(f"Dataset root               : {config_exp6.DATASET_DIR}")
    print(f"Illuminated frames root    : {config_exp6.PREPROCESSED_FRAMES_DIR}")
    print(f"OpenFace output dir        : {config_exp6.OPENFACE_FEATURES_DIR}\n")

    os.makedirs(config_exp6.OPENFACE_FEATURES_DIR, exist_ok=True)

    videos = find_video_files(config_exp6.DATASET_DIR, config_exp6.CLASSES)
    if not videos:
        print("[Error] No videos found in dataset directory. Run Step 1 first.")
        sys.exit(1)

    wrapper = OpenFaceWrapper()
    csv_temp_dir = os.path.join(config_exp6.OPENFACE_FEATURES_DIR, "raw_csvs")
    if wrapper.executable_path:
        os.makedirs(csv_temp_dir, exist_ok=True)

    success_count = 0
    skip_count = 0
    fallback_count = 0

    for video in tqdm(videos, desc="Extracting OpenFace features"):
        base_name = video["base_name"]
        class_name = video["class_name"]
        save_filename = f"{class_name}__{base_name}.npy"
        save_path = os.path.join(config_exp6.OPENFACE_FEATURES_DIR, save_filename)

        if os.path.exists(save_path):
            if wrapper.executable_path:
                try:
                    existing_arr = np.load(save_path)
                    if np.all(existing_arr == 0):
                        # Existing file is zero fallback from a previous run; re-extract with real OpenFace!
                        pass
                    else:
                        skip_count += 1
                        continue
                except Exception:
                    pass
            else:
                skip_count += 1
                continue

        frames_filename = f"{class_name}__{base_name}.npy"
        frames_path = os.path.join(config_exp6.PREPROCESSED_FRAMES_DIR, frames_filename)

        if not os.path.exists(frames_path):
            print(f"\n[Warning] Skipped '{base_name}': Illuminated frames not found in {config_exp6.PREPROCESSED_FRAMES_DIR}")
            fallback_count += 1
            continue

        try:
            illuminated_frames = np.load(frames_path)
            num_frames = illuminated_frames.shape[0]

            if wrapper.executable_path:
                csv_path = wrapper.run_extraction_on_illuminated_frames(illuminated_frames, base_name, csv_temp_dir)
                if csv_path:
                    features = wrapper.load_clean_features(csv_path, num_frames)
                    np.save(save_path, features)
                    success_count += 1
                else:
                    features = np.zeros((num_frames, config_exp6.OPENFACE_FALLBACK_DIM), dtype=np.float32)
                    np.save(save_path, features)
                    fallback_count += 1
            else:
                features = np.zeros((num_frames, config_exp6.OPENFACE_FALLBACK_DIM), dtype=np.float32)
                np.save(save_path, features)
                fallback_count += 1
        except Exception as e:
            print(f"\n[Warning] Error processing '{base_name}': {e}")
            fallback_count += 1

    print("\n" + "=" * 50)
    print("  OPENFACE EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Successfully extracted: {success_count}")
    print(f"Fallback (Zero Features): {fallback_count}")
    print(f"Resumed/Skipped        : {skip_count}")
    print(f"Total files in output   : {len(os.listdir(config_exp6.OPENFACE_FEATURES_DIR))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
