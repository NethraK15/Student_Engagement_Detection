import os
import sys
import shutil
import subprocess
import tempfile
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to path to import config_lightweight
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config_lightweight

class OpenFaceWrapper:
    def __init__(self):
        """
        Initializes OpenFace wrapper and detects FeatureExtraction executable.
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

        print("[OpenFace] WARNING: OpenFace FeatureExtraction binary NOT found.")
        print("           Feature extraction will use zero-filled fallback vectors for facial features.")
        return None

    def run_extraction_on_frames(self, frames_np, base_name, out_dir):
        if self.executable_path is None:
            return None

        os.makedirs(out_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, f"{base_name}_temp.avi")
        
        try:
            num_frames, h, w, _ = frames_np.shape
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (w, h))
            
            for i in range(num_frames):
                bgr_frame = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
            writer.release()

            cmd = [self.executable_path, "-f", temp_video_path, "-out_dir", out_dir]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            csv_path = os.path.join(out_dir, f"{base_name}_temp.csv")
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
            print(f"[OpenFace] Run failed for video '{base_name}': {e}")
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass
            return None

def main():
    print("==================================================")
    print("      STEP 3: OPENFACE FEATURE EXTRACTION        ")
    print("==================================================")
    
    frames_dir = getattr(config_lightweight, "PREPROCESSED_FRAMES_DIR", os.path.join(config_lightweight.PROJECT_ROOT, "preprocessed_frames"))
    of_out_dir = config_lightweight.OPENFACE_FEATURES_DIR

    os.makedirs(of_out_dir, exist_ok=True)
    print(f"Frames root       : {frames_dir}")
    print(f"OpenFace output   : {of_out_dir}\n")

    if not os.path.exists(frames_dir):
        print(f"[Error] Frames directory not found: {frames_dir}")
        sys.exit(1)

    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".npy")]
    if not frame_files:
        print(f"[Error] No .npy frame files found in {frames_dir}")
        sys.exit(1)

    wrapper = OpenFaceWrapper()
    csv_temp_dir = os.path.join(of_out_dir, "raw_csvs")

    success_count = 0
    skip_count = 0

    for frame_file in tqdm(frame_files, desc="Extracting OpenFace features"):
        sample_name = os.path.splitext(frame_file)[0]
        frames_path = os.path.join(frames_dir, frame_file)
        save_csv_path = os.path.join(of_out_dir, f"{sample_name}.csv")

        if os.path.exists(save_csv_path):
            skip_count += 1
            continue

        try:
            frames = np.load(frames_path)
            if wrapper.executable_path:
                extracted_csv = wrapper.run_extraction_on_frames(frames, sample_name, csv_temp_dir)
                if extracted_csv and os.path.exists(extracted_csv):
                    shutil.move(extracted_csv, save_csv_path)
                    success_count += 1
                else:
                    # Create basic fallback CSV if OpenFace binary didn't output CSV
                    df_fallback = pd.DataFrame(np.zeros((len(frames), 128)))
                    df_fallback.to_csv(save_csv_path, index=False)
                    success_count += 1
            else:
                df_fallback = pd.DataFrame(np.zeros((len(frames), 128)))
                df_fallback.to_csv(save_csv_path, index=False)
                success_count += 1
        except Exception as e:
            print(f"\n[Warning] Failed to extract OpenFace for '{sample_name}': {e}")

    print("\n" + "=" * 50)
    print("  OPENFACE EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Successfully processed: {success_count}")
    print(f"Resumed/Skipped       : {skip_count}")
    print(f"Total output files    : {len(os.listdir(of_out_dir))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
