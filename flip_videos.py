import cv2
import os

DATASET_DIR = r"d:\Amrita\Final_Year_Project\Sem_6 Work\Student_Engagement_DL\Customised_Classroom_Dataset"
OUTPUT_DATASET_DIR = r"d:\Amrita\Final_Year_Project\Sem_6 Work\Student_Engagement_DL\Customised_Classroom_Dataset_Flipped"
FOLDERS_TO_FLIP = ["DISENGAGED", "ENGAGED", "PARTIALLY ENGAGED"]

def flip_video(input_path, output_path):
    """Read a video, flip each frame horizontally, save to output_path."""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"  [ERROR] Could not open: {input_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fallback/sanity check for fps or frame size
    if fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0:
        print(f"  [ERROR] Invalid video dimensions for: {input_path}")
        cap.release()
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"  [ERROR] Could not open output video writer for: {output_path}")
        cap.release()
        return False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped = cv2.flip(frame, 1)   # 1 = horizontal flip
        out.write(flipped)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"  [OK] Done ({frame_idx}/{total} frames) -> {os.path.basename(output_path)}")
    return True

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory does not exist: {DATASET_DIR}")
        return

    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    print("=== Video Flipping Process ===")
    print(f"Input root: {DATASET_DIR}")
    print(f"Output root: {OUTPUT_DATASET_DIR}")
    print(f"Target folders: {FOLDERS_TO_FLIP}")
    print("==============================\n")

    for folder_name in FOLDERS_TO_FLIP:
        folder_path = os.path.join(DATASET_DIR, folder_name)
        output_folder_path = os.path.join(OUTPUT_DATASET_DIR, folder_name)

        if not os.path.isdir(folder_path):
            print(f"[WARNING] Directory not found, skipping: {folder_path}")
            continue

        os.makedirs(output_folder_path, exist_ok=True)

        print(f"\n[INFO] Processing folder: {folder_name}")
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
        if not video_files:
            print("  No .mp4 files found.")
            continue

        print(f"  [INFO] Found {len(video_files)} videos.")
        success_count = 0
        skip_count = 0

        for idx, filename in enumerate(video_files, 1):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder_path, filename)
            temp_path = os.path.join(output_folder_path, f"temp_flip_{filename}")

            if os.path.exists(output_path):
                print(f"  [{idx}/{len(video_files)}] Skipping '{filename}' (already flipped).")
                skip_count += 1
                success_count += 1
                continue

            print(f"  [{idx}/{len(video_files)}] Flipping '{filename}'...")
            
            try:
                if flip_video(input_path, temp_path):
                    # Safely rename the flipped temp file to the final output name
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(temp_path, output_path)
                    success_count += 1
                else:
                    # Clean up temp file on failure
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    print(f"  [ERROR] Failed to flip: {filename}")
            except Exception as e:
                print(f"  [ERROR] Exception occurred: {e}")
                # Wait briefly to ensure file handles are released before removing temp file
                import time
                time.sleep(0.5)
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as clean_err:
                        print(f"  [WARNING] Could not remove temp file '{temp_path}': {clean_err}")

        print(f"  [INFO] Completed {folder_name}: {success_count}/{len(video_files)} successfully flipped ({skip_count} skipped).")

    print("\n[INFO] All processes finished.")

if __name__ == "__main__":
    main()
