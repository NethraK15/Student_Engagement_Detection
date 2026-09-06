import os
import cv2
import glob
import argparse

def flip_videos_in_folder(input_folder, output_folder):
    """
    Finds all .mp4, .avi, .mov videos in input_folder, flips each frame horizontally,
    and saves the flipped video into output_folder with the same filename.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
        video_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    print(f"=== Video Flipping ===")
    print(f"Input Folder:  {input_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Found {len(video_files)} videos to process.\n")

    if not video_files:
        print("[WARNING] No videos found in input folder!")
        return

    for idx, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        out_path = os.path.join(output_folder, filename)

        if os.path.exists(out_path):
            print(f"[{idx}/{len(video_files)}] Skipping '{filename}' (already exists).")
            continue

        print(f"[{idx}/{len(video_files)}] Flipping '{filename}'...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [ERROR] Could not open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_out_path = os.path.join(output_folder, f"temp_{filename}")
        out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"  [ERROR] Could not initialize VideoWriter for: {temp_out_path}")
            cap.release()
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            flipped_frame = cv2.flip(frame, 1) # 1 = Horizontal flip
            out.write(flipped_frame)
            frame_count += 1

        cap.release()
        out.release()

        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(temp_out_path, out_path)
        print(f"  [OK] Saved flipped video ({frame_count}/{total_frames} frames) -> {filename}")

    print("\n[SUCCESS] Video flipping finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip classroom videos horizontally.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing original right classroom videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save flipped right classroom videos.")
    args = parser.parse_args()

    flip_videos_in_folder(args.input_dir, args.output_dir)
