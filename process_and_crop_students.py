import os
import cv2
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not installed. Please install via: pip install ultralytics")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Auto Crop and Build Dataset from Labeled Classroom Videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Folder containing classroom videos (or subfolder)")
    parser.add_argument("--output_dataset_dir", type=str, default="New_Dataset_Crafted", help="Output directory for cropped student face videos")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-face.pt", help="Path to YOLO face model")
    return parser.parse_args()

def process_single_video(video_path, output_dataset_dir, yolo_model):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n==========================================")
    print(f"Processing Video: {video_name}")
    print(f"==========================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    # 1. Run YOLO Tracking to detect all student tracks first
    track_crops = {}     # track_id -> dict(frame_idx -> face_crop)
    track_previews = {}  # track_id -> full classroom frame with yellow box around student face
    processed_frames = 0
    is_face_model = "face" in yolo_model.model_name.lower() if hasattr(yolo_model, 'model_name') else True

    print("[STEP 1] Detecting & Tracking student faces across frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1

        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if not is_face_model and cls != 0:
                    continue

                # Filter out low-confidence detections (< 0.50)
                if conf < 0.50:
                    continue

                track_id = int(box.id[0].item()) if box.id is not None else 0
                if track_id == 0:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = map(int, xyxy)

                xmin_c, ymin_c = max(0, xmin), max(0, ymin)
                xmax_c, ymax_c = min(width, xmax), min(height, ymax)

                if (xmax_c - xmin_c) > 0 and (ymax_c - ymin_c) > 0:
                    face_crop = frame[ymin_c:ymax_c, xmin_c:xmax_c].copy()
                    if track_id not in track_crops:
                        track_crops[track_id] = {}
                    track_crops[track_id][processed_frames] = face_crop

                    # Save context preview frame showing the FULL classroom video with yellow box on the detected person
                    if track_id not in track_previews:
                        preview_frame = frame.copy()
                        cv2.rectangle(preview_frame, (xmin_c, ymin_c), (xmax_c, ymax_c), (0, 255, 255), 4)
                        cv2.putText(preview_frame, f"STUDENT ID: {track_id}", (xmin_c, max(ymin_c-10, 25)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3, cv2.LINE_AA)
                        
                        # Add face zoom in top-right corner of classroom video preview
                        zoom_h, zoom_w = 180, 180
                        face_resized = cv2.resize(face_crop, (zoom_w, zoom_h))
                        margin = 20
                        if height > zoom_h + margin and width > zoom_w + margin:
                            preview_frame[margin:margin+zoom_h, width-zoom_w-margin:width-margin] = face_resized
                            cv2.rectangle(preview_frame, (width-zoom_w-margin, margin), (width-margin, margin+zoom_h), (0, 255, 255), 3)

                        track_previews[track_id] = preview_frame

    # Filter out short flicker detections/false positives (less than 15 frames)
    min_frames = 15
    track_crops = {tid: crops for tid, crops in track_crops.items() if len(crops) >= min_frames}

    if not track_crops:
        print(f"[WARNING] No valid student faces detected in {video_name} (all tracks were under {min_frames} frames).")
        return

    # 2. Interactive labeling per student track ID
    print(f"\n[STEP 2] Found {len(track_crops)} student tracks (>= {min_frames} frames) in this video.")
    print("Label each student track:")
    print("  [0] DISENGAGED")
    print("  [1] PARTIALLY ENGAGED")
    print("  [2] ENGAGED")
    print("  [s] Skip student track")

    class_subfolders = {
        0: "DISENGAGED",
        1: "PARTIALLY ENGAGED",
        2: "ENGAGED"
    }

    for label_subfolder in class_subfolders.values():
        os.makedirs(os.path.join(output_dataset_dir, label_subfolder), exist_ok=True)

    # Detect Colab / notebook environment
    for track_id, crops_dict in track_crops.items():
        # Clear previous images from cell output so notebook doesn't flood
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except Exception:
            pass

        # Get full classroom frame with yellow bounding box around this student
        preview_img = track_previews.get(track_id, list(crops_dict.values())[0])
        
        # Resize wide classroom frame for clear view in notebook (640x360)
        h, w = preview_img.shape[:2]
        if w > 0 and h > 0:
            classroom_display = cv2.resize(preview_img, (640, int(640 * (h / w))))
        else:
            classroom_display = preview_img

        preview_path = f"/content/preview_track_{track_id}.jpg"
        cv2.imwrite(preview_path, classroom_display)

        # Print track info
        print(f"\n" + "="*50)
        print(f"  TARGET STUDENT TRACK ID: {track_id} (Total {len(crops_dict)} frames)")
        print(f"="*50)

        # Display full classroom preview image directly in Colab
        try:
            from PIL import Image as PILImage
            display(PILImage.open(preview_path))
        except Exception:
            try:
                from google.colab.patches import cv2_imshow
                cv2_imshow(classroom_display)
            except Exception:
                print(f"  [Classroom Context Preview saved to: {preview_path}]")
        
        # Flush output and small pause so the image renders BEFORE the text input box opens
        import sys, time
        sys.stdout.flush()
        time.sleep(0.3)

        while True:
            lbl_input = input("\n👇 Enter Engagement Label (0: DISENGAGED, 1: PARTIALLY ENGAGED, 2: ENGAGED, s: Skip): ").strip().lower()
            if lbl_input in ['0', '1', '2', 's']:
                break
            print("  Invalid input! Please enter 0, 1, 2, or s.")

        if lbl_input == 's':
            print(f"  [-] Skipped Track ID {track_id}")
            continue

        label_code = int(lbl_input)
        target_folder = os.path.join(output_dataset_dir, class_subfolders[label_code])

        # 3. Save student face crop video (224x224)
        out_video_filename = f"{video_name}_student_{track_id}.mp4"
        out_video_path = os.path.join(target_folder, out_video_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (224, 224))

        sorted_frames = sorted(crops_dict.keys())
        for fnum in sorted_frames:
            resized_face = cv2.resize(crops_dict[fnum], (224, 224))
            out_writer.write(resized_face)

        out_writer.release()
        print(f"  [OK] Saved cropped video ({len(sorted_frames)} frames) -> {os.path.join(class_subfolders[label_code], out_video_filename)}")

def main():
    args = parse_args()
    if not os.path.exists(args.video_dir):
        print(f"[ERROR] Input video directory does not exist: {args.video_dir}")
        return

    print(f"[+] Loading YOLO model: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)

    video_files = glob.glob(os.path.join(args.video_dir, "*.mp4")) + glob.glob(os.path.join(args.video_dir, "*.avi"))
    print(f"[+] Found {len(video_files)} classroom videos in {args.video_dir}.")

    for v_path in video_files:
        process_single_video(v_path, args.output_dataset_dir, yolo_model)

    print("\n==========================================")
    print("ALL CLASSROOM VIDEOS PROCESSED & CROPPED SUCCESSFULLY!")
    print(f"Cropped videos saved inside '{args.output_dataset_dir}'")
    print("==========================================")

if __name__ == "__main__":
    main()
