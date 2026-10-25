import os
import sys
import cv2
import json
import argparse
import glob
import shutil
import platform
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup dynamic path for yolo_face_detector and local libraries
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, "lib")
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)
yolo_dir = os.path.join(script_dir, "yolo_face_detector")
if os.path.exists(yolo_dir) and yolo_dir not in sys.path:
    sys.path.append(yolo_dir)

try:
    from ultralytics import YOLO
except ImportError:
    print("[-] Ultralytics is not installed. Please install it using: pip install ultralytics")
    sys.exit(1)

try:
    from yolo_face_detector.face_detector_v4 import FaceQualityFilter, download_face_weights
except ImportError:
    # Fallback to local definitions if needed
    FaceQualityFilter = None
    download_face_weights = None

# Fallback implementation of download_face_weights if import failed
if download_face_weights is None:
    import urllib.request
    def download_face_weights(dest_path):
        urls = [
            "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
            "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face.pt"
        ]
        print(f"\n[+] Local face weights '{dest_path}' not found.")
        for url in urls:
            print(f"[+] Downloading from: {url}...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"[+] Download complete!")
                return True
            except Exception as e:
                print(f"[-] Download failed: {e}")
        return False

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive classroom student face tracking and annotation tool.")
    parser.add_argument("--video", type=str, required=True, help="Path to classroom video file.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-face.pt", help="YOLO face model weights.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Main training dataset directory.")
    parser.add_argument("--annotated_dir", type=str, default="Annotated_Folder", help="Output directory for cropped face videos.")
    parser.add_argument("--openface_dir", type=str, default=r"d:\Amrita\Final_Year_Project\Sem_6 Work\OpenFace_2.2.0_win_x64",
                        help="Path to OpenFace installation folder.")
    return parser.parse_args()

def run_openface(video_path, out_dir, openface_dir):
    """
    Runs OpenFace FaceLandmarkVidMulti.exe to extract features and aligned crops.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    processed_dir = os.path.join(out_dir, video_name)
    os.makedirs(processed_dir, exist_ok=True)

    csv_path = os.path.join(processed_dir, f"{video_name}.csv")
    aligned_dir = os.path.join(processed_dir, f"{video_name}_aligned")

    if os.path.exists(csv_path) and os.path.exists(aligned_dir) and len(os.listdir(aligned_dir)) > 0:
        print(f"[+] Found existing OpenFace outputs at {processed_dir}. Skipping landmark extraction.")
        return csv_path, aligned_dir

    if platform.system() != "Windows":
        print("[-] OpenFace execution is only supported on Windows.")
        return None, None

    exe_path = os.path.join(openface_dir, "FaceLandmarkVidMulti.exe")
    if not os.path.exists(exe_path):
        print(f"[-] Warning: Could not find OpenFace executable at {exe_path}. Landmark alignment mapping will be unavailable.")
        return None, None

    print(f"\n>>> Running OpenFace Landmark Extraction on {video_path}...")
    cmd = [exe_path, "-f", video_path, "-out_dir", processed_dir, "-aligned"]
    try:
        subprocess.run(cmd, check=True)
        print("[+] OpenFace processed successfully!")
    except Exception as e:
        print(f"[-] OpenFace execution failed: {e}")
        return None, None

    return csv_path, aligned_dir

def main():
    args = parse_args()
    
    if not os.path.exists(args.video):
        print(f"[-] Input video file does not exist: {args.video}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    
    # 1. OpenFace Integration setup
    processed_recordings_root = "processed_recordings"
    os.makedirs(processed_recordings_root, exist_ok=True)
    
    csv_path = os.path.join(processed_recordings_root, video_name, f"{video_name}.csv")
    aligned_dir = os.path.join(processed_recordings_root, video_name, f"{video_name}_aligned")
    
    openface_available = os.path.exists(csv_path) and os.path.exists(aligned_dir)
    
    if not openface_available:
        print(f"\n[?] OpenFace features not found for this video.")
        print(f"    Expected CSV: {csv_path}")
        print("    Would you like to run OpenFace landmark extraction now? (y/n): ", end="")
        choice = input().strip().lower()
        if choice == 'y':
            csv_path, aligned_dir = run_openface(args.video, processed_recordings_root, args.openface_dir)
            openface_available = csv_path is not None and os.path.exists(csv_path)
        else:
            print("[!] Continuing without OpenFace landmark alignment. Face crops will be cropped directly from raw frames.")

    openface_df = None
    if openface_available:
        try:
            openface_df = pd.read_csv(csv_path)
            openface_df.columns = openface_df.columns.str.strip()
            print(f"[+] Loaded {len(openface_df)} OpenFace feature records.")
        except Exception as e:
            print(f"[-] Failed to load OpenFace CSV: {e}. Falling back to raw cropping.")
            openface_available = False

    # 2. YOLO Model setup
    if not os.path.exists(args.yolo_model) and args.yolo_model == "yolov8n-face.pt":
        success = download_face_weights(args.yolo_model)
        if not success:
            print("[-] Downloading yolov8n-face.pt failed. Falling back to default yolo11n.pt if available.")
            args.yolo_model = "yolo11n.pt"

    print(f"[+] Loading YOLO Tracking Model: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    is_face_model = "face" in args.yolo_model.lower()

    # 3. Setup Folders
    class_folders = {
        0: "Disengaged",
        1: "Partially_engaged",
        2: "Engaged"
    }
    
    # Target dataset folder structure
    dataset_images_dir = os.path.join(args.dataset_dir, "images")
    dataset_flat_dir = os.path.join(args.dataset_dir, "images_flat")
    openface_dest_csv = os.path.join(args.dataset_dir, "openface", "images_flat.csv")
    frame_map_csv = os.path.join(args.dataset_dir, "frame_to_filename.csv")
    labels_csv = os.path.join(args.dataset_dir, "labels.csv")

    for val in class_folders.values():
        os.makedirs(os.path.join(dataset_images_dir, f"0_disengaged" if val == "Disengaged" else f"1_partial" if val == "Partially_engaged" else f"2_engaged"), exist_ok=True)
        os.makedirs(os.path.join(args.annotated_dir, val), exist_ok=True)
    os.makedirs(dataset_flat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_dir, "openface"), exist_ok=True)

    # Local video annotations progress tracker
    local_annotations_file = os.path.join(processed_recordings_root, video_name, "annotations.json")
    local_annotations = {}
    if os.path.exists(local_annotations_file):
        try:
            with open(local_annotations_file, "r") as f:
                local_annotations = json.load(f)
            print(f"[+] Loaded existing local progress containing {len(local_annotations)} labeled crops.")
        except Exception:
            pass

    # 4. Process Video and Track Faces
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[-] Cannot open video: {args.video}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    print(f"\n==================================================")
    # Map track_id -> List of face crops (dict mapping frame_num -> BGR image)
    track_crops = {}
    # Map track_id -> List of matched OpenFace CSV rows (dict mapping frame_num -> CSV row dict)
    track_of_rows = {}
    # Map track_id -> label (None initially, 0: Disengaged, 1: Partially engaged, 2: Engaged, -1: Skip)
    track_labels = {}

    processed_frames = 0
    cv2.namedWindow("Interactive Face Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interactive Face Annotator", 1000, 650)

    print("[+] Running YOLO tracker. When a new student is detected, you will be prompted.")
    print("    Press keys ON the OpenCV window:")
    print("    [0] -> Disengaged")
    print("    [1] -> Partially Engaged")
    print("    [2] -> Engaged")
    print("    [s] -> Skip student track")
    print("    [q] -> Quit and Save progress")
    print("    [Space] -> Pause / Resume playback")
    print("==================================================")

    paused = False

    for frame_idx in range(total_frames):
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frames += 1
            display_frame = frame.copy()
            
            # YOLO tracking
            results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            # Get OpenFace rows for the current frame
            of_rows_current = None
            if openface_df is not None:
                of_rows_current = openface_df[(openface_df["frame"] == processed_frames) & (openface_df["success"] == 1)]

            active_boxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if not is_face_model and cls != 0:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    if not is_face_model:
                        person_h = ymax - ymin
                        ymax = ymin + int(person_h * 0.25) # Top 25% fallback

                    track_id = int(box.id[0].item()) if box.id is not None else 0
                    if track_id == 0:
                        continue
                    
                    active_boxes.append((track_id, xmin, ymin, xmax, ymax))

            # Prompt labels for new tracks in this frame
            for track_id, xmin, ymin, xmax, ymax in active_boxes:
                # Store crop image
                xmin_c = max(0, xmin)
                ymin_c = max(0, ymin)
                xmax_c = min(width, xmax)
                ymax_c = min(height, ymax)
                
                if (xmax_c - xmin_c) > 0 and (ymax_c - ymin_c) > 0:
                    face_crop = frame[ymin_c:ymax_c, xmin_c:xmax_c].copy()
                    if track_id not in track_crops:
                        track_crops[track_id] = {}
                    track_crops[track_id][processed_frames] = face_crop

                # Match OpenFace record
                matched_row = None
                if of_rows_current is not None and not of_rows_current.empty:
                    best_iou = 0.0
                    for _, of_row in of_rows_current.iterrows():
                        try:
                            # 68 OpenFace landmarks coords
                            x_coords = [of_row[f"x_{i}"] for i in range(68)]
                            y_coords = [of_row[f"y_{i}"] for i in range(68)]
                            of_xmin, of_xmax = min(x_coords), max(x_coords)
                            of_ymin, of_ymax = min(y_coords), max(y_coords)
                            w_of, h_of = of_xmax - of_xmin, of_ymax - of_ymin
                            of_xmin = max(0, int(of_xmin - 0.1 * w_of))
                            of_xmax = min(width, int(of_xmax + 0.1 * w_of))
                            of_ymin = max(0, int(of_ymin - 0.15 * h_of))
                            of_ymax = min(height, int(of_ymax + 0.1 * h_of))
                            
                            iou = compute_iou([xmin, ymin, xmax, ymax], [of_xmin, of_ymin, of_xmax, of_ymax])
                            if iou > best_iou:
                                best_iou = iou
                                matched_row = of_row.to_dict()
                        except Exception:
                            continue
                    if best_iou > 0.2 and matched_row is not None:
                        if track_id not in track_of_rows:
                            track_of_rows[track_id] = {}
                        track_of_rows[track_id][processed_frames] = matched_row

                # Ask for label if this track is brand new
                if track_id not in track_labels:
                    # Pause video display to prompt user
                    canvas = display_frame.copy()
                    # Dim background
                    canvas = (canvas * 0.45).astype(np.uint8)
                    # Re-draw highlighting box around target student face
                    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
                    
                    # Zoom in on target face crop
                    zoom_h, zoom_w = 200, 200
                    zoom_panel = np.zeros((zoom_h, zoom_w, 3), dtype=np.uint8)
                    if (xmax_c - xmin_c) > 0 and (ymax_c - ymin_c) > 0:
                        face_resized = cv2.resize(frame[ymin_c:ymax_c, xmin_c:xmax_c], (zoom_w, zoom_h))
                        zoom_panel[:, :] = face_resized
                    
                    # Embed zoom panel in top-right corner
                    margin = 20
                    canvas[margin:margin+zoom_h, width-zoom_w-margin:width-margin] = zoom_panel
                    cv2.rectangle(canvas, (width-zoom_w-margin, margin), (width-margin, margin+zoom_h), (0, 255, 255), 2)
                    
                    # Text prompts
                    labels_text = [
                        f"Target Student Track ID: {track_id}",
                        f"Frame: {processed_frames}/{total_frames}",
                        "Engagement Classification:",
                        "  [0] Disengaged",
                        "  [1] Partially Engaged",
                        "  [2] Engaged",
                        "  [s] Skip this student",
                        "  [q] Save & Quit Annotator"
                    ]
                    
                    tx = 30
                    ty = 60
                    for line in labels_text:
                        cv2.putText(canvas, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                        ty += 30

                    cv2.imshow("Interactive Face Annotator", canvas)
                    
                    # Wait for interactive input
                    while True:
                        k = cv2.waitKey(0) & 0xFF
                        if k in [ord('0'), ord('1'), ord('2')]:
                            track_labels[track_id] = int(chr(k))
                            print(f"[+] Labeled Track {track_id} -> {class_folders[track_labels[track_id]]}")
                            break
                        elif k == ord('s'):
                            track_labels[track_id] = -1
                            print(f"[-] Skipped Track {track_id}")
                            break
                        elif k == ord('q'):
                            track_labels[track_id] = -1
                            paused = False
                            break
                    
                    if k == ord('q'):
                        print("[!] Quit command received. Terminating extraction...")
                        break

            # Draw labels on active frame display
            for track_id, xmin, ymin, xmax, ymax in active_boxes:
                label_val = track_labels.get(track_id, -1)
                if label_val == -1:
                    # Skipped/Unlabeled
                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                    cv2.putText(display_frame, f"ID:{track_id} [SKIPPED]", (xmin, max(ymin-5, 12)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    color = (0, 255, 0) if label_val == 2 else (0, 165, 255) if label_val == 1 else (50, 50, 255)
                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(display_frame, f"ID:{track_id} [{class_folders[label_val].split('_')[0]}]", (xmin, max(ymin-5, 12)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Draw generic controls at the top left of active display frame
            cv2.putText(display_frame, f"Frame: {processed_frames}/{total_frames} | Space: Pause | Q: Save & Quit", 
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Interactive Face Annotator", display_frame)

        # Check for pause/quit key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            paused = not paused
            print("[*] Paused" if paused else "[*] Resumed")
        elif key == ord('q'):
            print("[!] Quit requested...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # 5. Review Summary and modify labels before saving
    valid_tracks = {tid: lbl for tid, lbl in track_labels.items() if lbl in [0, 1, 2]}
    if valid_tracks:
        print("\n" + "="*50)
        print("          LABELING SUMMARY REVIEW")
        print("="*50)
        for tid, lbl in valid_tracks.items():
            print(f"  Track ID {tid:02d} -> {class_folders[lbl]} ({len(track_crops.get(tid, {}))} crops)")
        print("="*50)
        
        while True:
            print("\nEnter a Track ID to modify its label, 's' to skip a track, or press Enter to save and finish: ", end="")
            mod_input = input().strip()
            if not mod_input:
                break
            
            try:
                mod_tid = int(mod_input)
                if mod_tid in track_labels:
                    print(f"Current label for Track {mod_tid}: {class_folders.get(track_labels[mod_tid], 'Skipped/None')}")
                    print("Enter new label (0: Disengaged, 1: Partially engaged, 2: Engaged, 's' to skip): ", end="")
                    new_lbl_str = input().strip().lower()
                    if new_lbl_str == 's':
                        track_labels[mod_tid] = -1
                        print(f"[-] Track {mod_tid} marked as skipped.")
                    elif new_lbl_str in ['0', '1', '2']:
                        track_labels[mod_tid] = int(new_lbl_str)
                        print(f"[+] Track {mod_tid} updated -> {class_folders[track_labels[mod_tid]]}")
                else:
                    print("[-] Track ID not found in annotations.")
            except ValueError:
                print("[-] Invalid input. Please enter a numerical Track ID.")

    # 6. Save Crops & Compile Face Videos
    final_labeled_count = 0
    new_csv_rows = []
    
    # Load dataset features CSV if exists
    if os.path.exists(openface_dest_csv):
        try:
            main_features_df = pd.read_csv(openface_dest_csv)
            main_features_df.columns = main_features_df.columns.str.strip()
        except Exception:
            main_features_df = None
    else:
        main_features_df = None

    # Load dataset labels CSV if exists
    if os.path.exists(labels_csv):
        try:
            labels_df = pd.read_csv(labels_csv)
            labels_df.columns = labels_df.columns.str.strip()
        except Exception:
            labels_df = None
    else:
        labels_df = None

    # Load frame-to-filename CSV if exists
    if os.path.exists(frame_map_csv):
        try:
            frame_map_df = pd.read_csv(frame_map_csv)
            frame_map_df.columns = frame_map_df.columns.str.strip()
        except Exception:
            frame_map_df = None
    else:
        frame_map_df = None

    print("\n>>> Exporting annotated results...")
    
    for track_id, label_val in track_labels.items():
        if label_val not in [0, 1, 2]:
            continue
            
        crops = track_crops.get(track_id, {})
        of_records = track_of_rows.get(track_id, {})
        if not crops:
            continue

        label_name = class_folders[label_val]
        folder_label_name = "0_disengaged" if label_val == 0 else "1_partial" if label_val == 1 else "2_engaged"
        
        # A. Compile face crop video
        # Determine crop dimensions and resize to 224x224 for DL standard input
        video_out_path = os.path.join(args.annotated_dir, label_name, f"{video_name}_student_{track_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(video_out_path, fourcc, fps, (224, 224))
        
        sorted_frames = sorted(crops.keys())
        for fnum in sorted_frames:
            crop_resized = cv2.resize(crops[fnum], (224, 224))
            out_video.write(crop_resized)
        out_video.release()
        print(f"  [OK] Saved cropped video ({len(sorted_frames)} frames) -> {video_out_path}")

        # B. Save individual crops to main dataset folders
        for fnum in sorted_frames:
            face_img = crops[fnum]
            
            # Form crop filename in OpenFace format style
            # If OpenFace matched row exists, use the actual matched OpenFace face_id
            matched_of_row = of_records.get(fnum, None)
            of_face_id = int(matched_of_row["face_id"]) if matched_of_row is not None else 0
            
            crop_name = f"frame_det_{of_face_id:02d}_{fnum:06d}.jpg"
            new_filename = f"{video_name}_{crop_name}"
            
            # Write crop images
            dest_labeled_path = os.path.join(dataset_images_dir, folder_label_name, new_filename)
            dest_flat_path = os.path.join(dataset_flat_dir, new_filename)
            cv2.imwrite(dest_labeled_path, face_img)
            cv2.imwrite(dest_flat_path, face_img)
            
            # Save annotation mapping locally
            local_annotations[crop_name] = label_val
            
            # Store row details for dataset registration
            row_dict = {}
            if matched_of_row is not None:
                # Copy matched features row
                row_dict = matched_of_row.copy()
            else:
                # Mock OpenFace features with zeros if not matched
                if main_features_df is not None:
                    # Reuse headers from the CSV
                    metadata_cols = ["filename", "label", "frame", "face_id", "timestamp", "confidence", "success"]
                    feature_cols = [c for c in main_features_df.columns if c not in metadata_cols]
                    for col in main_features_df.columns:
                        row_dict[col] = 0
                else:
                    # Fallback default columns
                    row_dict = {"frame": fnum, "face_id": of_face_id, "timestamp": fnum/fps, "confidence": 0.98, "success": 1}
            
            row_dict["filename"] = new_filename
            row_dict["label"] = label_val
            new_csv_rows.append(row_dict)
            final_labeled_count += 1

    # C. Update dataset CSV databases
    if final_labeled_count > 0:
        # Save local progress annotations.json
        with open(local_annotations_file, "w") as f:
            json.dump(local_annotations, f, indent=4)
        print(f"  [OK] Saved local progress tracker -> {local_annotations_file}")

        # Re-map frames globally to avoid frame ID collisions in OpenFace dataset csv
        max_frame = frame_map_df["frame"].max() if (frame_map_df is not None and not frame_map_df.empty) else 0
        if pd.isna(max_frame):
            max_frame = 0

        temp_frame_mappings = []
        for index, row in enumerate(new_csv_rows):
            new_global_frame = int(max_frame + 1 + index)
            temp_frame_mappings.append({
                "frame": new_global_frame,
                "filename": row["filename"]
            })
            row["frame"] = new_global_frame

        # Append to frame_to_filename.csv
        new_frame_map_df = pd.DataFrame(temp_frame_mappings)
        if frame_map_df is not None:
            frame_map_df = pd.concat([frame_map_df, new_frame_map_df], ignore_index=True)
        else:
            frame_map_df = new_frame_map_df
        frame_map_df.to_csv(frame_map_csv, index=False)
        print(f"  [OK] Updated frame map CSV -> {frame_map_csv}")

        # Append features to images_flat.csv
        new_features_df = pd.DataFrame(new_csv_rows)
        # Drop filename and label columns to align with original OpenFace outputs format
        feature_cols_only = new_features_df.drop(columns=["filename", "label"], errors="ignore")
        
        if main_features_df is not None:
            feature_cols_only = feature_cols_only.reindex(columns=main_features_df.columns, fill_value=0)
            main_features_df = pd.concat([main_features_df, feature_cols_only], ignore_index=True)
        else:
            main_features_df = feature_cols_only
        main_features_df.to_csv(openface_dest_csv, index=False)
        print(f"  [OK] Updated features CSV -> {openface_dest_csv}")

        # Regenerate labels.csv
        print("  [OK] Re-generating training dataset labels.csv...")
        labels_rows = []
        for folder_lbl_name, val in {
            "0_disengaged": 0,
            "1_partial": 1,
            "2_engaged": 2
        }.items():
            folder_path = os.path.join(dataset_images_dir, folder_lbl_name)
            if os.path.exists(folder_path):
                for img in os.listdir(folder_path):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        labels_rows.append([img, val])
        
        with open(labels_csv, "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(labels_rows)
        print(f"  [OK] Updated labels CSV -> {labels_csv}")
        
        print("\n" + "="*50)
        print("          ANNOTATION PROCESS COMPLETE")
        print("="*50)
        print(f"Successfully integrated {final_labeled_count} face crops into database!")
        print("Videos are ready inside the 'Annotated_Folder' subfolders.")
        print("="*50)
    else:
        print("\n>>> No annotations were exported.")

if __name__ == "__main__":
    main()
