import os
import sys
import argparse
import glob
import shutil
import json
import pandas as pd
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive classroom student face crop annotation tool.")
    parser.add_argument("--video", type=str, required=True, help="Path to the classroom video MP4 file or video name.")
    parser.add_argument("--src_dir", type=str, default="", help="Path to aligned face crops. Defaults to processed_recordings/<video_name>/<video_name>_aligned/")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Main training dataset directory.")
    return parser.parse_args()

def parse_filename(filename):
    """
    Parses face_id and frame number from OpenFace aligned filenames like 'frame_det_00_000001.bmp'
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    try:
        # Expected parts: ['frame', 'det', 'face_id', 'frame_num']
        face_id = int(parts[2])
        frame_num = int(parts[3])
        return face_id, frame_num
    except (IndexError, ValueError):
        return None, None

def main():
    args = parse_args()
    
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    
    # 1. Setup Source Aligned Folder
    if args.src_dir == "":
        src_dir = os.path.join("processed_recordings", video_name, f"{video_name}_aligned")
    else:
        src_dir = args.src_dir
        
    if not os.path.exists(src_dir):
        print(f"!!! Error: Aligned crops folder does not exist at {src_dir}")
        print("    Please run test_classroom_videos.py on this video first to extract crops.")
        sys.exit(1)

    # Setup Video CSV Path
    video_csv_path = os.path.join("processed_recordings", video_name, f"{video_name}.csv")
    if not os.path.exists(video_csv_path):
        # Fallback search
        csv_files = glob.glob(os.path.join("processed_recordings", video_name, "*.csv"))
        if csv_files:
            video_csv_path = csv_files[0]
        else:
            print(f"!!! Error: OpenFace CSV features file not found for {video_name}")
            sys.exit(1)

    # 2. Setup Dataset Target Folders
    dataset_dir = args.dataset_dir
    images_dest_dir = os.path.join(dataset_dir, "images")
    flat_dest_dir = os.path.join(dataset_dir, "images_flat")
    openface_dest_csv = os.path.join(dataset_dir, "openface", "images_flat.csv")
    frame_map_csv = os.path.join(dataset_dir, "frame_to_filename.csv")
    labels_csv = os.path.join(dataset_dir, "labels.csv")

    # Map labels to folder structures
    class_folders = {
        0: "0_disengaged",
        1: "1_partial",
        2: "2_engaged"
    }

    for folder_name in class_folders.values():
        os.makedirs(os.path.join(images_dest_dir, folder_name), exist_ok=True)
    os.makedirs(flat_dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "openface"), exist_ok=True)

    # 3. Load or initialize annotations progress tracker
    progress_file = os.path.join("processed_recordings", video_name, "annotations.json")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    # Gather list of aligned image crops
    all_crops = [f for f in os.listdir(src_dir) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
    all_crops.sort()
    
    # Filter out already annotated crops
    unannotated_crops = [c for c in all_crops if c not in annotations]
    
    total_crops = len(all_crops)
    remaining_crops = len(unannotated_crops)
    
    print("="*60)
    print(f"   Interactive Annotator - {video_name.upper()}")
    print("="*60)
    print(f"Total face crops:      {total_crops}")
    print(f"Already annotated:     {total_crops - remaining_crops}")
    print(f"Remaining to annotate: {remaining_crops}")
    print("\n[Controls]:")
    print("  Press '0' -> Disengaged")
    print("  Press '1' -> Partially Engaged")
    print("  Press '2' -> Engaged")
    print("  Press 'ESC' or 'q' -> Save Progress and Exit")
    print("="*60)
    
    if remaining_crops == 0:
        print(">>> All crops are already annotated for this video!")
        sys.exit(0)

    # Load video CSV features to copy landmarks dynamically
    video_features_df = pd.read_csv(video_csv_path)
    video_features_df.columns = video_features_df.columns.str.strip()

    # Load main dataset feature and frame-to-filename CSVs
    if os.path.exists(openface_dest_csv):
        main_features_df = pd.read_csv(openface_dest_csv)
        main_features_df.columns = main_features_df.columns.str.strip()
    else:
        main_features_df = None

    if os.path.exists(frame_map_csv):
        frame_map_df = pd.read_csv(frame_map_csv)
        frame_map_df.columns = frame_map_df.columns.str.strip()
    else:
        frame_map_df = None

    cv2.namedWindow("Annotate Student Engagement", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate Student Engagement", 450, 450)

    new_annotations_count = 0
    new_csv_rows = []
    new_frame_map_rows = []

    for i, crop_file in enumerate(unannotated_crops):
        crop_path = os.path.join(src_dir, crop_file)
        img = cv2.imread(crop_path)
        if img is None:
            continue

        # Extract face_id and frame number
        face_id, frame_num = parse_filename(crop_file)
        if face_id is None or frame_num is None:
            print(f"    [Warning] Skipping malformed filename: {crop_file}")
            continue

        # Find OpenFace row matching this frame and face_id
        feat_row = video_features_df[
            (video_features_df["frame"] == frame_num) & 
            (video_features_df["face_id"] == face_id)
        ]
        
        if len(feat_row) == 0:
            print(f"    [Warning] OpenFace feature data not found for {crop_file} in CSV. Skipping.")
            continue

        # Annotation display loop
        while True:
            # Create a display copy with status bar at the bottom
            canvas = np.zeros((480, 400, 3), dtype=np.uint8)
            resized_face = cv2.resize(img, (400, 400))
            canvas[0:400, 0:400] = resized_face

            # Status bar drawing
            progress_str = f"Crop {i + 1}/{remaining_crops}  ({crop_file})"
            instruction_str = "Keys: [0] Diseng  [1] Partial  [2] Eng"
            exit_str = "Press ESC or 'q' to Quit"
            
            cv2.putText(canvas, progress_str, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, instruction_str, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, exit_str, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Annotate Student Engagement", canvas)
            key = cv2.waitKey(0) & 0xFF

            if key in [ord('0'), ord('1'), ord('2')]:
                label_val = int(chr(key))
                class_dir = class_folders[label_val]
                
                # Prepend video name to avoid filename collisions
                new_filename = f"{video_name}_{crop_file}"
                
                # A. Copy image to labeled class folder and images_flat
                src_img_path = os.path.join(src_dir, crop_file)
                shutil.copy(src_img_path, os.path.join(images_dest_dir, class_dir, new_filename))
                shutil.copy(src_img_path, os.path.join(flat_dest_dir, new_filename))
                
                # B. Save label classification mapping
                annotations[crop_file] = label_val
                
                # C. Record feature row for database update
                # Get the matching row as a dictionary
                row_dict = feat_row.iloc[0].to_dict()
                row_dict["filename"] = new_filename
                row_dict["label"] = label_val
                new_csv_rows.append(row_dict)
                
                new_annotations_count += 1
                break
                
            elif key == 27 or key == ord('q'): # ESC or q
                print("\n>>> Exiting annotator...")
                cv2.destroyAllWindows()
                break
        
        # If user exited using ESC/q
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()

    # 4. Save progress json and update dataset CSVs
    if new_annotations_count > 0:
        # Save local progress
        with open(progress_file, "w") as f:
            json.dump(annotations, f, indent=4)
            
        print(f"\n>>> Saving {new_annotations_count} annotations to the main training dataset...")

        # A. Append frame mappings to frame_to_filename.csv
        if frame_map_df is not None:
            max_frame = frame_map_df["frame"].max()
            if pd.isna(max_frame):
                max_frame = 0
        else:
            max_frame = 0

        # Construct new frame mappings
        temp_frame_mappings = []
        for index, row in enumerate(new_csv_rows):
            new_frame_num = int(max_frame + 1 + index)
            temp_frame_mappings.append({
                "frame": new_frame_num,
                "filename": row["filename"]
            })
            # Overwrite the temporary OpenFace frame column with this global unique frame id
            row["frame"] = new_frame_num

        new_frame_map_df = pd.DataFrame(temp_frame_mappings)
        if frame_map_df is not None:
            frame_map_df = pd.concat([frame_map_df, new_frame_map_df], ignore_index=True)
        else:
            frame_map_df = new_frame_map_df
        
        # Strip and save mapping
        frame_map_df.to_csv(frame_map_csv, index=False)
        print(f"    - Updated {frame_map_csv}")

        # B. Append features to openface/images_flat.csv
        new_features_df = pd.DataFrame(new_csv_rows)
        # Drop filename and label from feature data row to match original OpenFace format exactly
        feature_cols_only = new_features_df.drop(columns=["filename", "label"], errors="ignore")
        
        # Ensure column order matches the original images_flat.csv exactly
        if main_features_df is not None:
            # Reorder columns to match main CSV
            feature_cols_only = feature_cols_only.reindex(columns=main_features_df.columns, fill_value=0)
            main_features_df = pd.concat([main_features_df, feature_cols_only], ignore_index=True)
        else:
            main_features_df = feature_cols_only
            
        main_features_df.to_csv(openface_dest_csv, index=False)
        print(f"    - Updated {openface_dest_csv}")

        # C. Update labels.csv using create_labels.py logic
        print("    - Re-generating dataset labels.csv...")
        rows = []
        for folder, label in {
            "0_disengaged": 0,
            "1_partial": 1,
            "2_engaged": 2
        }.items():
            folder_path = os.path.join(images_dest_dir, folder)
            if os.path.exists(folder_path):
                for img in os.listdir(folder_path):
                    rows.append([img, label])
        
        with open(labels_csv, "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(rows)
            
        print("="*60)
        print("   DATASET UPDATE COMPLETED")
        print("="*60)
        print(f"Successfully integrated {new_annotations_count} labeled face crops!")
        print("You can now run training using the updated dataset files.")
        print("="*60)
    else:
        print("\n>>> No new annotations were added.")

if __name__ == "__main__":
    main()
