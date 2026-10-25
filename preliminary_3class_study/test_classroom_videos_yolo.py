import os
import sys
import cv2
import traceback
import time
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model architectures
from models.base_fusion_model import FusionModel as BaseFusionModel
from models.lstm_fusion_model import FusionModel as LSTMFusionModel
from models.transformer_fusion_model import FusionModel as TransformerFusionModel
from models.lstm_transformer_fusion_model import FusionModel as LSTMTransformerFusionModel

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

# Import YOLO and local dependencies
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Dynamically add script dir and yolo_face_detector subfolder to python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
yolo_dir = os.path.join(script_dir, "yolo_face_detector")
if os.path.exists(yolo_dir) and yolo_dir not in sys.path:
    sys.path.append(yolo_dir)

import_errors = []

try:
    from yolo_face_detector.face_detector_v4 import FaceQualityFilter, download_face_weights
    from yolo_face_detector.illumination_normalization import normalize_illumination
except ImportError as e:
    import_errors.append(f"Primary Import Error: {e}\n{traceback.format_exc()}")
    # Fallback to current directory import if run directly in Colab without folder structure
    try:
        from face_detector_v4 import FaceQualityFilter, download_face_weights
        from illumination_normalization import normalize_illumination
    except ImportError as e2:
        import_errors.append(f"Fallback Import Error: {e2}\n{traceback.format_exc()}")
        FaceQualityFilter = None
        download_face_weights = None
        normalize_illumination = None

def parse_args():
    parser = argparse.ArgumentParser(description="Test student engagement detection models using YOLO + Illumination Normalization.")
    parser.add_argument("--video", type=str, required=True, help="Path to the classroom video file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained PyTorch engagement weights (.pth file).")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "lstm", "transformer", "lstm_transformer", "base"],
                        help="Model architecture type. 'auto' will automatically detect based on weights keys.")
    parser.add_argument("--out_dir", type=str, default="yolo_test_output", help="Output directory for annotated videos and reports.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-face.pt", help="Path to YOLO face model weights.")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for temporal models (default 16).")
    parser.add_argument("--normalize", action="store_true", help="Apply CLAHE and sharpening illumination normalization to face crops.")
    
    # Filter thresholds
    parser.add_argument("--min_size", default=25, type=int, help="Minimum width/height for face crops")
    parser.add_argument("--blur_threshold", default=30.0, type=float, help="Laplacian variance threshold for blur")
    parser.add_argument("--min_brightness", default=40, type=int, help="Minimum mean brightness threshold")
    parser.add_argument("--max_brightness", default=230, type=int, help="Maximum mean brightness threshold")
    return parser.parse_args()

def detect_model_type(state_dict):
    keys = list(state_dict.keys())
    has_lstm = any("lstm" in k for k in keys)
    has_transformer = any("transformer" in k or "pos_encoder" in k for k in keys)
    if has_lstm and has_transformer:
        return "lstm_transformer"
    elif has_transformer:
        return "transformer"
    elif has_lstm:
        return "lstm"
    else:
        return "base"

def load_engagement_model(model_path, model_type, device):
    print(f">>> Loading engagement model weights from {model_path}...")
    if not os.path.exists(model_path):
        print(f"!!! Error: Model file not found at {model_path}")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=device)
    
    if model_type == "auto":
        model_type = detect_model_type(state_dict)
        print(f">>> Auto-detected engagement architecture: {model_type.upper()}")

    # Determine openface dimension from weights
    openface_dim = 128  # default fallback
    for key in ["openface_net.0.weight", "openface_net.linear1.weight"]:
        if key in state_dict:
            openface_dim = state_dict[key].shape[1]
            break
    print(f">>> Auto-detected OpenFace Feature Dimension: {openface_dim}")

    if model_type == "lstm":
        model = LSTMFusionModel(openface_dim=openface_dim)
    elif model_type == "transformer":
        model = TransformerFusionModel(openface_dim=openface_dim, transformer_dim=256)
    elif model_type == "lstm_transformer":
        model = LSTMTransformerFusionModel(openface_dim=openface_dim)
    elif model_type == "base":
        model = BaseFusionModel(openface_dim=openface_dim)
    else:
        print(f"!!! Error: Unknown model type: {model_type}")
        sys.exit(1)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(">>> Engagement model loaded successfully!")
    return model, model_type, openface_dim

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Testing on device: {device}")

    if YOLO is None:
        print("[-] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    if FaceQualityFilter is None:
        print("\n" + "="*70)
        print("[-] IMPORT ERROR: Could not locate 'face_detector_v4.py' or 'illumination_normalization.py'")
        print("="*70)
        print(f"Current Directory (os.getcwd()):      {os.getcwd()}")
        print(f"Directory of this script (__file__):   {os.path.dirname(os.path.abspath(__file__))}")
        print("\nTraceback / Errors encountered during import:")
        for err in import_errors:
            print("-" * 50)
            print(err)
            print("-" * 50)
        print("\nFiles/Folders in Current Directory:")
        try:
            for item in sorted(os.listdir('.')):
                print(f"  - {item}")
        except Exception as e:
            print(f"  (Failed to list files: {e})")
        print("\nSuggestions:")
        print("1. Ensure you have run: %cd \"/content/drive/MyDrive/Student_Engagement_DL\" (or your actual folder path) in Colab.")
        print("2. Ensure the 'yolo_face_detector' folder is uploaded inside that directory.")
        print("3. Ensure 'face_detector_v4.py' and 'illumination_normalization.py' are inside the 'yolo_face_detector' folder.")
        print("="*70 + "\n")
        sys.exit(1)

    # 1. Load trained engagement model
    engagement_model, model_type, openface_dim = load_engagement_model(args.model, args.model_type, device)

    # 2. Check and download face weight if needed
    if args.yolo_model == "yolov8n-face.pt" and not os.path.exists(args.yolo_model):
        success = download_face_weights(args.yolo_model)
        if not success:
            print("[-] Could not download yolov8n-face.pt. Falling back to default yolo11n.pt if available.")
            args.yolo_model = "yolo11n.pt"

    yolo_model = YOLO(args.yolo_model)
    is_face_model = "face" in args.yolo_model.lower()

    # 3. Open Video Source
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[-] Cannot open input video: {args.video}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output setup
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_folder = os.path.join(args.out_dir, video_name)
    os.makedirs(out_folder, exist_ok=True)

    output_video_path = os.path.join(out_folder, f"{video_name}_{model_type}_yolo_engagement.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize Filters and Transforms
    quality_filter = FaceQualityFilter(
        min_size=args.min_size,
        blur_threshold=args.blur_threshold,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Engagement config
    label_map = {0: "Disengaged", 1: "Partially Engaged", 2: "Engaged"}
    colors = {
        0: (0, 0, 255),    # Red
        1: (0, 165, 255),  # Orange/Yellow
        2: (0, 255, 0)     # Green
    }

    # Setup annotations and OpenFace CSV for mapping and accuracy calculation if they exist
    video_dir = os.path.dirname(os.path.abspath(args.video))
    annotations_path = os.path.join(video_dir, "annotations.json")
    
    openface_csv_path = None
    csv_files = glob.glob(os.path.join(video_dir, "*.csv"))
    if csv_files:
        openface_csv_path = csv_files[0]
        
    print(f"\n[DEBUG] Video directory: {video_dir}")
    print(f"[DEBUG] Looking for annotations at: {annotations_path} (Exists: {os.path.exists(annotations_path)})")
    print(f"[DEBUG] Looking for OpenFace CSV (glob '*.csv') in: {video_dir} (CSVs found: {csv_files})")
        
    annotations = None
    openface_df = None
    yolo_to_openface_matches = {} # track_id -> dict of of_face_id -> overlap_count
    y_true_list = []
    y_pred_list = []
    
    if os.path.exists(annotations_path) and openface_csv_path is not None:
        import json
        print(f"[+] Found annotations.json and OpenFace CSV ({openface_csv_path}) for mapping.")
        try:
            with open(annotations_path, "r") as f:
                annotations = json.load(f)
            openface_df = pd.read_csv(openface_csv_path)
            openface_df.columns = openface_df.columns.str.strip()
            print(f"[+] Loaded {len(annotations)} annotations and {len(openface_df)} OpenFace records.")
        except Exception as e:
            print(f"[!] Warning: Failed to load annotations/CSV for accuracy calculation: {e}")
    else:
        print("[!] Note: annotations.json or OpenFace CSV not found in video folder. Accuracy calculation will be skipped.")

    # Tracking sequence buffers for temporal models
    # Dictionary of track_id -> List of transformed image tensors (up to seq_len)
    face_buffers = {}
    
    # Store predictions for the summary report
    # Dictionary of track_id -> Dictionary of frame -> prediction_class
    predictions_history = {}

    is_sequence = "base" not in model_type
    processed_frames = 0

    print(f"\n[+] Running Inference with V4 Face Detector & {model_type.upper()} Model...")
    
    for frame_idx in tqdm(range(total_frames), desc="Running Pipeline"):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1

        # Get OpenFace rows for the current frame for track mapping
        of_rows_current = None
        if openface_df is not None:
            of_rows_current = openface_df[(openface_df["frame"] == processed_frames) & (openface_df["success"] == 1)]

        # Run YOLO ByteTrack tracking
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if not is_face_model and cls != 0:
                    # If not a custom face model, only keep class 0 (person)
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = map(int, xyxy)

                if not is_face_model:
                    # Fallback person box to face box
                    person_h = ymax - ymin
                    ymax = ymin + int(person_h * 0.25)

                track_id = int(box.id[0].item()) if box.id is not None else 0
                if track_id == 0:
                    continue

                # Overlap matching between YOLO track box and OpenFace CSV landmark boxes
                matched_of_face_id = None
                if of_rows_current is not None and not of_rows_current.empty:
                    best_iou = 0.0
                    best_face_id = None
                    for _, of_row in of_rows_current.iterrows():
                        try:
                            # Calculate OpenFace bounding box from 68 landmarks
                            x_coords = [of_row[f"x_{i}"] for i in range(68)]
                            y_coords = [of_row[f"y_{i}"] for i in range(68)]
                            of_xmin, of_xmax = min(x_coords), max(x_coords)
                            of_ymin, of_ymax = min(y_coords), max(y_coords)
                            
                            # Pad OpenFace box slightly (similar to test_classroom_videos.py overlay logic)
                            w_of, h_of = of_xmax - of_xmin, of_ymax - of_ymin
                            of_xmin = max(0, int(of_xmin - 0.1 * w_of))
                            of_xmax = min(width, int(of_xmax + 0.1 * w_of))
                            of_ymin = max(0, int(of_ymin - 0.15 * h_of))
                            of_ymax = min(height, int(of_ymax + 0.1 * h_of))
                            
                            iou = compute_iou([xmin, ymin, xmax, ymax], [of_xmin, of_ymin, of_xmax, of_ymax])
                            if iou > best_iou:
                                best_iou = iou
                                best_face_id = int(of_row["face_id"])
                        except Exception:
                            continue
                    
                    if best_iou > 0.2: # Threshold for overlap
                        if track_id not in yolo_to_openface_matches:
                            yolo_to_openface_matches[track_id] = {}
                        yolo_to_openface_matches[track_id][best_face_id] = yolo_to_openface_matches[track_id].get(best_face_id, 0) + 1

                # Crop coordinates
                xmin_c, ymin_c = max(0, xmin), max(0, ymin)
                xmax_c, ymax_c = min(width, xmax), min(height, ymax)

                if (xmax_c - xmin_c) <= 0 or (ymax_c - ymin_c) <= 0:
                    continue

                face_crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]
                is_valid, _ = quality_filter.check_quality(face_crop)
                
                if not is_valid:
                    # Draw red bounding box for low-quality / rejected crops
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                    cv2.putText(frame, f"ID:{track_id} [LOW QUALITY]", (xmin, max(ymin - 5, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    continue

                # Apply Illumination Normalization if enabled
                if args.normalize and normalize_illumination is not None:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    normalized_rgb = normalize_illumination(face_crop_rgb)
                    face_crop = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)

                # Preprocess image crop for DL model
                transformed_img = transform(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

                # Handle tracking sequences
                if track_id not in face_buffers:
                    face_buffers[track_id] = []
                    predictions_history[track_id] = {}

                face_buffers[track_id].append(transformed_img)
                if len(face_buffers[track_id]) > args.seq_len:
                    face_buffers[track_id].pop(0)

                # Run PyTorch Model Inference
                with torch.no_grad():
                    # Create mock zero openface features vector
                    openface_feat = torch.zeros((1, openface_dim), dtype=torch.float32).to(device)

                    if is_sequence:
                        # Ensure sequence is padded if we don't have enough history frames yet
                        seq_tensors = list(face_buffers[track_id])
                        while len(seq_tensors) < args.seq_len:
                            seq_tensors.insert(0, seq_tensors[0]) # Pad by repeating oldest frame

                        image_seq = torch.stack(seq_tensors).unsqueeze(0).to(device) # [1, seq_len, 3, 224, 224]
                        outputs = engagement_model(image_seq, openface_feat)
                    else:
                        # Base single frame model
                        image_tensor = transformed_img.unsqueeze(0).to(device) # [1, 3, 224, 224]
                        outputs = engagement_model(image_tensor, openface_feat)

                    pred_class = int(outputs.argmax(dim=1).item())
                    predictions_history[track_id][processed_frames] = pred_class

                    # Check accuracy if annotations are loaded
                    of_face_id = None
                    if track_id in yolo_to_openface_matches and yolo_to_openface_matches[track_id]:
                        of_face_id = max(yolo_to_openface_matches[track_id], key=yolo_to_openface_matches[track_id].get)
                        
                    if of_face_id is not None and annotations is not None:
                        gt_label = None
                        for ext in [".bmp", ".jpg", ".png", ".jpeg"]:
                            key = f"frame_det_{of_face_id:02d}_{processed_frames:06d}{ext}"
                            if key in annotations:
                                gt_label = annotations[key]
                                break
                        if gt_label is not None:
                            y_true_list.append(gt_label)
                            y_pred_list.append(pred_class)

                # Draw bounding box and labels based on prediction
                color = colors[pred_class]
                label_text = label_map[pred_class]
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                text_str = f"Student {track_id}: {label_text}"
                
                # Bounding box label header
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(text_str, font, font_scale, thickness)[0]
                
                cv2.rectangle(frame, (xmin, ymin - text_size[1] - 8), (xmin + text_size[0] + 6, ymin), color, -1)
                cv2.putText(frame, text_str, (xmin + 3, ymin - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out_video.write(frame)

    cap.release()
    out_video.release()
    print(f"\n>>> Annotated video saved to: {output_video_path}")

    # Generate Summary Report and Chart
    generate_report(predictions_history, label_map, out_folder, video_name, model_type)

    # Print Accuracy & Metrics if annotations were matched
    if len(y_true_list) > 0:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        accuracy = accuracy_score(y_true_list, y_pred_list)
        target_names = ["Disengaged", "Partially Engaged", "Engaged"]
        report = classification_report(y_true_list, y_pred_list, target_names=target_names, zero_division=0)
        cm = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1, 2])
        
        print("\n" + "="*70)
        print("          ACCURACY EVALUATION AGAINST GROUND TRUTH")
        print("="*70)
        print(f"Matched Detections:    {len(y_true_list)}")
        print(f"Overall Accuracy:      {accuracy * 100:.2f}%")
        print("\nClassification Report:\n")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        print("="*70 + "\n")

def generate_report(predictions, label_map, out_dir, video_name, model_type):
    report_path = os.path.join(out_dir, f"{video_name}_{model_type}_yolo_report.txt")
    report_lines = [
        f"Classroom Engagement Report (YOLO + Normalization) - {video_name}",
        "=" * 60,
        ""
    ]
    summary_data = []

    for face_id, frames_dict in predictions.items():
        total_tracked = len(frames_dict)
        if total_tracked == 0:
            continue
        
        counts = {0: 0, 1: 0, 2: 0}
        for pred in frames_dict.values():
            counts[pred] += 1
            
        disengaged_pct = 100 * counts[0] / total_tracked
        partial_pct = 100 * counts[1] / total_tracked
        engaged_pct = 100 * counts[2] / total_tracked
        avg_engagement = (counts[1]*1 + counts[2]*2) / total_tracked
        
        report_lines.append(f"Student {face_id}:")
        report_lines.append(f"  - Total Tracked Frames: {total_tracked}")
        report_lines.append(f"  - Engaged:             {engaged_pct:.1f}% ({counts[2]} frames)")
        report_lines.append(f"  - Partially Engaged:   {partial_pct:.1f}% ({counts[1]} frames)")
        report_lines.append(f"  - Disengaged:          {disengaged_pct:.1f}% ({counts[0]} frames)")
        report_lines.append(f"  - Engagement Score:    {avg_engagement:.2f} / 2.00")
        report_lines.append("-" * 30)

        summary_data.append({
            "Student": f"Student {face_id}",
            "Engaged": engaged_pct,
            "Partial": partial_pct,
            "Disengaged": disengaged_pct
        })

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f">>> Text report saved to: {report_path}")

    # Plot Chart
    if summary_data:
        plt.figure(figsize=(10, 6))
        students = [d["Student"] for d in summary_data]
        engaged = [d["Engaged"] for d in summary_data]
        partial = [d["Partial"] for d in summary_data]
        disengaged = [d["Disengaged"] for d in summary_data]

        bar_width = 0.5
        r = np.arange(len(students))

        plt.bar(r, engaged, color='g', width=bar_width, label='Engaged')
        plt.bar(r, partial, bottom=engaged, color='orange', width=bar_width, label='Partially Engaged')
        bottom_disengaged = np.array(engaged) + np.array(partial)
        plt.bar(r, disengaged, bottom=bottom_disengaged, color='r', width=bar_width, label='Disengaged')

        plt.xlabel('Students')
        plt.ylabel('Percentage of Time (%)')
        plt.title(f'YOLO Engagement Distribution - {video_name}')
        plt.xticks(r, students)
        plt.ylim(0, 100)
        plt.legend(loc='upper right')
        
        chart_path = os.path.join(out_dir, f"{video_name}_{model_type}_yolo_chart.png")
        plt.savefig(chart_path)
        plt.close()
        print(f">>> Engagement chart saved to: {chart_path}")

if __name__ == "__main__":
    main()
