import os
import sys
import cv2
import time
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")  # Headless mode for server / Colab compatibility
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure parent directory and colab_exp4 are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config_exp4
from models_exp4 import EngagementLSTMClassifier
from preprocess.extract_frames_illumination import apply_illumination_enhancement

# Import YOLO and local face detector tools if available
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from yolo_face_detector.face_detector_v4 import FaceQualityFilter, download_face_weights
except ImportError:
    try:
        from face_detector_v4 import FaceQualityFilter, download_face_weights
    except ImportError:
        FaceQualityFilter = None
        download_face_weights = None


class EfficientNetFeatureExtractor:
    """
    Extracts 1280-dim visual embeddings from 224x224 RGB face crops using EfficientNet-B0.
    """
    def __init__(self, device):
        self.device = device
        from torchvision import models
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            model = models.efficientnet_b0(pretrained=True)

        model.classifier = nn.Identity()
        model.eval()
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def extract_crop_feature(self, crop_bgr):
        """
        Extracts 1280-dim feature vector from a single BGR image crop.
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = self.normalize(tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.squeeze(0).cpu().numpy()  # [1280]


def parse_args():
    parser = argparse.ArgumentParser(description="Exp4 (BiLSTM) Multi-Student Classroom Video Testing Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input classroom video file")
    parser.add_argument("--model", type=str, default=None, help="Path to trained Exp4 BiLSTM weights (.pth)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for annotated video and reports")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-face.pt", help="Path to YOLO face model weights")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for temporal buffer")
    parser.add_argument("--no_normalize", action="store_true", help="Disable CLAHE + Laplacian Sharpening illumination normalization")
    
    # Filter thresholds
    parser.add_argument("--min_size", default=25, type=int, help="Minimum face width/height in pixels")
    parser.add_argument("--blur_threshold", default=30.0, type=float, help="Laplacian variance blur threshold")
    parser.add_argument("--min_brightness", default=40, type=int, help="Minimum average brightness threshold")
    parser.add_argument("--max_brightness", default=230, type=int, help="Maximum average brightness threshold")
    return parser.parse_args()


def load_exp4_lstm_model(model_path, device):
    """
    Loads EngagementLSTMClassifier model weights.
    """
    target_path = model_path if (model_path and os.path.exists(model_path)) else config_exp4.MODEL_WEIGHTS_PATH

    feature_dim = 1408  # default feature dimension
    hidden_dim = config_exp4.HIDDEN_DIM
    num_layers = config_exp4.NUM_LAYERS
    state_dict = None

    if target_path and os.path.exists(target_path):
        print(f"[+] Loading trained Exp4 BiLSTM model from: {target_path}")
        checkpoint = torch.load(target_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            feature_dim = checkpoint.get("input_dim", 1408)
            hidden_dim = checkpoint.get("hidden_dim", config_exp4.HIDDEN_DIM)
            num_layers = checkpoint.get("num_layers", config_exp4.NUM_LAYERS)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint

        if state_dict is not None and "input_norm.weight" in state_dict:
            feature_dim = state_dict["input_norm.weight"].shape[0]
            print(f"[+] Auto-detected model feature dimension from layer norm: {feature_dim}")
    else:
        if model_path:
            print(f"[!] Warning: Specified model path '{model_path}' not found. Using initialized model.")
        else:
            print(f"[!] Warning: Default weights '{config_exp4.MODEL_WEIGHTS_PATH}' not found. Using initialized model.")

    model = EngagementLSTMClassifier(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=3,
        dropout=config_exp4.DROPOUT
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f" EXPERIMENT 4: BILSTM CLASSROOM VIDEO PIPELINE   ")
    print(f" Device: {device}")
    print(f" Input Video: {args.video}")
    print(f"==================================================")

    if YOLO is None:
        print("[-] Error: Ultralytics package is not installed. Install with: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"[-] Error: Video file not found: {args.video}")
        sys.exit(1)

    # 1. Load trained BiLSTM engagement classifier
    lstm_model = load_exp4_lstm_model(args.model, device)

    # 2. Check and download face weight if needed
    yolo_weight = args.yolo_model
    if yolo_weight == "yolov8n-face.pt" and not os.path.exists(yolo_weight):
        if download_face_weights is not None:
            success = download_face_weights(yolo_weight)
            if not success:
                yolo_weight = "yolo11n.pt"
        else:
            yolo_weight = "yolo11n.pt"

    print(f"[+] Loading YOLO model from: {yolo_weight}")
    yolo = YOLO(yolo_weight)
    is_face_model = "face" in yolo_weight.lower()

    # 3. Open Video Source
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[-] Error: Could not open video file: {args.video}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output setup
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out_dir if args.out_dir else os.path.join(config_exp4.EVAL_OUT_DIR, f"{video_name}_exp4_results")
    os.makedirs(out_dir, exist_ok=True)

    output_video_path = os.path.join(out_dir, f"{video_name}_exp4_lstm_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize Filters and Feature Extractors
    quality_filter = FaceQualityFilter(
        min_size=args.min_size,
        blur_threshold=args.blur_threshold,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness
    ) if FaceQualityFilter is not None else None

    effnet_extractor = EfficientNetFeatureExtractor(device)

    # Class mappings & visual styling
    class_names = ["Disengaged", "Partially Engaged", "Engaged"]
    colors = {
        0: (0, 0, 255),     # Red for Disengaged
        1: (0, 165, 255),   # Orange/Yellow for Partially Engaged
        2: (0, 255, 0)      # Green for Engaged
    }

    # Tracking & prediction data structures
    face_buffers = {}       # track_id -> list of [1408] feature arrays (up to seq_len)
    predictions_history = {} # track_id -> dict of frame_idx -> pred_class
    detailed_records = []    # list of dicts for CSV export

    print(f"\n[+] Processing Video Frames and Tracking Students...")
    processed_frames = 0
    start_time = time.time()

    for frame_idx in tqdm(range(total_frames), desc="Exp4 BiLSTM Pipeline"):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1
        timestamp_sec = round(processed_frames / fps, 2)

        # Run YOLO tracking with ByteTrack
        results = yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

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
                    ymax = ymin + int(person_h * 0.25)

                track_id = int(box.id[0].item()) if box.id is not None else 0
                if track_id <= 0:
                    continue

                xmin_c, ymin_c = max(0, xmin), max(0, ymin)
                xmax_c, ymax_c = min(width, xmax), min(height, ymax)

                if (xmax_c - xmin_c) <= 0 or (ymax_c - ymin_c) <= 0:
                    continue

                face_crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]

                # Check Quality Filter
                if quality_filter is not None:
                    is_valid, reason = quality_filter.check_quality(face_crop)
                    if not is_valid:
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                        cv2.putText(frame, f"ID:{track_id} [{reason}]", (xmin, max(ymin - 5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        continue

                # Apply Illumination Normalization (CLAHE + Laplacian Sharpening)
                if not args.no_normalize:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_crop_rgb = apply_illumination_enhancement(face_crop_rgb)
                    face_crop = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)

                # Extract 1280-dim visual feature vector
                effnet_feat = effnet_extractor.extract_crop_feature(face_crop)  # [1280]

                # Dynamic facial feature vector dimension matching model's expected input dimension
                model_input_dim = getattr(lstm_model, "input_dim", 1408)
                openface_dim = max(0, model_input_dim - 1280)
                openface_feat = np.zeros(openface_dim, dtype=np.float32)

                # Fuse features -> [model_input_dim]
                fused_feat = np.concatenate([effnet_feat, openface_feat], axis=0)

                # Buffer features for temporal tracking sequence
                if track_id not in face_buffers:
                    face_buffers[track_id] = []
                    predictions_history[track_id] = {}

                face_buffers[track_id].append(fused_feat)
                if len(face_buffers[track_id]) > args.seq_len:
                    face_buffers[track_id].pop(0)

                # Form sequence tensor: [1, seq_len, 1408]
                seq_list = list(face_buffers[track_id])
                while len(seq_list) < args.seq_len:
                    seq_list.insert(0, seq_list[0])  # Repeat oldest frame for padding

                seq_tensor = torch.tensor(np.array(seq_list), dtype=torch.float32).unsqueeze(0).to(device)

                # Run BiLSTM model prediction
                with torch.no_grad():
                    logits = lstm_model(seq_tensor, seq_lens=[args.seq_len])
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pred_class = int(np.argmax(probs))
                    confidence = float(probs[pred_class])

                predictions_history[track_id][processed_frames] = pred_class

                detailed_records.append({
                    "frame": processed_frames,
                    "timestamp_sec": timestamp_sec,
                    "track_id": track_id,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "predicted_class": pred_class,
                    "class_name": class_names[pred_class],
                    "confidence": round(confidence * 100, 2),
                    "disengaged_prob": round(float(probs[0]), 4),
                    "partial_prob": round(float(probs[1]), 4),
                    "engaged_prob": round(float(probs[2]), 4)
                })

                # Annotate Bounding Box and Engagement Label
                color = colors[pred_class]
                label_text = f"Student {track_id}: {class_names[pred_class]} ({confidence*100:.0f}%)"

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]

                cv2.rectangle(frame, (xmin, ymin - text_size[1] - 8), (xmin + text_size[0] + 6, ymin), color, -1)
                cv2.putText(frame, label_text, (xmin + 3, ymin - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out_video.write(frame)

    cap.release()
    out_video.release()
    elapsed = time.time() - start_time
    avg_fps = processed_frames / elapsed if elapsed > 0 else 0

    print(f"\n==================================================")
    print(f"        EXP4 BILSTM PROCESSING COMPLETE           ")
    print(f"==================================================")
    print(f"Processed Frames : {processed_frames}")
    print(f"Processing Speed : {avg_fps:.2f} FPS")
    print(f"Annotated Video  : {output_video_path}")

    # Generate CSV details, text report, and summary chart
    export_reports_and_charts(predictions_history, detailed_records, class_names, out_dir, video_name)


def export_reports_and_charts(predictions_history, detailed_records, class_names, out_dir, video_name):
    # 1. Save detailed CSV log
    csv_path = os.path.join(out_dir, f"{video_name}_exp4_lstm_details.csv")
    if detailed_records:
        df = pd.DataFrame(detailed_records)
        df.to_csv(csv_path, index=False)
        print(f"Detailed CSV Log : {csv_path}")

    # 2. Save Text Summary Report
    txt_path = os.path.join(out_dir, f"{video_name}_exp4_lstm_report.txt")
    report_lines = [
        f"Experiment 4: Illuminated BiLSTM Classroom Engagement Report",
        f"Video Name: {video_name}",
        "=" * 65,
        ""
    ]

    summary_data = []

    for track_id, frames_dict in predictions_history.items():
        total_tracked = len(frames_dict)
        if total_tracked == 0:
            continue

        counts = {0: 0, 1: 0, 2: 0}
        for pred in frames_dict.values():
            counts[pred] += 1

        disengaged_pct = 100.0 * counts[0] / total_tracked
        partial_pct    = 100.0 * counts[1] / total_tracked
        engaged_pct    = 100.0 * counts[2] / total_tracked
        avg_score      = (counts[1]*1 + counts[2]*2) / total_tracked

        report_lines.append(f"Student Track ID #{track_id}:")
        report_lines.append(f"  - Total Tracked Frames: {total_tracked}")
        report_lines.append(f"  - Engaged:             {engaged_pct:.1f}% ({counts[2]} frames)")
        report_lines.append(f"  - Partially Engaged:   {partial_pct:.1f}% ({counts[1]} frames)")
        report_lines.append(f"  - Disengaged:          {disengaged_pct:.1f}% ({counts[0]} frames)")
        report_lines.append(f"  - Average Engagement:  {avg_score:.2f} / 2.00")
        report_lines.append("-" * 40)

        summary_data.append({
            "Student": f"Student {track_id}",
            "Engaged": engaged_pct,
            "Partial": partial_pct,
            "Disengaged": disengaged_pct
        })

    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Text Summary     : {txt_path}")

    # 3. Generate Stacked Bar Chart Plot
    if summary_data:
        chart_path = os.path.join(out_dir, f"{video_name}_exp4_lstm_chart.png")
        plt.figure(figsize=(10, 6))

        students = [d["Student"] for d in summary_data]
        engaged = [d["Engaged"] for d in summary_data]
        partial = [d["Partial"] for d in summary_data]
        disengaged = [d["Disengaged"] for d in summary_data]

        bar_width = 0.45
        r = np.arange(len(students))

        plt.bar(r, engaged, color='green', width=bar_width, label='Engaged')
        plt.bar(r, partial, bottom=engaged, color='orange', width=bar_width, label='Partially Engaged')
        bottom_disengaged = np.array(engaged) + np.array(partial)
        plt.bar(r, disengaged, bottom=bottom_disengaged, color='red', width=bar_width, label='Disengaged')

        plt.xlabel('Students Detected')
        plt.ylabel('Percentage of Time (%)')
        plt.title(f'Exp4 BiLSTM Engagement Distribution - {video_name}')
        plt.xticks(r, students)
        plt.ylim(0, 100)
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.savefig(chart_path, dpi=200)
        plt.close()
        print(f"Summary Chart    : {chart_path}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
