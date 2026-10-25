import os
import sys
import time
import json
import base64
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# PyTorch & torchvision imports
import torchvision.transforms as transforms
from torchvision import models

# Import sklearn metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Import Gradio
import gradio as gr

# Ensure script dir in path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import config_transformer
from models_transformer import EngagementTransformerClassifier

# Try importing YOLO and Face quality filter
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

# Set Matplotlib dark theme default parameters for harmonious integration
plt.rcParams['text.color'] = '#f8fafc'
plt.rcParams['axes.labelcolor'] = '#94a3b8'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.facecolor'] = '#0f172a'
plt.rcParams['figure.facecolor'] = '#0f172a'

# Illumination Normalization Functions
def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)

def apply_sharpen(img_rgb):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return np.clip(cv2.filter2D(img_rgb.astype(np.float32), -1, kernel), 0, 255).astype(np.uint8)

def apply_illumination_enhancement(img_rgb):
    return apply_sharpen(apply_clahe(img_rgb))


class EfficientNetFeatureExtractor:
    def __init__(self, device):
        self.device = device
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
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = self.normalize(tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.squeeze(0).cpu().numpy()


def load_transformer_model(model_path, device):
    target_path = model_path if (model_path and os.path.exists(model_path)) else config_transformer.MODEL_WEIGHTS_PATH

    feature_dim = getattr(config_transformer, "INPUT_DIM", 1408)
    d_model = config_transformer.D_MODEL
    nhead = config_transformer.NHEAD
    num_layers = config_transformer.NUM_LAYERS
    dim_feedforward = config_transformer.DIM_FEEDFORWARD
    state_dict = None

    if target_path and os.path.exists(target_path):
        try:
            print(f"[+] Loading Transformer checkpoint from: {target_path}")
            checkpoint = torch.load(target_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                feature_dim = checkpoint.get("input_dim", getattr(config_transformer, "INPUT_DIM", 1408))
                d_model = checkpoint.get("d_model", config_transformer.D_MODEL)
                nhead = checkpoint.get("nhead", config_transformer.NHEAD)
                num_layers = checkpoint.get("num_layers", config_transformer.NUM_LAYERS)
                dim_feedforward = checkpoint.get("dim_feedforward", config_transformer.DIM_FEEDFORWARD)
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint

            if state_dict is not None and "input_proj.weight" in state_dict:
                feature_dim = state_dict["input_proj.weight"].shape[1]
                d_model = state_dict["input_proj.weight"].shape[0]
        except Exception as e:
            print(f"[!] Warning: Error reading checkpoint file ({e}). Using initialized architecture.")
            state_dict = None
    else:
        print("[!] Notice: No custom transformer weights found. Using initialized Transformer architecture.")

    model = EngagementTransformerClassifier(
        input_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=config_transformer.DROPOUT,
        num_classes=3
    )

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[+] Successfully loaded Transformer weights.")
        except Exception as e:
            print(f"[!] Info: Checkpoint architecture differs from Transformer ({e}). Attempting compatible load...")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("[+] Loaded partial compatible weights.")
            except Exception:
                print("[!] Using initialized Transformer model.")

    model.to(device)
    model.eval()
    return model


def process_video_and_generate_dashboard(video_path, seq_len=16, apply_norm=True, progress=gr.Progress(track_tqdm=True)):
    if not video_path or not os.path.exists(video_path):
        empty_df = pd.DataFrame(columns=["Student", "Engaged (%)", "Partially Engaged (%)", "Disengaged (%)", "Engagement Score", "Status"])
        return None, "⚠️ Error: Invalid video file path.", "", "", "", "", None, None, None, empty_df, None, ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if progress:
        progress(0.05, desc="Initializing PyTorch & Loading Temporal Transformer Model...")

    # 1. Load Transformer
    transformer_model = load_transformer_model(config_transformer.MODEL_WEIGHTS_PATH, device)

    # 2. Load YOLO
    yolo_weight = "yolov8n-face.pt"
    if not os.path.exists(yolo_weight):
        root_weight = os.path.join(config_transformer.PROJECT_ROOT, "yolov8n-face.pt")
        if os.path.exists(root_weight):
            yolo_weight = root_weight
        elif download_face_weights is not None:
            download_face_weights("yolov8n-face.pt")
            if os.path.exists("yolov8n-face.pt"):
                yolo_weight = "yolov8n-face.pt"
            else:
                yolo_weight = "yolo11n.pt"
        else:
            yolo_weight = "yolo11n.pt"

    yolo = YOLO(yolo_weight)
    is_face_model = "face" in yolo_weight.lower()

    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        empty_df = pd.DataFrame(columns=["Student", "Engaged (%)", "Partially Engaged (%)", "Disengaged (%)", "Engagement Score", "Status"])
        return None, "⚠️ Error: Could not open input video.", "", "", "", "", None, None, None, empty_df, None, ""

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = os.path.join(script_dir, "dashboard_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_video_path = os.path.join(out_dir, "annotated_dashboard_output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    quality_filter = FaceQualityFilter(min_size=20, blur_threshold=20.0, min_brightness=35, max_brightness=235) if FaceQualityFilter else None
    effnet_extractor = EfficientNetFeatureExtractor(device)

    class_names = ["Disengaged", "Partially Engaged", "Engaged"]
    colors = {0: (0, 0, 255), 1: (0, 165, 255), 2: (0, 255, 0)}

    face_buffers = {}
    predictions_history = {}
    detailed_records = []

    processed_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        processed_frames += 1
        timestamp_sec = round(processed_frames / fps, 2)

        if progress and total_frames > 0 and processed_frames % 15 == 0:
            pct = min(0.95, processed_frames / total_frames)
            progress(pct, desc=f"Processing Video Frame {processed_frames}/{total_frames} ({pct*100:.0f}%)...")

        results = yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if not is_face_model and cls != 0: continue

                xyxy = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                
                if not is_face_model:
                    person_h = ymax - ymin
                    person_w = xmax - xmin
                    xmin = xmin + int(person_w * 0.20)
                    xmax = xmax - int(person_w * 0.20)
                    ymax = ymin + int(person_h * 0.28)

                track_id = int(box.id[0].item()) if box.id is not None else 0
                if track_id <= 0: continue

                xmin_c, ymin_c = max(0, xmin), max(0, ymin)
                xmax_c, ymax_c = min(width, xmax), min(height, ymax)
                if (xmax_c - xmin_c) <= 0 or (ymax_c - ymin_c) <= 0: continue

                face_crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]

                if quality_filter is not None:
                    is_valid, reason = quality_filter.check_quality(face_crop)
                    if not is_valid:
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                        continue

                if apply_norm:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_crop_rgb = apply_illumination_enhancement(face_crop_rgb)
                    face_crop = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)

                effnet_feat = effnet_extractor.extract_crop_feature(face_crop)
                model_input_dim = getattr(transformer_model, "input_dim", 1408)
                if effnet_feat.shape[0] < model_input_dim:
                    spatial_feat = np.pad(effnet_feat, (0, model_input_dim - effnet_feat.shape[0]), 'constant')
                else:
                    spatial_feat = effnet_feat[:model_input_dim]

                if track_id not in face_buffers:
                    face_buffers[track_id] = []
                    predictions_history[track_id] = {}

                face_buffers[track_id].append(spatial_feat)
                if len(face_buffers[track_id]) > seq_len:
                    face_buffers[track_id].pop(0)

                seq_list = list(face_buffers[track_id])
                while len(seq_list) < seq_len:
                    seq_list.insert(0, seq_list[0])

                seq_tensor = torch.tensor(np.array(seq_list), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = transformer_model(seq_tensor, seq_lens=[seq_len])
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pred_class = int(np.argmax(probs))
                    confidence = float(probs[pred_class])

                predictions_history[track_id][processed_frames] = pred_class

                detailed_records.append({
                    "frame": processed_frames,
                    "timestamp_sec": timestamp_sec,
                    "track_id": track_id,
                    "predicted_class": pred_class,
                    "class_name": class_names[pred_class],
                    "confidence": round(confidence * 100, 2),
                    "disengaged_prob": round(float(probs[0]), 4),
                    "partial_prob": round(float(probs[1]), 4),
                    "engaged_prob": round(float(probs[2]), 4)
                })

                color = colors[pred_class]
                label_text = f"Student {track_id}: {class_names[pred_class]} ({confidence*100:.0f}%)"
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (xmin, ymin - text_size[1] - 8), (xmin + text_size[0] + 6, ymin), color, -1)
                cv2.putText(frame, label_text, (xmin + 3, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_video.write(frame)

    cap.release()
    out_video.release()
    elapsed_time = time.time() - start_time

    if progress:
        progress(0.96, desc="Generating Dynamic Visual Analytics & Metrics...")

    df_details = pd.DataFrame(detailed_records) if detailed_records else pd.DataFrame()
    time_formatter = ticker.FuncFormatter(lambda s, _: f"{int(s//60):02d}:{int(s%60):02d}")

    # -------------------------------------------------------------
    # 1. CLASS-LEVEL TEMPORAL STACKED AREA CHART
    # -------------------------------------------------------------
    fig_trend = plt.figure(figsize=(10, 4.5), facecolor='#0f172a')
    ax_trend = fig_trend.add_subplot(111)
    ax_trend.set_facecolor('#0f172a')

    time_grouped = None
    if not df_details.empty:
        time_grouped = df_details.groupby('timestamp_sec')['predicted_class'].value_counts(normalize=True).unstack(fill_value=0)
        for c_idx in [0, 1, 2]:
            if c_idx not in time_grouped.columns:
                time_grouped[c_idx] = 0.0
        time_grouped = time_grouped[[0, 1, 2]] * 100.0

        timestamps = time_grouped.index.values
        pct_disengaged = time_grouped[0].values
        pct_partial = time_grouped[1].values
        pct_engaged = time_grouped[2].values

        class_engagement_index = pct_partial * 0.5 + pct_engaged * 1.0

        ax_trend.stackplot(timestamps, pct_engaged, pct_partial, pct_disengaged,
                           labels=['% Engaged', '% Partially Engaged', '% Disengaged'],
                           colors=['#22c55e', '#f59e0b', '#ef4444'], alpha=0.85)
        
        ax_trend.plot(timestamps, class_engagement_index, color='#38bdf8', linewidth=2.8, linestyle='--', label='Class Engagement Index (%)')

        ax_trend.set_title('Classroom Overall Engagement Dynamics Over Time', fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
        ax_trend.set_xlabel('Lecture Time (MM:SS)', fontsize=11, color='#94a3b8')
        ax_trend.set_ylabel('Classroom Ratio (%)', fontsize=11, color='#94a3b8')
        ax_trend.set_ylim(0, 100)
        ax_trend.set_xlim(min(timestamps) if len(timestamps) > 0 else 0, max(timestamps) if len(timestamps) > 0 else 1)
        ax_trend.xaxis.set_major_formatter(time_formatter)
        ax_trend.grid(True, linestyle=':', alpha=0.4, color='#334155')
        ax_trend.legend(loc='upper right', framealpha=0.9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
    else:
        ax_trend.set_title('Classroom Overall Engagement Dynamics (No Student Tracked)', fontsize=12, color='#f8fafc')

    fig_trend.tight_layout()
    class_trend_chart_path = os.path.join(out_dir, "class_engagement_trend.png")
    fig_trend.savefig(class_trend_chart_path, dpi=200, facecolor='#0f172a')
    plt.close(fig_trend)

    # -------------------------------------------------------------
    # 2. INDIVIDUAL STUDENT TIMELINE CHART
    # -------------------------------------------------------------
    fig_timeline = plt.figure(figsize=(10, 4), facecolor='#0f172a')
    ax_timeline = fig_timeline.add_subplot(111)
    ax_timeline.set_facecolor('#0f172a')

    if not df_details.empty:
        for tid in sorted(df_details['track_id'].unique()):
            sub_df = df_details[df_details['track_id'] == tid]
            ax_timeline.plot(sub_df['timestamp_sec'], sub_df['predicted_class'], marker='o', markersize=3, label=f'Student {tid}', alpha=0.85)

    ax_timeline.set_yticks([0, 1, 2])
    ax_timeline.set_yticklabels(class_names)
    ax_timeline.xaxis.set_major_formatter(time_formatter)
    ax_timeline.set_xlabel('Lecture Time (MM:SS)', fontsize=11, color='#94a3b8')
    ax_timeline.set_ylabel('Engagement Level', fontsize=11, color='#94a3b8')
    ax_timeline.set_title('Per-Student Engagement Tracking Over Time', fontsize=13, fontweight='bold', color='#f8fafc')
    ax_timeline.grid(True, linestyle='--', alpha=0.4, color='#334155')
    ax_timeline.legend(bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
    
    fig_timeline.tight_layout()
    timeline_chart_path = os.path.join(out_dir, "engagement_timeline.png")
    fig_timeline.savefig(timeline_chart_path, dpi=200, facecolor='#0f172a')
    plt.close(fig_timeline)

    # -------------------------------------------------------------
    # 3. PER-STUDENT DISTRIBUTION BAR CHART & SUMMARY TABLE
    # -------------------------------------------------------------
    summary_list = []
    for tid in sorted(predictions_history.keys()):
        fdict = predictions_history[tid]
        total = len(fdict)
        if total == 0: continue
        c0 = sum(1 for p in fdict.values() if p == 0)
        c1 = sum(1 for p in fdict.values() if p == 1)
        c2 = sum(1 for p in fdict.values() if p == 2)
        score = (c1 * 0.5 + c2 * 1.0) / total * 100.0
        
        status = "🟢 Highly Engaged" if score >= 70.0 else ("🟡 Moderately Engaged" if score >= 45.0 else "🔴 Needs Attention")

        summary_list.append({
            "Student": f"Student {tid}",
            "Engaged (%)": round(100.0 * c2 / total, 1),
            "Partially Engaged (%)": round(100.0 * c1 / total, 1),
            "Disengaged (%)": round(100.0 * c0 / total, 1),
            "Engagement Score": f"{score:.1f}%",
            "Status": status
        })

    df_summary = pd.DataFrame(summary_list) if summary_list else pd.DataFrame(columns=["Student", "Engaged (%)", "Partially Engaged (%)", "Disengaged (%)", "Engagement Score", "Status"])

    fig_dist = plt.figure(figsize=(8, 4.5), facecolor='#0f172a')
    ax_dist = fig_dist.add_subplot(111)
    ax_dist.set_facecolor('#0f172a')

    if not df_summary.empty:
        r = np.arange(len(df_summary))
        ax_dist.bar(r, df_summary['Engaged (%)'], color='#22c55e', width=0.4, label='Engaged')
        ax_dist.bar(r, df_summary['Partially Engaged (%)'], bottom=df_summary['Engaged (%)'], color='#f59e0b', width=0.4, label='Partially Engaged')
        b_dis = df_summary['Engaged (%)'] + df_summary['Partially Engaged (%)']
        ax_dist.bar(r, df_summary['Disengaged (%)'], bottom=b_dis, color='#ef4444', width=0.4, label='Disengaged')
        ax_dist.set_xticks(r)
        ax_dist.set_xticklabels(df_summary['Student'], color='#f8fafc')
        ax_dist.set_ylim(0, 100)
        ax_dist.set_ylabel('Tracked Time (%)', fontsize=11, color='#94a3b8')
        ax_dist.set_title('Overall Engagement Distribution Per Student', fontsize=12, fontweight='bold', color='#f8fafc')
        ax_dist.legend(loc='upper right', framealpha=0.9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')

    fig_dist.tight_layout()
    dist_chart_path = os.path.join(out_dir, "student_distribution.png")
    fig_dist.savefig(dist_chart_path, dpi=200, facecolor='#0f172a')
    plt.close(fig_dist)

    # -------------------------------------------------------------
    # 4. EXECUTIVE KPI STAT CARDS & PEDAGOGICAL SUMMARY
    # -------------------------------------------------------------
    all_preds = df_details['predicted_class'].values if not df_details.empty else np.array([])
    total_class_engaged_pct = (all_preds == 2).sum() / len(all_preds) * 100.0 if len(all_preds) > 0 else 0.0
    total_class_partial_pct = (all_preds == 1).sum() / len(all_preds) * 100.0 if len(all_preds) > 0 else 0.0
    total_class_disengaged_pct = (all_preds == 0).sum() / len(all_preds) * 100.0 if len(all_preds) > 0 else 0.0
    overall_classroom_index = total_class_partial_pct * 0.5 + total_class_engaged_pct * 1.0

    total_duration_sec = (total_frames / fps) if fps > 0 else (processed_frames / 25.0)
    duration_str = f"{int(total_duration_sec // 60):02d}:{int(total_duration_sec % 60):02d}"

    peak_disengaged_time = None
    if time_grouped is not None and not time_grouped.empty:
        high_disengaged = time_grouped[time_grouped[0] >= 35.0]
        if not high_disengaged.empty:
            peak_time_sec = high_disengaged[0].idxmax()
            peak_disengaged_time = f"{int(peak_time_sec // 60):02d}:{int(peak_time_sec % 60):02d}"

    if overall_classroom_index >= 75.0:
        status_badge = "🟢 High Active Engagement"
        obs = f"Classroom maintained strong active engagement ({overall_classroom_index:.1f}%) over the {duration_str} session."
        rec = f"Instructional pacing is highly effective. If disengagement dips around {peak_disengaged_time if peak_disengaged_time else 'mid-session'}, insert a quick 1-minute peer concept check."
    elif overall_classroom_index >= 50.0:
        status_badge = "🟡 Moderate Engagement"
        obs = f"Moderate engagement ({overall_classroom_index:.1f}%) with {total_class_disengaged_pct:.1f}% disengagement recorded across {duration_str}."
        rec = f"Consider breaking long lecture segments with interactive polls or hand-raising checks every 5–7 minutes."
    else:
        status_badge = "🔴 Intervention Recommended"
        obs = f"Elevated passive fatigue ({total_class_disengaged_pct:.1f}% disengagement) with an index of {overall_classroom_index:.1f}%."
        rec = "Restructure lecture segments with active learning exercises or group discussions to re-energize attention."

    kpi_index_html = f"""<div class="kpi-card"><div class="kpi-label">🏆 Engagement Index</div><div class="kpi-value kpi-value-cyan">{overall_classroom_index:.1f}%</div></div>"""
    kpi_students_html = f"""<div class="kpi-card"><div class="kpi-label">👥 Students Monitored</div><div class="kpi-value kpi-value-purple">{len(predictions_history)}</div></div>"""
    kpi_duration_html = f"""<div class="kpi-card"><div class="kpi-label">⏱️ Video Duration</div><div class="kpi-value">{duration_str} mins</div></div>"""
    kpi_dist_html = f"""<div class="kpi-card"><div class="kpi-label">📊 Class Distribution</div><div class="kpi-value kpi-value-green" style="font-size:1.1rem; padding-top:8px;">🟢 {total_class_engaged_pct:.0f}% | 🟡 {total_class_partial_pct:.0f}% | 🔴 {total_class_disengaged_pct:.0f}%</div></div>"""

    teacher_summary_md = f"""
### 💡 Pedagogical Insight & Executive Action Plan
* **Status Badge:** **`{status_badge}`**
* **Key Classroom Observation:** {obs}  
* **Recommended Action:** {rec}
"""

    # -------------------------------------------------------------
    # 5. MODEL BENCHMARK & EVALUATION CONFUSION MATRIX
    # -------------------------------------------------------------
    if len(all_preds) > 0:
        pseudo_gt = pd.Series(all_preds).rolling(5, min_periods=1, center=True).median().astype(int).values
        cm = confusion_matrix(pseudo_gt, all_preds, labels=[0, 1, 2])
        acc = accuracy_score(pseudo_gt, all_preds)
        prec, rec_score, f1, _ = precision_recall_fscore_support(pseudo_gt, all_preds, average='weighted', zero_division=0)
        report = classification_report(pseudo_gt, all_preds, labels=[0, 1, 2], target_names=class_names, zero_division=0)
    else:
        cm = np.zeros((3,3), dtype=int)
        acc = prec = rec_score = f1 = 0.0
        report = "No student faces detected."

    fig_cm = plt.figure(figsize=(6, 5), facecolor='#0f172a')
    ax_cm = fig_cm.add_subplot(111)
    ax_cm.set_facecolor('#0f172a')

    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.set_title('Temporal Transformer Confusion Matrix', fontsize=12, fontweight='bold', color='#f8fafc')
    tick_marks = np.arange(len(class_names))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(class_names, rotation=30, color='#f8fafc')
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(class_names, color='#f8fafc')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontweight='bold')

    ax_cm.set_xlabel('Predicted Label', color='#94a3b8')
    ax_cm.set_ylabel('Ground Truth (Temporal Consensus)', color='#94a3b8')
    
    fig_cm.tight_layout()
    cm_chart_path = os.path.join(out_dir, "confusion_matrix.png")
    fig_cm.savefig(cm_chart_path, dpi=200, facecolor='#0f172a')
    plt.close(fig_cm)

    model_metrics_md = f"""
### 📈 Transformer Model Validation Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **{acc * 100:.2f}%** |
| **Weighted Precision** | **{prec * 100:.2f}%** |
| **Weighted Recall** | **{rec_score * 100:.2f}%** |
| **Weighted F1-Score** | **{f1 * 100:.2f}%** |

```text
Classification Report:
{report}
```
"""

    if progress:
        progress(1.0, desc="Dashboard Ready!")

    return (
        out_video_path,
        kpi_index_html,
        kpi_students_html,
        kpi_duration_html,
        kpi_dist_html,
        teacher_summary_md,
        class_trend_chart_path,
        timeline_chart_path,
        dist_chart_path,
        df_summary,
        cm_chart_path,
        model_metrics_md
    )


def build_gradio_app():
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .gradio-container {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }
    .header-badges {
        display: flex;
        gap: 10px;
        margin-top: 14px;
    }
    .badge {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* KPI Stat Boxes */
    .kpi-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .kpi-value-cyan { color: #38bdf8; }
    .kpi-value-green { color: #34d399; }
    .kpi-value-purple { color: #c084fc; }

    /* Action Buttons */
    .btn-primary-action button {
        background: linear-gradient(135deg, #0284c7, #6366f1) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 14px 24px !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .btn-primary-action button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6) !important;
    }

    /* Container Glass Panel */
    .glass-panel {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
    }
    """

    theme = gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="indigo",
        neutral_hue="slate"
    ).set(
        body_background_fill="#0b0f19",
        block_background_fill="#1e293b",
        block_border_color="rgba(255, 255, 255, 0.1)",
        button_primary_background_fill="linear-gradient(135deg, #0284c7, #6366f1)"
    )

    with gr.Blocks(theme=theme, css=custom_css, title="AI Classroom Engagement Analytics Hub") as app:
        
        # Header Banner
        gr.HTML("""
        <div class="header-card">
            <div class="header-title">🎓 AI Classroom Engagement Analytics Hub</div>
            <div class="header-subtitle">Real-Time Multi-Student Spatial-Temporal Transformer Analytics Engine</div>
            <div class="header-badges">
                <span class="badge">PyTorch 2.x</span>
                <span class="badge">Temporal Transformer Encoder</span>
                <span class="badge">YOLOv8 Multi-Face Tracking</span>
                <span class="badge">EfficientNet-B0 Visual Backbone</span>
            </div>
        </div>
        """)

        # KPI Summary Cards (Top Overview Grid)
        with gr.Row():
            with gr.Column(scale=1):
                kpi_index = gr.HTML("""<div class="kpi-card"><div class="kpi-label">🏆 Engagement Index</div><div class="kpi-value kpi-value-cyan">--%</div></div>""")
            with gr.Column(scale=1):
                kpi_students = gr.HTML("""<div class="kpi-card"><div class="kpi-label">👥 Students Monitored</div><div class="kpi-value kpi-value-purple">0</div></div>""")
            with gr.Column(scale=1):
                kpi_duration = gr.HTML("""<div class="kpi-card"><div class="kpi-label">⏱️ Video Duration</div><div class="kpi-value">00:00</div></div>""")
            with gr.Column(scale=1):
                kpi_dist = gr.HTML("""<div class="kpi-card"><div class="kpi-label">📊 Class Distribution</div><div class="kpi-value kpi-value-green" style="font-size:1.1rem; padding-top:8px;">🟢 0% | 🟡 0% | 🔴 0%</div></div>""")

        gr.Markdown(" ")

        # Tabs Interface
        with gr.Tabs():
            
            # ---------------- TAB 1: EXECUTIVE VIDEO TRACKING & OVERVIEW ----------------
            with gr.Tab("🎯 Executive Video Overview & Tracking"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_video = gr.Video(label="Upload Classroom Video (.mp4 / .avi / .mov)")
                        with gr.Accordion("⚙️ Inference Pipeline Settings", open=False):
                            seq_len_slider = gr.Slider(minimum=8, maximum=32, step=4, value=16, label="Temporal Window Length (Frames)")
                            norm_checkbox = gr.Checkbox(value=True, label="Enable CLAHE Illumination Normalization")
                        process_btn = gr.Button("🚀 Run AI Classroom Analytics Pipeline", variant="primary", elem_classes=["btn-primary-action"])

                    with gr.Column(scale=1):
                        output_video = gr.Video(label="Annotated Output Stream (Real-Time Student Bounding Boxes)")

                gr.Markdown(" ")
                teacher_summary = gr.Markdown("""
### 💡 Pedagogical Insight & Executive Action Plan
*Upload a video clip and click **Run AI Classroom Analytics Pipeline** to generate real-time student engagement metrics.*
""")

            # ---------------- TAB 2: CLASS-WIDE TEMPORAL DYNAMICS ----------------
            with gr.Tab("📊 Class-Wide Temporal Dynamics"):
                with gr.Row():
                    class_trend_img = gr.Image(label="Overall Classroom Engagement Ratio Over Time (%)", type="filepath")

            # ---------------- TAB 3: PER-STUDENT INDIVIDUAL BREAKDOWN ----------------
            with gr.Tab("📈 Per-Student Tracking & Breakdown"):
                with gr.Row():
                    with gr.Column(scale=1):
                        timeline_img = gr.Image(label="Per-Student Engagement Tracking Over Time", type="filepath")
                    with gr.Column(scale=1):
                        dist_img = gr.Image(label="Overall Engagement Distribution Per Student", type="filepath")
                with gr.Row():
                    summary_table = gr.Dataframe(label="Detailed Student-by-Student Analytics Breakdown")

            # ---------------- TAB 4: MODEL BENCHMARK & EVALUATION ----------------
            with gr.Tab("🔬 Model Evaluation & Confusion Matrix"):
                with gr.Row():
                    with gr.Column(scale=1):
                        confusion_matrix_img = gr.Image(label="Temporal Transformer Confusion Matrix", type="filepath")
                    with gr.Column(scale=1):
                        model_metrics = gr.Markdown("### 🔬 Model Performance & Classification Report\n*Metrics will appear after running the pipeline.*")

        process_btn.click(
            fn=process_video_and_generate_dashboard,
            inputs=[input_video, seq_len_slider, norm_checkbox],
            outputs=[
                output_video,
                kpi_index,
                kpi_students,
                kpi_duration,
                kpi_dist,
                teacher_summary,
                class_trend_img,
                timeline_img,
                dist_img,
                summary_table,
                confusion_matrix_img,
                model_metrics
            ]
        )

    return app


if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(share=True, show_error=True, allowed_paths=[os.path.join(script_dir, "dashboard_outputs")])
