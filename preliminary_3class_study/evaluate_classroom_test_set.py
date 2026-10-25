import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import model architectures
# New models
from models.base_fusion_model import FusionModel as BaseFusionModel
from models.lstm_fusion_model import FusionModel as LSTMFusionModel
from models.transformer_fusion_model import FusionModel as TransformerFusionModel
from models.lstm_transformer_fusion_model import FusionModel as LSTMTransformerFusionModel

# Optional illumination normalization
try:
    from yolo_face_detector.illumination_normalization import normalize_illumination
except ImportError:
    try:
        from illumination_normalization import normalize_illumination
    except ImportError:
        normalize_illumination = None

# Old models (for pretrained ResNet-18 models)
from old.models.fusion_model import FusionModel as ResNetBaseFusionModel
from old.models.fusion_model_T import FusionModel as ResNetLSTMFusionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate student engagement models on an annotated classroom test dataset.")
    parser.add_argument("--dataset_dir", type=str, default="classroom_test_dataset", help="Path to the classroom test dataset directory.")
    parser.add_argument("--model", type=str, required=True, help="Path to the PyTorch trained weights (.pth file).")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "lstm", "transformer", "lstm_transformer", "base", "resnet_lstm", "resnet_base"],
                        help="Model architecture type. 'auto' will auto-detect based on state_dict keys.")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for temporal models (default 16).")
    parser.add_argument("--normalize", action="store_true", help="Apply CLAHE and sharpening illumination normalization to face crops.")
    return parser.parse_args()

def parse_test_filename(filename):
    """
    Parses the video name, face_id, and original frame number from filenames like:
    'Classroom_Recording_6_frame_det_00_000001.bmp'
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    try:
        # Find index of "frame"
        f_idx = parts.index("frame")
        video_name = "_".join(parts[:f_idx])
        face_id = int(parts[f_idx+2])
        frame_num = int(parts[f_idx+3])
        return video_name, face_id, frame_num
    except (ValueError, IndexError):
        return None, None, None

def detect_model_type(state_dict):
    keys = list(state_dict.keys())
    is_resnet = any("cnn.conv1" in k for k in keys) or any("cnn.bn1" in k for k in keys)
    has_lstm = any("lstm" in k for k in keys)
    has_transformer = any("transformer" in k or "pos_encoder" in k for k in keys)
    has_proj = any("proj" in k for k in keys)

    if is_resnet:
        if has_lstm:
            return "resnet_lstm"
        else:
            return "resnet_base"
    else:
        if has_lstm and has_transformer:
            return "lstm_transformer"
        elif has_transformer:
            return "transformer"
        elif has_lstm:
            return "lstm"
        else:
            return "base"

def load_model(model_path, model_type, openface_dim, device):
    if not os.path.exists(model_path):
        print(f"!!! Error: Model file not found at {model_path}")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=device)
    
    if model_type == "auto":
        model_type = detect_model_type(state_dict)
        print(f">>> Auto-detected model architecture: {model_type.upper()}")

    if model_type == "resnet_lstm":
        model = ResNetLSTMFusionModel(openface_dim=openface_dim)
    elif model_type == "resnet_base":
        model = ResNetBaseFusionModel(openface_dim=openface_dim)
    elif model_type == "lstm":
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
    print(">>> Model loaded successfully!")
    return model, model_type

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Evaluating on device: {device}")

    # 1. Paths verification
    dataset_dir = args.dataset_dir
    labels_csv = os.path.join(dataset_dir, "labels.csv")
    frame_map_csv = os.path.join(dataset_dir, "frame_to_filename.csv")
    openface_csv = os.path.join(dataset_dir, "openface", "images_flat.csv")
    images_flat_dir = os.path.join(dataset_dir, "images_flat")

    if not all(os.path.exists(p) for p in [labels_csv, frame_map_csv, openface_csv, images_flat_dir]):
        print(f"!!! Error: Could not locate all dataset files in {dataset_dir}")
        print("    Please run annotate_crops.py with '--dataset_dir' set to this directory first.")
        sys.exit(1)

    # Load label mappings
    labels_df = pd.read_csv(labels_csv)
    labels_df.columns = labels_df.columns.str.strip()
    filename_to_label = dict(zip(labels_df["filename"], labels_df["label"]))

    # Load frame to filename mappings
    frame_map_df = pd.read_csv(frame_map_csv)
    frame_map_df.columns = frame_map_df.columns.str.strip()
    frame_to_filename = dict(zip(frame_map_df["frame"], frame_map_df["filename"]))

    # Load OpenFace Features CSV
    features_df = pd.read_csv(openface_csv)
    features_df.columns = features_df.columns.str.strip()
    
    metadata_cols = ["filename", "label", "frame", "face_id", "timestamp", "confidence", "success"]
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]
    openface_dim = len(feature_cols)
    print(f">>> OpenFace Feature Dimension: {openface_dim}")

    # 2. Load Model
    model, model_type = load_model(args.model, args.model_type, openface_dim, device)
    is_sequence = "base" not in model_type
    seq_len = args.seq_len

    # 3. Associate global frame index in CSV to original videos, students, and sub-frames
    # We will build a structured dataframe containing: [global_frame, filename, video_name, face_id, frame_num, label]
    records = []
    for _, row in features_df.iterrows():
        global_frame = int(row["frame"])
        filename = frame_to_filename.get(global_frame, None)
        if filename is None or filename not in filename_to_label:
            continue
        
        label = filename_to_label[filename]
        video_name, face_id, frame_num = parse_test_filename(filename)
        
        if video_name is None:
            continue
            
        records.append({
            "global_frame": global_frame,
            "filename": filename,
            "video_name": video_name,
            "face_id": face_id,
            "frame_num": frame_num,
            "label": label,
            "feature_row_index": _
        })

    structured_df = pd.DataFrame(records)
    if len(structured_df) == 0:
        print("!!! Error: No valid annotated sequences could be constructed.")
        sys.exit(1)

    print(f">>> Loaded {len(structured_df)} annotated face crops.")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    y_true = []
    y_pred = []
    results_details = []

    # Group by video and student to construct chronological sequence histories
    grouped = structured_df.groupby(["video_name", "face_id"])
    print(f">>> Running validation on {len(grouped)} student sequences...")

    with torch.no_grad():
        for (video, face_id), group in grouped:
            # Sort chronologically by original frame number
            sorted_group = group.sort_values("frame_num").reset_index(drop=True)
            total_student_frames = len(sorted_group)

            for idx in range(total_student_frames):
                row = sorted_group.iloc[idx]
                global_frame = row["global_frame"]
                filename = row["filename"]
                ground_truth = row["label"]

                # Extract OpenFace features
                csv_row = features_df.iloc[row["feature_row_index"]]
                openface_feat = torch.from_numpy(csv_row[feature_cols].values.astype(np.float32)).float().unsqueeze(0).to(device)

                if is_sequence:
                    # Construct sequence of 16 aligned frames
                    seq_images = []
                    
                    for step in range(seq_len):
                        # Go back in time
                        history_idx = idx - (seq_len - 1) + step
                        if history_idx < 0:
                            history_idx = 0  # Pad by repeating the first tracked frame
                        
                        hist_row = sorted_group.iloc[history_idx]
                        hist_filename = hist_row["filename"]
                        
                        img_path = os.path.join(images_flat_dir, hist_filename)
                        img = cv2.imread(img_path)
                        if img is None:
                            img = np.zeros((224, 224, 3), dtype=np.uint8)
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            if args.normalize and normalize_illumination is not None:
                                img = normalize_illumination(img)
                            
                        seq_images.append(transform(img))
                        
                    image_seq = torch.stack(seq_images).unsqueeze(0).to(device)
                    outputs = model(image_seq, openface_feat)
                else:
                    # Single frame
                    img_path = os.path.join(images_flat_dir, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        img = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if args.normalize and normalize_illumination is not None:
                            img = normalize_illumination(img)
                        
                    image_tensor = transform(img).unsqueeze(0).to(device)
                    outputs = model(image_tensor, openface_feat)

                pred_class = int(outputs.argmax(dim=1).item())
                y_true.append(ground_truth)
                y_pred.append(pred_class)

                results_details.append({
                    "filename": filename,
                    "video_name": video,
                    "face_id": int(face_id),
                    "frame_num": int(row["frame_num"]),
                    "ground_truth": ground_truth,
                    "prediction": pred_class
                })

    # 4. Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    target_names = ["Disengaged", "Partially Engaged", "Engaged"]
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "="*60)
    print(f"      MODEL EVALUATION REPORT - CLASSROOM TEST DATASET")
    print("="*60)
    print(f"Overall Test Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("="*60)

    # Save report to text file
    report_out_path = os.path.join(dataset_dir, f"evaluation_report_{model_type}.txt")
    with open(report_out_path, "w") as f:
        f.write(f"Model Evaluation Report - Classroom Test Dataset\n")
        f.write(f"Model File: {args.model}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write("="*50 + "\n")
        f.write(f"Overall Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")

    print(f">>> Text report saved to {report_out_path}")

    # Plot and save Confusion Matrix chart
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names, yticklabels=target_names,
           title=f'Confusion Matrix - {model_type.upper()}',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    cm_path = os.path.join(dataset_dir, f"confusion_matrix_{model_type}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f">>> Confusion matrix saved to {cm_path}")

    # 5. Save detailed predictions to CSV
    detailed_df = pd.DataFrame(results_details)
    detailed_csv_path = os.path.join(dataset_dir, f"detailed_predictions_{model_type}.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f">>> Detailed predictions CSV saved to {detailed_csv_path}")

    # 6. Generate visually stunning HTML dashboard
    html_out_path = os.path.join(dataset_dir, f"evaluation_dashboard_{model_type}.html")
    correct_count = sum(1 for r in results_details if r["ground_truth"] == r["prediction"])
    incorrect_count = len(results_details) - correct_count

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Dashboard - {model_type.upper()}</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #0f172a;
            color: #f1f5f9;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .meta-info {{
            color: #94a3b8;
            margin-bottom: 24px;
            font-size: 0.95rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: #1e293b;
            border: 1px solid #334155;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .stat-val {{
            font-size: 2rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 4px;
        }}
        .stat-val.accuracy {{
            color: #34d399;
        }}
        .stat-val.incorrect {{
            color: #f87171;
        }}
        .filters {{
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-btn {{
            background-color: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .filter-btn:hover {{
            background-color: #334155;
        }}
        .filter-btn.active {{
            background-color: #38bdf8;
            color: #0f172a;
            border-color: #38bdf8;
        }}
        .search-bar {{
            background-color: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
            padding: 8px 16px;
            border-radius: 8px;
            min-width: 250px;
            margin-left: auto;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
        }}
        .card {{
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-4px);
        }}
        .card.correct-card {{
            border-color: rgba(52, 211, 153, 0.3);
        }}
        .card.correct-card:hover {{
            border-color: #34d399;
        }}
        .card.incorrect-card {{
            border-color: rgba(248, 113, 113, 0.3);
        }}
        .card.incorrect-card:hover {{
            border-color: #f87171;
        }}
        .card-img-wrapper {{
            position: relative;
            background-color: #0f172a;
            height: 180px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .card-img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .badge {{
            position: absolute;
            top: 12px;
            right: 12px;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge.correct-badge {{
            background-color: rgba(52, 211, 153, 0.2);
            color: #34d399;
            border: 1px solid #34d399;
        }}
        .badge.incorrect-badge {{
            background-color: rgba(248, 113, 113, 0.2);
            color: #f87171;
            border: 1px solid #f87171;
        }}
        .card-content {{
            padding: 16px;
        }}
        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            margin: 0 0 12px 0;
            color: #cbd5e1;
        }}
        .detail-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }}
        .detail-label {{
            color: #94a3b8;
        }}
        .detail-val {{
            font-weight: 500;
        }}
        .val-disengaged {{ color: #ef4444; }}
        .val-partial {{ color: #f59e0b; }}
        .val-engaged {{ color: #10b981; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Dashboard</h1>
        <div class="meta-info">
            Model Type: <strong>{model_type.upper()}</strong> | Weights: <strong>{os.path.basename(args.model)}</strong>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="detail-label">Overall Accuracy</div>
                <div class="stat-val accuracy">{accuracy*100:.2f}%</div>
            </div>
            <div class="stat-card">
                <div class="detail-label">Total Face Crops</div>
                <div class="stat-val">{len(results_details)}</div>
            </div>
            <div class="stat-card">
                <div class="detail-label">Correct Predictions</div>
                <div class="stat-val" style="color: #34d399;">{correct_count}</div>
            </div>
            <div class="stat-card">
                <div class="detail-label">Incorrect Predictions</div>
                <div class="stat-val incorrect">{incorrect_count}</div>
            </div>
        </div>

        <div class="filters">
            <button class="filter-btn active" onclick="filterCards('all', this)">Show All</button>
            <button class="filter-btn" onclick="filterCards('incorrect', this)">Show Misclassifications ({incorrect_count})</button>
            <button class="filter-btn" onclick="filterCards('correct', this)">Show Correct ({correct_count})</button>
            <input type="text" class="search-bar" id="search" placeholder="Search by Student ID or Video..." oninput="handleSearch()">
        </div>

        <div class="card-grid" id="cardGrid">
"""

    for r in results_details:
        status_class = "correct-card" if r["ground_truth"] == r["prediction"] else "incorrect-card"
        badge_text = "CORRECT" if r["ground_truth"] == r["prediction"] else "MISCLASSIFIED"
        badge_class = "correct-badge" if r["ground_truth"] == r["prediction"] else "incorrect-badge"

        gt_label = target_names[r["ground_truth"]]
        pred_label = target_names[r["prediction"]]

        gt_color_class = f"val-{gt_label.lower().split()[0]}"
        pred_color_class = f"val-{pred_label.lower().split()[0]}"

        img_src = f"images_flat/{r['filename']}"

        html_content += f"""
            <div class="card {status_class}" data-status="{"correct" if r["ground_truth"] == r["prediction"] else "incorrect"}" data-student="{r["face_id"]}" data-video="{r["video_name"]}">
                <div class="card-img-wrapper">
                    <img class="card-img" src="{img_src}" alt="{r["filename"]}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'200\\' height=\\'200\\' viewBox=\\'0 0 100 100\\'><text y=\\'50%\\' x=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'%2364748b\\'>Image Missing</text></svg>'">
                    <span class="badge {badge_class}">{badge_text}</span>
                </div>
                <div class="card-content">
                    <h3 class="card-title">Student {r["face_id"]}</h3>
                    <div class="detail-row">
                        <span class="detail-label">Video</span>
                        <span class="detail-val">{r["video_name"]}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Frame Num</span>
                        <span class="detail-val">#{r["frame_num"]}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">True Engagement</span>
                        <span class="detail-val {gt_color_class}">{gt_label}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Model Predicted</span>
                        <span class="detail-val {pred_color_class}">{pred_label}</span>
                    </div>
                </div>
            </div>"""

    html_content += """
        </div>
    </div>

    <script>
        let currentFilter = 'all';
        let searchQuery = '';

        function filterCards(filterType, btnEl) {
            currentFilter = filterType;

            // Update active button styling
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            if (btnEl) btnEl.classList.add('active');

            applyFilters();
        }

        function handleSearch() {
            searchQuery = document.getElementById('search').value.toLowerCase().trim();
            applyFilters();
        }

        function applyFilters() {
            const cards = document.querySelectorAll('.card');
            cards.forEach(card => {
                const status = card.getAttribute('data-status');
                const student = card.getAttribute('data-student').toLowerCase();
                const video = card.getAttribute('data-video').toLowerCase();

                const matchesFilter = (currentFilter === 'all') || (status === currentFilter);
                const matchesSearch = searchQuery === '' ||
                                      student.includes(searchQuery) ||
                                      video.includes(searchQuery) ||
                                      ("student " + student).includes(searchQuery);

                if (matchesFilter && matchesSearch) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f">>> HTML evaluation dashboard generated at {html_out_path}")

if __name__ == "__main__":
    main()
