import os
import sys
import glob
import torch
import pandas as pd

# Add parent directory to path to import config_lightweight
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config_lightweight

def load_openface_csv(csv_path):
    """
    Reads OpenFace 2.0 output CSV and extracts action units (AUs), 
    gaze directions, and head pose features per frame.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    feature_cols = []
    # Action Units (intensity _r and presence _c)
    feature_cols += [c for c in df.columns if c.startswith('AU') and (c.endswith('_r') or c.endswith('_c'))]
    # Gaze features
    feature_cols += [c for c in df.columns if c.startswith('gaze_')]
    # Pose features (pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz)
    feature_cols += [c for c in df.columns if c.startswith('pose_')]

    if not feature_cols:
        # Fallback to all numeric columns except frame/timestamp/confidence
        skip = {'frame', 'face_id', 'timestamp', 'confidence', 'success'}
        feature_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    feats = df[feature_cols].fillna(0.0).values
    return torch.tensor(feats, dtype=torch.float32)

def fuse_features_for_sample(openface_csv_path, efficientnet_pt_path):
    """
    Fuses OpenFace tabular frame features with EfficientNet spatial embeddings.
    Returns: Tensor of shape [N_frames, feature_dim]
    """
    of_tensor = load_openface_csv(openface_csv_path)
    
    eff_data = torch.load(efficientnet_pt_path, map_location="cpu")
    if isinstance(eff_data, dict) and "features" in eff_data:
        eff_tensor = eff_data["features"]
    elif isinstance(eff_data, torch.Tensor):
        eff_tensor = eff_data
    else:
        raise ValueError(f"Unknown format in EfficientNet file {efficientnet_pt_path}")

    if eff_tensor.dim() == 1:
        eff_tensor = eff_tensor.unsqueeze(0)

    n_of = of_tensor.shape[0]
    n_eff = eff_tensor.shape[0]

    # Align sequence length across modalities
    min_len = min(n_of, n_eff)
    if min_len == 0:
        raise ValueError(f"Zero frames found for sample: {openface_csv_path}")

    of_trimmed = of_tensor[:min_len]
    eff_trimmed = eff_tensor[:min_len]

    # Concatenate frame-level features
    fused_seq = torch.cat([of_trimmed, eff_trimmed], dim=1)
    return fused_seq

def get_class_label(file_name):
    """
    Extracts numerical class label (0: engaged, 1: partially engaged, 2: disengaged)
    from filename.
    """
    fn_lower = file_name.lower()
    if "partially" in fn_lower or "partial" in fn_lower:
        return 1
    elif "disengaged" in fn_lower or "disengage" in fn_lower:
        return 2
    elif "engaged" in fn_lower or "engage" in fn_lower:
        return 0
    else:
        return 0

def main():
    print("==================================================")
    print("  PREPROCESSING: FUSING SEQUENCE FEATURES FOR     ")
    print("           LIGHTWEIGHT TEMPORAL MODELS            ")
    print("==================================================")

    of_dir = config_lightweight.OPENFACE_FEATURES_DIR
    eff_dir = config_lightweight.EFFICIENTNET_FEATURES_DIR
    out_dir = config_lightweight.SEQUENCE_FUSED_DIR

    os.makedirs(out_dir, exist_ok=True)

    print(f"OpenFace Features Dir:      {of_dir}")
    print(f"EfficientNet Features Dir:  {eff_dir}")
    print(f"Output Fused Sequence Dir:  {out_dir}")

    if not os.path.exists(of_dir):
        print(f"[Error] OpenFace feature directory not found: {of_dir}")
        sys.exit(1)

    if not os.path.exists(eff_dir):
        print(f"[Error] EfficientNet feature directory not found: {eff_dir}")
        sys.exit(1)

    of_files = sorted([f for f in os.listdir(of_dir) if f.endswith(".csv")])
    if not of_files:
        print(f"[Error] No OpenFace CSV files found in {of_dir}")
        sys.exit(1)

    print(f"\nFound {len(of_files)} OpenFace CSV feature files. Starting fusion...")

    success_count = 0
    skipped_count = 0

    for of_file in of_files:
        sample_name = os.path.splitext(of_file)[0]
        of_path = os.path.join(of_dir, of_file)
        
        # Look for corresponding EfficientNet feature file (.pt)
        eff_path = os.path.join(eff_dir, f"{sample_name}.pt")
        if not os.path.exists(eff_path):
            # Try alternate extension matching
            matching_eff = glob.glob(os.path.join(eff_dir, f"*{sample_name}*.pt"))
            if matching_eff:
                eff_path = matching_eff[0]
            else:
                print(f"[Warning] Missing EfficientNet pair for '{sample_name}'. Skipping.")
                skipped_count += 1
                continue

        try:
            fused_seq = fuse_features_for_sample(of_path, eff_path)
            label = get_class_label(sample_name)

            out_path = os.path.join(out_dir, f"{sample_name}_fused.pt")
            torch.save({
                "sample_id": sample_name,
                "sequence": fused_seq,  # [N_frames, feature_dim]
                "label": label,
                "num_frames": fused_seq.shape[0],
                "feature_dim": fused_seq.shape[1]
            }, out_path)

            success_count += 1
        except Exception as e:
            print(f"[Warning] Failed to fuse '{sample_name}': {e}")
            skipped_count += 1

    print("\n==================================================")
    print(f"Sequence Feature Fusion Complete!")
    print(f"Successfully processed: {success_count} samples")
    print(f"Skipped / Failed:      {skipped_count} samples")
    print(f"Saved to: {out_dir}")
    print("==================================================")

if __name__ == "__main__":
    main()
