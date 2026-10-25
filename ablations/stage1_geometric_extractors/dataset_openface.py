import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class OpenFaceBinaryDataset(Dataset):
    """
    Dataset to load sequence or pooled features and extract ONLY OpenFace features
    for binary engagement classification (0: Engaged, 1: Disengaged).
    """
    def __init__(self, file_paths, max_seq_len=120, openface_slice_mode="auto"):
        """
        Args:
            file_paths: List of file paths (.pt or .npy)
            max_seq_len: Max frames for temporal sequences
            openface_slice_mode: 'auto', 'all', 'first_128', or 'last_128'
        """
        self.samples = []
        self.max_seq_len = max_seq_len
        self.openface_slice_mode = openface_slice_mode
        self._load_and_filter_files(file_paths)

    def _extract_feature_and_label(self, path):
        """Extracts the feature tensor and binary label from a single file."""
        if path.endswith(".pt"):
            data = torch.load(path, map_location="cpu")
        elif path.endswith(".npy"):
            data = np.load(path, allow_pickle=True)
            if data.dtype == object and data.size == 1:
                data = data.item()
        else:
            return None, None

        # 1. Extract feature tensor
        raw_feat = None
        if isinstance(data, dict):
            for key in ["sequence", "feature", "features", "fused_seq", "data"]:
                if key in data:
                    raw_feat = data[key]
                    break
        elif isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            raw_feat = data

        if raw_feat is None:
            return None, None

        if isinstance(raw_feat, np.ndarray):
            raw_feat = torch.from_numpy(raw_feat).float()
        elif isinstance(raw_feat, torch.Tensor):
            raw_feat = raw_feat.float()

        # Ensure feature is 2D: [Frames, Dims]
        if raw_feat.dim() == 1:
            raw_feat = raw_feat.unsqueeze(0)  # Shape: [1, Dims]

        # 2. Extract ONLY OpenFace features (Ablation: drop EfficientNet)
        n_frames, feat_dim = raw_feat.shape

        if feat_dim > 500:
            # Fused vector detected (e.g. 1408 = 1280 EfficientNet + 128 OpenFace)
            # Baseline placed CNN first (0:1280), OpenFace second (1280:)
            # Lightweight placed OpenFace first (0:128), CNN second (128:)
            if self.openface_slice_mode == "first_128" or (feat_dim == 1408 and self.openface_slice_mode == "auto"):
                # Check variance to identify OpenFace vs CNN
                var_first = raw_feat[:, :128].var().item()
                var_last = raw_feat[:, -128:].var().item()
                # CNN embeddings typically have smaller normalized variance or standard distribution
                # Extract the 128-dim OpenFace portion
                openface_feat = raw_feat[:, -128:] if (feat_dim == 1408 and raw_feat.shape[1] > 1280) else raw_feat[:, :128]
            elif self.openface_slice_mode == "last_128":
                openface_feat = raw_feat[:, -128:]
            else:
                openface_feat = raw_feat[:, :128]
        else:
            # Feature dimension <= 500: already pure OpenFace features!
            openface_feat = raw_feat

        # 3. Extract and map label to binary (0: Engaged, 1: Disengaged)
        raw_label = None
        if isinstance(data, dict) and "label" in data:
            raw_label = data["label"]
        else:
            # Fallback to filename parsing
            fn = os.path.basename(path).lower()
            if "disengaged" in fn or "disengage" in fn:
                raw_label = 1
            elif "engaged" in fn or "engage" in fn:
                raw_label = 0

        if raw_label is None:
            return None, None

        # Binary normalization
        if isinstance(raw_label, str):
            lbl_lower = raw_label.lower().strip()
            if "disengaged" in lbl_lower or "disengage" in lbl_lower:
                binary_label = 1
            elif "engaged" in lbl_lower:
                binary_label = 0
            else:
                return None, None  # Skip partial or ambiguous classes if any
        else:
            int_lbl = int(raw_label)
            if int_lbl == 0:
                binary_label = 0      # Engaged
            elif int_lbl in [1, 2]:
                binary_label = 1      # Disengaged (or 2 mapped to 1)
            else:
                return None, None

        return openface_feat, binary_label

    def _load_and_filter_files(self, file_paths):
        for path in file_paths:
            try:
                feat, lbl = self._extract_feature_and_label(path)
                if feat is not None and lbl is not None:
                    self.samples.append({
                        "path": path,
                        "feature": feat,
                        "label": lbl
                    })
            except Exception as e:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feat = sample["feature"]
        label = sample["label"]

        # Truncate sequence to max_seq_len if too long
        n_frames = feat.shape[0]
        if n_frames > self.max_seq_len:
            feat = feat[:self.max_seq_len]

        return feat, label

def collate_temporal_fn(batch):
    """
    Collate function to batch variable-length sequences.
    Pads sequences and returns sequence lengths.
    """
    sequences, labels = zip(*batch)
    batch_size = len(sequences)
    lengths = [s.shape[0] for s in sequences]
    max_len = max(lengths)
    feat_dim = sequences[0].shape[1]

    padded_seqs = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    for i, seq in enumerate(sequences):
        padded_seqs[i, :seq.shape[0], :] = seq

    return padded_seqs, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def get_ablation_dataloaders(dataset_dir, batch_size=16, random_state=42):
    """
    Scans dataset_dir, loads samples, performs stratified 70/15/15 split,
    and returns (train_loader, val_loader, test_loader, feature_dim).
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Feature dataset directory not found: {dataset_dir}")

    all_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".pt") or f.endswith(".npy")
    ]

    if len(all_files) == 0:
        raise ValueError(f"No .pt or .npy files found in {dataset_dir}")

    # Temporary probe dataset to get labels and valid files
    probe_dataset = OpenFaceBinaryDataset(all_files)
    if len(probe_dataset) == 0:
        raise ValueError(f"No valid samples could be extracted from {dataset_dir}")

    valid_paths = [s["path"] for s in probe_dataset.samples]
    labels = [s["label"] for s in probe_dataset.samples]

    n_engaged = labels.count(0)
    n_disengaged = labels.count(1)
    sample_feat_dim = probe_dataset.samples[0]["feature"].shape[1]

    print("==================================================")
    print("           DATASET INVENTORY (ABLATION)           ")
    print("==================================================")
    print(f"Total Valid Samples:    {len(valid_paths)}")
    print(f"  • Engaged (Class 0):    {n_engaged} samples")
    print(f"  • Disengaged (Class 1): {n_disengaged} samples")
    print(f"OpenFace Feature Dim:   {sample_feat_dim}")
    print("==================================================")

    # Stratified Train (70%) and Temp (30%)
    try:
        train_p, temp_p, train_l, temp_l = train_test_split(
            valid_paths, labels, test_size=0.30, stratify=labels, random_state=random_state
        )
        val_p, test_p, val_l, test_l = train_test_split(
            temp_p, temp_l, test_size=0.50, stratify=temp_l, random_state=random_state
        )
    except Exception:
        # Fallback to random split if a class has too few instances
        train_p, temp_p = train_test_split(valid_paths, test_size=0.30, random_state=random_state)
        val_p, test_p = train_test_split(temp_p, test_size=0.50, random_state=random_state)

    train_set = OpenFaceBinaryDataset(train_p)
    val_set = OpenFaceBinaryDataset(val_p)
    test_set = OpenFaceBinaryDataset(test_p)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_temporal_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_fn)

    print(f"Split distribution: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}\n")

    return train_loader, val_loader, test_loader, sample_feat_dim
