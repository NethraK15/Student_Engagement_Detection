import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config_gru

class SequenceDataset(Dataset):
    """
    Dataset to load pre-extracted 2D sequence feature tensors.
    Applies global dataset-level feature scaling (preserving absolute expression levels)
    and pads/truncates sequences to max_seq_len.
    """
    def __init__(self, file_paths, max_seq_len=config_gru.MAX_SEQ_LEN, global_mean=None, global_std=None):
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu")
        sequence = data["sequence"]  # Shape: [N_frames, feature_dim]
        label = int(data["label"])   # Class label: 0, 1, 2
        
        n_frames, feat_dim = sequence.shape

        # Global feature scaling (if provided) to keep absolute expression levels intact
        if self.global_mean is not None and self.global_std is not None:
            normalized_sequence = (sequence - self.global_mean) / self.global_std
        else:
            normalized_sequence = sequence

        # Truncate or Pad to max_seq_len
        if n_frames >= self.max_seq_len:
            padded_sequence = normalized_sequence[:self.max_seq_len]
            actual_len = self.max_seq_len
        else:
            actual_len = n_frames
            padding = torch.zeros((self.max_seq_len - n_frames, feat_dim), dtype=torch.float32)
            padded_sequence = torch.cat([normalized_sequence, padding], dim=0)

        return padded_sequence, actual_len, label

def compute_global_stats(file_paths):
    """
    Computes global feature mean and std across all frames of the training split
    to preserve absolute facial Action Unit intensity baselines.
    """
    all_frames = []
    for f in file_paths:
        try:
            data = torch.load(f, map_location="cpu")
            all_frames.append(data["sequence"])
        except Exception:
            continue
    if not all_frames:
        return None, None
    concat_frames = torch.cat(all_frames, dim=0)
    global_mean = concat_frames.mean(dim=0, keepdim=True)
    global_std = concat_frames.std(dim=0, keepdim=True) + 1e-6
    return global_mean, global_std

def get_sequence_dataloaders(sequence_fused_dir, batch_size=config_gru.BATCH_SIZE, random_state=config_gru.RANDOM_STATE):
    """
    Loads sequence feature files, performs stratified Train (70%), Val (15%), 
    and Test (15%) split, computes global feature stats on train set, and returns DataLoaders.
    """
    if not os.path.exists(sequence_fused_dir):
        raise FileNotFoundError(f"Sequence feature folder not found: {sequence_fused_dir}. Run fuse_sequence_features.py first.")
        
    all_files = [
        os.path.join(sequence_fused_dir, f) 
        for f in os.listdir(sequence_fused_dir) 
        if f.endswith(".pt")
    ]

    if len(all_files) == 0:
        raise ValueError(f"No .pt sequence files found in {sequence_fused_dir}. Run fuse_sequence_features.py first.")

    # Determine expected feature dimension
    dims = []
    for f in all_files[:min(10, len(all_files))]:
        try:
            data = torch.load(f, map_location="cpu")
            dims.append(data["sequence"].shape[1])
        except Exception:
            continue
    expected_dim = max(set(dims), key=dims.count) if dims else None

    labels = []
    filtered_files = []
    for f in all_files:
        try:
            data = torch.load(f, map_location="cpu")
            feat_dim = data["sequence"].shape[1]
            if expected_dim is not None and feat_dim != expected_dim:
                print(f"[DataLoader] Skipping stale file '{os.path.basename(f)}' with feature dim {feat_dim} (expected {expected_dim}).")
                continue
            labels.append(data["label"])
            filtered_files.append(f)
        except Exception as e:
            print(f"[DataLoader] Warning: Failed to load '{os.path.basename(f)}' ({e}). Skipping.")
            continue

    all_files = filtered_files

    try:
        # Stratified splits
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_files, labels, test_size=0.30, stratify=labels, random_state=random_state
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=random_state
        )
        print("[DataLoader] Successfully performed stratified GRU sequence splitting.")
    except ValueError as e:
        print(f"[DataLoader] WARNING: Stratified splitting failed ({e}). Falling back to simple random split.")
        train_files, temp_files = train_test_split(all_files, test_size=0.30, random_state=random_state)
        val_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=random_state)

    # Compute global feature mean & std on training split only
    global_mean, global_std = compute_global_stats(train_files)
    print(f"[DataLoader] Global feature scaling computed on train set ({len(train_files)} samples).")

    # Datasets
    train_dataset = SequenceDataset(train_files, global_mean=global_mean, global_std=global_std)
    val_dataset = SequenceDataset(val_files, global_mean=global_mean, global_std=global_std)
    test_dataset = SequenceDataset(test_files, global_mean=global_mean, global_std=global_std)

    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[DataLoader] GRU sequence splits created:")
    print(f"             - Train: {len(train_dataset)} samples")
    print(f"             - Val:   {len(val_dataset)} samples")
    print(f"             - Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, test_files
