import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config_lstm

class SequenceDataset(Dataset):
    """
    Dataset to load pre-extracted 2D sequence feature tensors for each video.
    Pads shorter sequences with zeros and truncates longer sequences to max_seq_len.
    Also applies z-score normalization across feature dimensions.
    """
    def __init__(self, file_paths, max_seq_len=config_lstm.MAX_SEQ_LEN):
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu")
        sequence = data["sequence"]  # Tensor of shape [N_frames, feature_dim]
        label = int(data["label"])   # Class label: 0, 1, 2
        
        n_frames, feat_dim = sequence.shape

        # Z-score normalization per feature channel to bring OpenFace & EfficientNet features to comparable scales
        mean = sequence.mean(dim=0, keepdim=True)
        std = sequence.std(dim=0, keepdim=True) + 1e-6
        normalized_sequence = (sequence - mean) / std

        # Truncate or Pad to max_seq_len
        if n_frames >= self.max_seq_len:
            padded_sequence = normalized_sequence[:self.max_seq_len]
            actual_len = self.max_seq_len
        else:
            actual_len = n_frames
            padding = torch.zeros((self.max_seq_len - n_frames, feat_dim), dtype=torch.float32)
            padded_sequence = torch.cat([normalized_sequence, padding], dim=0)

        return padded_sequence, actual_len, label

def get_sequence_dataloaders(sequence_fused_dir, batch_size=config_lstm.BATCH_SIZE, random_state=config_lstm.RANDOM_STATE):
    """
    Loads all sequence feature files, performs a stratified Train (70%), Val (15%), 
    and Test (15%) split, and returns PyTorch DataLoaders.
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
        print("[DataLoader] Successfully performed stratified sequence splitting.")
    except ValueError as e:
        print(f"[DataLoader] WARNING: Stratified splitting failed ({e}). Falling back to simple random split.")
        train_files, temp_files = train_test_split(all_files, test_size=0.30, random_state=random_state)
        val_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=random_state)

    # Datasets
    train_dataset = SequenceDataset(train_files)
    val_dataset = SequenceDataset(val_files)
    test_dataset = SequenceDataset(test_files)

    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[DataLoader] Sequence splits created:")
    print(f"             - Train: {len(train_dataset)} samples")
    print(f"             - Val:   {len(val_dataset)} samples")
    print(f"             - Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, test_files
