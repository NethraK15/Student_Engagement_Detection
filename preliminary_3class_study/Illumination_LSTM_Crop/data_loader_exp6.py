import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config_exp6

class SequenceDataset(Dataset):
    """
    Dataset to load pre-extracted 2D sequence feature tensors for each video in Experiment 6.
    Pads shorter sequences with zeros and truncates longer sequences if max_seq_len is specified.
    Applies z-score normalization across feature dimensions.
    """
    def __init__(self, file_paths, max_seq_len=None):
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

        # Truncate or Pad if max_seq_len is specified
        if self.max_seq_len is not None and self.max_seq_len > 0:
            if n_frames >= self.max_seq_len:
                padded_sequence = normalized_sequence[:self.max_seq_len]
                actual_len = self.max_seq_len
            else:
                actual_len = n_frames
                padding = torch.zeros((self.max_seq_len - n_frames, feat_dim), dtype=torch.float32)
                padded_sequence = torch.cat([normalized_sequence, padding], dim=0)
            return padded_sequence, actual_len, label
        else:
            return normalized_sequence, n_frames, label

def pad_sequence_collate_fn(batch):
    """
    Custom collate function that dynamically pads sequences in the batch to the max length of the batch.
    Returns:
        padded_sequences: [Batch, Max_Batch_Seq_Len, Feature_Dim]
        lengths: [Batch]
        labels: [Batch]
    """
    sequences, lengths, labels = zip(*batch)
    
    feat_dim = sequences[0].shape[1]
    max_len = max(lengths)
    batch_size = len(sequences)
    
    padded_sequences = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded_sequences[i, :seq_len, :] = seq
        
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return padded_sequences, lengths_tensor, labels_tensor

def get_sequence_dataloaders(sequence_fused_dir=config_exp6.SEQUENCE_FUSED_DIR, 
                             batch_size=config_exp6.BATCH_SIZE, 
                             random_state=config_exp6.RANDOM_STATE):
    """
    Loads all sequence feature files from Drive, performs a stratified Train (70%), Val (15%), 
    and Test (15%) split, and returns PyTorch DataLoaders with dynamic sequence padding.
    """
    if not os.path.exists(sequence_fused_dir):
        raise FileNotFoundError(f"Sequence feature folder not found: {sequence_fused_dir}. Run fuse_sequence_features_exp6.py first.")
        
    all_files = [
        os.path.join(sequence_fused_dir, f) 
        for f in os.listdir(sequence_fused_dir) 
        if f.endswith(".pt")
    ]

    if len(all_files) == 0:
        raise ValueError(f"No .pt sequence files found in {sequence_fused_dir}. Run fuse_sequence_features_exp6.py first.")

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
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_files, labels, test_size=0.30, stratify=labels, random_state=random_state
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=random_state
        )
        print("[DataLoader] Successfully performed stratified sequence splitting for Experiment 6.")
    except ValueError as e:
        print(f"[DataLoader] WARNING: Stratified splitting failed ({e}). Falling back to simple random split.")
        train_files, temp_files = train_test_split(all_files, test_size=0.30, random_state=random_state)
        val_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=random_state)

    train_dataset = SequenceDataset(train_files)
    val_dataset = SequenceDataset(val_files)
    test_dataset = SequenceDataset(test_files)

    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_collate_fn, num_workers=num_workers)

    print(f"[DataLoader] Experiment 6 sequence splits created:")
    print(f"             - Train: {len(train_dataset)} samples")
    print(f"             - Val:   {len(val_dataset)} samples")
    print(f"             - Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, test_files
