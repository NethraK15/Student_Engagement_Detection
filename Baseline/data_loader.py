import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config

class FusedFeaturesDataset(Dataset):
    """
    Dataset to load pre-extracted, fused feature vectors for each video.
    Each file is a dictionary saved as a PyTorch tensor .pt file.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu")
        feature = data["feature"]  # Shape: [1280 + openface_dim]
        label = int(data["label"]) # Integer class: 0, 1, 2
        return feature, label

def get_dataloaders(fused_dir, batch_size=config.BATCH_SIZE, random_state=config.RANDOM_STATE):
    """
    Loads all pre-extracted fused feature files, splits them into Train (70%),
    Val (15%), and Test (15%) subsets, and returns PyTorch DataLoaders.
    """
    if not os.path.exists(fused_dir):
        raise FileNotFoundError(f"Fused feature folder not found: {fused_dir}. Run preprocessing scripts first.")
        
    all_files = [
        os.path.join(fused_dir, f) 
        for f in os.listdir(fused_dir) 
        if f.endswith(".pt")
    ]

    if len(all_files) == 0:
        raise ValueError(f"No .pt fused feature files found in {fused_dir}. Run fuse_features.py first.")

    # Determine the expected feature dimension to filter out mismatched/stale files
    dims = []
    for f in all_files[:min(10, len(all_files))]:
        try:
            data = torch.load(f, map_location="cpu")
            dims.append(data["feature"].shape[0])
        except Exception:
            continue
    expected_dim = max(set(dims), key=dims.count) if dims else None

    # Retrieve labels and filter files
    labels = []
    filtered_files = []
    for f in all_files:
        try:
            data = torch.load(f, map_location="cpu")
            feat_dim = data["feature"].shape[0]
            if expected_dim is not None and feat_dim != expected_dim:
                print(f"[DataLoader] Skipping file '{os.path.basename(f)}' with mismatching dimension {feat_dim} (expected {expected_dim}).")
                continue
            labels.append(data["label"])
            filtered_files.append(f)
        except Exception as e:
            print(f"[DataLoader] Warning: Failed to load '{os.path.basename(f)}' ({e}). Skipping.")
            continue

    all_files = filtered_files

    try:
        # Try stratified splits:
        # First split: Train (70%) and Temp (30%)
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_files, 
            labels, 
            test_size=0.30, 
            stratify=labels, 
            random_state=random_state
        )

        # Second split: Val (15% of total) and Test (15% of total)
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, 
            temp_labels, 
            test_size=0.50, 
            stratify=temp_labels, 
            random_state=random_state
        )
        print("[DataLoader] Successfully performed stratified splitting.")
        
    except ValueError as e:
        # Fall back to standard random split (non-stratified) if there are too few class members
        print(f"[DataLoader] WARNING: Stratified splitting failed ({e}). Falling back to simple random splitting.")
              
        train_files, temp_files = train_test_split(
            all_files, 
            test_size=0.30, 
            random_state=random_state
        )
        val_files, test_files = train_test_split(
            temp_files, 
            test_size=0.50, 
            random_state=random_state
        )

    # Create Datasets
    train_dataset = FusedFeaturesDataset(train_files)
    val_dataset = FusedFeaturesDataset(val_files)
    test_dataset = FusedFeaturesDataset(test_files)

    # Create DataLoaders
    # On Windows, num_workers must be 0 to prevent multiprocessing hangs. On Colab/Linux, 2 works well.
    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[DataLoader] Splits created:")
    print(f"             - Train: {len(train_dataset)} samples")
    print(f"             - Val:   {len(val_dataset)} samples")
    print(f"             - Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, test_files
