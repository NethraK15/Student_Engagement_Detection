import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config_binary_lstm as config

# =====================================================================
# SLIDING WINDOW DATASET
# =====================================================================
class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        file_paths,
        window_size=config.WINDOW_SIZE,
        window_stride=config.WINDOW_STRIDE,
        global_mean=None,
        global_std=None
    ):
        self.window_size = window_size
        self.window_stride = window_stride
        self.global_mean = global_mean
        self.global_std = global_std

        self.windows = []
        for fp in file_paths:
            try:
                data = torch.load(fp, map_location="cpu")
                sequence = data["sequence"]
                label    = int(data["label"])
                n_frames = sequence.shape[0]

                start = 0
                while True:
                    end = start + window_size
                    if end > n_frames:
                        actual_len = n_frames - start
                        if actual_len > 0:
                            self.windows.append((fp, start, actual_len, label))
                        break
                    else:
                        self.windows.append((fp, start, window_size, label))

                    start += window_stride
                    if start >= n_frames:
                        break

            except Exception as e:
                print(f"[Dataset] Warning: Failed to index '{os.path.basename(fp)}': {e}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fp, start, actual_len, label = self.windows[idx]

        data = torch.load(fp, map_location="cpu")
        sequence = data["sequence"]
        feat_dim = sequence.shape[1]

        window_seq = sequence[start : start + actual_len]

        if self.global_mean is not None and self.global_std is not None:
            mean = self.global_mean.to(window_seq.device)
            std = self.global_std.to(window_seq.device)
            window_seq = (window_seq - mean) / std

        if actual_len < self.window_size:
            padding = torch.zeros(
                (self.window_size - actual_len, feat_dim), dtype=torch.float32
            )
            window_seq = torch.cat([window_seq, padding], dim=0)

        return window_seq, actual_len, label


# =====================================================================
# VIDEO-LEVEL DATASET FOR INFERENCE
# =====================================================================
class VideoLevelDataset(Dataset):
    def __init__(
        self,
        file_paths,
        window_size=config.WINDOW_SIZE,
        window_stride=config.WINDOW_STRIDE,
        global_mean=None,
        global_std=None
    ):
        self.window_size = window_size
        self.window_stride = window_stride
        self.global_mean = global_mean
        self.global_std = global_std
        self.file_paths = file_paths

        self.windows = []
        self.video_map = {}
        for fp in file_paths:
            try:
                data = torch.load(fp, map_location="cpu")
                sequence = data["sequence"]
                label    = int(data["label"])
                n_frames = sequence.shape[0]

                self.video_map[fp] = []
                start = 0
                while True:
                    end = start + window_size
                    if end > n_frames:
                        actual_len = n_frames - start
                        if actual_len > 0:
                            self.video_map[fp].append(len(self.windows))
                            self.windows.append((fp, start, actual_len, label))
                        break
                    else:
                        self.video_map[fp].append(len(self.windows))
                        self.windows.append((fp, start, window_size, label))

                    start += window_stride
                    if start >= n_frames:
                        break

            except Exception as e:
                print(f"[Dataset] Warning: Could not index '{os.path.basename(fp)}': {e}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fp, start, actual_len, label = self.windows[idx]

        data = torch.load(fp, map_location="cpu")
        sequence = data["sequence"]
        feat_dim = sequence.shape[1]

        window_seq = sequence[start : start + actual_len]
        if self.global_mean is not None and self.global_std is not None:
            mean = self.global_mean.to(window_seq.device)
            std = self.global_std.to(window_seq.device)
            window_seq = (window_seq - mean) / std

        if actual_len < self.window_size:
            padding = torch.zeros(
                (self.window_size - actual_len, feat_dim), dtype=torch.float32
            )
            window_seq = torch.cat([window_seq, padding], dim=0)

        return window_seq, actual_len, label


# =====================================================================
# GLOBAL STATS CALCULATION
# =====================================================================
def compute_global_stats(file_paths):
    all_frames = []
    for f in file_paths:
        try:
            data = torch.load(f, map_location="cpu")
            all_frames.append(data["sequence"])
        except Exception:
            continue
    if not all_frames:
        return None, None
    concat = torch.cat(all_frames, dim=0)
    mean = concat.mean(dim=0, keepdim=True)
    std  = concat.std(dim=0,  keepdim=True) + 1e-6
    return mean, std


# =====================================================================
# BINARY DATALOADER BUILDER
# =====================================================================
def get_binary_dataloaders(
    sequence_fused_dir,
    batch_size=config.BATCH_SIZE,
    random_state=config.RANDOM_STATE
):
    if not os.path.exists(sequence_fused_dir):
        parent = os.path.dirname(sequence_fused_dir)
        available_dirs = os.listdir(parent) if os.path.exists(parent) else []
        raise FileNotFoundError(
            f"\n[Error] Sequence features folder not found at:\n  '{sequence_fused_dir}'\n"
            f"Available folders in '{parent}':\n  {available_dirs}\n"
            "Please check that your Google Drive is mounted and the folder path is correct."
        )

    all_files = [
        os.path.join(sequence_fused_dir, f)
        for f in os.listdir(sequence_fused_dir)
        if f.endswith(".pt")
    ]

    if not all_files:
        raise ValueError(f"No .pt files found in '{sequence_fused_dir}'.")

    VALID_LABELS = {0, 1}
    labels = []
    filtered_files = []

    dims = []
    for f in all_files[:min(20, len(all_files))]:
        try:
            d = torch.load(f, map_location="cpu")
            lbl = int(d["label"])
            if lbl in VALID_LABELS:
                dims.append(d["sequence"].shape[1])
        except Exception:
            continue
    expected_dim = max(set(dims), key=dims.count) if dims else config.FUSED_INPUT_DIM

    skipped_partial = 0
    for f in all_files:
        try:
            d = torch.load(f, map_location="cpu")
            lbl = int(d["label"])
            feat_dim = d["sequence"].shape[1]

            if lbl not in VALID_LABELS:
                skipped_partial += 1
                continue

            if expected_dim is not None and feat_dim != expected_dim:
                continue

            labels.append(lbl)
            filtered_files.append(f)
        except Exception:
            pass

    if skipped_partial > 0:
        print(f"[DataLoader] Excluded {skipped_partial} non-binary/partial samples.")

    print(f"[DataLoader] Binary dataset: {len(filtered_files)} videos "
          f"({labels.count(0)} Engaged, {labels.count(1)} Disengaged)")

    try:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            filtered_files, labels, test_size=0.30, stratify=labels, random_state=random_state
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=random_state
        )
    except ValueError:
        train_files, temp_files = train_test_split(filtered_files, test_size=0.30, random_state=random_state)
        val_files, test_files   = train_test_split(temp_files, test_size=0.50, random_state=random_state)

    global_mean, global_std = compute_global_stats(train_files)

    train_ds = SlidingWindowDataset(train_files, global_mean=global_mean, global_std=global_std)
    val_ds   = SlidingWindowDataset(val_files,   global_mean=global_mean, global_std=global_std)
    test_ds  = VideoLevelDataset(test_files,     global_mean=global_mean, global_std=global_std)

    num_workers = 0 if os.name == "nt" else 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[DataLoader] Partition Summary:")
    print(f"             - Train : {len(train_files)} videos -> {len(train_ds)} windows")
    print(f"             - Val   : {len(val_files)} videos -> {len(val_ds)} windows")
    print(f"             - Test  : {len(test_files)} videos -> {len(test_ds)} windows")

    return train_loader, val_loader, test_loader, test_files, global_mean, global_std
