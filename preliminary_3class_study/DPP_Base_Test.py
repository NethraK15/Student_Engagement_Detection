import torch
import os
import platform
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset_loaders.base_dataset import EngagementDataset
from models.base_fusion_model import FusionModel

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Testing on device: {device}")

# OS-aware num_workers setting
num_workers = 0 if platform.system() == "Windows" else 4
print(f">>> Using num_workers={num_workers} (OS: {platform.system()})")

# 2. Load Full Dataset (Single-frame)
dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat"
)

# 3. Recreate the Test Split (using random_state=42 to match training split exactly)
indices = list(range(len(dataset)))
_, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
test_ds = Subset(dataset, test_idx)

loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=num_workers)
print(f">>> Test split size: {len(test_ds)}")

# 4. Model Setup
_, openface_sample, _ = dataset[0]
openface_dim = openface_sample.shape[0]

model = FusionModel(openface_dim=openface_dim).to(device)

# 5. Load Trained Weights
model_path = "models/engagement_base.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f">>> Loaded weights from {model_path}")
else:
    print(f"!!! Error: Could not find weights at {model_path}. Run training first.")
    exit()

model.eval()

# 6. Evaluation Loop
correct = 0
total = 0

print(">>> Starting evaluation...")
with torch.no_grad():
    for images, openface_feats, labels in loader:
        images, openface_feats, labels = images.to(device), openface_feats.to(device), labels.to(device)
        
        outputs = model(images, openface_feats)
        preds = outputs.argmax(dim=1)
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()

# 7. Final Results
final_acc = 100 * correct / total
print("-" * 30)
print(f"Final Base-Fusion Test Accuracy: {final_acc:.2f}%")
print(f"Total Samples Tested: {total}")
print("-" * 30)