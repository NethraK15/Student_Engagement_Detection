import torch
import os
import platform
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset_loaders.seq_dataset import EngagementDataset
from models.lstm_fusion_model import FusionModel

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Using device: {device}")

# OS-aware num_workers setting
num_workers = 0 if platform.system() == "Windows" else 4
print(f">>> Using num_workers={num_workers} (OS: {platform.system()})")

# 2. Load Full Dataset (16-frame sequence)
full_dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat",
    seq_len=16
)

# 3. Create Train/Test Split (80% Train, 20% Test)
indices = list(range(len(full_dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=num_workers)

print(f">>> Total Sequences: {len(full_dataset)} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# 4. Get OpenFace feature dimension
_, openface_sample, _ = full_dataset[0]
openface_dim = openface_sample.shape[0]

# 5. Model Initialization
model = FusionModel(openface_dim=openface_dim, lstm_hidden_dim=256).to(device)

# Freeze CNN backbone
if hasattr(model, 'cnn'):
    for param in model.cnn.parameters():
        param.requires_grad = False
    print(">>> CNN backbone frozen.")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 6. Training & Validation Loop
epochs = 10 
for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()
    total_loss = 0
    for images, openface_feats, labels in train_loader:
        images, openface_feats, labels = images.to(device), openface_feats.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, openface_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # --- VALIDATION PHASE (Evaluation) ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, openface_feats, labels in test_loader:
            images, openface_feats, labels = images.to(device), openface_feats.to(device), labels.to(device)
            outputs = model(images, openface_feats)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {val_acc:.2f}%")

# 7. Save the final model weights
os.makedirs("models", exist_ok=True)
save_path = "models/engagement_lstm.pth"
torch.save(model.state_dict(), save_path)
print(f">>> Training complete. Model saved to {save_path}")