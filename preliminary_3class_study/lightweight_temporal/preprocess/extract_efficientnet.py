import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# Add parent directory to path to import config_lightweight
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config_lightweight

class EfficientNetExtractor:
    def __init__(self):
        """
        Loads ImageNet pretrained EfficientNet-B0 and replaces classifier head with nn.Identity
        to extract 1280-dimensional visual feature vectors per frame.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EfficientNet] Using device: {self.device}")

        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            model = models.efficientnet_b0(pretrained=True)

        model.classifier = nn.Identity()  # Outputs 1280-dim feature vector
        model.eval()
        model.to(self.device)

        for p in model.parameters():
            p.requires_grad = False
            
        self.model = model

        # ImageNet normalization transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def extract_features(self, frames_np, batch_size=64):
        """
        Extracts 1280-dim features for all frames of a video clip in batches.
        Args:
            frames_np: Numpy array of shape [num_frames, 224, 224, 3] (RGB uint8)
        Returns:
            features: Tensor of shape [num_frames, 1280] (float32)
        """
        num_frames = len(frames_np)
        
        # Convert to tensor: [N, H, W, 3] -> [N, 3, H, W] normalized float [0, 1]
        tensor = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0
        
        features_list = []
        
        with torch.no_grad():
            for idx in range(0, num_frames, batch_size):
                batch_tensor = tensor[idx : idx + batch_size]
                batch_normalized = torch.stack([self.normalize(img) for img in batch_tensor]).to(self.device)
                batch_feats = self.model(batch_normalized)
                features_list.append(batch_feats.cpu())
                
        return torch.cat(features_list, dim=0)  # [num_frames, 1280]

def main():
    print("==================================================")
    print("    STEP 2: EFFICIENTNET FEATURE EXTRACTION      ")
    print("==================================================")
    
    frames_dir = getattr(config_lightweight, "PREPROCESSED_FRAMES_DIR", os.path.join(config_lightweight.PROJECT_ROOT, "preprocessed_frames"))
    eff_out_dir = config_lightweight.EFFICIENTNET_FEATURES_DIR

    os.makedirs(eff_out_dir, exist_ok=True)

    print(f"Preprocessed Frames Root : {frames_dir}")
    print(f"EfficientNet Output Dir  : {eff_out_dir}\n")

    if not os.path.exists(frames_dir):
        print(f"[Error] Preprocessed frames directory not found: {frames_dir}. Run extract_frames.py first.")
        sys.exit(1)

    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".npy")]
    if not frame_files:
        print(f"[Error] No .npy frame files found in {frames_dir}")
        sys.exit(1)

    extractor = EfficientNetExtractor()

    success_count = 0
    skip_count = 0
    error_count = 0

    for frame_file in tqdm(frame_files, desc="Extracting EfficientNet features"):
        sample_name = os.path.splitext(frame_file)[0]
        frames_path = os.path.join(frames_dir, frame_file)
        save_path = os.path.join(eff_out_dir, f"{sample_name}.pt")

        if os.path.exists(save_path):
            skip_count += 1
            continue

        try:
            frames = np.load(frames_path)
            features = extractor.extract_features(frames)
            
            torch.save({
                "sample_id": sample_name,
                "features": features  # [num_frames, 1280]
            }, save_path)
            
            success_count += 1
        except Exception as e:
            print(f"\n[Warning] Failed to extract features for '{sample_name}': {e}")
            error_count += 1

    print("\n" + "=" * 50)
    print("  EFFICIENTNET EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Successfully extracted: {success_count}")
    print(f"Resumed/Skipped        : {skip_count}")
    print(f"Failed                 : {error_count}")
    print(f"Total output files     : {len(os.listdir(eff_out_dir))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
