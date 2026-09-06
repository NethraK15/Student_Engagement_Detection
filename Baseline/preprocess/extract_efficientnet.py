import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from extract_frames import find_video_files

class EfficientNetExtractor:
    def __init__(self):
        """
        Loads the pretrained EfficientNet-B0 model and replaces the classifier with nn.Identity
        to extract the raw 1280-dimensional feature representations.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EfficientNet] Using device: {self.device}")

        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            model = models.efficientnet_b0(pretrained=True)

        model.classifier = nn.Identity()  # Outputs 1280-dim feature vector
        model.eval()
        model.to(self.device)

        # Freeze all layers
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
        Extracts 1280-dim features for all frames of a video in batches.
        Args:
            frames_np: Numpy array of shape [num_frames, 224, 224, 3] (RGB, uint8)
        Returns:
            features: Numpy array of shape [num_frames, 1280] (float32)
        """
        num_frames = len(frames_np)
        
        # Convert to tensor: [N, H, W, 3] -> [N, 3, H, W] and normalize float range to [0, 1]
        tensor = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0
        
        features_list = []
        
        with torch.no_grad():
            for idx in range(0, num_frames, batch_size):
                batch_tensor = tensor[idx : idx + batch_size]
                
                # Apply ImageNet normalization channel-wise
                batch_normalized = torch.stack([self.normalize(img) for img in batch_tensor]).to(self.device)
                
                # Model inference
                batch_feats = self.model(batch_normalized)
                features_list.append(batch_feats.cpu().numpy())
                
        return np.concatenate(features_list, axis=0) # [N, 1280]

def main():
    print("==================================================")
    print("      STEP 3: MODULAR EFFICIENTNET EXTRACTION     ")
    print("==================================================")
    print(f"Preprocessed frames root : {config.PREPROCESSED_FRAMES_DIR}")
    print(f"Output dir               : {config.EFFICIENTNET_FEATURES_DIR}\n")

    os.makedirs(config.EVAL_OUT_DIR, exist_ok=True)
    os.makedirs(config.EFFICIENTNET_FEATURES_DIR, exist_ok=True)

    videos = find_video_files(config.DATASET_DIR, config.CLASSES)
    if not videos:
        print("[Error] No videos found in the dataset. Run Step 1 first.")
        sys.exit(1)

    extractor = EfficientNetExtractor()

    success_count = 0
    skip_count = 0
    error_count = 0

    for video in tqdm(videos, desc="Extracting EfficientNet features"):
        base_name = video["base_name"]
        class_name = video["class_name"]
        
        frames_filename = f"{class_name}__{base_name}.npy"
        frames_path = os.path.join(config.PREPROCESSED_FRAMES_DIR, frames_filename)
        
        save_path = os.path.join(config.EFFICIENTNET_FEATURES_DIR, frames_filename)

        # Resume check
        if os.path.exists(save_path):
            skip_count += 1
            continue

        if not os.path.exists(frames_path):
            print(f"\n[Warning] Skipped '{base_name}': Preprocessed frames (.npy) not found. Run Step 1 first.")
            error_count += 1
            continue

        try:
            # Load preprocessed frames
            frames = np.load(frames_path)
            
            # Extract features
            features = extractor.extract_features(frames)
            
            # Save features
            np.save(save_path, features)
            success_count += 1
        except Exception as e:
            print(f"\n[Warning] Failed to extract features for '{base_name}': {e}")
            error_count += 1

    print("\n" + "=" * 50)
    print("  EFFICIENTNET EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Successfully extracted: {success_count}")
    print(f"Resumed/Skipped        : {skip_count}")
    print(f"Failed                 : {error_count}")
    print(f"Total files in output   : {len(os.listdir(config.EFFICIENTNET_FEATURES_DIR))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
