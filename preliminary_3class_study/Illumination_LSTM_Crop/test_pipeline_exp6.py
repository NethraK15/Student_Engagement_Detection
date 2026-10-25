import os
import sys
import tempfile
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config_exp6
from preprocess.extract_frames_crop_illumination import apply_illumination_enhancement, apply_clahe, apply_sharpen, detect_and_crop_face
from models_exp6 import EngagementLSTMClassifier
from data_loader_exp6 import SequenceDataset, pad_sequence_collate_fn
from torch.utils.data import DataLoader

def test_face_crop_and_illumination():
    print("[Test 1/4] Testing Face Detection + Crop + CLAHE + Sharpening Pipeline...")
    dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    crop_out = detect_and_crop_face(dummy_img)
    assert crop_out.ndim == 3 and crop_out.shape[2] == 3, f"Expected 3D BGR image, got {crop_out.shape}"
    
    rgb_crop = crop_out[:, :, ::-1] # BGR to RGB
    enhanced_out = apply_illumination_enhancement(rgb_crop)
    assert enhanced_out.dtype == np.uint8
    print("  --> Passed! Face crop & illumination pipeline outputs valid image arrays.\n")

def test_bilstm_model():
    print("[Test 2/4] Testing Bidirectional EngagementLSTMClassifier...")
    batch_size = 4
    seq_len = 15
    feature_dim = 1408  # 1280 (EfficientNet) + 128 (OpenFace fallback)
    num_classes = 3

    dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    dummy_lens = torch.tensor([15, 12, 10, 8])

    model = EngagementLSTMClassifier(
        input_dim=feature_dim,
        hidden_dim=config_exp6.HIDDEN_DIM,
        num_layers=config_exp6.NUM_LAYERS,
        num_classes=num_classes,
        dropout=config_exp6.DROPOUT
    )
    
    logits = model(dummy_input, seq_lens=dummy_lens)
    assert logits.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
    print("  --> Passed! BiLSTM forward pass completed successfully.\n")

def test_dataloader_and_collation():
    print("[Test 3/4] Testing Dynamic Sequence DataLoader & Collation...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_files = []
        for i in range(5):
            seq_len = np.random.randint(10, 25)
            dummy_seq = torch.randn(seq_len, 1408)
            file_path = os.path.join(tmp_dir, f"sample_{i}.pt")
            torch.save({
                "sequence": dummy_seq,
                "seq_len": seq_len,
                "label": i % 3,
                "video_name": f"video_{i}.mp4",
                "class_name": "engaged",
                "feature_dim": 1408
            }, file_path)
            dummy_files.append(file_path)

        dataset = SequenceDataset(dummy_files)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=pad_sequence_collate_fn)

        for padded_seqs, lens, labels in loader:
            assert padded_seqs.ndim == 3, f"Expected 3D tensor, got {padded_seqs.ndim}D"
            assert padded_seqs.shape[2] == 1408, f"Expected feature dim 1408, got {padded_seqs.shape[2]}"
            assert len(lens) == len(labels), "Length and label count mismatch"

    print("  --> Passed! DataLoader dynamic padding and collation verified.\n")

def test_drive_config_paths():
    print("[Test 4/4] Verifying Dedicated Google Drive Output Config Paths...")
    assert "exp6_cropped_illuminated_frames" in config_exp6.PREPROCESSED_FRAMES_DIR
    assert "exp6_openface_features" in config_exp6.OPENFACE_FEATURES_DIR
    assert "exp6_efficientnet_features" in config_exp6.EFFICIENTNET_FEATURES_DIR
    assert "exp6_sequence_fused_features" in config_exp6.SEQUENCE_FUSED_DIR
    assert "models_exp6" in config_exp6.MODEL_OUT_DIR
    assert "evaluation_results_exp6" in config_exp6.EVAL_OUT_DIR
    print("  --> Passed! All Experiment 6 output paths target dedicated Drive folders.\n")

def main():
    print("==================================================")
    print("      TESTING EXPERIMENT 6 PIPELINE INTEGRITY     ")
    print("==================================================")
    test_face_crop_and_illumination()
    test_bilstm_model()
    test_dataloader_and_collation()
    test_drive_config_paths()
    print("==================================================")
    print("   ALL EXPERIMENT 6 UNIT TESTS PASSED CLEANLY!    ")
    print("==================================================")

if __name__ == "__main__":
    main()
