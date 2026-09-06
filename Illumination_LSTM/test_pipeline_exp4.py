import os
import sys
import tempfile
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config_exp4
from preprocess.extract_frames_illumination import apply_illumination_enhancement, apply_clahe, apply_sharpen
from models_exp4 import EngagementLSTMClassifier
from data_loader_exp4 import SequenceDataset, pad_sequence_collate_fn
from torch.utils.data import DataLoader

def test_illumination_enhancement():
    print("[Test 1/4] Testing CLAHE + Laplacian Sharpening Illumination Pipeline...")
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    clahe_out = apply_clahe(dummy_img)
    assert clahe_out.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {clahe_out.shape}"
    assert clahe_out.dtype == np.uint8, f"Expected uint8 dtype, got {clahe_out.dtype}"
    
    sharpen_out = apply_sharpen(clahe_out)
    assert sharpen_out.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {sharpen_out.shape}"
    assert sharpen_out.dtype == np.uint8, f"Expected uint8 dtype, got {sharpen_out.dtype}"
    
    full_out = apply_illumination_enhancement(dummy_img)
    assert full_out.shape == (224, 224, 3)
    print("  --> Passed! CLAHE + Sharpening outputs valid [224, 224, 3] uint8 images.\n")

def test_lstm_model():
    print("[Test 2/4] Testing Bidirectional EngagementLSTMClassifier...")
    batch_size = 4
    seq_len = 15
    feature_dim = 1408  # 1280 (EfficientNet) + 128 (OpenFace fallback)
    num_classes = 3

    dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    dummy_lens = torch.tensor([15, 12, 10, 8])

    model = EngagementLSTMClassifier(
        input_dim=feature_dim,
        hidden_dim=config_exp4.HIDDEN_DIM,
        num_layers=config_exp4.NUM_LAYERS,
        num_classes=num_classes,
        dropout=config_exp4.DROPOUT
    )
    
    logits = model(dummy_input, seq_lens=dummy_lens)
    assert logits.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
    print("  --> Passed! Bidirectional LSTM forward pass completed successfully.\n")

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
    print("[Test 4/4] Verifying Google Drive Output Config Paths...")
    assert "exp4_preprocessed_frames" in config_exp4.PREPROCESSED_FRAMES_DIR
    assert "exp4_openface_features" in config_exp4.OPENFACE_FEATURES_DIR
    assert "exp4_efficientnet_features" in config_exp4.EFFICIENTNET_FEATURES_DIR
    assert "exp4_sequence_fused_features" in config_exp4.SEQUENCE_FUSED_DIR
    assert "models_exp4" in config_exp4.MODEL_OUT_DIR
    assert "evaluation_results_exp4" in config_exp4.EVAL_OUT_DIR
    print("  --> Passed! All Experiment 4 output paths explicitly target Drive folders.\n")

def main():
    print("==================================================")
    print("      TESTING EXPERIMENT 4 PIPELINE INTEGRITY     ")
    print("==================================================")
    test_illumination_enhancement()
    test_lstm_model()
    test_dataloader_and_collation()
    test_drive_config_paths()
    print("==================================================")
    print("   ALL EXPERIMENT 4 UNIT TESTS PASSED CLEANLY!    ")
    print("==================================================")

if __name__ == "__main__":
    main()
