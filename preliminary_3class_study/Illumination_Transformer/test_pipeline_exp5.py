import os
import sys
import tempfile
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config_exp5
from models_exp5 import EngagementTransformerClassifier, PositionalEncoding
from data_loader_exp5 import SequenceDataset, pad_sequence_collate_fn
from torch.utils.data import DataLoader

def test_positional_encoding():
    print("[Test 1/4] Testing Sinusoidal Positional Encoding...")
    d_model = 128
    seq_len = 20
    batch_size = 4
    
    pos_enc = PositionalEncoding(d_model=d_model, max_len=100)
    dummy_x = torch.randn(batch_size, seq_len, d_model)
    out = pos_enc(dummy_x)
    
    assert out.shape == (batch_size, seq_len, d_model), f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {out.shape}"
    print("  --> Passed! PositionalEncoding output shape verified.\n")

def test_transformer_model():
    print("[Test 2/4] Testing Temporal EngagementTransformerClassifier...")
    batch_size = 4
    seq_len = 18
    feature_dim = 1408
    num_classes = 3

    dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    dummy_lens = torch.tensor([18, 14, 10, 6])

    model = EngagementTransformerClassifier(
        input_dim=feature_dim,
        d_model=config_exp5.D_MODEL,
        nhead=config_exp5.NHEAD,
        num_layers=config_exp5.NUM_LAYERS,
        dim_feedforward=config_exp5.DIM_FEEDFORWARD,
        dropout=config_exp5.DROPOUT,
        num_classes=num_classes
    )
    
    logits = model(dummy_input, seq_lens=dummy_lens)
    assert logits.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
    print("  --> Passed! Transformer Encoder forward pass completed successfully.\n")

def test_dataloader_and_collation():
    print("[Test 3/4] Testing Sequence DataLoader & Collation...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_files = []
        for i in range(6):
            seq_len = torch.randint(10, 25, (1,)).item()
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
    assert "exp4_sequence_fused_features" in config_exp5.SEQUENCE_FUSED_DIR
    assert "models_exp5" in config_exp5.MODEL_OUT_DIR
    assert "evaluation_results_exp5" in config_exp5.EVAL_OUT_DIR
    print("  --> Passed! All Experiment 5 output paths target Google Drive folders.\n")

def main():
    print("==================================================")
    print("      TESTING EXPERIMENT 5 PIPELINE INTEGRITY     ")
    print("==================================================")
    test_positional_encoding()
    test_transformer_model()
    test_dataloader_and_collation()
    test_drive_config_paths()
    print("==================================================")
    print("   ALL EXPERIMENT 5 UNIT TESTS PASSED CLEANLY!    ")
    print("==================================================")

if __name__ == "__main__":
    main()
