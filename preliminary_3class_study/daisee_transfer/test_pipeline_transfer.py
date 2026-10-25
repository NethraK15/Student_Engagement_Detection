import os
import sys
import torch
import torch.nn as nn

import config_daisee
from models_transfer import get_transfer_model, load_pretrained_daisee_weights

def main():
    print("==================================================")
    print("  TESTING TEMPORAL TRANSFORMER DAISEE PIPELINE   ")
    print("==================================================")

    batch_size = 4
    seq_len = 16
    input_dim = 1989
    
    mock_daisee_input = torch.randn(batch_size, seq_len, input_dim)
    daisee_targets = torch.tensor([0, 1, 2, 3])
    
    # 1. Test Phase 1 (DAiSEE Pre-training Transformer Model - 4 classes)
    print("\n--- Testing Phase 1: DAiSEE Pre-training Temporal Transformer ---")
    daisee_model = get_transfer_model("transformer", input_dim=input_dim, num_classes=4)
    out1 = daisee_model(mock_daisee_input)
    print(f"DAiSEE Pretraining Output Shape: {out1.shape} (Expected: [4, 4])")
    assert out1.shape == (4, 4)
    
    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(out1, daisee_targets)
    loss1.backward()
    print("Phase 1 Backward Pass Successful!")

    # Save mock DAiSEE checkpoint
    test_ckpt_path = os.path.join(config_daisee.MODEL_OUT_DIR, "test_daisee_transformer_pretrained.pth")
    os.makedirs(os.path.dirname(test_ckpt_path), exist_ok=True)
    torch.save({"model_state_dict": daisee_model.state_dict(), "input_dim": input_dim}, test_ckpt_path)
    print(f"Saved mock DAiSEE Transformer checkpoint to {test_ckpt_path}")

    # 2. Test Phase 2 (Target Model Transfer - 3 classes)
    print("\n--- Testing Phase 2: Target Temporal Transformer Fine-Tuning ---")
    target_model = get_transfer_model("transformer", input_dim=input_dim, num_classes=3)
    target_model = load_pretrained_daisee_weights(target_model, test_ckpt_path, num_target_classes=3)
    
    mock_target_input = torch.randn(batch_size, seq_len, input_dim)
    target_labels = torch.tensor([0, 1, 2, 0])
    
    out2 = target_model(mock_target_input)
    print(f"Target Fine-Tuning Output Shape: {out2.shape} (Expected: [4, 3])")
    assert out2.shape == (4, 3)

    loss2 = criterion(out2, target_labels)
    loss2.backward()
    print("Phase 2 Backward Pass Successful!")

    if os.path.exists(test_ckpt_path):
        os.remove(test_ckpt_path)

    print("\n==================================================")
    print(" TEMPORAL TRANSFORMER DAISEE TRANSFER TEST PASSED! ")
    print("==================================================")

if __name__ == "__main__":
    main()
