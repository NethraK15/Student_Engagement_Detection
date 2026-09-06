import os
import sys
import torch
import torch.nn as nn

import config_lightweight
from models_lightweight import get_lightweight_model, AttentionPoolingClassifier, TCNClassifier, SingleLayerGRUClassifier

def test_model_forward(model_name, model, input_tensor, seq_lens):
    print(f"\n--- Testing Model: {model_name} ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    try:
        output = model(input_tensor, seq_lens)
        print(f"Forward Pass Output Shape: {output.shape} (Expected: [Batch=4, Num_Classes=3])")
        assert output.shape == (4, 3), f"Expected shape (4, 3), got {output.shape}"
        
        # Test Loss and Backward Pass
        criterion = nn.CrossEntropyLoss()
        target = torch.tensor([0, 1, 2, 0])
        loss = criterion(output, target)
        loss.backward()
        print(f"Loss Calculation & Backward Pass Successful! (Loss: {loss.item():.4f})")
    except Exception as e:
        print(f"[FAIL] Error testing {model_name}: {e}")
        raise e

def main():
    print("==================================================")
    print("  TESTING LIGHTWEIGHT TEMPORAL PIPELINE & MODELS  ")
    print("==================================================")

    batch_size = 4
    seq_len = 16
    input_dim = 1989 # OpenFace (128) + EfficientNet (1861)
    
    mock_input = torch.randn(batch_size, seq_len, input_dim)
    seq_lens = [16, 12, 16, 10]
    
    # 1. Test Attention Pooling
    attn_model = get_lightweight_model("attention_pool", input_dim=input_dim)
    test_model_forward("Attention Pooling", attn_model, mock_input, seq_lens)
    
    # 2. Test TCN
    tcn_model = get_lightweight_model("tcn", input_dim=input_dim)
    test_model_forward("1-2 Layer TCN", tcn_model, mock_input, seq_lens)

    # 3. Test GRU
    gru_model = get_lightweight_model("gru", input_dim=input_dim)
    test_model_forward("Single-Layer GRU", gru_model, mock_input, seq_lens)

    print("\n==================================================")
    print("  ALL LIGHTWEIGHT TEMPORAL MODEL TESTS PASSED!   ")
    print("==================================================")

if __name__ == "__main__":
    main()
