"""
Video-Level Inference via Window Soft Voting
Implements Section 3.6 of Paper:
Averages window softmax posteriors over all K windows of a video to produce a stable video-level prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np

def video_level_soft_voting(window_logits):
    """
    Computes video-level engagement classification by averaging window softmax posteriors.
    
    Args:
        window_logits: Tensor or numpy array of shape [K_windows, num_classes=2]
    Returns:
        video_probs: Averaged probability vector of shape [2]
        predicted_class: 0 (Engaged) or 1 (Disengaged)
        confidence: Posterior probability of the predicted class
    """
    if isinstance(window_logits, np.ndarray):
        window_logits = torch.from_numpy(window_logits)

    # Compute softmax posteriors for each window: [K, 2]
    posteriors = F.softmax(window_logits, dim=-1)

    # Average posteriors across all K windows: \hat{p} = \frac{1}{K} \sum_{k=1}^K p_k
    video_probs = torch.mean(posteriors, dim=0).cpu().numpy()

    predicted_class = int(np.argmax(video_probs))
    confidence = float(video_probs[predicted_class])

    return video_probs, predicted_class, confidence
