"""
Statistical Wilson Score Confidence Interval Calculation
Formula from Section 4.3 of Paper:
Wilson score interval for binomial proportions with small sample sizes (N=20 test videos).
"""

import math

def wilson_score_interval(successes, total, confidence=0.95):
    """
    Computes the Wilson score interval for a binomial proportion.
    Args:
        successes: Number of correct predictions (e.g. 17 out of 20)
        total: Total number of test samples (e.g. 20)
        confidence: Confidence level (default: 0.95, z=1.96)
    Returns:
        lower_pct, upper_pct: Confidence interval in percentages (e.g. 64.0%, 94.8%)
    """
    if total == 0:
        return 0.0, 0.0

    p_hat = successes / total
    z = 1.959963984540054  # 95% two-sided normal quantile

    denominator = 1.0 + (z**2) / total
    centre_adjusted_probability = p_hat + (z**2) / (2.0 * total)
    adjusted_standard_deviation = math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)

    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator

    lower_pct = max(0.0, lower_bound * 100.0)
    upper_pct = min(100.0, upper_bound * 100.0)

    return round(lower_pct, 1), round(upper_pct, 1)

if __name__ == "__main__":
    # Test for 17/20 correct (reported in paper: 85.00% [64.0, 94.8])
    low, high = wilson_score_interval(17, 20)
    print(f"17/20: 85.00% -> 95% Wilson CI: [{low}, {high}] (Matches paper: [64.0, 94.8])")
    # Test for 18/20 correct (LSTM: 90.00% [69.9, 97.2])
    low, high = wilson_score_interval(18, 20)
    print(f"18/20: 90.00% -> 95% Wilson CI: [{low}, {high}] (Matches paper: [69.9, 97.2])")
