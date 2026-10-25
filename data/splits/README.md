# 📁 Dataset Splits (Stratified 70/15/15)

In accordance with Section 4.1 (Table 3) of our paper:

| Partition | Videos | Engaged (0) | Disengaged (1) | 120-frame Windows (Stride 30) |
| :--- | :---: | :---: | :---: | :---: |
| **Training (70%)** | 93 | 45 | 48 | 727 |
| **Validation (15%)** | 20 | 10 | 10 | 142 |
| **Test (15%)** | 20 | 10 | 10 | 150 |
| **Total** | **133** | **65** | **68** | **1,019** |

Videos are strictly separated at the video level so that no video or window is shared across splits.
