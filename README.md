# 🎓 A Dual-Stream Temporal Transformer Framework for Student Engagement Detection in Classroom Videos

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-green.svg)](https://opencv.org/)
[![OpenFace](https://img.shields.io/badge/OpenFace-2.0-red.svg)](https://github.com/TadasBaltrusaitis/OpenFace)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Official Research Codebase for:**  
> *"A Dual-Stream Temporal Transformer Framework for Student Engagement Detection in Classroom Videos"*  
> **Authors:** Nethra Kanagaraj, S. Adwaitha, Srividya Manikandan, Aadhithya Bharathi A, and Dr. Manjusha R.*  
> **Affiliation:** Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore, India.  
> *\*Corresponding author / Research Supervisor: Dr. Manjusha R., Assistant Professor*

---

## 📌 Abstract & Overview

Recognising whether students are engaged during a lesson can provide instructors with objective feedback. However, natural classroom video remains challenging due to oblique angles, student occlusions, lighting shifts, and subtle temporal behaviors.

This repository implements a **dual-stream temporal Transformer framework** for binary (**Engaged** vs. **Disengaged**) student engagement detection:
1. **Appearance Stream**: A frozen **EfficientNet-B0** convolutional backbone supplying a 1,280-dimensional spatial descriptor per frame.
2. **Facial Behavior Stream**: **OpenFace 2.0** supplying a 709-dimensional structured descriptor of facial action units (17 AUs), 3D eye gaze, head pose (6-DoF), and landmarks.
3. **Temporal Modeling**: Concatenated into a **1,989-dimensional** fused stream, segmented into **120-frame windows (4s at 30 fps)** with **75% overlap (stride 30 frames)**.
4. **Pre-LN Transformer**: A compact 3-layer, 4-head pre-layer-normalized Transformer encoder with **663,074 trainable parameters** (4.67M total, only 14% trained).
5. **Video-Level Inference**: Window posteriors are aggregated using **soft voting** to deliver robust video-level engagement predictions.

---

## 🔬 System Architecture

```text
               [ Classroom Video Stream (30 fps) ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [ Stream 1: Appearance ]              [ Stream 2: Facial Behavior ]
   • EfficientNet-B0 (Frozen)            • OpenFace 2.0 Feature Engine
   • 224x224 RGB Frame Input             • 17 Facial Action Units (AUs)
   • Global Average Pooling              • 3D Eye Gaze Vectors & Angles
   • Dimension: 1,280                    • 6-DoF Head Pose (Yaw, Pitch, Roll)
            │                            • Dimension: 709
            └──────────────────┬──────────────────┘
                               ▼
               [ Early Feature Concatenation ]
                  Dimension: 1,989 features/frame
                               │
                               ▼
             [ Sliding Window Segmentation (W=120, S=30) ]
                  75% Temporal Overlap (4s windows)
                               │
                               ▼
             [ Linear Projection & Positional Encoding ]
                  1,989 → d_model = 128 (Pre-LN)
                               │
                               ▼
             [ 3-Layer Pre-LN Transformer Encoder ]
                  4 Attention Heads, FFN Dim = 256
                  (663,074 Trainable Parameters)
                               │
                               ▼
               [ Masked Temporal Mean Pooling ]
                               │
                               ▼
              [ Tapering MLP Classification Head ]
                  Linear(128→64) → GELU → Linear(64→32) → GELU → Linear(32→2)
                               │
                               ▼
               [ Video-Level Soft Voting Posterior ]
                  p_video = (1 / K) * sum(p_k)
                               │
                               ▼
                { Engaged (0)  |  Disengaged (1) }
```

---

## 📁 Repository Structure & Team Task Allocations

```text
Student_Engagement_Detection/
├── configs/
│   └── config.py                           # Central configuration (W=120, S=30, 1989-dim, AdamW)
│
├── src/                                    # Proposed Framework Core Modules (Lead)
│   ├── preprocess/
│   │   └── fuse_sequence_features_binary.py # Concatenates OpenFace (709) + EfficientNet (1280) -> 1989-D .pt
│   ├── models/
│   │   └── transformer_encoder.py          # Proposed 3-layer Pre-LN Transformer (663,074 params)
│   └── inference/
│       ├── soft_voting.py                  # Video-level window posterior averaging
│       └── wilson_interval.py              # Statistical 95% Wilson confidence intervals
│
├── ablations/                              # 4-Stage Controlled Ablation Study Suite
│   ├── stage1_geometric_extractors/        # Stage 1: Dlib vs MediaPipe vs OpenFace 2.0 (Adwaitha)
│   │   ├── config_ablation.py              # Stage 1 hyperparameter config
│   │   ├── dataset_openface.py             # OpenFace feature slice data loader
│   │   ├── model_openface.py               # Geometric MLP classifier head
│   │   ├── train_ablation.py               # Training loop
│   │   ├── evaluate_ablation.py            # Validation & test metrics evaluator
│   │   ├── extract_classical_dlib_features.py # 6-dim EAR & MAR extractor
│   │   ├── extract_mediapipe_features.py   # 55-dim 3D landmark mesh extractor
│   │   ├── OpenFace_Binary_Ablation_Colab.ipynb
│   │   ├── Geometric_Alternatives_Ablation_Colab.ipynb
│   │   └── README_Colab_Guide.md
│   │
│   ├── stage2_visual_backbones/            # Stage 2: ResNet-18 vs EfficientNet-B0 (Srividya)
│   │   ├── ablation_efficientnet_binary.py # EfficientNet-B0 visual stream extraction & classification
│   │   ├── ablation_resnet18_binary.py     # ResNet-18 spatial feature extraction & benchmark
│   │   ├── Ablation_EfficientNet_Binary.ipynb
│   │   ├── Ablation_ResNet18_Binary.ipynb
│   │   └── README_Colab_Guide.md
│   │
│   ├── stage3_stream_fusion/               # Stage 3: Unimodal vs Dual-Stream Fused (Joint Collaboration)
│   │   ├── config_binary_fusion.py         # Fusion experiment configuration
│   │   ├── data_loader_binary_fusion.py    # Unimodal & concatenated data loader
│   │   ├── models_binary_fusion.py         # Static order-invariant pooling baseline
│   │   ├── train_binary_fusion.py          # Stage 3 training routine
│   │   ├── evaluate_binary_fusion.py       # Single-stream vs Fused comparative evaluation
│   │   ├── Ablation_Binary_Fusion_Colab.ipynb
│   │   └── Colab_Ablation_Guide.md
│   │
│   └── stage4_temporal_models/             # Stage 4: Static vs BiLSTM vs LSTM vs Transformer (Aadhithya)
│       ├── colab_binary_lstm/              # 2-layer forward LSTM sequence model
│       │   ├── config_binary_lstm.py
│       │   ├── data_loader_binary_lstm.py
│       │   ├── models_binary_lstm.py
│       │   ├── train_binary_lstm.py
│       │   ├── evaluate_binary_lstm.py
│       │   ├── Ablation_Binary_LSTM_Colab.ipynb
│       │   └── Colab_LSTM_Guide.md
│       ├── colab_binary_bilstm/            # 2-layer bidirectional LSTM sequence model
│       │   ├── config_binary_bilstm.py
│       │   ├── data_loader_binary_bilstm.py
│       │   ├── models_binary_bilstm.py
│       │   ├── train_binary_bilstm.py
│       │   ├── evaluate_binary_bilstm.py
│       │   ├── Ablation_Binary_BiLSTM_Colab.ipynb
│       │   └── Colab_BiLSTM_Guide.md
│       └── colab_binary_transformer/       # Windowed temporal Transformer model
│           ├── config_binary_transformer.py
│           ├── data_loader_binary_transformer.py
│           ├── models_binary_transformer.py
│           ├── train_binary_transformer.py
│           ├── evaluate_binary_transformer.py
│           ├── test_classroom_binary_transformer.py
│           ├── Binary_Transformer_Colab.ipynb
│           └── Colab_Binary_Transformer_Guide.md
│
├── preliminary_3class_study/               # Preserved Review 1 Exploratory 3-Class Experiments
│   ├── Baseline/                           # Early single-frame models
│   ├── Baseline_transformer/               # Initial transformer trials
│   ├── baseline_LSTM/                      # Initial LSTM experiments
│   ├── Illumination_LSTM/                  # Illumination robustness tests
│   ├── Illumination_LSTM_Crop/             # Cropped face illumination trials
│   ├── Illumination_Transformer/           # Illumination transformer trials
│   ├── daisee_transfer/                    # DAiSEE benchmark transfer learning
│   ├── lightweight_temporal/               # Preliminary TCN, GRU & Attention Pooling
│   ├── lstm_gru/                           # Hybrid LSTM-GRU comparative trials
│   └── README.md                           # Documentation on Review 1 exploratory findings
│
├── data/                                   # Dataset Documentation & Demonstration Samples
│   ├── sample/
│   │   └── Multi_student_video_sample/
│   │       └── multi_student_video.mp4     # Multi-student classroom video demonstration
│   ├── splits/
│   │   └── README.md                       # 70/15/15 stratified video-level split documentation
│   └── README.md                           # Cloud feature download instructions & format
│
├── dashboard/                              # Real-time Interactive Web Dashboard
├── docs/                                   # Comprehensive Project Documentation
│   ├── installation_guide.md               # Step-by-step setup (Windows, Linux, CUDA)
│   └── ablation_results.md                 # Full ablation benchmark tables & Wilson CI analysis
│
├── scripts/                                # Helper Utilities
│   └── download_dataset.py                 # Automated dataset fetcher
│
├── .gitignore
├── CITATION.cff                            # Research paper citation metadata
├── CONTRIBUTING.md                         # Open-source contribution guidelines
├── LICENSE                                 # MIT Open Source License
├── README.md                               # Research paper showcase documentation
└── requirements.txt                        # Python dependencies
```

---

## 📊 Experimental Results & Ablation Analysis

Evaluated on **133 naturalistic classroom video recordings** with a stratified **70/15/15** split (93 train, 20 val, 20 test).

### Stage 1: Geometric Feature Extractor Ablation
| Extractor | Feature Dim | Trainable Params | Validation Acc | Test Acc [95% Wilson CI] | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dlib (EAR / MAR)** | 6 | 1,154 | 55.56% | 52.63% [31.7, 72.7] | 27.70% | 52.63% | 36.30% |
| **MediaPipe FaceMesh** | 55 | 7,426 | 66.67% | 68.42% [46.0, 84.6] | 68.92% | 68.42% | 67.88% |
| **OpenFace 2.0 (Proposed)** | **709** | **16,770** | **85.00%** | **70.00% [48.1, 85.5]** | **73.81%** | **70.00%** | **68.75%** |

*Takeaway:* OpenFace 2.0 provides semantic facial action units (AUs), 3D gaze, and head pose that survive head rotations where geometry-only heuristics fail.

### Stage 2: Visual CNN Backbone Ablation
| Backbone | Feature Dim | Backbone Params | Validation Acc | Test Acc [95% Wilson CI] | Parameter Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 512 | $\approx 11.18\,\text{M}$ | 85.00% | 75.00% [53.1, 88.8] | Baseline |
| **EfficientNet-B0 (Proposed)** | **1,280** | **$\approx 4.01\,\text{M}$** | **75.00%** | **75.00% [53.1, 88.8]** | **−64.1% reduction** |

*Takeaway:* EfficientNet-B0 matches ResNet-18 test accuracy with a 64.1% reduction in backbone parameters while providing a richer 1,280-dim representation.

### Stage 3: Complementarity of Visual & Behavioral Streams
| Configuration | Modalities | Feature Dim | Test Acc [95% Wilson CI] | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stream 1 Only** | EfficientNet-B0 | 1,280 | 75.00% [53.1, 88.8] | 75.00% | 75.00% | 75.00% |
| **Stream 2 Only** | OpenFace 2.0 | 709 | 70.00% [48.1, 85.5] | 73.81% | 70.00% | 68.75% |
| **Dual-Stream Fused** | **EfficientNet-B0 + OpenFace** | **1,989** | **85.00% [64.0, 94.8]** | **85.00%** | **85.00%** | **85.00%** |

*Takeaway:* Fusing appearance and facial behavior yields a +10.0% and +15.0% accuracy improvement over either stream alone.

### Stage 4: Temporal Sequence Modeling on Fused Stream
| Architecture | Sequence Engine | Trainable Params | Val Acc (142 windows) | Test Acc [95% Wilson CI] | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Dual-Stream** | Mean Pooling | 265,378 | 86.62% | 85.00% [64.0, 94.8] | 85.00% | 85.00% | 85.00% |
| **Dual-Stream + BiLSTM** | 2-layer BiLSTM | 464,034 | 70.42% | 80.00% [58.4, 91.9] | 80.00% | 80.00% | 80.00% |
| **Dual-Stream + LSTM** | 2-layer LSTM | 529,826 | 76.06% | 90.00% [69.9, 97.2] | 90.00% | 90.00% | 90.00% |
| **Dual-Stream + Transformer** | **3-layer Pre-LN MHSA** | **663,074** | **83.80%** | **85.00% [64.0, 94.8]** | **85.71%** | **85.00%** | **85.35%** |

*Takeaway:* The Transformer achieves higher validation stability (83.80%) than the LSTM (76.06%) and BiLSTM (70.42%), and enables temporal attention weight inspectability.

---

## 💻 Quick Start & Installation

```bash
# 1. Clone the repository
git clone https://github.com/NethraK15/Student_Engagement_Detection.git
cd Student_Engagement_Detection

# 2. Set up virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 👥 Authors & Contributions

* **Nethra Kanagaraj**: Dual-stream pipeline design, OpenFace stream integration, Transformer architecture, soft-voting inference, and repository coordination.
* **S. Adwaitha**: Stage 1 ablation (Dlib EAR/MAR vs. MediaPipe FaceMesh geometry vs. OpenFace).
* **Srividya Manikandan**: Stage 2 ablation (EfficientNet-B0 vs. ResNet-18 spatial backbone profiling).
* **Aadhithya Bharathi A**: Stage 4 ablation (Temporal sequence modeling: Static, LSTM, BiLSTM baselines).
* **Dr. Manjusha R.**: Research supervisor, methodology review, and experimental design guidance.

---

## 📜 Citation

If you find this research work or codebase useful in your academic research, please cite our paper:

```bibtex
@article{kanagaraj2024dualstream,
  title={A Dual-Stream Temporal Transformer Framework for Student Engagement Detection in Classroom Videos},
  author={Kanagaraj, Nethra and Adwaitha, S. and Manikandan, Srividya and Bharathi A, Aadhithya and Manjusha, R.},
  institution={Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore, India},
  year={2024}
}
```

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE). See `LICENSE` for details.
