# 🎓 Multi-Modal Student Engagement Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-green.svg)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end deep learning framework for **automated student engagement detection** in online and physical classroom environments. This system fuses fine-grained facial behavioral cues (head pose, eye gaze, facial action units from OpenFace) with deep visual representations (EfficientNet embeddings), processed through temporal architectures and transfer learning to predict engagement levels in real time.

---

## 📌 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Methodology & Model Architectures](#-methodology--model-architectures)
  - [1. Multimodal Feature Extraction](#1-multimodal-feature-extraction)
  - [2. Baseline & Deep Temporal Models](#2-baseline--deep-temporal-models)
  - [3. Lightweight Temporal Models](#3-lightweight-temporal-models)
  - [4. DAiSEE Transfer Learning](#4-daisee-transfer-learning)
  - [5. Interactive Analytics Dashboard](#5-interactive-analytics-dashboard)
- [Installation & Setup](#-installation--setup)
- [Usage & Execution](#-usage--execution)
  - [Baseline Single-Frame & Temporal Models](#baseline-single-frame--temporal-models)
  - [Lightweight Temporal Experiment](#lightweight-temporal-experiment)
  - [DAiSEE Transfer Learning Pipeline](#daisee-transfer-learning-pipeline)
  - [Interactive Dashboard](#interactive-dashboard)
- [Evaluation & Results](#-evaluation--results)
- [Contributors](#-contributors)

---

## 🔬 Architecture Overview

```
[ Classroom / Video Stream ]
           │
     ┌─────┴─────────────────────────────────────┐
     ▼                                           ▼
[ OpenFace 2.0 Feature Extraction ]      [ EfficientNet-B0 Visual Backbone ]
  • Head Pose (Yaw, Pitch, Roll)           • Deep spatial representations
  • Eye Gaze Vectors & Angles              • Pre-trained convolutional features
  • Facial Action Units (AUs)              • Temporal sequence frames
     │                                           │
     └─────────────────────┬─────────────────────┘
                           ▼
             [ Feature Fusion & Normalization ]
             (16-frame Temporal Window Sequences)
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
      [ Attention Pool ] [ TCN Layer ] [ Bi-GRU / Transformer ]
             │             │             │
             └─────────────┬─────────────┘
                           ▼
          [ Multi-Class Engagement Classification ]
          • High Engagement
          • Partial / Nominal Engagement
          • Disengaged / Distracted
                           │
                           ▼
      [ Interactive Analytics & Web Dashboard ]
      • Real-time video processing & overlay
      • Engagement timeline & distribution
      • Temporal attention heatmaps & metrics
```

---

## 🚀 Key Features

- **Multi-Modal Feature Representation**: Combines geometric & behavioral cues (head pose angles, gaze direction, AU intensities) with CNN spatial feature representations.
- **Comprehensive Benchmarks**: Includes baseline single-frame CNNs, fused feature classifiers, multimodal LSTM-Transformers, and lightweight temporal heads.
- **Lightweight Temporal Modeling**: Low-parameter architectures (Attention Pooling, TCN, Single-layer GRU) designed to prevent overfitting on smaller sample sizes while capturing dynamic engagement shifts.
- **DAiSEE Transfer Learning**: Pre-training on the large-scale **DAiSEE** dataset (thousands of annotated video sequences) followed by domain-adaptation fine-tuning on custom classroom target clips.
- **Illumination Robustness**: Integrated CLAHE (Contrast Limited Adaptive Histogram Equalization) and face quality filters to handle varying lighting conditions.
- **Interactive Web Dashboard**: Gradio-powered interactive UI with real-time video analytics, frame-level attention breakdowns, engagement gauges, and downloadable session reports.

---

## 📁 Repository Structure

```
.
├── DPP_LSTM_Transformer_Train.py      # Baseline LSTM-Transformer training pipeline
├── DPP_LSTM_Transformer_Test.py       # Baseline inference & test script
│
├── Baseline/                          # Baseline single-frame & temporal benchmark
│   ├── preprocess/
│   │   ├── extract_frames.py          # Frame extraction from raw video
│   │   ├── extract_openface.py        # Facial landmark & AU extraction
│   │   ├── extract_efficientnet.py    # Spatial feature extraction via EfficientNet
│   │   └── fuse_features.py           # Feature fusion script for baseline
│   ├── config.py                      # Baseline experiment configuration
│   ├── data_loader.py                 # Baseline dataset loader
│   ├── models.py                      # Baseline models
│   ├── train.py                       # Baseline training loop
│   ├── evaluate.py                    # Baseline evaluation
│   ├── test_pipeline_local.py         # Baseline test pipeline
│   └── Colab_Instructions.md          # Baseline Colab guide
│
├── lightweight_temporal/              # Lightweight temporal modeling subproject
│   ├── preprocess/
│   │   ├── extract_frames.py          # Frame extraction from raw video
│   │   ├── extract_openface.py        # Facial landmark & AU extraction
│   │   ├── extract_efficientnet.py    # Spatial feature extraction via EfficientNet
│   │   └── fuse_sequence_features.py  # Synchronized temporal sequence fusion
│   ├── config_lightweight.py          # Hyperparameters & path configurations
│   ├── data_loader_lightweight.py     # PyTorch Dataset & DataLoader
│   ├── models_lightweight.py          # Attention Pooling, TCN, & GRU models
│   ├── train_lightweight.py           # Training routine with early stopping
│   ├── evaluate_lightweight.py        # Confusion matrix, ROC, & metric reports
│   ├── test_pipeline_lightweight.py   # Unit & forward pass pipeline tests
│   ├── Colab_Lightweight_Instructions.md # Step-by-step Google Colab execution guide
│   └── Lightweight_Temporal_Colab_Experiment.ipynb # Interactive notebook
│
├── daisee_transfer/                   # DAiSEE transfer learning & fine-tuning
│   ├── config_daisee.py               # Pre-training & fine-tuning configuration
│   ├── data_loader_daisee.py          # DAiSEE batch loader & dataset formatting
│   ├── models_transfer.py             # Transferable Transformer architecture
│   ├── pretrain_daisee.py             # Phase 1: Pre-training on DAiSEE benchmark
│   ├── finetune_target.py             # Phase 2: Domain-adaptation fine-tuning
│   ├── evaluate_transfer.py           # Transfer model performance evaluation
│   ├── test_pipeline_transfer.py      # Transfer pipeline verification script
│   ├── Colab_DAiSEE_Transfer_Instructions.md # Transfer learning execution guide
│   └── DAiSEE_Transfer_Colab_Experiment.ipynb # Transfer experiment notebook
│
├── dashboard/                         # Real-time interactive web dashboard
│   ├── dashboard_transformer.py       # Gradio analytics dashboard (v2)
│   ├── dashboard_transformer_v1.py    # Initial dashboard implementation (v1)
│   └── Dashboard_Colab_Guide.ipynb    # Notebook for hosting dashboard on Colab
│
├── requirements.txt                   # Project Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # Comprehensive documentation
```

---

## 🧠 Methodology & Model Architectures

### 1. Multimodal Feature Extraction
- **OpenFace Behavioral Descriptors**:
  - Head Pose Translation & Rotation ($R_x, R_y, R_z$)
  - 2D/3D Eye Gaze Vectors
  - 18 Facial Action Units (Presence and Intensity)
- **Deep Spatial Embeddings**:
  - Pre-trained EfficientNet-B0 backbone extracting dense visual feature representations per frame.
- **Temporal Alignment**:
  - Features are aligned into sequences of length $T=16$ frames with stride-based sampling.

### 2. Lightweight Temporal Models
To optimize computational efficiency and avoid overfitting on compact datasets:
- **Attention Pooling** (`--model attention_pool`): Uses a single learned query to dynamically weigh salient frames (e.g. distraction moments) with only ~30k parameters.
- **Temporal Convolutional Network (TCN)** (`--model tcn`): 1D dilated convolutions with BatchNorm and ReLU for local temporal receptive fields (~30k params).
- **Single-Layer GRU** (`--model gru`): Recurrent unit capturing chronological dependencies with low parameter overhead (~35k params).

### 3. DAiSEE Transfer Learning
- **Stage 1 (Pre-training)**: Train a 2-layer Multi-Head Self-Attention Transformer Encoder (`d_model=128`, `nhead=4`) on the DAiSEE dataset.
- **Stage 2 (Domain Adaptation)**: Freeze low-level representations, swap the classification head to target categories, and fine-tune on target classroom recordings.

---

## 💻 Installation & Setup

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NethraK15/Student_Engagement_Detection.git
   cd Student_Engagement_Detection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage & Execution

### Baseline Models
```bash
# Train baseline model
python Baseline/train.py

# Evaluate baseline
python Baseline/evaluate.py
```

### Lightweight Temporal Experiment

1. **Fuse sequence features**:
   ```bash
   python lightweight_temporal/preprocess/fuse_sequence_features.py
   ```

2. **Run pipeline test**:
   ```bash
   python lightweight_temporal/test_pipeline_lightweight.py
   ```

3. **Train a model variant**:
   ```bash
   # Train Attention Pooling Model
   python lightweight_temporal/train_lightweight.py --model attention_pool --epochs 50 --lr 0.001

   # Train TCN Model
   python lightweight_temporal/train_lightweight.py --model tcn --epochs 50 --lr 0.001

   # Train GRU Model
   python lightweight_temporal/train_lightweight.py --model gru --epochs 50 --lr 0.001
   ```

4. **Evaluate performance**:
   ```bash
   python lightweight_temporal/evaluate_lightweight.py --model attention_pool
   ```

### DAiSEE Transfer Learning Pipeline

1. **Pre-train on DAiSEE**:
   ```bash
   python daisee_transfer/pretrain_daisee.py --model transformer --epochs 30 --lr 0.0005
   ```

2. **Fine-tune on Target Dataset**:
   ```bash
   python daisee_transfer/finetune_target.py --model transformer --epochs 40 --lr 0.0001
   ```

3. **Evaluate Transfer Model**:
   ```bash
   python daisee_transfer/evaluate_transfer.py --model transformer
   ```

### Interactive Dashboard

Launch the Gradio dashboard for real-time video analysis and engagement tracking:

```bash
python dashboard/dashboard_transformer.py
```

---

## 📊 Evaluation & Results

The system evaluates student engagement across three primary states:
- **Engaged**: Focused eye gaze, active posture, relevant facial action unit activations.
- **Partially Engaged**: Nominal focus with minor gaze wandering or low-intensity engagement cues.
- **Disengaged**: Significant head turn, closed eyes/drowsiness, persistent off-screen gaze.

Performance is assessed using **Accuracy**, **Precision**, **Recall**, **Macro F1-Score**, and **ROC-AUC Curves**.

---

## 👥 Contributors

- **Team Srividya**
- Final Year Project (FYP) — Student Engagement Detection System
