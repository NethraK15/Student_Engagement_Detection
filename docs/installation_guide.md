# 🛠️ Comprehensive Installation & Setup Guide

This guide provides step-by-step instructions to set up the **Student Engagement Detection** environment on Windows, Linux, and Google Colab.

---

## 1. System Requirements

- **Operating System**: Windows 10/11 (64-bit) or Ubuntu 20.04/22.04 LTS
- **Python**: Python 3.8 to 3.10
- **RAM**: Minimum 8 GB (16 GB recommended for sequence feature caching)
- **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA 11.8 or 12.1+

---

## 2. Step-by-Step Installation

### Step 2.1: Clone the Repository
```bash
git clone https://github.com/NethraK15/Student_Engagement_Detection.git
cd Student_Engagement_Detection
```

### Step 2.2: Set Up Virtual Environment
Creating an isolated virtual environment prevents dependency conflicts.

**On Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2.3: Upgrade Package Manager
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 2.4: Install PyTorch with GPU Support (CUDA)
Check your CUDA version (`nvidia-smi`). If you have an NVIDIA GPU, install the CUDA-enabled PyTorch build first:

* **For CUDA 11.8:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
* **For CUDA 12.1:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
* **For CPU Only:**
  ```bash
  pip install torch torchvision
  ```

### Step 2.5: Install Project Dependencies
Install the remaining packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 3. Verify Installation

Run the quick verification script to confirm PyTorch, OpenCV, and CUDA status:
```bash
python -c "import torch, cv2; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | OpenCV: {cv2.__version__}')"
```

If `CUDA Available: True` is displayed, your GPU acceleration is properly configured.

---

## 4. Google Colab Quick Setup

If running experiments on Google Colab:
```python
# In a Colab code cell:
!git clone https://github.com/NethraK15/Student_Engagement_Detection.git
%cd Student_Engagement_Detection
!pip install -r requirements.txt
```
