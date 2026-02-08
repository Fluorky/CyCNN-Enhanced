# CyCNN - Setup Guide

## System & Hardware (Host)

- **CPU:** AMD Ryzen 7 5700X
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB) / RTX 3070 Ti (8 GB)
- **OS:** Ubuntu 24.04.2 LTS on Windows 10 and Windows 11 (WSL2)
- **NVIDIA Driver:** 581.57 (`nvidia-smi`)
- **Reported CUDA Version:** 13.0
- **Python (recommended):** 3.10–3.11
- **Docker (optional):** 24.x+ with NVIDIA Container Toolkit

Note: Even if `nvidia-smi` reports “CUDA Version: 13.0”, the drivers are backward compatible - PyTorch wheels for cu12.x will work properly.

---

## Quick Start (Docker) - Recommended

The repository already includes:
- `Dockerfile` - GPU build (CUDA 12.1)
- `Dockerfile.cpu` - CPU-only build
- `docker-compose.yml` - for convenient multi-service setup
- `README-docker.md` - detailed container instructions

### 1. Install NVIDIA Container Toolkit

On Ubuntu 24.04 inside WSL2:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |   sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure
sudo systemctl restart docker
```

Check GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

### 2. Build Images

From the project root (where `cycnn/` and `cycnn-extension/` are located):

```bash
# GPU build
docker build -t cycnn:gpu -f Dockerfile .

# CPU build
docker build -t cycnn:cpu -f Dockerfile.cpu .
```

or via Compose:
```bash
docker compose build cycnn-gpu
docker compose build cycnn-cpu
```

---

### 3. Run Containers

GPU (help menu):
```bash
docker run --rm --gpus all   -v $PWD/cycnn/logs:/app/cycnn/logs   -v $PWD/cycnn/saves:/app/cycnn/saves   -v $PWD/data:/app/cycnn/data   cycnn:gpu python main.py --help
```

CPU (help menu):
```bash
docker run --rm   -v $PWD/cycnn/logs:/app/cycnn/logs   -v $PWD/cycnn/saves:/app/cycnn/saves   -v $PWD/data:/app/cycnn/data   cycnn:cpu python main.py --help
```

Example - train CyVGG19 on MNIST with LinearPolar transform:
```bash
docker run --rm --gpus all   -v $PWD/cycnn/logs:/app/cycnn/logs   -v $PWD/cycnn/saves:/app/cycnn/saves   -v $PWD/data:/app/cycnn/data   cycnn:gpu   python main.py --model cyvgg19 --train --dataset mnist                  --polar-transform linearpolar --batch-size 128 --num-epochs 10
```

For Ampere GPUs (RTX 3060 / 3070 Ti = `sm_86`), you can set:
```bash
-e TORCH_CUDA_ARCH_LIST="8.6"
```

---

## Native Installation (without Docker)

Works on Ubuntu 24.04 WSL2 with NVIDIA GPU passthrough enabled.
Make sure `nvidia-smi` runs correctly inside Ubuntu.

### 1. System dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip     build-essential git libgl1 libglib2.0-0
```

### 2. Virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install PyTorch + dependencies

Compatible CUDA 12.x wheels:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121     torch torchvision torchaudio
pip install -r cycnn/requirements.txt
```

(Skip any `CyConv2d==...` entry in requirements if present.)

### 4. Build the CyCNN extension

```bash
cd cycnn-extension
export TORCH_CUDA_ARCH_LIST="8.6"   # optional, for Ampere
python setup.py install
cd ..
```

### 5. Verify installation

```python
python - <<'PY'
import torch, importlib
print("CUDA available:", torch.cuda.is_available())
m = importlib.import_module("CyConv2d_cuda")
print("CyConv2d_cuda OK:", hasattr(m, "forward"))
PY
```

---

## Project Structure

```text
cycnn/
├── cycnn/
│   ├── data/                   (training data)
│   ├── logs/                   (training logs)
│   ├── models/                 (CNN and CyCNN model definitions)
│   │   ├── cyconvlayer.py
│   │   ├── cyresnet.py
│   │   ├── cyvgg.py
│   │   ├── getmodel.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── saves/                  (saved model checkpoints)
│   ├── main.py                 (main script for training/testing)
│   ├── data.py                 (dataset loading)
│   ├── image_transforms.py     (image transformation logic)
│   └── utils.py
├── cycnn-extension/            (CUDA extension for CyCNN)
│   ├── cycnn.cpp
│   ├── cycnn_cuda.cu
│   └── setup.py
└── README.md
```

---

## How to Run (Train / Test)

Main script: `cycnn/main.py`

```text
usage: main.py [-h] [--model MODEL] [--train] [--test]
               [--polar-transform POLAR_TRANSFORM]
               [--augmentation AUGMENTATION] [--data-dir DATA_DIR]
               [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS] [--lr LR]
               [--dataset DATASET] [--redirect]
               [--early-stop-epochs EARLY_STOP_EPOCHS] [--test-while-training]
```

Train CyVGG19 on MNIST (LinearPolar):
```bash
python main.py --train --model cyvgg19 --dataset mnist                --polar-transform linearpolar --batch-size 128 --num-epochs 10
```

Test a saved checkpoint:
```bash
python main.py --test --model cyvgg19 --dataset mnist                --polar-transform linearpolar
```

Results are stored in `cycnn/saves/` and logs in `cycnn/logs/`.

---

## Compatibility & Notes

- PyTorch/CUDA: cu12.x builds run fine with driver 581.57 (reported CUDA 13.0).
- GPU architecture: RTX 3060/3070 Ti = SM 8.6 → `TORCH_CUDA_ARCH_LIST="8.6"`.
- WSL2: If `nvidia-smi` fails, ensure GPU integration is enabled on Windows.
- OpenCV: Missing `libGL` or `libglib` causes `cv2` import errors - installed above.
- Memory: For 8 GB VRAM (3070 Ti), lower `--batch-size` (e.g., 64 or 32).
- Docker Compose (GPU): If the `deploy:` block is ignored, run with:
  ```bash
  docker compose run --gpus all cycnn-gpu ...
  ```

---

> **Fork Notice:**  
This project is based on a fork of the original repository - https://github.com/mcrl/CyCNN. 
It includes numerous fixes, improvements, and additional functionalities built on top of the original work.
