# CyCNN - Setup Guide

## Tested Hardware / Environments

This repository has been tested on the following local GPU environments:

| Environment | CPU | System RAM | GPU | GPU VRAM | Notes |
|---|---:|---:|---|---------:|---|
| Secondary workstation | AMD Ryzen 7 5700X | 32 GB | NVIDIA GeForce RTX 3070 Ti | 8 GB | Ampere / SM 8.6 |
| Main workstation | AMD Ryzen 9 5950X | 128 GB | NVIDIA GeForce RTX 5070 Ti | 16 GB | Blackwell / SM 12.0 |

Recommended software stack:

- **OS:** Ubuntu 24.04.x LTS, including WSL2 setups
- **Python:** 3.10–3.12 depending on installation path
- **Docker:** recommended for GPU builds
- **GPU stack:** NVIDIA NGC PyTorch 25.02 image for the GPU Docker build
- **CUDA extension arch list:** `TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"`

Notes:

- The RTX 3070 Ti setup targets Ampere / SM 8.6.
- The RTX 5070 Ti setup targets Blackwell / SM 12.0 and should use a CUDA 12.8+ capable PyTorch stack or the provided NGC PyTorch Docker image.
- The CUDA version shown by `nvidia-smi` is the maximum CUDA runtime supported by the installed driver, not necessarily the CUDA runtime used by PyTorch.

---

## Quick Start (Docker) - Recommended

The repository already includes:
- `Dockerfile` - full GPU build based on NVIDIA NGC PyTorch 25.02, suitable for CUDA 12.8 / Blackwell-capable environments
- `Dockerfile.light` - lightweight GPU build using the same CUDA/PyTorch stack, with large local artifacts excluded from the build context
- `Dockerfile.light.dockerignore` - ignore rules used only by the lightweight Docker build
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
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.02-py3 nvidia-smi
```

---

### 2. Build Images

From the project root (where `cycnn/` and `cycnn-extension/` are located), you can build either a full image or a lightweight image.

#### Full GPU image

Use the default `Dockerfile` when you want a self-contained image that may include the current repository contents, including local datasets, saved models, logs, or experiment artifacts if they are present in the build context.

```bash
docker build -t cycnn:gpu -f Dockerfile .
```

This is useful for archival or portable experiment images. The trade-off is that the Docker build context can become very large when datasets, checkpoints, logs, or generated outputs are stored inside the repository tree.

#### Lightweight GPU image

Use `Dockerfile.light` for faster development builds. This variant uses the same CUDA/PyTorch stack as the full GPU image, but large local artifacts are excluded from the build context by `Dockerfile.light.dockerignore`.

```bash
docker build -t cycnn:gpu-light -f Dockerfile.light .
```

For lightweight images, mount datasets, logs, and checkpoints at runtime:

```bash
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD/data:/app/cycnn/data" \
  -v "$PWD/cycnn/logs:/app/cycnn/logs" \
  -v "$PWD/cycnn/saves:/app/cycnn/saves" \
  cycnn:gpu-light \
  python main.py --help
```

Use the full image when you intentionally want local artifacts baked into the image. Use the lightweight image when you want faster rebuilds and runtime-mounted datasets/checkpoints.

#### CPU image

For CPU-only usage:

```bash
docker build -t cycnn:cpu -f Dockerfile.cpu .
```

#### Docker Compose

Docker Compose builds the default GPU and CPU services:

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

For Ampere and Blackwell GPUs, the Dockerfile already sets:
```bash
-e TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r cycnn/requirements.txt
```

(Skip any `CyConv2d==...` entry in requirements if present.)

### 4. Build the CyCNN extension

```bash
cd cycnn-extension
export TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"   # Ampere + RTX 50xx / Blackwell
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

- PyTorch/CUDA: use CUDA 12.8+ PyTorch builds or the provided NGC PyTorch image for RTX 5070 Ti / Blackwell.
- GPU architecture: RTX 3060/3070 Ti = SM 8.6; RTX 5070 Ti / Blackwell = SM 12.0 → `TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"`.
- WSL2: If `nvidia-smi` fails, ensure GPU integration is enabled on Windows.
- OpenCV: Missing `libGL` or `libglib` causes `cv2` import errors - installed above.
- Memory: 32 GB system RAM is sufficient for normal MNIST/SVHN-style experiments; the 128 GB workstation is recommended for larger datasets, many parallel runs, or heavier Docker workflows.
- GPU memory: For 8 GB VRAM GPUs such as RTX 3070 Ti, lower `--batch-size` when needed, e.g. 64 or 32.
- Docker Compose (GPU): If the `deploy:` block is ignored, run with:
  ```bash
  docker compose run --gpus all cycnn-gpu ...
  ```

---

> **Fork Notice:**  
This project is based on a fork of the original repository - https://github.com/mcrl/CyCNN. 
It includes numerous fixes, improvements, and additional functionalities built on top of the original work.
