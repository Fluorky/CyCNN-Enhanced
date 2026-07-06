# CyCNN - Setup Guide

## Tested / Available Hardware Environments

This repository is maintained and tested against local NVIDIA GPU environments.  
The primary supported path for CyCNN models is the GPU Docker image.

| Environment | CPU | System RAM | GPU | GPU VRAM | Notes |
|---|---:|---:|---|---------:|---|
| Main workstation | AMD Ryzen 9 5950X | 128 GB | NVIDIA GeForce RTX 5070 Ti | 16 GB | Blackwell / SM 12.0 |
| Secondary workstation | AMD Ryzen 7 5700X | 32 GB | NVIDIA GeForce RTX 3070 Ti | 8 GB | Ampere / SM 8.6 |
| Legacy laptop | Intel Core i7-11800H | 64 GB | NVIDIA GeForce RTX 3060 Laptop GPU | 6 GB | Ampere / SM 8.6; use small batches |

Recommended software stack:

- **OS:** Ubuntu 24.04.x LTS, including WSL2 setups
- **Python:** 3.10-3.12 depending on installation path
- **Docker:** recommended for GPU builds
- **GPU stack:** NVIDIA NGC PyTorch 25.02 image for the GPU Docker build
- **CUDA extension arch list:** `TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"`

Notes:

- RTX 3060 Laptop / RTX 3070 Ti target Ampere / SM 8.6.
- RTX 5070 Ti targets Blackwell / SM 12.0 and should use a CUDA 12.8+ capable PyTorch stack or the provided NGC PyTorch Docker image.
- The CUDA version shown by `nvidia-smi` is the maximum CUDA runtime supported by the installed driver, not necessarily the CUDA runtime used by PyTorch.
- CyCNN models use a CUDA-only `CyConv2d` extension. CPU-only environments are supported only for classic CNN baselines such as `vgg19` and `resnet20`.

---

## Quick Start (Docker) - Recommended

The repository includes:

- `Dockerfile` - full GPU build based on NVIDIA NGC PyTorch 25.02, suitable for CUDA 12.8 / Blackwell-capable environments
- `Dockerfile.light` - lightweight GPU build using the same CUDA/PyTorch stack, with large local artifacts excluded from the build context
- `Dockerfile.light.dockerignore` - ignore rules used only by the lightweight Docker build
- `Dockerfile.cpu` - CPU-only helper image for classic CNN baselines and utility checks
- `docker-compose.yml` - convenient multi-service setup
- `cycnn/README-docker.md` - detailed container instructions

The recommended runtime for CyCNN models is the GPU image. Use:

- `Dockerfile` when you intentionally want a full/self-contained image
- `Dockerfile.light` for faster development builds with mounted datasets, checkpoints, and logs
- `Dockerfile.cpu` only for CPU-compatible baseline models and basic utility checks

---

## 1. Install NVIDIA Container Toolkit

On Ubuntu 24.04 inside WSL2:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

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

## 2. Build Images

From the project root, where `cycnn/` and `cycnn-extension/` are located, build one of the available image variants.

### Full GPU image

Use the default `Dockerfile` when you want a self-contained image that may include the current repository contents, including local datasets, saved models, logs, or experiment artifacts if they are present in the build context.

```bash
docker build -t cycnn:gpu -f Dockerfile .
```

This is useful for archival or portable experiment images. The trade-off is that the Docker build context can become very large when datasets, checkpoints, logs, or generated outputs are stored inside the repository tree.

### Lightweight GPU image

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

### CPU helper image

The CPU image is not a CyCNN runtime.

It is intended for:

- classic CNN baselines such as `vgg19` and `resnet20`
- checking CLI/help output
- lightweight utility scripts
- dataset/log inspection workflows

It does not support CyCNN models such as `cyvgg19` or `cyresnet56`, because those models require the CUDA-only `CyConv2d` extension.

Build the CPU helper image with:

```bash
docker build -t cycnn:cpu -f Dockerfile.cpu .
```

Expected behavior:

```text
vgg19 / resnet20       -> supported in CPU image
cyvgg* / cyresnet*     -> require GPU image
```

### Docker Compose

Docker Compose builds the default GPU and CPU services:

```bash
docker compose build cycnn-gpu
docker compose build cycnn-cpu
```

---

## 3. Run Containers

### GPU help menu

Full GPU image:

```bash
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD/data:/app/cycnn/data" \
  -v "$PWD/cycnn/logs:/app/cycnn/logs" \
  -v "$PWD/cycnn/saves:/app/cycnn/saves" \
  cycnn:gpu \
  python main.py --help
```

Lightweight GPU image:

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

### CPU help menu

```bash
docker run --rm \
  -v "$PWD/data:/app/cycnn/data" \
  -v "$PWD/cycnn/logs:/app/cycnn/logs" \
  -v "$PWD/cycnn/saves:/app/cycnn/saves" \
  cycnn:cpu \
  python main.py --help
```

### Example - train CyVGG19 on MNIST

```bash
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD/data:/app/cycnn/data" \
  -v "$PWD/cycnn/logs:/app/cycnn/logs" \
  -v "$PWD/cycnn/saves:/app/cycnn/saves" \
  cycnn:gpu-light \
  python main.py \
    --model cyvgg19 \
    --train \
    --dataset mnist \
    --polar-transform linearpolar \
    --batch-size 16 \
    --num-epochs 1
```

For Ampere and Blackwell GPUs, the GPU Dockerfiles set:

```bash
TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"
```

---

## Native Installation without Docker

Native installation is mainly intended for NVIDIA GPU environments.  
For reproducible GPU builds, Docker is recommended.

### 1. System dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.11 \
  python3.11-venv \
  python3-pip \
  build-essential \
  git \
  libgl1 \
  libglib2.0-0
```

### 2. Virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

### 3. Install PyTorch and Python dependencies

Install CUDA 12.8 PyTorch wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining project dependencies while skipping packages that are provided by the selected PyTorch stack or built separately:

```bash
sed -e '/^torch==/d' \
    -e '/^torchvision==/d' \
    -e '/^torchaudio==/d' \
    -e '/^triton==/d' \
    -e '/^nvidia-/d' \
    -e '/^CyConv2d==/d' \
    cycnn/requirements.txt > /tmp/cycnn-requirements.noext.txt

pip install -r /tmp/cycnn-requirements.noext.txt
```

### 4. Build the CyCNN CUDA extension

```bash
cd cycnn-extension

export TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"

pip install . --no-build-isolation

cd ..
```

### 5. Verify GPU installation

```bash
python - <<'PY'
import torch
import importlib

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

m = importlib.import_module("CyConv2d_cuda")
print("CyConv2d_cuda OK:", hasattr(m, "forward"))
PY
```

---

## Project Structure

```text
CyCNN-Enhanced/
├── cycnn/
│   ├── data/                   # training data
│   ├── logs/                   # training logs
│   ├── models/                 # CNN and CyCNN model definitions
│   │   ├── cyconvlayer.py
│   │   ├── cyresnet.py
│   │   ├── cyvgg.py
│   │   ├── getmodel.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── saves/                  # saved model checkpoints
│   ├── main.py                 # main script for training/testing
│   ├── data.py                 # dataset loading
│   ├── image_transforms.py     # image transformation logic
│   └── utils.py
├── cycnn-extension/            # CUDA extension for CyCNN
│   ├── cycnn.cpp
│   ├── cycnn_cuda.cu
│   └── setup.py
├── Dockerfile                  # full GPU image
├── Dockerfile.light            # lightweight GPU image
├── Dockerfile.cpu              # CPU helper image
├── docker-compose.yml
└── README.md
```

---

## How to Run Train / Test

Main script:

```text
cycnn/main.py
```

Inside the Docker images, the working directory is already:

```text
/app/cycnn
```

For native runs from the repository root:

```bash
cd cycnn
```

To see all available options:

```bash
python main.py --help
```

### Train CyVGG19 on MNIST with LinearPolar

```bash
python main.py \
  --train \
  --model cyvgg19 \
  --dataset mnist \
  --polar-transform linearpolar \
  --batch-size 16 \
  --num-epochs 1
```

### Test a saved checkpoint

```bash
python main.py \
  --test \
  --model cyvgg19 \
  --dataset mnist \
  --polar-transform linearpolar \
  --model-path saves/mnist-cyvgg19-linearpolar.pt \
  --output-dir logs/main_smoke_test
```

Results are stored in:

```text
cycnn/saves/
cycnn/logs/
```

When running inside the container, those correspond to:

```text
/app/cycnn/saves/
/app/cycnn/logs/
```

---

## Smoke-Tested Checks

The lightweight GPU image has been smoke-tested on RTX 5070 Ti with NVIDIA NGC PyTorch 25.02 / CUDA 12.8 for:

- CUDA availability and GPU capability detection
- `get_model()` for `vgg19`, `resnet20`, `cyvgg19`, and `cyresnet56`
- lazy `CyConv2d` workspace allocation
- `CyConv2d` forward/backward
- `main.py --help`
- `main.py` import
- one-epoch MNIST CyVGG19 training smoke
- checkpoint test smoke with confusion matrix output

Observed GPU environment:

```text
torch: 2.7.0a0+ecf3bae40a.nv25.02
torch cuda: 12.8
device: NVIDIA GeForce RTX 5070 Ti
capability: (12, 0)
arch list includes: sm_86, sm_120, compute_120
```

The CPU helper image has been smoke-tested for classic CNN baselines:

- `vgg19`
- `resnet20`

CyCNN models intentionally require the GPU image.

Expected CPU behavior:

```text
vgg19 / resnet20       -> OK
cyvgg19 / cyresnet56   -> clear RuntimeError explaining that the GPU image is required
```

---

## Compatibility and Notes

- **Primary CyCNN runtime:** GPU Docker image.
- **CPU image scope:** helper/baseline-only image. It does not support CyCNN models.
- **PyTorch/CUDA:** use CUDA 12.8+ PyTorch builds or the provided NGC PyTorch image for RTX 5070 Ti / Blackwell.
- **GPU architecture:** RTX 3060 Laptop / RTX 3070 Ti = SM 8.6; RTX 5070 Ti / Blackwell = SM 12.0.
- **CUDA arch list:** `TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"`.
- **WSL2:** if `nvidia-smi` fails, ensure GPU integration is enabled on Windows.
- **OpenCV:** missing `libGL` or `libglib` can cause `cv2` import errors. The Dockerfiles install the required system libraries.
- **System RAM:** 32 GB is sufficient for normal MNIST/SVHN-style experiments; the 128 GB workstation is useful for larger datasets, parallel runs, and heavier Docker workflows.
- **GPU memory:** for 6-8 GB VRAM GPUs such as RTX 3060 Laptop or RTX 3070 Ti, lower `--batch-size` when needed, for example `16` or `8`.
- **CyConv2d workspace:** CyCNN models allocate a large CUDA workspace lazily on the first forward pass. Importing `CyConv2d` should not allocate the workspace, but running a CyCNN model will require sufficient free VRAM.
- **Docker Compose GPU:** if the `deploy:` block is ignored by your Docker Compose setup, run with:
  ```bash
  docker compose run --gpus all cycnn-gpu ...
  ```

---

## Fork Notice

This project is based on a fork of the original repository:

```text
https://github.com/mcrl/CyCNN
```

It includes fixes, Docker updates, runtime compatibility improvements, and additional functionality built on top of the original work.
