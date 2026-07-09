# CyCNN — Docker Usage

This document describes the Docker images provided for the CyCNN-Enhanced project.

## Image Types

The project provides three Docker image variants:

| Image | Dockerfile | Purpose |
|---|---|---|
| Full GPU image | `Dockerfile` | Primary CyCNN runtime. Builds the CUDA extension and may include local datasets/checkpoints/logs in the image context. |
| Lightweight GPU image | `Dockerfile.light` | Primary development image. Builds the CUDA extension but excludes large local artifacts from the build context. |
| CPU helper image | `Dockerfile.cpu` | CPU-only helper image for classic CNN baselines and utility checks. Not a CyCNN runtime. |

## Files

From the project root:

- `Dockerfile` — full GPU build based on NVIDIA NGC PyTorch 25.02, suitable for CUDA 12.8 / Blackwell-capable environments, including RTX 5070 Ti.
- `Dockerfile.light` — lightweight GPU build using the same CUDA/PyTorch stack as the full GPU image.
- `Dockerfile.light.dockerignore` — ignore rules used only by the lightweight build.
- `Dockerfile.cpu` — CPU-only helper build for classic CNN baselines such as `vgg19` and `resnet20`.
- `docker-compose.yml` — convenience services for GPU, lightweight GPU, and CPU images.

## Important Runtime Scope

CyCNN models such as `cyvgg19` and `cyresnet56` require the CUDA-only `CyConv2d` extension from `cycnn-extension/`.

Therefore:

```text
cyvgg* / cyresnet*     -> GPU image required
vgg* / resnet*         -> CPU image supported
```

The GPU images build the `CyConv2d` CUDA extension automatically after PyTorch is installed.

The CPU image intentionally does not build the CUDA extension. It is intended for:

- classic CNN baselines
- CLI/help checks
- dataset/log utility workflows
- lightweight CPU sanity checks

It is not intended to run CyCNN models.

---

## Build

Run all build commands from the repository root, where `cycnn/` and `cycnn-extension/` are located.

### Full GPU Image

Use the full GPU image when you want a self-contained image that may include local datasets, saved models, logs, or experiment artifacts if they are present in the build context.

```bash
docker build -t cycnn:gpu -f Dockerfile .
```

Compose equivalent:

```bash
docker compose -f docker-compose.yml build cycnn-gpu
```

### Lightweight GPU Image

Use the lightweight GPU image for faster development builds.

```bash
docker build -t cycnn:gpu-light -f Dockerfile.light .
```

This build uses `Dockerfile.light.dockerignore` to exclude large local artifacts such as datasets, checkpoints, logs, NumPy arrays, archives, and experiment outputs from the Docker build context.

Compose equivalent:

```bash
docker compose -f docker-compose.yml build cycnn-gpu-light
```

For lightweight images, mount datasets, checkpoints, and logs at runtime:

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

### CPU Helper Image

Use the CPU image only for classic CNN baselines and helper workflows.

```bash
docker build -t cycnn:cpu -f Dockerfile.cpu .
```

Compose equivalent:

```bash
docker compose -f docker-compose.yml build cycnn-cpu
```

Expected behavior:

```text
vgg19 / resnet20       -> supported
cyvgg19 / cyresnet56   -> clear RuntimeError; GPU image required
```

---

## Run

### GPU Help Menu

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

### CPU Help Menu

```bash
docker run --rm \
  -v "$PWD/data:/app/cycnn/data" \
  -v "$PWD/cycnn/logs:/app/cycnn/logs" \
  -v "$PWD/cycnn/saves:/app/cycnn/saves" \
  cycnn:cpu \
  python main.py --help
```

---

## Example Runs

### Train CyVGG19 on MNIST

Use a GPU image for CyCNN models:

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

### Test a Saved CyVGG19 Checkpoint

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
    --test \
    --dataset mnist \
    --polar-transform linearpolar \
    --model-path saves/mnist-cyvgg19-linearpolar.pt \
    --output-dir logs/main_smoke_test
```

### Train a Classic CPU Baseline

Use the CPU image only for non-CyCNN baseline models:

```bash
docker run --rm \
  -v "$PWD/data:/app/cycnn/data" \
  -v "$PWD/cycnn/logs:/app/cycnn/logs" \
  -v "$PWD/cycnn/saves:/app/cycnn/saves" \
  cycnn:cpu \
  python main.py \
    --model resnet20 \
    --train \
    --dataset mnist \
    --batch-size 16 \
    --num-epochs 1
```

---

## Smoke Checks

### GPU Smoke Check

```bash
docker run --rm -i --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  cycnn:gpu-light python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())

assert torch.cuda.is_available()
PY
```

### Model Registry Smoke Check

GPU image should support both classic CNN and CyCNN models:

```bash
docker run --rm -i --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  cycnn:gpu-light python - <<'PY'
from models.getmodel import get_model

for name in ["vgg19", "resnet20", "cyvgg19", "cyresnet56"]:
    model = get_model(name, dataset="mnist")
    print("OK", name, type(model).__name__)
PY
```

CPU image should support only classic CNN models and reject CyCNN models with a clear error:

```bash
docker run --rm -i cycnn:cpu python - <<'PY'
from models.getmodel import get_model

for name in ["vgg19", "resnet20", "cyvgg19", "cyresnet56"]:
    try:
        model = get_model(name, dataset="mnist")
        print("OK", name, type(model).__name__)
    except Exception as e:
        print("FAIL", name, repr(e))
PY
```

Expected CPU result:

```text
OK vgg19 VGG
OK resnet20 ResNet
FAIL cyvgg19 RuntimeError(...)
FAIL cyresnet56 RuntimeError(...)
```

---

## Notes and Troubleshooting

- **CUDA/GPU:** The GPU images use NVIDIA's NGC PyTorch 25.02 base image. This is the recommended path for RTX 5070 Ti / Blackwell because it provides a CUDA 12.8-capable PyTorch stack.
- **CUDA architecture list:** The GPU Dockerfiles set `TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"` for Ampere and Blackwell compatibility.
- **Extension build:** GPU images install PyTorch first, then build `cycnn-extension` so PyTorch extension headers are available.
- **CPU image:** The CPU image intentionally skips the CUDA extension. Use it only for classic CNN baselines or helper workflows.
- **OpenCV:** The Dockerfiles install `libgl1` and `libglib2.0-0` so `cv2` imports cleanly inside the container.
- **Data:** Place datasets under `./data`, mounted into `/app/cycnn/data`.
- **Logs and checkpoints:** Persist logs and saved models with volume mounts to `/app/cycnn/logs` and `/app/cycnn/saves`.
- **GPU memory:** For 6-8 GB VRAM GPUs such as RTX 3060 Laptop or RTX 3070 Ti, lower `--batch-size` when needed, for example `16` or `8`.
- **CyConv2d workspace:** CyCNN models allocate a large CUDA workspace lazily on the first forward pass. Importing `CyConv2d` should not allocate the workspace, but running a CyCNN model requires sufficient free VRAM.
- **NVIDIA Container Toolkit:** GPU containers require NVIDIA Container Toolkit on the host.
- **Docker Compose GPU:** Some Compose versions require explicit runtime flags. If `deploy:` is ignored, run:
  ```bash
  docker compose run --gpus all cycnn-gpu
  ```
