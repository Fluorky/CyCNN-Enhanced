# CyCNN — Docker usage

This folder contains ready-to-use Docker files for the CyCNN-Enhanced project.

## Files
- `Dockerfile` — GPU build (CUDA 12.1). Requires NVIDIA driver and `nvidia-container-toolkit`.
- `Dockerfile.cpu` — CPU-only build (no GPU required).
- `docker-compose.yml` — convenience commands to run either image with mounted volumes.

> The project expects a compiled local extension **CyConv2d** (provided in `cycnn-extension/`). Both Dockerfiles build it automatically after installing PyTorch.

## Build

From the root of the project (the folder that contains `cycnn/` and `cycnn-extension/`):

### GPU
```bash
docker build -t cycnn:gpu -f Dockerfile .
# or with compose
docker compose -f docker-compose.yml build cycnn-gpu
```

> Ensure you have the NVIDIA toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

### CPU
```bash
docker build -t cycnn:cpu -f Dockerfile.cpu .
# or with compose
docker compose -f docker-compose.yml build cycnn-cpu
```

## Run

### Quick help
```bash
# GPU
docker run --rm --gpus all -v $PWD/cycnn/logs:/app/cycnn/logs -v $PWD/cycnn/saves:/app/cycnn/saves -v $PWD/data:/app/cycnn/data cycnn:gpu python main.py --help

# CPU
docker run --rm -v $PWD/cycnn/logs:/app/cycnn/logs -v $PWD/cycnn/saves:/app/cycnn/saves -v $PWD/data:/app/cycnn/data cycnn:cpu python main.py --help
```

### Example runs
Train CyVGG on MNIST with linear polar transform:
```bash
docker run --rm --gpus all -v $PWD/cycnn/logs:/app/cycnn/logs -v $PWD/cycnn/saves:/app/cycnn/saves -v $PWD/data:/app/cycnn/data cycnn:gpu   python main.py --model cyvgg19 --train --dataset mnist --polar-transform linearpolar --batch-size 128 --num-epochs 10
```

Test an already trained model:
```bash
docker run --rm --gpus all -v $PWD/cycnn/logs:/app/cycnn/logs -v $PWD/cycnn/saves:/app/cycnn/saves -v $PWD/data:/app/cycnn/data cycnn:gpu   python main.py --model cyvgg19 --test --dataset mnist --polar-transform linearpolar
```

## Notes & Troubleshooting

- **CUDA/GPU**: The GPU image uses CUDA 12.1. If you have a different driver/toolkit version, adjust the `FROM nvidia/cuda:...` tag accordingly.
- **Extension build**: We install PyTorch first, *then* build `cycnn-extension` so headers are available. If your GPU has a very new/old compute capability, you can pass custom NVCC flags by setting `TORCH_CUDA_ARCH_LIST`, e.g.:
  ```bash
  docker run -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" ...
  ```
- **OpenCV**: We install `libgl1` and `libglib2.0-0` so `cv2` imports cleanly inside the container.
- **Data**: Place datasets under `./data` (mounted into `/app/cycnn/data`). Pretrained weights and logs are persisted via volume mounts.
- **Compose GPU**: Some Compose versions require explicit runtime flags; if `deploy:` is ignored, run with `docker compose run --gpus all cycnn-gpu`.
