# ---- GPU-enabled build (CUDA 13 / PyTorch from NGC) ----
FROM nvcr.io/nvidia/pytorch:25.02-py3

# Build extension for both Ampere (3060/3070Ti) and Blackwell (5070)
ENV TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for building C++/CUDA extensions and runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for caching
COPY cycnn/requirements.txt /app/requirements.txt

# Install Python deps BUT do not override the PyTorch/CUDA stack from the base image
RUN python -m pip install --upgrade pip && \
    sed -e '/^torch==/d' \
        -e '/^torchvision==/d' \
        -e '/^triton==/d' \
        -e '/^nvidia-/d' \
        -e '/^CyConv2d==/d' \
        /app/requirements.txt > /app/requirements.noext.txt && \
    pip install --no-cache-dir -r /app/requirements.noext.txt

# Copy and build the CUDA extension after torch is present (from base image)
COPY cycnn-extension/ /app/cycnn-extension/
RUN cd /app/cycnn-extension && pip install . --no-build-isolation

# Finally copy the rest of the project
COPY . /app

ENV PYTHONPATH=/app
WORKDIR /app/cycnn

CMD ["python", "main.py", "--help"]
