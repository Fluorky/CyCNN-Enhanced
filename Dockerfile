# ---- GPU-enabled build (CUDA 12.1) ----
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV TORCH_CUDA_ARCH_LIST="8.6"
# (RTX 3060 / 3070 Ti = Ampere, SM 8.6)
ENV DEBIAN_FRONTEND=noninteractive     PIP_DISABLE_PIP_VERSION_CHECK=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

# Copy only what's needed first to leverage Docker layer caching
COPY cycnn/requirements.txt /app/requirements.txt

# Install Python deps except the local CyConv2d (we'll build it from source)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir ninja \
 && sed -e '/^CyConv2d==/d' requirements.txt > requirements.noext.txt \
 && pip install --no-cache-dir -r requirements.noext.txt

# Now copy and build the CUDA extension after torch is present
COPY cycnn-extension/ /app/cycnn-extension/
RUN cd /app/cycnn-extension \
 && python setup.py install

# Finally copy the rest of the project
COPY . /app

# Helpful defaults
ENV PYTHONPATH=/app
WORKDIR /app/cycnn

# Default command prints help; override in docker run / compose
CMD ["python", "main.py", "--help"]
