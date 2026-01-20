FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="diiulio-MoroThesis"

# Zero interaction
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /workspace
ENV APP_PATH=/workspace

# Install general-purpose dependencies 
RUN apt-get update -y && \
    apt-get install -y curl git bash nano python3.11 python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# Copy requirements from the root (we will run build from project root)
COPY requirements.txt .

# Install your project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attn (optional but recommended in guide)
RUN VLLM_FLASH_ATTN_VERSION=2 MAX_JOBS=16 pip install flash-attn --no-build-isolation

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
