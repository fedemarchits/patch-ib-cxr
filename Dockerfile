# Use a base image with PyTorch and CUDA pre-installed.
# This version matches the server requirements (usually Linux/x86).
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system utilities (git is needed to clone your repo later)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python libraries
# We add --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt


# TO RUN LATER: docker push fedemarchits/thesis-env:latest