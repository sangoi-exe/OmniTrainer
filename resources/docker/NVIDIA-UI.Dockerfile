# Note: as of April 2025, this Dockerfile is outdated and requires adjustments

# Inspiration for setup @ https://dev.to/ordigital/nvidia-525-cuda-118-python-310-pytorch-gpu-docker-image-1l4a
FROM docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
    nvidia-driver-525 \
 && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
RUN mkdir -p /OneTrainer
WORKDIR /OneTrainer

# Copy the current directory's contents to the container image
COPY . /OneTrainer
WORKDIR /OneTrainer

# Install uv-managed Python and CUDA requirements
RUN export OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt \
 && ./install.sh

# Run the training UI
CMD ["./start-ui.sh"]
