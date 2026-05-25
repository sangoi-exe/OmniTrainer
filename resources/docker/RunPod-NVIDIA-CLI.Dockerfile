#To build from the OneTrainer repository root, run
#    docker build -t <image-name> . -f resources/docker/RunPod-NVIDIA-CLI.Dockerfile
#    docker tag <image-name> <dockerhub-username>/<repository-name>:<tag>
#    docker push <dockerhub-username>/<repository-name>:<tag>

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#the base image is barely used, pytorch is the wrong version. However, by using
#a base image that is popular on RunPod, the base image likely is already available
#in the image cache of a pod, and no download is necessary

WORKDIR /OneTrainer
RUN apt-get update --yes \
 && apt-get install --yes --no-install-recommends curl ca-certificates git \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
COPY . /OneTrainer
RUN export OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt \
 && export OT_LAZY_UPDATES=true \
 && ./install.sh \
 && ./.uv/bin/uv cache clean
RUN apt-get update --yes \
 && apt-get install --yes --no-install-recommends \
      joe \
	  less \
	  gh \
	  iputils-ping \
	  nano \
	  nethogs \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN ./.uv/bin/uv pip install --python .venv/bin/python nvitop \
 && ./.uv/bin/uv cache clean
COPY resources/docker/RunPod-NVIDIA-CLI-start.sh.patch /start.sh.patch
RUN patch /start.sh < /start.sh.patch
