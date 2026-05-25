#To build from the OneTrainer repository root, run
#    docker build -t <image-name> . -f resources/docker/Vast-NVIDIA-CLI.Dockerfile
#    docker tag <image-name> <dockerhub-username>/<repository-name>:<tag>
#    docker push <dockerhub-username>/<repository-name>:<tag>

FROM vastai/pytorch:cuda-12.8.1-auto

WORKDIR /OneTrainer
USER root
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
RUN mkdir /workspace && ln -snf /OneTrainer /workspace/OneTrainer
