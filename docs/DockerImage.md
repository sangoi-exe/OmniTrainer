# Docker Image

A UI Dockerfile based on `nvidia/cuda:11.8.0-devel-ubuntu22.04` is provided at `resources/docker/NVIDIA-UI.Dockerfile`.

This image requires `nvidia-driver-525` and `nvidia-docker2` installed on the host.

> [!WARNING]
> The UI Dockerfile is an old NVIDIA example. It now uses the POSIX uv-managed bootstrap, but the base CUDA image is still dated and should be treated as an example, not validated training infrastructure.

## Building Image

Build using:

```
docker build -t myuser/onetrainer:latest -f resources/docker/NVIDIA-UI.Dockerfile .
```

## Running Image

This is an example

```
docker run \
  --gpus all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -i \
  --tty \
  --shm-size=512m \
  myuser/onetrainer:latest
```
