FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git clang cmake ninja-build build-essential \
        libvulkan1 libxext6 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

COPY docker/ubuntu-cu126/nvidia_egl_icd.json /opt/cressim-neo/nvidia_egl_icd.json
RUN install -d -m 700 /tmp/cressim-xdg-runtime

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/micromamba/bin:${PATH}
# NVIDIA's GLX Vulkan ICD requires an X11 path that a headless GPU container
# does not provide. The EGL ICD is an NVIDIA-supported Vulkan ICD for this
# environment; the host driver libraries remain mounted by the toolkit.
ENV VK_ICD_FILENAMES=/opt/cressim-neo/nvidia_egl_icd.json
ENV XDG_RUNTIME_DIR=/tmp/cressim-xdg-runtime
WORKDIR /workspace
