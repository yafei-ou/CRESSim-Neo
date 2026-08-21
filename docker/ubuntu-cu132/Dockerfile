FROM nvidia/cuda:13.2.1-devel-ubuntu22.04

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
ENV VK_ICD_FILENAMES=/opt/cressim-neo/nvidia_egl_icd.json
ENV XDG_RUNTIME_DIR=/tmp/cressim-xdg-runtime
WORKDIR /workspace
