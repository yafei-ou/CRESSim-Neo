FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 ca-certificates curl \
        libvulkan1 mesa-vulkan-drivers vulkan-tools \
        libx11-6 libxcursor1 libxext6 libxi6 libxinerama1 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/micromamba/bin:${PATH}
WORKDIR /workspace
