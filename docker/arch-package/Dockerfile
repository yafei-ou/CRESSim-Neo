FROM archlinux:latest

RUN pacman -Syu --noconfirm --needed \
        base-devel clang cmake cuda git ninja python python-numpy python-pybind11 \
        python-pytorch vulkan-icd-loader \
    && useradd --create-home builder \
    && pacman -Scc --noconfirm

WORKDIR /build
