FROM quay.io/pypa/manylinux_2_34_x86_64

# The base release lane uses the same glibc floor and viewer build dependencies
# as the CUDA lane, but deliberately has no CUDA Toolkit or NVIDIA libraries.
RUN dnf install -y clang ninja-build git gcc-toolset-13-gcc-c++ \
        libX11-devel libXcursor-devel libXext-devel libXi-devel \
        libXinerama-devel libXrandr-devel \
    && dnf clean all
