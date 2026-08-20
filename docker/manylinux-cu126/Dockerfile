FROM quay.io/pypa/manylinux_2_28_x86_64

# CUDA is a build-time dependency only. The resulting wheel deliberately keeps
# NVIDIA libraries external and resolves them through its Python bootstrap.
RUN dnf install -y dnf-plugins-core clang ninja-build git gcc-toolset-13-gcc-c++ \
    && dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y cuda-toolkit-12-6 \
    && test -x /usr/local/cuda-12.6/bin/nvcc \
    && dnf clean all

ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV CUDAToolkit_ROOT=${CUDA_HOME}
ENV CUDAHOSTCXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
