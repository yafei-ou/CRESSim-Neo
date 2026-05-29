#ifndef CRESSIM_NEO_GPU_CUDA_INTEROP_TYPES_H
#define CRESSIM_NEO_GPU_CUDA_INTEROP_TYPES_H

namespace cressim::neo::gpu::interop
{

struct NativeHandle
{
#if defined(_WIN32)
    void *win32Handle = nullptr;
#else
    int fd = -1;
#endif
};

} // namespace cressim::neo::gpu::interop

#endif // CRESSIM_NEO_GPU_CUDA_INTEROP_TYPES_H
