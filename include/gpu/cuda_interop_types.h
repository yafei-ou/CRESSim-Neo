#ifndef CRESSIM_NEO_GPU_CUDA_INTEROP_TYPES_H
#define CRESSIM_NEO_GPU_CUDA_INTEROP_TYPES_H

/// @file cuda_interop_types.h
/// @brief Platform-native shared memory primitive handle types for CUDA/graphics memory interop.

namespace cressim::neo::gpu::interop
{

/// @brief Platform-specific operating system handle for exported GPU memory sharing.
struct NativeHandle
{
#if defined(_WIN32)
    void *win32Handle = nullptr; ///< Windows NT shared handle (`HANDLE`).
#else
    int fd = -1;                 ///< POSIX file descriptor for Vulkan memory handle sharing (Opaque Fd).
#endif
};

} // namespace cressim::neo::gpu::interop

#endif // CRESSIM_NEO_GPU_CUDA_INTEROP_TYPES_H
