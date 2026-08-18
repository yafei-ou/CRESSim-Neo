from __future__ import annotations

import gc

try:
    import torch
except ImportError as exc:
    raise RuntimeError("cressim_neo Torch env support requires PyTorch to be installed.") from exc

from . import _cressim_neo as neo


def make_tensor(
    runtime: neo.Runtime,
    handle: neo.SharedBufferHandle,
    shape: list[int],
    dtype_code: neo.SharedBufferTensorDTypeCode,
    *,
    dtype_bits: int = 32,
    dtype_lanes: int = 1,
) -> "torch.Tensor":
    desc = neo.SharedBufferTensorDesc()
    desc.shape = shape
    desc.dtype_code = dtype_code
    desc.dtype_bits = dtype_bits
    desc.dtype_lanes = dtype_lanes
    return torch.utils.dlpack.from_dlpack(runtime.shared_buffer_to_dlpack(handle, desc))


class TorchStagedVectorEnvBase:
    def __init__(self, env_count: int, *, delta_seconds: float = 1.0 / 60.0) -> None:
        self.env_count = env_count
        self._frame = neo.FrameContext()
        self._frame.delta_seconds = delta_seconds
        self._frame.frame_index = 0
        self._frame.time_seconds = 0.0
        self._shared_handles: list[neo.SharedBufferHandle] = []
        self._custom_pass_handles: list[neo.CustomComputePassHandle] = []

    def _register_shared_buffer(
        self,
        runtime: neo.Runtime,
        name: str,
        count: int,
        dtype_code: neo.SharedBufferTensorDTypeCode,
        *,
        element_stride_bytes: int = 4,
        shape: list[int] | None = None,
    ) -> tuple[neo.SharedBufferHandle, "torch.Tensor"]:
        desc = neo.SharedBufferDesc()
        desc.debug_name = name
        desc.element_stride_bytes = element_stride_bytes
        desc.element_count = count
        desc.access = neo.SharedBufferAccess.ReadWrite
        desc.bind_flags = (
            neo.SharedBufferBindFlags.ShaderResource | neo.SharedBufferBindFlags.UnorderedAccess
        )
        handle = runtime.create_shared_buffer(desc)
        if not handle.is_valid():
            raise RuntimeError(f"Failed to create shared buffer '{name}'.")
        self._shared_handles.append(handle)
        tensor = make_tensor(runtime, handle, shape or [count], dtype_code)
        return handle, tensor

    def _register_custom_pass(self, runtime: neo.Runtime, desc: neo.CustomComputePassDesc) -> neo.CustomComputePassHandle:
        handle = runtime.create_custom_compute_pass(desc)
        if not handle.is_valid():
            raise RuntimeError(f"Failed to create custom compute pass '{desc.debug_name}'.")
        self._custom_pass_handles.append(handle)
        return handle

    def _sync_from_cuda(self, runtime: neo.Runtime, handles: tuple[neo.SharedBufferHandle, ...] | list[neo.SharedBufferHandle]) -> None:
        for handle in handles:
            if not runtime.sync_shared_buffer_from_cuda(handle):
                raise RuntimeError(f"Failed to synchronize shared buffer {handle.id} from CUDA.")

    def _sync_to_cuda(
        self,
        runtime: neo.Runtime,
        handles: tuple[neo.SharedBufferHandle, ...] | list[neo.SharedBufferHandle],
        *,
        device: "torch.device | None" = None,
    ) -> None:
        for handle in handles:
            if not runtime.sync_shared_buffer_to_cuda(handle):
                raise RuntimeError(f"Failed to synchronize shared buffer {handle.id} to CUDA.")
        if device is not None:
            torch.cuda.synchronize(device=device)

    def _end_frame(self, runtime: neo.Runtime, *, advance: bool) -> None:
        runtime.end_frame(self._frame)
        if advance:
            self._frame.frame_index += 1
            self._frame.time_seconds += self._frame.delta_seconds

    def close_runtime(self, runtime: neo.Runtime | None) -> None:
        if runtime is None:
            return
        # Tensors created through DLPack retain CUDA imports of the runtime's
        # shared Vulkan buffers.  Drop those tensors before the runtime and its
        # CUDA interop objects are destroyed; otherwise a later collection can
        # release an import whose owning Vulkan device no longer exists.
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except RuntimeError:
            # Preserve close() as best-effort cleanup after a prior CUDA error.
            pass
        for attribute_name, value in tuple(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, attribute_name, None)
        gc.collect()
        for handle in self._custom_pass_handles:
            if handle.is_valid():
                runtime.destroy_custom_compute_pass(handle)
        for handle in self._shared_handles:
            if handle.is_valid():
                runtime.destroy_shared_buffer(handle)
        runtime.shutdown()
        self._custom_pass_handles.clear()
        self._shared_handles.clear()
