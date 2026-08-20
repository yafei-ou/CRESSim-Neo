#!/usr/bin/env python3
"""GPU integration smoke test for the managed CUDA 12.6 wheel lane."""

from __future__ import annotations

import ctypes
import torch

import cressim_neo as neo


def main() -> None:
    if torch.version.cuda is None or not torch.version.cuda.startswith("12.6"):
        raise RuntimeError(f"Expected a CUDA 12.6 PyTorch build, got {torch.version.cuda!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the CUDA 12.6 interop test.")

    # Verify the Ultrasound CUDA dependencies selected by the package bootstrap.
    for soname, function in (("libcufft.so.11", "cufftGetVersion"), ("libcurand.so.10", "curandGetVersion")):
        library = ctypes.CDLL(soname)
        version = ctypes.c_int()
        get_version = getattr(library, function)
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        if get_version(ctypes.byref(version)) != 0 or version.value <= 0:
            raise RuntimeError(f"{function} failed for {soname}.")

    runtime = neo.Runtime()
    try:
        if not runtime.initialize(neo.RuntimeConfig()):
            raise RuntimeError("CRESSim runtime initialization failed.")
        info = runtime.get_info()
        if not info.cuda_interop_supported or not info.ultrasound_supported:
            raise RuntimeError("The installed wheel does not include CUDA interop and Ultrasound.")

        desc = neo.SharedBufferDesc()
        desc.debug_name = "cuda126-smoke"
        desc.element_stride_bytes = 4
        desc.element_count = 16
        desc.access = neo.SharedBufferAccess.ReadWrite
        desc.bind_flags = neo.SharedBufferBindFlags.ShaderResource | neo.SharedBufferBindFlags.UnorderedAccess
        handle = runtime.create_shared_buffer(desc)
        if not handle.is_valid():
            raise RuntimeError("Failed to create CUDA-exportable shared buffer.")

        tensor_desc = neo.SharedBufferTensorDesc()
        tensor_desc.shape = [16]
        tensor_desc.dtype_code = neo.SharedBufferTensorDTypeCode.Float
        tensor_desc.dtype_bits = 32
        tensor_desc.dtype_lanes = 1
        tensor = torch.utils.dlpack.from_dlpack(runtime.shared_buffer_to_dlpack(handle, tensor_desc))
        tensor.fill_(3.0)
        if not torch.all(tensor == 3.0):
            raise RuntimeError("Torch DLPack shared-buffer round trip failed.")
        if not runtime.sync_shared_buffer_from_cuda(handle):
            raise RuntimeError("Failed to synchronize the shared buffer from CUDA.")
        runtime.destroy_shared_buffer(handle)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
