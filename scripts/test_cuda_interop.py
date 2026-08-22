#!/usr/bin/env python3
"""GPU integration smoke test for a managed CUDA release wheel."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import torch

import cressim_neo as neo


CUDA_LANES = {"cu126": "12.6", "cu130": "13.0", "cu132": "13.2"}


def _check_library(path: str, function: str) -> None:
    library = ctypes.WinDLL(path) if sys.platform == "win32" else ctypes.CDLL(path)
    version = ctypes.c_int()
    get_version = getattr(library, function)
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    if get_version(ctypes.byref(version)) != 0 or version.value <= 0:
        raise RuntimeError(f"{function} failed for {path}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=CUDA_LANES, required=True)
    args = parser.parse_args()
    expected_cuda = CUDA_LANES[args.lane]

    if torch.version.cuda is None or not torch.version.cuda.startswith(expected_cuda):
        raise RuntimeError(f"Expected a CUDA {expected_cuda} PyTorch build, got {torch.version.cuda!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError(f"A CUDA GPU is required for the {args.lane} interop test.")

    diagnostics = neo.get_cuda_runtime_diagnostics()
    loaded = {Path(path).name: path for path in diagnostics["loaded_libraries"]}
    library_checks = (
        (("cufft64_", "cufftGetVersion"), ("curand64_", "curandGetVersion"))
        if sys.platform == "win32"
        else (("libcufft.so", "cufftGetVersion"), ("libcurand.so", "curandGetVersion"))
    )
    for prefix, function in library_checks:
        path = next((path for name, path in loaded.items() if name.startswith(prefix)), None)
        if path is None:
            raise RuntimeError(f"Managed runtime bootstrap did not load {prefix}.")
        _check_library(path, function)

    runtime = neo.Runtime()
    try:
        if not runtime.initialize(neo.RuntimeConfig()):
            raise RuntimeError("CRESSim runtime initialization failed.")
        info = runtime.get_info()
        if not info.cuda_interop_supported or not info.ultrasound_supported:
            raise RuntimeError("The installed wheel does not include CUDA interop and Ultrasound.")

        desc = neo.SharedBufferDesc()
        desc.debug_name = f"{args.lane}-smoke"
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
