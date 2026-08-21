#!/usr/bin/env python3
"""Import smoke test for the no-CUDA/no-Ultrasound base wheel lane."""

from __future__ import annotations

import cressim_neo as neo


def main() -> None:
    diagnostics = neo.get_cuda_runtime_diagnostics()
    if diagnostics["cuda_enabled"]:
        raise RuntimeError(f"Base wheel unexpectedly enables CUDA: {diagnostics}")

    runtime = neo.Runtime()
    try:
        if not runtime.initialize(neo.RuntimeConfig()):
            raise RuntimeError("Base wheel runtime initialization failed.")
        info = runtime.get_info()
        if info.cuda_interop_supported or info.ultrasound_supported:
            raise RuntimeError("Base wheel unexpectedly includes CUDA interop or Ultrasound.")
        print({"cuda": diagnostics, "runtime": info})
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
