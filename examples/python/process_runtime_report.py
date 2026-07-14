from __future__ import annotations

from pathlib import Path
import os
import sys


def _safe_torch_info() -> tuple[str, str, str]:
    try:
        import torch
    except ImportError:
        return ("<not installed>", "<not available>", "<not available>")

    torch_version = getattr(torch, "__version__", "<unknown>")
    torch_cuda = getattr(torch.version, "cuda", None) or "<not available>"
    torch_core_path = getattr(getattr(torch, "_C", None), "__file__", "<not available>")
    return (str(torch_version), str(torch_cuda), str(torch_core_path))


def _safe_cressim_core_path() -> str:
    try:
        import cressim_neo as neo
    except ImportError:
        return "<not imported>"

    core_module = getattr(neo, "_cressim_neo", None)
    return str(getattr(core_module, "__file__", "<not available>"))


def _collect_loaded_cuda_libraries() -> list[str]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return []

    prefixes = (
        "libcuda",
        "libcudart",
        "libcublas",
        "libcublasLt",
        "libcudnn",
        "libcufft",
        "libcurand",
        "libcusparse",
        "libcusparseLt",
        "libcufile",
        "libcupti",
        "libnvrtc",
        "libnccl",
        "libcru_interface",
    )
    loaded: list[str] = []
    seen: set[str] = set()
    for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        name = os.path.basename(path)
        if not name.startswith(prefixes):
            continue
        if path in seen:
            continue
        seen.add(path)
        loaded.append(path)
    return loaded


def print_process_runtime_report(header: str = "Process runtime report") -> None:
    torch_version, torch_cuda, torch_core_path = _safe_torch_info()
    cressim_core_path = _safe_cressim_core_path()
    print(header)
    print(f"python_executable: {sys.executable}")
    print(f"torch_version: {torch_version}")
    print(f"torch_cuda: {torch_cuda}")
    print(f"torch_core: {torch_core_path}")
    print(f"cressim_core: {cressim_core_path}")
    print("loaded_cuda_libraries:")
    loaded = _collect_loaded_cuda_libraries()
    if not loaded:
        print("  <none found>")
        return
    for path in loaded:
        print(f"  {path}")
