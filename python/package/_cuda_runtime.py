"""Resolve the CUDA shared libraries before loading CRESSim native modules.

The native package deliberately keeps CUDA libraries outside the wheel.  This
module selects one coherent provider (system CUDA, an active Conda environment,
or pip-installed NVIDIA packages) and preloads it with global symbol visibility.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from ._cuda_runtime_config import (
    CUDA_ENABLED,
    CUDA_REQUIRES_ULTRASOUND_LIBRARIES,
    CUDA_CUFFT_SONAME,
    CUDA_CURAND_SONAME,
    CUDA_RUNTIME_MAJOR,
    CUDA_RUNTIME_MINIMUM,
    CUDA_RUNTIME_PROVIDER,
)


_LOADED = False


def _library_mode() -> int:
    return getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)


def _soname(name: str) -> str:
    if name == "cudart":
        return f"libcudart.so.{CUDA_RUNTIME_MAJOR}"
    if name == "cufft":
        return CUDA_CUFFT_SONAME
    if name == "curand":
        return CUDA_CURAND_SONAME
    raise ValueError(f"Unknown CUDA library '{name}'.")


def _managed_directories() -> list[Path]:
    directories: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        directories.append(Path(conda_prefix) / "lib")

    package_directories = (
        "cuda_runtime",
        "cufft",
        "curand",
        f"cu{CUDA_RUNTIME_MAJOR}",
    )
    for entry in map(Path, sys.path):
        if not entry.is_dir():
            continue
        nvidia_root = entry / "nvidia"
        for directory in package_directories:
            library_dir = nvidia_root / directory / "lib"
            if library_dir.is_dir():
                directories.append(library_dir)

    # Preserve deterministic precedence while avoiding duplicate probes.
    return list(dict.fromkeys(directories))


def _explicit_runtime() -> Path | None:
    configured = os.environ.get("CRESSIM_NEO_CUDA_RUNTIME_PATH")
    if not configured:
        return None
    candidate = Path(configured).expanduser()
    if candidate.is_dir():
        candidate /= _soname("cudart")
    return candidate


def _load_path(path: Path) -> ctypes.CDLL:
    return ctypes.CDLL(str(path), mode=_library_mode())


def _load_from_directories(name: str, directories: list[Path]) -> ctypes.CDLL | None:
    soname = _soname(name)
    for directory in directories:
        candidate = directory / soname
        if candidate.exists():
            return _load_path(candidate)
    return None


def _validate_cudart(library: ctypes.CDLL) -> None:
    version = ctypes.c_int()
    get_version = library.cudaRuntimeGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    if get_version(ctypes.byref(version)) != 0:
        raise RuntimeError("cudaRuntimeGetVersion failed while loading the CRESSim CUDA runtime.")
    if version.value // 1000 != CUDA_RUNTIME_MAJOR or version.value < CUDA_RUNTIME_MINIMUM:
        raise RuntimeError(
            "CRESSim requires CUDA runtime "
            f"{CUDA_RUNTIME_MAJOR} (minimum encoded version {CUDA_RUNTIME_MINIMUM}), "
            f"but loaded runtime version {version.value}."
        )


def _runtime_version(library: ctypes.CDLL, symbol: str) -> int | None:
    version = ctypes.c_int()
    try:
        get_version = getattr(library, symbol)
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        if get_version(ctypes.byref(version)) == 0:
            return version.value
    except AttributeError:
        pass
    return None


def _mapped_cuda_libraries() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    names = ("libcudart.so", "libcufft.so", "libcurand.so")
    paths: list[str] = []
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[-1].startswith("/"):
            continue
        path = fields[-1]
        if any(name in path for name in names):
            paths.append(path)
    return list(dict.fromkeys(paths))


def get_cuda_runtime_diagnostics() -> dict[str, object]:
    """Return the active CRESSim CUDA runtime contract and loaded library paths.

    This is intentionally Python data rather than a native binding so it is
    available before a detailed import failure and is useful in support logs.
    """

    result: dict[str, object] = {
        "cuda_enabled": CUDA_ENABLED,
        "provider": CUDA_RUNTIME_PROVIDER,
        "required_major": CUDA_RUNTIME_MAJOR,
        "minimum_runtime_version": CUDA_RUNTIME_MINIMUM,
        "ultrasound_libraries_required": CUDA_REQUIRES_ULTRASOUND_LIBRARIES,
    }
    if not CUDA_ENABLED:
        return result

    ensure_cuda_runtime()
    cudart = ctypes.CDLL(_soname("cudart"), mode=_library_mode())
    result["runtime_version"] = _runtime_version(cudart, "cudaRuntimeGetVersion")
    result["driver_version"] = _runtime_version(cudart, "cudaDriverGetVersion")
    result["loaded_libraries"] = _mapped_cuda_libraries()
    return result


def ensure_cuda_runtime() -> None:
    """Preload the CUDA libraries required by this native package once."""

    global _LOADED
    if _LOADED or not CUDA_ENABLED:
        return
    if sys.platform != "linux":
        raise RuntimeError(
            "This CRESSim CUDA build currently supports Linux only. Install a CPU build on this platform."
        )

    errors: list[str] = []
    cudart: ctypes.CDLL | None = None
    explicit = _explicit_runtime()
    if explicit is not None:
        try:
            cudart = _load_path(explicit)
        except OSError as exc:
            errors.append(f"{explicit}: {exc}")
    elif CUDA_RUNTIME_PROVIDER in {"MANAGED", "AUTO"}:
        try:
            cudart = _load_from_directories("cudart", _managed_directories())
        except OSError as exc:
            errors.append(str(exc))

    if cudart is None and CUDA_RUNTIME_PROVIDER in {"SYSTEM", "AUTO"}:
        try:
            cudart = ctypes.CDLL(_soname("cudart"), mode=_library_mode())
        except OSError as exc:
            errors.append(str(exc))

    if cudart is None:
        details = "\n  ".join(errors) if errors else "no matching managed or system runtime was found"
        raise RuntimeError(
            f"CRESSim requires {_soname('cudart')}. Activate the supported environment, "
            "set CRESSIM_NEO_CUDA_RUNTIME_PATH, or install a matching system CUDA runtime.\n  "
            f"{details}"
        )

    _validate_cudart(cudart)
    if CUDA_REQUIRES_ULTRASOUND_LIBRARIES:
        directories = _managed_directories()
        if explicit is not None:
            directories.insert(0, explicit.parent)
        for name in ("cufft", "curand"):
            try:
                library = _load_from_directories(name, directories)
                if library is None and CUDA_RUNTIME_PROVIDER in {"SYSTEM", "AUTO"}:
                    ctypes.CDLL(_soname(name), mode=_library_mode())
                elif library is None:
                    raise OSError(f"{_soname(name)} was not found in managed CUDA package paths")
            except OSError as exc:
                raise RuntimeError(
                    f"CRESSim Ultrasound requires {_soname(name)} from the selected CUDA runtime: {exc}"
                ) from exc
    _LOADED = True
