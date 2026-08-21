from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER_PATH = REPO_ROOT / "python/package/_cuda_runtime.py"


def load_loader(*, enabled: bool, provider: str = "AUTO") -> types.ModuleType:
    package = types.ModuleType("cressim_neo")
    package.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType("cressim_neo._cuda_runtime_config")
    config.CUDA_ENABLED = enabled
    config.CUDA_RUNTIME_PROVIDER = provider
    config.CUDA_RUNTIME_MAJOR = 12
    config.CUDA_RUNTIME_MINIMUM = 12060
    config.CUDA_REQUIRES_ULTRASOUND_LIBRARIES = False
    config.CUDA_CUFFT_SONAME = "libcufft.so.11"
    config.CUDA_CURAND_SONAME = "libcurand.so.10"
    old_modules = {name: sys.modules.get(name) for name in ("cressim_neo", "cressim_neo._cuda_runtime_config")}
    sys.modules["cressim_neo"] = package
    sys.modules["cressim_neo._cuda_runtime_config"] = config
    try:
        spec = importlib.util.spec_from_file_location("cressim_neo._cuda_runtime", LOADER_PATH)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in old_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class CudaRuntimeLoaderTests(unittest.TestCase):
    def test_disabled_build_does_not_attempt_a_load(self) -> None:
        loader = load_loader(enabled=False)
        with mock.patch.object(loader.ctypes, "CDLL") as cdll:
            loader.ensure_cuda_runtime()
        cdll.assert_not_called()
        self.assertEqual(
            loader.get_cuda_runtime_diagnostics(),
            {
                "cuda_enabled": False,
                "provider": "AUTO",
                "required_major": 12,
                "minimum_runtime_version": 12060,
                "ultrasound_libraries_required": False,
            },
        )

    def test_managed_directories_prefer_conda_prefix(self) -> None:
        loader = load_loader(enabled=True, provider="MANAGED")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conda_lib = root / "conda/lib"
            pip_lib = root / "site/nvidia/cuda_runtime/lib"
            conda_lib.mkdir(parents=True)
            pip_lib.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"CONDA_PREFIX": str(root / "conda")}, clear=False), \
                 mock.patch.object(loader.sys, "path", [str(root / "site")]):
                directories = loader._managed_directories()
        self.assertEqual(directories[:2], [conda_lib, pip_lib])

    def test_missing_runtime_has_actionable_error(self) -> None:
        loader = load_loader(enabled=True, provider="SYSTEM")
        with mock.patch.object(loader.ctypes, "CDLL", side_effect=OSError("not found")):
            with self.assertRaisesRegex(RuntimeError, "libcudart.so.12"):
                loader.ensure_cuda_runtime()


if __name__ == "__main__":
    unittest.main()
