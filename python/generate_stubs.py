"""Generate stubs for a freshly built extension module.

Windows Python uses restricted DLL search paths for extension modules.  Keep
the supplied handles alive while pybind11-stubgen imports the module.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--module-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dll-directory", action="append", default=[], type=Path)
    args = parser.parse_args()

    module_directory = args.module_directory.resolve()
    sys.path.insert(0, str(module_directory))

    dll_directories: list[object] = []
    if sys.platform == "win32":
        for directory in [module_directory, *args.dll_directory]:
            if directory.is_dir():
                dll_directories.append(os.add_dll_directory(str(directory.resolve())))

    # Keep dll_directories alive until pybind11-stubgen has imported the module.
    sys.argv = ["pybind11-stubgen", args.module, "-o", str(args.output)]
    runpy.run_module("pybind11_stubgen", run_name="__main__")


if __name__ == "__main__":
    main()
