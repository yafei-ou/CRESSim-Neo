# Testing and Code Quality

The supported CMake CI presets are lean, headless verification profiles. They
disable the viewer and examples and enable `BUILD_TESTING`.

## CTest presets

On Linux:

```bash
cmake --preset linux-ci
cmake --build --preset linux-ci --parallel
ctest --preset linux-ci
```

On macOS, with `VULKAN_SDK` set:

```bash
cmake --preset macos-ci
cmake --build --preset macos-ci --parallel
ctest --preset macos-ci
```

On Windows, run from a Visual Studio 2022 developer shell:

```powershell
cmake --preset windows-vs2022-ci
cmake --build --preset windows-vs2022-ci --parallel
ctest --preset windows-vs2022-ci
```

## Static analysis and formatting

Enable clang-tidy in any compatible CMake configuration with:

```bash
cmake --preset linux-debug -DCRESSIM_NEO_ENABLE_CLANG_TIDY=ON
cmake --build --preset linux-debug --parallel
```

The option warns and continues if `clang-tidy` is unavailable. C++ changes
must also follow the repository `.clang-format` rules. Release-wheel tests are
separate from these development tests; see {doc}`packaging`.
