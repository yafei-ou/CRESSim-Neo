# C++ SDK

C++ is the primary engine interface. Build the SDK and examples from the
repository:

```bash
scripts/configure_builds.sh
cmake --build build/linux-release --parallel
cmake --install build/linux-release --component CXXSDK
```

Consume an installed SDK with CMake:

```cmake
find_package(CRESSimNeo CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE CRESSimNeo::engine)
```

The repository README describes system prerequisites, install components, and
asset-path configuration.
