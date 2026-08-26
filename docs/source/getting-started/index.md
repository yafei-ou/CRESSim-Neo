# Getting Started

CRESSim-Neo is implemented as a standalone C++ simulation engine with a
high-level Python binding layer. The C++ runtime can execute independently of
Python; the Python module exposes the same runtime and world API for scripting,
task construction, and learning workflows.

Both interfaces use the same staged execution model: author a scene, prepare
and upload its GPU resources, then step physics, sensors, rendering, and
optional custom compute. A Vulkan-capable GPU and graphics driver are required
for the standard runtime. CUDA interoperability and ultrasound simulation are
optional build features.

Build CRESSim-Neo using the interface that suits your application, then create
and run a minimal scene. The {doc}`../guides/index` explains the engine model
and features in depth.

```{toctree}
:maxdepth: 1

license
build
first-scene
```
