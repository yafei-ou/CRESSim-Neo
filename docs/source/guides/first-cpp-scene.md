# First C++ scene

Create and initialize a `Runtime`, then author its `World` before preparing it
for GPU execution.

```cpp
cressim::neo::engine::Runtime runtime;
cressim::neo::engine::RuntimeConfig config{};
runtime.initialize(config);

auto& world = runtime.getWorld();
const auto entity = world.createEntity();
// Set transform, rigid body, colliders, renderers, cameras, and lights here.

runtime.prepare();
runtime.uploadWorld();
```

The source at `examples/physics/basic.cpp` is the complete reference. Continue
with {doc}`author-a-scene` and {doc}`step-a-simulation`.
