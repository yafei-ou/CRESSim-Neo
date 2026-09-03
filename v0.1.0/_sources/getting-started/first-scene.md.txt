# First Scene

The runtime has two phases: scene authoring and frame stepping. First create
the runtime and author a world. Then prepare and upload the GPU layout. Once
initialized, step physics, simulation sensors, visual sensors, and any
task-specific compute passes as needed.

::::{tab-set}

:::{tab-item} C++

```cpp
#include "common/frame_context.h"
#include "engine/runtime.h"

using namespace cressim::neo;

int main() {
    engine::Runtime runtime;
    engine::RuntimeConfig config{};
    if (!runtime.initialize(config)) {
        return 1;
    }

    auto& world = runtime.getWorld();
    const auto entity = world.createEntity();
    // Add transforms, renderers, physics components, and sensors here.

    runtime.prepare();
    if (!runtime.uploadWorld()) {
        return 1;
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    runtime.stepPhysics(frame);
    runtime.stepSimulationSensors(frame);
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);
}
```

See `examples/physics/basic.cpp` for a complete rigid-body scene.

:::

:::{tab-item} Python

```python
import cressim_neo as neo

runtime = neo.Runtime()
if not runtime.initialize(neo.RuntimeConfig()):
    raise RuntimeError("Could not initialize CRESSim-Neo")

world = runtime.world()
entity = world.create_entity()
# Add transforms, renderers, physics components, and sensors here.

runtime.prepare()
if not runtime.upload_world():
    raise RuntimeError("Could not upload the world")

frame = neo.FrameContext()
frame.delta_seconds = 1.0 / 60.0
runtime.step_physics(frame)
runtime.step_simulation_sensors(frame)
runtime.step_visual_sensors(frame)
runtime.end_frame(frame)
```

See `examples/python/runtime_frame_readback.py` for a complete direct-runtime
example.

:::

::::

The {doc}`../guides/index` explains scene authoring, frame stepping, batched
environments, physics, sensors, and GPU integration.
