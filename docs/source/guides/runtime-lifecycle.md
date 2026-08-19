# Runtime Lifecycle

The engine executes in a staged runtime model with two phases: authoring and
stepping.

## Startup and Shutdown

Construct a {cpp:struct}`RuntimeConfig <cressim::neo::engine::RuntimeConfig>`
or {py:class}`RuntimeConfig <cressim_neo.RuntimeConfig>`, then initialize the
runtime with {cpp:func}`Runtime::initialize <cressim::neo::engine::Runtime::initialize>`
or {py:meth}`Runtime.initialize <cressim_neo.Runtime.initialize>`. Handle an
unsuccessful initialization result before authoring a scene.

Call {cpp:func}`Runtime::shutdown <cressim::neo::engine::Runtime::shutdown>`
or {py:meth}`Runtime.shutdown <cressim_neo.Runtime.shutdown>` when resources
must be released before the runtime object reaches the end of its lifetime. In
C++, the `Runtime` destructor calls `shutdown()`, so an explicit call is
normally unnecessary.

## Scene Authoring

A scene is configured by registering entities, components, and physics
constraints through the C++ or Python API. The engine compiles the required HLSL
shaders, allocates device memory, and generates structural layout mapping
caches.

Configure the scene layout before initializing the runtime. Each entity is
created with the environment index to which its simulation and rendering state
belongs.

| C++ | Python |
| --- | --- |
| {cpp:struct}`RuntimeConfig <cressim::neo::engine::RuntimeConfig>` | {py:class}`RuntimeConfig <cressim_neo.RuntimeConfig>` |
| {cpp:struct}`SceneLayoutDesc <cressim::neo::common::SceneLayoutDesc>` | {py:class}`SceneLayoutDesc <cressim_neo.SceneLayoutDesc>` |
| {cpp:func}`Runtime::initialize <cressim::neo::engine::Runtime::initialize>` | {py:meth}`Runtime.initialize <cressim_neo.Runtime.initialize>` |
| {cpp:func}`Runtime::getWorld <cressim::neo::engine::Runtime::getWorld>` | {py:meth}`Runtime.world <cressim_neo.Runtime.world>` |
| {cpp:func}`World::createEntity <cressim::neo::engine::World::createEntity>` | {py:meth}`World.create_entity <cressim_neo.World.create_entity>` |

The authored world is prepared and uploaded at initialization and whenever
host-authored scene state changes need to reach GPU resources. It is not
required for an otherwise unchanged steady-state frame.

| C++ | Python |
| --- | --- |
| {cpp:func}`Runtime::prepare <cressim::neo::engine::Runtime::prepare>` | {py:meth}`Runtime.prepare <cressim_neo.Runtime.prepare>` |
| {cpp:func}`Runtime::uploadWorld <cressim::neo::engine::Runtime::uploadWorld>` | {py:meth}`Runtime.upload_world <cressim_neo.Runtime.upload_world>` |

::::{tab-set}

:::{tab-item} C++

```cpp
Runtime runtime;
runtime.initialize(config);

World& world = runtime.getWorld();
// Register entities, components, constraints, and sensors.

runtime.prepare();
runtime.uploadWorld();
```

:::

:::{tab-item} Python

```python
runtime = neo.Runtime()
runtime.initialize(config)

world = runtime.world()
# Register entities, components, constraints, and sensors.

runtime.prepare()
runtime.upload_world()
```

:::

::::

## Frame Stepping

Once initialized, the runtime steps through a steady-state frame execution
loop:

1. **Data upload** writes dynamic updates, such as kinematic target transforms
   or user inputs, into device buffers when those updates are needed.
2. **Physics step** dispatches the HLSL PBD physics solvers for rigid-body and
   deformable constraints, collisions, and other physics work on the GPU.
3. **Sensor and rendering step** processes cameras and simulation sensors, such
   as ultrasound synthesis, and writes to GPU render targets.
4. Optional user-defined **custom compute passes** are dispatched for custom
   tasks, such as post-processing, data packaging, and post-physics
   calculations. These passes can be inserted between any stages of the frame
   loop.
5. **Frame finalization** synchronizes the GPU and completes any requested data
   readbacks.

| C++ | Python |
| --- | --- |
| {cpp:func}`Runtime::stepPhysics <cressim::neo::engine::Runtime::stepPhysics>` | {py:meth}`Runtime.step_physics <cressim_neo.Runtime.step_physics>` |
| {cpp:func}`Runtime::stepSimulationSensors <cressim::neo::engine::Runtime::stepSimulationSensors>` | {py:meth}`Runtime.step_simulation_sensors <cressim_neo.Runtime.step_simulation_sensors>` |
| {cpp:func}`Runtime::stepVisualSensors <cressim::neo::engine::Runtime::stepVisualSensors>` | {py:meth}`Runtime.step_visual_sensors <cressim_neo.Runtime.step_visual_sensors>` |
| {cpp:func}`Runtime::executeCustomComputePass <cressim::neo::engine::Runtime::executeCustomComputePass>` | {py:meth}`Runtime.execute_custom_compute_pass <cressim_neo.Runtime.execute_custom_compute_pass>` |
| {cpp:func}`Runtime::endFrame <cressim::neo::engine::Runtime::endFrame>` | {py:meth}`Runtime.end_frame <cressim_neo.Runtime.end_frame>` |

::::{tab-set}

:::{tab-item} C++

```cpp
while (running) {
    runtime.stepPhysics(frame);
    runtime.stepSimulationSensors(frame);
    runtime.stepVisualSensors(frame);

    // Optional: insert custom compute where the task requires it.
    runtime.executeCustomComputePass(customPass);

    runtime.endFrame(frame);
}
```

:::

:::{tab-item} Python

```python
while running:
    runtime.step_physics(frame)
    runtime.step_simulation_sensors(frame)
    runtime.step_visual_sensors(frame)

    # Optional: insert custom compute where the task requires it.
    runtime.execute_custom_compute_pass(custom_pass)

    runtime.end_frame(frame)
```

:::

::::

This design makes synchronization boundaries explicit and allows
{doc}`custom GPU computations <gpu-integration-and-custom-compute>` to be
interleaved with simulation in a straightforward manner.

For a complete runnable-style example, see {doc}`../getting-started/first-scene`.
