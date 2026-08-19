# Runtime Lifecycle

The engine executes in a staged runtime model with two phases: authoring and
stepping.

## Scene Authoring

A scene is configured by registering entities, components, and physics
constraints through the C++ or Python API. The engine compiles the required HLSL
shaders, allocates device memory, and generates structural layout mapping
caches.

The authored world is prepared and uploaded at initialization and whenever
host-authored scene state changes need to reach GPU resources. It is not
required for an otherwise unchanged steady-state frame.

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

::::{tab-set}

:::{tab-item} C++

```cpp
// One-time authoring
Runtime runtime;
runtime.initialize(config);

World& world = runtime.getWorld();
// Register entities, components, constraints, and sensors.

runtime.prepare();
runtime.uploadWorld();

// Steady-state frame loop
while (running) {
    runtime.prepare();
    runtime.uploadWorld();

    runtime.stepPhysics(frame);

    // Simulated sensors, e.g., ultrasound.
    runtime.stepSimulationSensors(frame);

    // Visual sensors, e.g., color/depth.
    runtime.stepVisualSensors(frame);

    // Optional custom compute pass.
    runtime.executeCustomComputePass(pass);

    // End-frame synchronize.
    runtime.endFrame(frame);
}
```

:::

:::{tab-item} Python

```python
# One-time authoring
runtime = neo.Runtime()
runtime.initialize(config)

world = runtime.world()
# Register entities, components, constraints, and sensors.

runtime.prepare()
runtime.upload_world()

# Steady-state frame loop
while running:
    runtime.prepare()
    runtime.upload_world()

    runtime.step_physics(frame)

    # Simulated sensors, e.g., ultrasound.
    runtime.step_simulation_sensors(frame)

    # Visual sensors, e.g., color/depth.
    runtime.step_visual_sensors(frame)

    # Optional custom compute pass.
    runtime.execute_custom_compute_pass(custom_pass)

    # End-frame synchronize.
    runtime.end_frame(frame)
```

:::

::::

The code follows the execution-pipeline example in the framework figure. For an
otherwise unchanged scene, omit the repeated `prepare` and upload calls; they
are needed when host-authored scene changes must be prepared and uploaded.

This design makes synchronization boundaries explicit and allows
{doc}`custom GPU computations <gpu-integration-and-custom-compute>` to be
interleaved with simulation in a straightforward manner.
