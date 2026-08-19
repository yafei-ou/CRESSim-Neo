# Runtime model

`engine::Runtime` coordinates a `World`, the GPU device, physics, rendering,
sensors, shared buffers, and custom compute. It is deliberately staged rather
than hiding synchronization or GPU work behind a single `simulate()` call.

1. **Author:** populate the `World` and register resources.
2. **Prepare:** resolve authored state into fixed GPU layouts.
3. **Upload:** transfer the prepared world to device resources.
4. **Execute:** step physics, sensors, and optional custom compute.
5. **End frame:** finalize submissions and requested readbacks.

```text
world authoring → prepare → upload → physics → sensors → custom compute → end frame
```

Topology changes invalidate prepared layouts. Make them between frames, then
prepare and upload again. See {doc}`world-and-components` for authored state.
