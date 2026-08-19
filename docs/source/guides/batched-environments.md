# Batched Environments

CRESSim-Neo packs multiple environments into one batched scene. The scene-layout
descriptor defines the number of environments and per-environment capacities for
renderable objects, lights, and cameras. An entity is created with an
environment index that determines its simulation and rendering state associated
with that environment.

On the host side, the scene is authored through a unified entity-component
world. Entities may carry transforms, mesh renderers, rigid bodies, colliders,
soft bodies, fluids, strands, cameras, lights, or ultrasound-related
components. During `prepare()` and `uploadWorld()`, this authored state is
converted into GPU-resident scene and physics layouts together with mappings
that preserve entity ownership and environment membership. For renderables,
cameras, and lights, the runtime allocates fixed-capacity per-environment slots.
For physics objects, including rigid bodies, colliders, soft bodies, fluids, and
strands, the runtime stores environment indices and owner-to-buffer mappings
rather than using the same fixed-slot scheme as rendering.

```{figure} ../_static/batched-data-layout.png
:alt: Batched CRESSim-Neo scene data layout with shared resources and per-environment state.
:width: 100%

Batched scene data layout.
```

This representation separates shared resources from per-environment state. Mesh,
texture, material, and shader resources are shared, while dynamic simulation
state, per-environment lighting, camera state, environment fluid and IBL
settings, and sensor execution are environment-specific.
