# Batched Environments

CRESSim-Neo packs multiple environments into one batched scene. The scene-layout
descriptor defines the number of environments and per-environment capacities for
renderable objects, lights, and cameras. An entity is created with an
environment index that determines its simulation and rendering state associated
with that environment.

| C++ | Python |
| --- | --- |
| {cpp:struct}`SceneLayoutDesc <cressim::neo::common::SceneLayoutDesc>` | {py:class}`SceneLayoutDesc <cressim_neo.SceneLayoutDesc>` |
| {cpp:func}`World::createEntity <cressim::neo::engine::World::createEntity>` | {py:meth}`World.create_entity <cressim_neo.World.create_entity>` |
| {cpp:func}`World::setEntityEnvironment <cressim::neo::engine::World::setEntityEnvironment>` | {py:meth}`World.set_entity_environment <cressim_neo.World.set_entity_environment>` |

For example, the following configures 64 environments, then creates an entity
in environment 17. The capacity values apply to each environment.

::::{tab-set}

:::{tab-item} C++

```cpp
RuntimeConfig config{};
config.sceneLayout.envCount = 64;
config.sceneLayout.maxRenderableObjectsPerEnv = 128;
config.sceneLayout.maxLightsPerEnv = 4;
config.sceneLayout.maxCamerasPerEnv = 2;

Runtime runtime;
runtime.initialize(config);

World& world = runtime.getWorld();
const auto entity = world.createEntity(17);
```

:::

:::{tab-item} Python

```python
config = neo.RuntimeConfig()
config.scene_layout.env_count = 64
config.scene_layout.max_renderable_objects_per_env = 128
config.scene_layout.max_lights_per_env = 4
config.scene_layout.max_cameras_per_env = 2

runtime = neo.Runtime()
runtime.initialize(config)

world = runtime.world()
entity = world.create_entity(env_index=17)
```

:::

::::

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
