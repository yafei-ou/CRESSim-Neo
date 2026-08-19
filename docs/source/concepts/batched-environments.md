# Batched environments

One runtime can host many independent environments. Static resources such as
meshes, materials, and shaders are shared; dynamic state, cameras, lights, and
physics ownership are indexed per environment.

Use batching when environments share a layout but differ in state, actions, or
randomization. Choose capacities in the scene layout before preparation; memory
use grows roughly with the environment count and those capacities.

Set `RuntimeConfig::sceneLayout` before initialization, then create each entity
with its environment index. The same layout is exposed in Python as
`RuntimeConfig.scene_layout`.
