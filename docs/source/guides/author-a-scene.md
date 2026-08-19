# Author a scene

Author a scene through `engine::World` before calling `Runtime::prepare()`.

1. Set capacities in `RuntimeConfig::sceneLayout`.
2. Register meshes, materials, textures, and optional environment resources.
3. Create entities with `World::createEntity(envIndex)`.
4. Attach transforms, renderers, physics components, cameras, and lights.
5. Add colliders, joints, and cross-object constraints as required.
6. Call `prepare()` and `uploadWorld()`.

Components describe source state; preparation assigns the GPU slot and buffer
layouts. Keep entity ownership and environment indices consistent across
constraints.
