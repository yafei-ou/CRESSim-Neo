# Big-Bang Refactor Plan: GPU Compute Core + Clean Physics/Renderer Boundaries

## Summary
Refactor into four clean modules with shared GPU ownership and explicit data flow:

1. `gpu`: owns Diligent device/context/swapchain/target/compute resource lifecycle.
2. `graphics`: rendering/presentation only, built on `gpu`.
3. `physics`: GPU-compute simulation pipelines (PBD), built on `gpu`.
4. `engine::Runtime`: orchestration only (tick order + sync systems), no backend-specific leakage.

Chosen authority model: **physics authoritative** for simulated transform state.  
Current integration path: **GPU physics -> CPU readback -> render world copy** (explicitly temporary).  
Architecture leaves a direct GPU->GPU path for later via shared `gpu` resources/handles.

---

## Target Module Boundaries

### `gpu` module (new primary backend layer)
Public API (minimal wrapping, Diligent-first):
- `include/gpu/gpu_device.h`
- `include/gpu/gpu_types.h`
- `include/gpu/export.h`

Core types:
- `GpuBackend` (`Null`, `Vulkan`)
- `GpuDeviceDesc` (moves device/presentation setup from `graphics::GraphicsDeviceDesc`)
- `GpuContext`:
  - `Diligent::IRenderDevice* renderDevice`
  - `Diligent::IDeviceContext* immediateContext`
  - `Diligent::ISwapChain* primarySwapChain` (nullable)
- `GpuTextureHandle`, `GpuBufferHandle`, `GpuPipelineHandle`, `GpuReadbackRequest` (opaque ids)
- `GpuRenderTargetDesc` (color/depth target descriptors)
- `GpuReadbackEvent` (generalized for texture/buffer readback)

API responsibilities:
- initialize/shutdown Diligent backend
- create/destroy/resize render targets
- frame begin/end + present
- create shader/pipeline for compute + graphics (pass-through to Diligent descriptors)
- dispatch compute
- readback buffers/textures

Implementation:
- Move/adapt `src/graphics/device/graphics_device_impl_*` into `src/gpu/*` as `GpuDeviceImpl`.
- Keep direct Diligent types in signatures where practical.

### `graphics` module (renderer only)
Public API:
- `include/graphics/renderer.h`
- `include/graphics/render_world.h`
- `include/graphics/render_resource_manager.h`

Changes:
- Remove device bootstrap/ownership from `graphics`.
- `Renderer` takes `gpu::GpuDevice&` (or `gpu::GpuContext` + helper service).
- Replace `GraphicsDeviceImpl` static-casts in passes with `gpu::GpuDevice`/context calls.
- Keep current forward pipeline behavior intact.

### `physics` module (new compute simulation module)
Public API:
- `include/physics/physics_world.h`
- `include/physics/physics_solver.h`
- `include/physics/physics_types.h`
- `include/physics/export.h`

Responsibilities:
- Own simulation data layout (CPU metadata + GPU SoA buffers)
- Compile/manage compute pipelines
- Execute PBD solve schedule per tick
- Read back minimal state needed by engine/render sync (transforms initially)

### `engine` module (orchestration only)
`engine::Runtime` responsibilities:
- Own `gpu::GpuDevice`, `graphics::Renderer`, `physics::PhysicsSolver`, `World`
- Tick order:
  1. sync ECS -> physics world (non-sim authoring fields)
  2. physics step on GPU
  3. sync physics results -> ECS transform + render world
  4. render
- Remove direct backend exposure (`getGraphicsDevice()`); expose subsystem interfaces instead.

---

## Shared Shader and GPU Cache Policy

### Shader source provider ownership
1. Move `ShaderSourceProvider` out of `graphics::detail` into `gpu` as a shared service (for example `gpu::ShaderLibrary`).
2. Keep behavior simple and Diligent-first:
   - resolve shader root once (`shaderDirectory`, `CRESSIM_NEO_SHADER_SOURCE_DIR`, `shaders`)
   - create one `Diligent::IShaderSourceInputStreamFactory`
   - expose resolved path + stream factory directly
3. Both `graphics` and `physics` must consume this same service for shader compilation/loading.
4. Initial file migration:
   - from `src/graphics/renderer/services/shader_source_provider.*`
   - to `src/gpu/shader_library.*`
5. Future rule: no module-local shader directory resolvers in renderer/physics.

### Mesh GPU cache ownership
1. Keep `MeshGpuCache` in `graphics` and treat it as render-only infrastructure.
2. Do not reuse `MeshGpuCache` for physics buffers.
3. Add a separate `physics` buffer owner (for example `physics::PhysicsGpuBuffers`) for SoA simulation state and high-frequency updates.
4. Keep shared low-level resource creation in `gpu` (buffer/texture creation helpers) so `graphics::MeshGpuCache` and `physics::PhysicsGpuBuffers` both depend on `gpu`, not on each other.
5. Interop is postponed by design:
   - current path remains GPU physics -> CPU copy -> render upload/sync
   - future direct GPU->GPU bridge will be a dedicated interop layer, not a merged cache type.

---

## Public API / Type Changes (Decision Complete)

### Remove / replace
- Remove `include/graphics/graphics_device.h` as public entrypoint.
- Replace with `include/gpu/gpu_device.h`.
- Update all includes currently pulling `graphics/graphics_device.h`:
  - `include/engine/components.h`
  - `include/engine/runtime.h`
  - `include/graphics/render_world.h`
  - renderer pass headers

### New `engine::RuntimeConfig`
- `gpu::GpuDeviceDesc gpuDeviceDesc{}`
- `graphics::RendererDesc rendererDesc{}`
- `physics::PhysicsSolverDesc physicsDesc{}`
- Keep compatibility shim field names for one commit if needed, then delete.

### Engine component split for physics
Add physics-facing components (no Diligent backend leakage):
- `RigidBodyComponent`
- `ColliderComponent` (shape type + params)
- `SoftBodyComponent` (particle/cluster topology refs)
- `PhysicsMaterialComponent`
- Optional `PhysicsProxyComponent` with stable solver handle

Keep `TransformComponent` as visualization/output transform.

### Render target references
- Camera output target type changes from `graphics::RenderTargetHandle` to `gpu::GpuRenderTargetHandle`.

---

## Data-Oriented Physics Layout (CPU + GPU)

### CPU-side metadata (sparse authoring, stable ids)
- `PhysicsEntityMap`: entityId <-> solver index
- `RigidBodyMeta[]`: type, sleep flags, shape ref, material ref
- `SoftBodyMeta[]`: particle range, constraint ranges, rest volumes/distances
- `ShapeMeta[]`: sphere/capsule/box/convex params
- Dirty bitsets for incremental uploads

### GPU-side SoA buffers
Rigid bodies:
- positions `float4` (xyz + invMass)
- orientations `float4` (quat)
- linear velocities `float4`
- angular velocities `float4`
- shape index/type arrays
- inertia tensors / inverse inertia (packed)

Soft bodies:
- particle positions (current/predicted)
- particle velocities + invMass
- constraints:
  - distance constraints
  - volume constraints
  - bending constraints (extensible later)
- body/cluster index ranges

Shared:
- contact pair buffers
- constraint lambdas/accumulators
- scratch/scan/indirect dispatch buffers

Readback (temporary path):
- rigid body transforms buffer
- optional soft body debug positions (gated by debug flag)

---

## PBD Compute Pass Architecture

Per frame (with configurable substeps/iterations):
1. `integrate_external_forces.cs.hlsl`
2. `build_broadphase_keys.cs.hlsl`
3. `sort_or_bucket_pairs` (initially simple uniform grid, no full radix requirement yet)
4. `generate_contacts.cs.hlsl`
5. `build_constraints_rigid.cs.hlsl`
6. `build_constraints_soft.cs.hlsl`
7. Solver loop (`iterations`):
   - `solve_contacts.cs.hlsl`
   - `solve_rigid_shape_constraints.cs.hlsl`
   - `solve_soft_constraints.cs.hlsl`
   - `solve_rigid_soft_coupling.cs.hlsl` (stub first)
8. `update_velocities.cs.hlsl`
9. `writeback_transforms.cs.hlsl`
10. `optional_readback_pack.cs.hlsl` (debug/CPU sync support)

Initial implementation for this big-bang:
- Only placeholder pass + scaffolding wired:
  - `physics_placeholder_integrate.cs.hlsl`
  - updates rigid transform from velocity and dt
- All non-placeholder passes created as pipeline slots with TODOs and no-op dispatch guards.

---

## Placeholder Compute Deliverable (Required)

### Shader
Add:
- `assets/shaders/physics/physics_placeholder_integrate.cs.hlsl`

Behavior:
- Input buffer: rigid states (`position`, `rotation`, `linearVelocity`)
- Constants: `dt`, `bodyCount`
- Output buffer: updated position and transform matrix/quaternion (choose quaternion output)
- Deterministic and minimal.

### Physics solver wiring
Add in `physics`:
- pipeline creation for placeholder compute
- per-frame constants buffer update
- dispatch dimensions `(bodyCount + groupSize - 1)/groupSize`
- readback request for rigid transform output buffer (CPU path)

---

## Sync Systems and Ownership

### New sync files
- `src/engine/world_to_physics_world_sync.cpp`
- `src/engine/physics_world_to_world_sync.cpp`
- keep `world_to_render_world_sync.cpp`, but now read transforms after physics sync

Flow:
- ECS authoring values (initial spawn/config) copied into `PhysicsWorld`
- Each tick physics updates canonical transforms
- physics results copied back to ECS `TransformComponent`
- render sync continues from ECS to `RenderWorld`

This keeps renderer decoupled from solver internals and preserves current render pipeline.

---

## Build System Refactor

### New targets
- `cressim_neo_gpu`
- `cressim_neo_physics`

### Dependency graph
- `common` -> (base)
- `gpu` -> `common` + Diligent libs
- `graphics` -> `common` + `gpu`
- `physics` -> `common` + `gpu`
- `engine` -> `common` + `graphics` + `physics`
- `viewer/tests` -> `engine` (+ `gpu` where direct handles used)

### CMake edits
- `src/CMakeLists.txt`: enable `gpu` and `physics` subdirs, keep `graphics`, `engine`
- Move Diligent engine backend link rules from `src/graphics/CMakeLists.txt` to `src/gpu/CMakeLists.txt`
- Keep shader copy post-build in one place (prefer `gpu` target, include `assets/shaders/physics`)

---

## Runtime Orchestration Contract

`Runtime::tick(frame)` exact order:
1. `syncWorldToPhysicsWorld(mWorld, mPhysicsWorld)`
2. `mPhysicsSolver.step(frame, mPhysicsWorld)`
3. `syncPhysicsWorldToWorld(mPhysicsWorld, mWorld)`
4. `syncWorldToRenderWorld(mWorld, mRenderWorld)`
5. `mLastRenderStats = mRenderer.render(frame, mRenderWorld)`

Runtime public accessors:
- `getGpuDevice()`
- `getPhysicsSolver()`
- `getRenderer()`
- keep `getWorld()`, `getResources()`
- remove backend-specific methods from runtime that bypass boundaries.

---

## Migration Sequence (Big-Bang but Controlled)

1. Introduce `gpu` target and migrate current graphics device impl into it.
2. Replace all `graphics::GraphicsDevice*` references with `gpu::GpuDevice*`.
3. Update renderer/passes to consume `gpu` context and remove `GraphicsDeviceImpl` knowledge.
4. Introduce `physics` target with `PhysicsWorld` + `PhysicsSolver` skeleton.
5. Add placeholder compute shader + pipeline + dispatch + readback.
6. Add new engine sync systems (`world<->physics`).
7. Rewire `Runtime` orchestration and config.
8. Re-enable/adjust tests and viewer compile path.
9. Add TODO markers in all planned PBD pass stages and constraints buffers.
10. Remove legacy `graphics_device` API surface and dead paths.

---

## Tests and Scenarios

### New required test
`tests/physics/physics_placeholder_compute.cpp`:
- Initialize runtime with Vulkan backend.
- Spawn one rigid physics entity with known linear velocity.
- Tick N frames.
- Assert transform position changed by ~`velocity * dt * N` (tolerance).
- Assert renderer still runs and returns nonzero camera stats when camera exists.

### Additional boundary tests
1. `tests/engine/runtime_orchestration_order.cpp`
   - verifies physics executes before render sync (via observable counters/hooks).
2. `tests/graphics/renderer_gpu_boundary.cpp`
   - ensures renderer compiles/executes with `gpu::GpuDevice` only; no physics includes.
3. `tests/physics/physics_gpu_context_lifecycle.cpp`
   - solver survives device init/shutdown and rejects calls when uninitialized.

### Existing tests to update
- `tests/smoke/smoke_runtime.cpp`
- `tests/graphics/render_target_resize_policy.cpp`
- `tests/forward_pipeline/forward_pipeline.cpp`
- replace old runtime graphics accessors/handle types with `gpu` equivalents.

---

## TODO Strategy in Code
Add explicit TODO blocks (not hidden comments) for:
- full rigid shape constraint solve
- soft body distance/volume constraints
- rigid-soft coupling
- broadphase optimization (sort/scan)
- direct GPU->GPU physics/render interop path
- async compute queue split (future)

---

## Assumptions and Defaults
1. Backend target is Vulkan first; Null path may remain limited for physics compute.
2. Physics is authoritative for simulated transforms.
3. CPU copy from physics output to render input is accepted initially.
4. Diligent types are used directly in module APIs where practical.
5. Breaking API changes are acceptable in this refactor.
6. Placeholder compute validates architecture and execution chain, not final PBD correctness.
