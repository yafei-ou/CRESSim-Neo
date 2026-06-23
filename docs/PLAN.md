# Pre-Vector-Env Prototype Plan

## Summary
Build a first Python-facing prototype that stops just before the Python vector env layer. The milestone will deliver: a minimal C++ Python binding surface for authoring and staged stepping, a minimal Torch/CUDA interop path for structured buffers, and a minimal engine-side custom compute-pass API that Python can configure and dispatch. The plan assumes a fixed-layout prototype workflow: build scenes once, do not perform topology-changing authoring during stepping, and defer the scene duplicator and camera-based observations until after this milestone.

## Key Changes
### 1. Add a minimal Python binding module
- Introduce a new Python extension target built from CMake; use `pybind11` as the binding layer.
- Expose only the narrow authoring/runtime API needed for the prototype:
  - `Runtime`
  - `RuntimeConfig` subset
  - `World`
  - `FrameContext`
  - simple math/value types needed by rigid-body authoring
- Expose staged runtime calls exactly as they exist now:
  - `prepare()`
  - `step_physics(frame_ctx)`
  - `step_simulation_sensors(frame_ctx)`
  - `step_visual_sensors(frame_ctx)`
  - `end_frame(frame_ctx)`
- Expose a narrow rigid-body authoring slice only:
  - create entity with `envIndex`
  - set transform
  - set rigid body
  - add/update collider
  - optional mesh renderer and camera only if needed for smoke coverage, not for RL yet
- Do not expose soft bodies, strands, fluids, ultrasound, or rendering-resource authoring in v1 unless required to keep bindings coherent.

### 2. Expose stable prototype identity/mapping data
- Add a small public query API from C++ to Python for rigid-body prototype use:
  - rigid body `entity_ids`
  - rigid body `environment_indices`
  - rigid body count
  - current rigid layout/binding generation
- For v1, return this mapping metadata on the host side to Python; do not require a polished GPU mapping registry yet.
- Document the fixed-layout prototype contract:
  - cache mappings after `prepare()`
  - mappings remain valid while no topology-changing authoring occurs
  - users must treat add/remove/rebuild operations as invalidating cached indices
- Constraint enable/disable and collider enable/disable are considered layout-preserving for the prototype; structural add/remove is not.

### 3. Add minimal Torch/CUDA interop for structured state
- Add a small engine-side Torch interop layer for one structured state path first:
  - recommended target: rigid body live state and/or kinematic target write path
- Support two operations:
  - read/copy engine state into a Torch CUDA tensor provided by Python
  - write/copy a Torch CUDA tensor into an engine control/state buffer provided by Python
- Torch owns destination/source tensors; the engine does not allocate rollout or history buffers.
- Validate at call time:
  - tensor is CUDA
  - tensor dtype matches expected layout
  - tensor is contiguous
  - tensor shape/count matches the current scene count
- Keep the prototype to linear structured buffers only. Defer camera/render-target export and image packing until later.

### 4. Add a minimal custom compute-pass API
- Expose a minimal C++ API that Python can drive to create and run custom GPU compute work without exposing raw backend internals.
- The initial API should support:
  - create/load a compute pass from shader path + entry point + declared threadgroup info
  - bind named engine buffers/resources
  - bind Torch-owned CUDA buffers where the implementation path supports it
  - bind a small constants/uniform block
  - dispatch with explicit dimensions
- Keep the first public resource set curated and small:
  - rigid positions/orientations/velocities
  - rigid kinematic target buffers
  - optionally collider buffers if needed for a simple reward/termination kernel
- Python defines orchestration and bindings; the engine executes GPU work.
- Do not build the full generalized resource registry or a fully stable public buffer catalog yet. The prototype should expose only the resources needed for one rigid-body task.

### 5. Define the prototype workflow boundary
- The milestone stops before creating any Python vector env wrapper.
- The intended end state is that Python can:
  - author a small fixed-layout rigid-body multi-env scene
  - call staged stepping
  - query rigid mapping metadata
  - provide Torch CUDA tensors for actions/outputs
  - invoke minimal custom compute passes for observation/reward-style kernels
- The next milestone, explicitly out of scope here, is implementing the Python vector env wrapper on top of these capabilities.

## Test Plan
- Build test:
  - Python extension target compiles and links cleanly with the existing engine build.
- Binding smoke test:
  - Python can create a runtime, create a few entities in multiple envs, author transforms/rigid bodies/colliders, call `prepare()`, and step one frame through the staged API.
- Mapping stability test:
  - after `prepare()`, rigid `entity_ids` and env indices are queryable and match authored entities
  - repeated steps with no structural edits keep the same mapping generation
- Torch interop test:
  - Python allocates a CUDA tensor for a rigid-body readback target, engine fills it, and contents match expected body state
  - Python allocates a CUDA tensor for a write-in target, engine consumes it, and kinematic/control state changes accordingly
- Compute-pass smoke test:
  - Python creates a minimal compute pass that reads rigid state and writes a simple output tensor
  - dispatch succeeds across multiple env instances without CPU readback in the hot path
- Negative validation test:
  - wrong device, wrong dtype, wrong shape, or stale count produces a clear error

## Assumptions and Defaults
- Use `pybind11` for the first Python extension.
- Prototype scope is `Bindings + Compute`, stopping before any env wrapper.
- Rigid-body-only authoring is sufficient for the first milestone.
- Fixed-layout mode is assumed for prototype usage: no topology-changing authoring during stepping.
- Scene duplicator/fixed-layout replication support is deferred to the next architecture milestone.
- Camera observations and render-target export are deferred; first Torch interop targets structured buffers only.
- Rollout/replay ownership remains on the Torch/Python side; the engine will not own RL history buffers in this milestone.
