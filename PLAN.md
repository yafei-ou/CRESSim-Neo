# GPU-Only Rigid Body Solver Milestone

## Summary

Implement a correctness-first rigid body solver on the GPU only, covering:

- dynamic-vs-dynamic collisions
- dynamic-vs-static collisions
- gravity
- stable normal contact resolution with angular response
- `Sphere`, `Box`, and `Capsule` support for all mixed pairs
- authored static ground via existing rigid bodies with `inverseMass = 0`

Inertia will be user-authored, not derived from mass or primitive shape. Use local-space inverse inertia diagonal as the public API.

No CPU rigid solver in this slice.

If a GPU physics step fails, physics fails closed:

- `PhysicsSolver::step()` returns `false`
- runtime does not advance rigid bodies through any CPU fallback path

## Public API / Interface Changes

### `include/engine/components.h`

Extend `RigidBodyComponent` with:

- `Diligent::float3 inverseInertiaLocal{1.0f, 1.0f, 1.0f};`

Semantics:

- local-space inverse inertia diagonal
- aligned to the rigid body's local axes
- static bodies use `{0, 0, 0}` together with `inverseMass = 0`

### `include/physics/physics_types.h`

Extend `RigidBodyState` with:

- `Diligent::float3 scale{1.0f, 1.0f, 1.0f};`
- `Diligent::float3 inverseInertiaLocal{1.0f, 1.0f, 1.0f};`

Extend `RigidBodySoAHost` with:

- `std::vector<Diligent::float4> scales;`
- `std::vector<Diligent::float4> inverseInertiaLocal;`

Store both as `float4` for alignment.

Define collider parameter semantics explicitly:

- `Sphere`: `colliderParams.x = baseRadius`
- `Box`: `colliderParams.xyz = baseHalfExtents`
- `Capsule`: `colliderParams.x = baseRadius`, `colliderParams.y = baseHalfHeight`
- capsule axis is local `+Y`

Define effective-size rules:

- sphere radius = `baseRadius * max(abs(scale.x), abs(scale.y), abs(scale.z))`
- box half extents = `baseHalfExtents * abs(scale.xyz)`
- capsule radius = `baseRadius * max(abs(scale.x), abs(scale.z))`
- capsule half height = `baseHalfHeight * abs(scale.y)`

### `include/physics/physics_solver.h`

Remove:

- `enableCollisionPipelineScaffold`

Keep:

- `enableGpuCompute`
- `substeps`
- `solverIterations`
- `enableBlockingReadback`

No new public material/gravity knobs in this slice.

## World Sync Changes

### `src/engine/world_to_physics_world_sync.cpp`

Sync into physics state:

- `state.scale = transform->worldTransform.scale`
- `state.inverseInertiaLocal = rigidBody->inverseInertiaLocal`

### `src/engine/physics_world_to_world_sync.cpp`

Do not sync scale or inertia back to world.

### `src/physics/physics_world.cpp`

Update SoA conversion helpers to include:

- scale
- inverse inertia local

Preserve snapshot/SoA parity for both.

## Solver Architecture

## C++ Side

Refactor `src/physics/physics_solver.cpp` into a real rigid-body GPU solver orchestrator.

Add internal shared helpers only as needed:

- `src/physics/rigid_body_common.h`
- `src/physics/rigid_body_common.cpp`

Responsibilities:

- effective primitive dimensions from `colliderParams + scale`
- world-space inverse inertia from authored local diagonal and orientation
- rigid contact/schema structs shared with shader packing
- quaternion / rotation-vector helper math used by CPU-side setup and validation

Do not compute inertia from mass or shape anywhere in the solver.

Do not add a CPU rigid solver.

### Runtime behavior

Update `src/engine/runtime.cpp`:

- remove fallback to `mPhysicsWorld.integrateRigidBodiesCpu(...)`
- if `mPhysicsSolver->step(...)` fails, leave physics state unchanged for that frame

## GPU / HLSL Design

Replace the placeholder shader with rigid-only multi-pass shaders.

### New shader files

Add:

- `assets/shaders/physics/include/physics_rigid_common.hlsli`
- `assets/shaders/physics/physics_rigid_predict.cs.hlsl`
- `assets/shaders/physics/physics_rigid_generate_contacts.cs.hlsl`
- `assets/shaders/physics/physics_rigid_solve_gather.cs.hlsl`
- `assets/shaders/physics/physics_rigid_apply_corrections.cs.hlsl`
- `assets/shaders/physics/physics_rigid_update_velocities.cs.hlsl`

Retire:

- `assets/shaders/physics/physics_placeholder_integrate.cs.hlsl`

### Passes

#### 1. Predict

One thread per body:

- skip integration for `inverseMass == 0`
- apply gravity `(0, -9.81, 0)`
- store previous position/orientation
- integrate predicted position from linear velocity
- integrate predicted orientation from angular velocity
- normalize quaternion

#### 2. Generate Contacts

Use brute-force pair generation for correctness.

Pair count:

- `pairCount = bodyCount * (bodyCount - 1) / 2`

Per pair thread:

- skip if both bodies are static
- compute shape AABBs
- reject non-overlap
- run narrowphase
- emit up to 4 contacts into a deterministic fixed block for that pair

No append atomics.

Contact buffer stores per active slot:

- `bodyA`
- `bodyB`
- `normal`
- `point`
- `penetration`
- `active`

#### 3. Solve Constraints

Use Jacobi gather/apply to avoid float atomics.

Per iteration:

- `physics_rigid_solve_gather.cs.hlsl`
  - one thread per body
  - scan pair-contact blocks
  - gather contacts touching that body
  - compute accumulated translation correction
  - compute accumulated rotation-vector correction using:
    - authored `inverseMass`
    - authored `inverseInertiaLocal` transformed into world space

- `physics_rigid_apply_corrections.cs.hlsl`
  - one thread per body
  - apply translation correction
  - apply orientation correction
  - normalize quaternion
  - clear correction buffers

#### 4. Update Velocities

One thread per body:

- linear velocity = `(predictedPosition - previousPosition) / dt`
- angular velocity = quaternion delta / `dt`
- skip static bodies
- normalize output orientation

Defaults:

- restitution = `0.0`
- friction = `0.0`
- contact slop = `1e-3`

## Narrowphase Coverage

All enum pairs must work:

- sphere-sphere
- sphere-box
- sphere-capsule
- box-box
- box-capsule
- capsule-capsule

Applies to:

- dynamic-dynamic
- dynamic-static

Requirements:

- box-box supports face contacts and edge-edge fallback
- sphere/capsule contacts can use 1 contact point
- box contacts may emit up to 4 contact points

Static bodies are existing rigid bodies with:

- `simulated = true`
- `inverseMass = 0`
- `inverseInertiaLocal = {0, 0, 0}`

Ground is a static box, not a special plane primitive.

## GPU Buffer Layout

### Persistent buffers

Use:

- positions + inverse mass
- orientations
- linear velocities
- angular velocities
- collider shape types
- collider params
- rigid scales
- rigid inverse inertia local

### Transient buffers

Use only implemented buffers:

- previous positions
- previous orientations
- predicted positions
- predicted orientations
- predicted linear velocities
- predicted angular velocities
- contact buffer
- translation correction buffer
- rotation correction buffer

Remove unused transient allocations:

- spatial keys
- sorted indices
- candidate pairs
- unused generic scratch buffers

## Physics Solver Stages

Executed:

- `PredictState`
- `GenerateContacts`
- `SolveConstraints`
- `UpdateVelocities`
- `CommitResults` when blocking readback is enabled

Intentionally skipped:

- `BuildSpatialIndices`
- `SortSpatialIndices`
- `BuildConstraintData`

Reason:
- no broadphase or compaction in this correctness-first slice

## Initialization / Failure Behavior

### `PhysicsSolver::initialize()`

Create PSOs/SRBs for:

- predict
- generate contacts
- solve gather
- apply corrections
- update velocities

Initialization fails if any required shader or PSO fails.

### `PhysicsSolver::step()`

Behavior:

- if not initialized, return `false`
- if GPU compute is disabled, return `false`
- if GPU backend/context missing, return `false`
- if any pass dispatch/readback fails, return `false`

No CPU rigid solving path.

## Viewer / Demo Updates

Update `tests/physics_viewer/physics_viewer.cpp`:

- static ground as a large box rigid body with `inverseMass = 0`
- set `inverseInertiaLocal = {0,0,0}` for static ground
- set explicit collider params for all bodies
- set explicit authored inertia for dynamic bodies
- include at least:
  - dynamic sphere
  - dynamic box
  - dynamic capsule

## Documentation Updates

Update `docs/PhysicsPlan.md` to match the implemented slice:

- rigid-only
- GPU-only
- no CPU rigid solver
- inertia is authored by the user as local inverse inertia diagonal
- solver does not derive inertia from primitive shape or mass
- brute-force pair generation for first pass
- Jacobi gather/apply contact solve
- static authored via zero-inverse-mass rigid bodies
- all sphere/box/capsule mixed pairs
- transform scale participates in collider sizing

Deferred:

- broadphase optimization
- contact adjacency/compaction optimization
- restitution tuning
- friction
- CPU parity solver
- fluids
- soft bodies
- manual constraints

## Test Cases and Scenarios

### Update existing tests

#### `tests/physics/physics_world_soa_layout.cpp`

Extend to verify:

- scale is preserved
- authored inverse inertia is preserved
- both survive removal/compaction and upsert

#### `tests/engine/runtime_physics_sync_fields.cpp`

Extend to verify:

- transform scale syncs into physics world
- authored inverse inertia syncs into physics world
- shape and collider params still round-trip

### Replace placeholder compute test

Replace `tests/physics/physics_placeholder_soa_compute.cpp` with a Vulkan rigid collision test:

Scenario:

- static box ground
- dynamic sphere above it
- authored inverse inertia on the sphere
- run several frames

Verify:

- sphere falls
- sphere settles above ground within tolerance
- penetration stays below tolerance
- orientation remains normalized
- `GenerateContacts`, `SolveConstraints`, and `UpdateVelocities` executed

### Add GPU mixed-pair coverage test

Add a Vulkan physics test covering:

- sphere-box
- sphere-capsule
- box-box
- box-capsule
- capsule-capsule

For each case:

- set explicit authored inverse inertia per body
- initialize overlapping or colliding setup
- run enough frames/iterations
- verify near-non-penetration
- verify off-center impact can produce angular velocity where expected
- verify no NaNs

### Add GPU static-contact shape test

Add a Vulkan physics test for:

- sphere on static box ground
- box on static box ground
- capsule on static box ground

Verify:

- final center height matches expected support height within tolerance
- body remains above ground
- final speed is near zero after settle window

## Acceptance Criteria

Implementation is complete when all are true:

- Vulkan GPU path performs rigid collision solving correctly
- no CPU rigid solver exists
- no CPU fallback runs on GPU step failure
- static authored bodies work through `inverseMass = 0`
- all sphere/box/capsule mixed pairs resolve
- transform scale affects collider size with the chosen primitive-preserving rules
- authored inertia controls angular response
- off-center contacts produce angular response
- placeholder shader path is removed
- only used rigid buffers remain allocated
- `PhysicsPlan.md` matches the implemented GPU-only milestone

## Assumptions and Defaults

- GPU-only rigid solving
- fail-closed behavior on GPU step failure
- inertia is user-authored local inverse inertia diagonal
- no inertia derivation from mass or shape
- static bodies use `inverseMass = 0` and `inverseInertiaLocal = {0,0,0}`
- no separate static collider component
- no plane primitive
- no restitution or friction tuning in this slice
- gravity = `(0, -9.81, 0)`
- contact slop = `1e-3`
- broadphase optimization deferred to the next pass
- correctness first, HLSL performance later
