## Remove Readback Coupling And Move Soft Render Derivations To GPU

### Summary
Make GPU physics state authoritative across frames, so rigid and soft simulation no longer depend on blocking CPU readback for correctness. Decouple CPU-authored world invalidation from GPU writeback, then move soft-body dynamic normals and bounds generation to GPU passes so steady-state multi-scene scaling is not limited by CPU recomputation or upload churn.

### Key Changes

#### 1. Separate authored state from simulation/readback state
- Replace the current single physics revision usage with two explicit tracks:
  - `authoredRevision`: increments only for CPU-authored physics mutations
  - `simulationRevision`: optional/debug-only, increments on GPU readback if needed
- Update physics upload invalidation to use only `authoredRevision` and topology revisions.
- Update render invalidation to stop treating GPU writeback as a scene mutation.
- Keep blocking readback available for tools/debugging, but make it observational only.

#### 2. Make GPU rigid state authoritative across frames
- Treat persistent GPU rigid buffers as the canonical frame-to-frame simulation state.
- Commit final predicted rigid state back into persistent GPU rigid buffers at the end of the step, not only between substeps.
- Change next-frame upload so CPU rigid data is uploaded only for explicitly authored dirty bodies:
  - create/remove
  - teleports or transform edits
  - property changes
  - kinematic target changes
- Ensure readback does not mark rigid bodies dirty for re-upload.
- Preserve CPU readback only for inspection/gameplay consumers that explicitly request it.

#### 3. Make soft-state readback optional and non-invalidating
- Remove readback-driven soft-particle re-upload behavior.
- `finalizeSoftParticleWriteback()` must not affect authored invalidation.
- Keep soft topology uploads keyed only to actual soft-body topology/configuration changes.
- Maintain GPU soft particle buffers as the authoritative deformation source for rendering.

#### 4. Move soft-body normals to GPU
- Remove per-frame CPU normal generation from `World::refreshSoftBodyRenderNormals(...)`.
- Add a GPU compute pass that rebuilds soft-body vertex normals every frame from deformed particle positions.
- Build and upload any required render-side adjacency/topology data once when soft-body render bindings are created:
  - either triangle-to-particle data
  - or vertex adjacency data sufficient for normal accumulation
- Store the output in a GPU soft-body normal buffer read directly by the forward/shadow passes.
- Keep CPU/rest normals only as initialization fallback or debug fallback.

#### 5. Move soft-body bounds to GPU
- Remove steady-state CPU bounds recomputation from deformed soft particles.
- Add a GPU reduction pass that computes one world-space AABB per soft body each frame from current deformed particle positions.
- Add/upload compact per-soft-body particle ranges once so the reduction pass knows which particles belong to each soft body.
- Feed render metadata/culling from the GPU-generated AABBs, avoiding full particle readback.
- Keep a conservative rest/inflated bounds fallback for initialization and failure cases.

#### 6. Narrow steady-state CPU/render uploads
- Upload soft-body render binding data only when topology or render bindings change.
- Stop steady-state CPU upload of soft-body normals entirely.
- Keep runtime render uploads focused on true scene changes, not simulation progression.
- Preserve vertex-shader deformation from GPU particle position buffers.

### Public APIs / Interface Changes
- `PhysicsWorld`
  - add or rename revision accessors so authored changes and simulation/writeback changes are distinct
  - authored revision becomes the only revision used by upload/render invalidation
- `PhysicsSceneGpuState`
  - upload logic switches to authored/topology revisions only
  - add GPU buffers for per-soft-body ranges, dynamic soft-body AABBs, and soft-body render normals
- Render/engine side
  - render metadata consumes GPU-produced soft-body bounds instead of CPU-computed bounds
  - soft-body shading consumes GPU normal buffer populated by compute
- Pass definitions
  - add compute pass definitions for soft-body normal generation and soft-body bounds reduction

### Test Plan
- Rigid continuity without readback:
  - run multiple frames with blocking readback disabled
  - verify rigid bodies continue correctly frame-to-frame
- CPU override behavior:
  - edit one rigid body on CPU between frames and verify only that body is uploaded/overridden
- Soft readback decoupling:
  - enable blocking readback and verify it no longer causes next-frame soft re-upload or render invalidation churn
- Soft-body normals:
  - verify compute-generated normals update with deformation and match/improve on current CPU lighting behavior
- Soft-body bounds:
  - verify GPU AABBs track deformed bodies and do not over-cull in the multi-env viewers
- Perf validation:
  - compare FPS and max parallel scene count before/after on `physics_viewer_soft_particles_toroid_multi_env`
  - compare CPU frame time and GPU frame time to confirm CPU-side scaling improvement

### Assumptions
- GPU compute is the preferred implementation path for both dynamic normals and dynamic bounds.
- Blocking readback remains supported for debugging, but not as part of normal simulation correctness.
- GPU-generated soft-body bounds are authoritative for rendering/culling once available.
- Any gameplay systems that need CPU-simulated state must opt into readback explicitly rather than receiving it implicitly every frame.
