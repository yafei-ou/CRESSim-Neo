# Forward Renderer Refactor Plan (Diligent-First Culling)

## Summary
Adopt a Diligent-first policy for common math/visibility algorithms.  
For culling, use Diligent’s `AdvancedMath.hpp` helpers (`ExtractViewFrustumPlanesFromMatrix`, `BoundBox`, `GetBoxVisibility`) instead of custom frustum-plane math.  
Overall roadmap stays the same: architecture-first pass scheduler, shadows next, transparent execution later.

## Important Public API / Interface / Type Changes
1. `include/graphics/render_resource_manager.h`
   - Add `ShadingModel { Pbr, Phong }`.
   - Add `BlendMode { Opaque, Transparent }`.
   - Extend `MaterialResourceDesc` with `shadingModel`, `blendMode`, `opacity`, `castsShadows`, `receivesShadows`.
2. `include/graphics/render_world.h`
   - Extend `RenderableInstance` with resolved render policy fields copied from material.
3. `include/engine/components.h`
   - Keep `MeshRendererComponent` material-driven (`mesh`, `material`, `visible`) with no extra shadow/shading flags.
4. `include/graphics/renderer.h`
   - Extend `RenderStats` with queue/culling/pass counters.
5. `include/graphics/graphics_device.h`
   - Add explicit render-pass begin descriptor so multi-pass target load/clear behavior is controllable.

## Implementation Plan

### Milestone 1: Architecture-First Opaque Pipeline (with Diligent Culling)
1. Refactor sync in `src/engine/world_to_render_world_sync.cpp` to copy material-driven render policy into `RenderableInstance`.
2. Add frame/queue types in `src/graphics/renderer/passes/` for static pass scheduling.
3. Replace custom culling math plan with Diligent utilities:
   - Use `Diligent::BoundBox` for per-mesh local bounds.
   - Compute local mesh bounds once on mesh registration (simple min/max vertex scan).
   - Build camera frustum via `ExtractViewFrustumPlanesFromMatrix(viewProj, frustum, bIsOpenGL)`.
   - Use `BoundBox::Transform(modelMatrix)` for world bounds.
   - Use `GetBoxVisibility(frustum, worldBounds)` for visibility classification.
4. Implement static pass scheduler sequence:
   - Opaque pass executes.
   - Shadow pass is scaffolded/no-op.
   - Transparent pass is scaffolded/no-op.
5. Refactor `src/graphics/renderer/renderer.cpp`:
   - Per camera: cull, build queues, sort deterministically, execute scheduler.
6. Keep `PbrPass` as working opaque path, adapted to queue input.
7. Update `docs/GraphicsRoadmap.md` with pass scheduler + Diligent-first culling policy.

### Milestone 2: Shadows + Phong
1. Implement single directional-light shadow map pass.
2. Add shadow sampling integration to opaque lighting path.
3. Implement CSM and PCF
4. Add `PhongPass` and shading dispatch (`Pbr` + `Phong`).
5. Enforce material policy:
   - `castsShadows` controls shadow queue inclusion.
   - `receivesShadows` controls shadow application in lighting.

### Milestone 3: Transparent Execution
1. Enable transparent pass rendering.
2. Use back-to-front alpha blending.
3. Use `BlendMode::Transparent` and `opacity`.
4. Keep transparent shadow-casting disabled by default in this milestone.

### Mileston 4: Advanced
1. Implement material diffuse/specular maps

## Test Cases and Scenarios
1. Add forward-pipeline tests (null backend) for:
   - Diligent-based frustum culling counts.
   - Stable queue ordering across frames.
   - Opaque draw-call counters.
2. Add shadow tests (Milestone 2):
   - Caster/non-caster policy verification.
3. Add transparent tests:
   - Milestone 1: transparent queue populated, transparent draws zero.
   - Milestone 3: transparent draws nonzero.
4. Keep existing `smoke` and `cube_depth` tests green.

## Assumptions and Defaults
1. Use Diligent algorithms first when available for shared math/culling logic.
2. For current Vulkan/null path, `bIsOpenGL=false` in frustum extraction.
3. API breaks are acceptable in this refactor.
4. `PLAN.md` is not present in repo; roadmap documentation updates go to `docs/GraphicsRoadmap.md`.
