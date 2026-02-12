# Graphics Runtime Notes

This document summarizes what the current graphics scaffolding does and what to implement next.

## Current Architecture

1. `engine::Runtime` builds a render snapshot every tick by copying ECS data into `graphics::RenderWorld`.
2. `graphics::Renderer` sorts cameras by `renderOrder` and renders each camera output target.
3. `graphics::IGraphicsDevice` owns render targets and backend-specific frame/target operations.
4. Vulkan implementation is currently headless/offscreen and clears targets per camera pass.

## Camera Output Flow

Camera fields in `engine::CameraComponent`:

- `outputTarget`: explicit output target. Invalid handle means fallback to default target.
- `outputWidth` / `outputHeight`: optional resize request for the selected target.
- `viewport`: normalized region in the selected output target.
- `renderOrder`: deterministic camera scheduling order.
- `requestReadback`: asks device to emit a readback completion event.

The renderer copies these into `graphics::CameraData` and applies them per camera.

## Readback State Machine (Current)

1. `requestReadback(target)` adds `target` to `mPendingReadbacks`.
2. `endRenderTarget(target, frame)` moves the request to `mCompletedReadbacks`.
3. `tryPopReadbackEvent(...)` pops completion metadata for the runtime/app.

Important: this is currently logical completion metadata only. It does **not** include CPU pixel bytes yet.

## Next Implementation Steps

1. Replace logical readback completion with real GPU staging copy + fence completion.
2. Add pipeline/pass execution in renderer:
   - culling
   - pipeline/material binding
   - draw submission
3. Add explicit render graph/pass descriptors:
   - color/depth load-store behavior
   - dependency edges between passes
4. Add GPU-resident texture handoff APIs for physics/ultrasound interop without CPU copies.
5. Replace full `RenderWorld` rebuild with dirty/incremental sync for large scenes.

## Practical Milestone Suggestions

1. Milestone A: Draw one mesh with one material to one camera target.
2. Milestone B: Add second camera and verify deterministic order and viewport split.
3. Milestone C: Implement real readback bytes for one target with fence-based completion.
4. Milestone D: Add a compute pass consuming camera color/depth directly on GPU.
