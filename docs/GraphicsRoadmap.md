# Graphics Runtime Notes

This document summarizes what the current graphics scaffolding does and what to implement next.

## Current Architecture

1. `engine::Runtime` builds a render snapshot every tick by copying ECS data into `graphics::RenderWorld`.
2. `graphics::Renderer` sorts cameras by `renderOrder` and renders each camera output target.
3. `graphics::GraphicsDevice` is focused on backend primitives (render targets, frame/target lifecycle, presentation, readback).
4. `graphics::Renderer` now builds per-camera render queues (`shadow`, `opaque`, `transparent`) and executes them through a static forward pass scheduler.
5. `graphics::Renderer` uses Diligent-first culling with `AdvancedMath.hpp` (`ExtractViewFrustumPlanesFromMatrix`, `BoundBox`, `GetBoxVisibility`).
6. `PbrPass` is the active opaque pass implementation; shadow and transparent execution are scaffolded for later milestones.
7. Shader source lookup/loading lives in `ShaderSourceProvider`, shared by renderer passes.
8. Vulkan implementation is currently headless/offscreen.
9. Material now drives shading and render policy (`ShadingModel`, blend mode, shadow casting/receiving, opacity).

## Camera Output Flow

Camera fields in `engine::CameraComponent`:

- `outputTarget`: explicit output target. Invalid handle means fallback to default target.
- `outputWidth` / `outputHeight`: optional resize request for the selected target.
- `viewport`: normalized region in the selected output target.
- `renderOrder`: deterministic camera scheduling order.

The renderer copies these into `graphics::CameraData` and applies them per camera.

## Readback State Machine (Current)

1. `requestRenderTargetReadback(target)` adds `target` to `mPendingReadbacks`.
2. `endRenderTarget(target, frame)` queues GPU copy into staging + fence signal.
3. `endFrame(...)` waits/maps completed copies and appends payload events.
4. `tryGetRenderTargetReadback(...)` pops completion metadata plus optional RGBA payload.

## Next Implementation Steps

1. Implement shadow pass execution (single directional light map first) and shadow sampling in opaque shading.
2. Add functional `Phong` pass alongside `Pbr`.
3. Enable transparent pass execution with back-to-front blending.
4. Add GPU-resident texture handoff APIs for physics/ultrasound interop without CPU copies.
5. Replace full `RenderWorld` rebuild with dirty/incremental sync for large scenes.
