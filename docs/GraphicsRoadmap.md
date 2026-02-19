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

## Coordinate System Conventions

1. Runtime math uses Diligent/Direct3D-style row vectors and row-major host-side matrices.
2. Canonical world/camera basis is left-handed: `+X` right, `+Y` up, `+Z` forward.
3. Transform composition follows Diligent order: `World = Scale * Rotation * Translation`.
4. Camera projection uses Diligent defaults for Vulkan/Null: NDC `Z` range is `[0, 1]`.
5. Vulkan viewport Y handling relies on Diligent's backend behavior (negative viewport height under the hood); no extra app-side Y inversion should be added.

## Invariants Checklist

1. Do not introduce manual right-handed projection/view formulas.
2. Do not assume camera forward is `-Z`; default forward is `+Z`.
3. Keep matrix multiplication order in Diligent row-vector form (`World * View * Proj`).
4. Keep frustum extraction in non-OpenGL mode for current Vulkan/Null runtime (`bIsOpenGL = false`).

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
