# Debug Viewer Libraryization Plan (Main Project Module)

## Summary
Move debug viewer functionality out of `tests/debug_viewer/debug_viewer.cpp` into a new first-class library module `cressim_neo_viewer`, with GLFW dependency isolated to that module only.  
The viewer library attaches to a caller-provided `engine::Runtime` and caller-owned scene/camera entity, and provides a high-level run loop API (no product standalone app target).

## Public API / Interface Changes
1. Add public viewer namespace headers under `include/viewer/`.
2. Add `include/viewer/export.h` with `CRESSIM_NEO_VIEWER_API`.
3. Add `include/viewer/debug_viewer_app.h` with:
   - `DebugViewerAppDesc`
   - `DebugViewerCameraBinding`
   - `DebugViewerCallbacks`
   - `DebugViewerApp`
4. `DebugViewerApp` API:
   - `bool initialize(DebugViewerAppDesc desc, engine::RuntimeConfig& inOutRuntimeConfig);`
   - `bool run(engine::Runtime& runtime, DebugViewerCameraBinding camera, DebugViewerCallbacks callbacks = {});`
   - `void requestExit();`
   - `void shutdown();`
5. `engine`/`graphics` APIs remain unchanged.

## Runtime Usage Contract
1. Caller creates and fills `DebugViewerAppDesc`.
2. Caller invokes `DebugViewerApp::initialize(...)`, passing mutable `RuntimeConfig`.
3. Caller initializes `engine::Runtime` using that updated config.
4. Caller creates scene entities/components (including camera entity).
5. Caller invokes `DebugViewerApp::run(...)` with camera binding and optional callbacks.

## Test Strategy
1. Build with `CRESSIM_NEO_BUILD_VIEWER=ON` and verify `cressim_neo_viewer` links.
2. Keep smoke and cube-depth regressions passing.
3. Keep debug viewer test folder as integration validation only, calling viewer API.
4. Verify viewer module is the only place that links GLFW.

## Defaults
1. Module placement: `cressim_neo_viewer`.
2. GLFW policy: viewer-only optional dependency.
3. API level: high-level app API.
4. Scene ownership: caller-owned.
5. Runtime ownership: caller-provided runtime.
6. Product standalone binary: none.
