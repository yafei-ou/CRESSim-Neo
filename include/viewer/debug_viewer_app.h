#ifndef CRESSIM_NEO_VIEWER_DEBUG_VIEWER_APP_H
#define CRESSIM_NEO_VIEWER_DEBUG_VIEWER_APP_H

#include "common/frame_context.h"
#include "common/id.h"
#include "engine/runtime.h"
#include "viewer/export.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace cressim::neo::viewer
{

struct DebugViewerKeymap
{
    // Letter keys follow GLFW key values (ASCII uppercase).
    int moveForward  = 87; // W
    int moveBackward = 83; // S
    int moveLeft     = 65; // A
    int moveRight    = 68; // D
    int moveUp       = 69; // E
    int moveDown     = 81; // Q

    int speedBoostPrimary   = 340; // Left Shift
    int speedBoostSecondary = 344; // Right Shift
    int speedSlowPrimary    = 341; // Left Ctrl
    int speedSlowSecondary  = 345; // Right Ctrl

    int lookButton = 1; // Right mouse button

    int resetCamera                  = 70;  // F
    int toggleStats                  = 72;  // H
    int togglePresentedOutputMode    = 85;  // U
    int cyclePresentedCameraPrevious = 91;  // [
    int cyclePresentedCameraNext     = 93;  // ]
    int cyclePresentedProbePrevious  = 44;  // ,
    int cyclePresentedProbeNext      = 46;  // .
    int quit                         = 256; // Escape
};

struct DebugViewerAppDesc
{
    std::string windowTitle      = "CRESSim Neo Debug Viewer";
    std::uint32_t width          = 1280;
    std::uint32_t height         = 720;
    bool windowEnabled           = true;
    bool windowVisible           = true;
    bool startFullscreen         = false;
    bool startFullscreenWindowed = false;
    bool vSync                   = true;

    float inputSensitivity = 0.08f;
    float moveSpeed        = 3.0f;
    float speedBoostScale  = 3.0f;
    float speedSlowScale   = 0.35f;
    float wheelSpeedScale  = 0.12f;
    float minMoveSpeed     = 0.25f;
    float maxMoveSpeed     = 80.0f;

    float fixedDeltaSeconds           = 1.0f / 60.0f;
    bool useFixedTimestep             = false;
    std::uint64_t maxFrames           = 0;
    bool showStats                    = true;
    bool enableDebugParticles         = false;
    std::uint32_t statsIntervalFrames = 120;

    DebugViewerKeymap keymap{};
};

struct DebugViewerCameraBinding
{
    common::EntityId cameraEntity = common::kInvalidEntityId;
    float moveSpeed               = 0.0f;
    float inputSensitivity        = 0.0f;
    float speedBoostScale         = 0.0f;
    float speedSlowScale          = 0.0f;
};

struct DebugViewerCallbacks
{
    std::function<void(const common::FrameContext &, engine::Runtime &)> beforeTick{};
    std::function<void(const common::FrameContext &, engine::Runtime &)> afterTick{};
};

class CRESSIM_NEO_VIEWER_API DebugViewerApp
{
public:
    DebugViewerApp();
    ~DebugViewerApp();

    DebugViewerApp(const DebugViewerApp &)            = delete;
    DebugViewerApp &operator=(const DebugViewerApp &) = delete;
    DebugViewerApp(DebugViewerApp &&) noexcept;
    DebugViewerApp &operator=(DebugViewerApp &&) noexcept;

    bool initialize(DebugViewerAppDesc desc, engine::RuntimeConfig &inOutRuntimeConfig);
    bool run(engine::Runtime &runtime, DebugViewerCameraBinding camera,
             DebugViewerCallbacks callbacks = {});
    void requestExit();
    void shutdown();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::viewer

#endif // CRESSIM_NEO_VIEWER_DEBUG_VIEWER_APP_H
