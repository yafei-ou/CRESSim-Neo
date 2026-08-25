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

/// @file debug_viewer_app.h
/// @brief Interactive GLFW/ImGui desktop debug viewer application, fly camera controller, and video capture runner.

namespace cressim::neo::viewer
{

/// @brief GLFW keycode mappings for first-person fly camera navigation and application hotkeys.
struct DebugViewerKeymap
{
    int moveForward  = 87; ///< Forward movement key (default: W).
    int moveBackward = 83; ///< Backward movement key (default: S).
    int moveLeft     = 65; ///< Strafe left key (default: A).
    int moveRight    = 68; ///< Strafe right key (default: D).
    int moveUp       = 69; ///< Ascend elevation key (default: E).
    int moveDown     = 81; ///< Descend elevation key (default: Q).

    int speedBoostPrimary   = 340; ///< Primary movement speed boost key (default: Left Shift).
    int speedBoostSecondary = 344; ///< Secondary movement speed boost key (default: Right Shift).
    int speedSlowPrimary    = 341; ///< Primary movement precision slow key (default: Left Ctrl).
    int speedSlowSecondary  = 345; ///< Secondary movement precision slow key (default: Right Ctrl).

    int lookButton = 1; ///< Mouse button held to rotate the camera view (default: Right Mouse Button = 1).

    int resetCamera                  = 70;  ///< Reset camera pose to initial authored transform (default: F).
    int toggleStats                  = 72;  ///< Toggle on-screen statistics and telemetry overlay (default: H).
    int togglePresentedOutputMode    = 85;  ///< Toggle between Color, Depth, and Segmentation display modes (default: U).
    int cyclePresentedCameraPrevious = 91;  ///< Switch active presentation view to previous camera (default: [).
    int cyclePresentedCameraNext     = 93;  ///< Switch active presentation view to next camera (default: ]).
    int cyclePresentedProbePrevious  = 44;  ///< Cycle previous probe or sensor attachment (default: ,).
    int cyclePresentedProbeNext      = 46;  ///< Cycle next probe or sensor attachment (default: .).
    int quit                         = 256; ///< Exit application key (default: Escape).
};

/// @brief Launch configuration parameters for the interactive debug viewer application.
struct DebugViewerAppDesc
{
    std::string windowTitle      = "CRESSim Neo Debug Viewer"; ///< Window caption text.
    std::uint32_t width          = 1280;                        ///< Window client area width in pixels.
    std::uint32_t height         = 720;                         ///< Window client area height in pixels.
    bool windowEnabled           = true;                        ///< Whether a desktop window is opened (false runs headless).
    bool windowVisible           = true;                        ///< Initial window visibility.
    bool startFullscreen         = false;                       ///< Start in exclusive fullscreen mode.
    bool startFullscreenWindowed = false;                       ///< Start in borderless windowed fullscreen mode.
    bool vSync                   = true;                        ///< Enable vertical synchronization.

    float inputSensitivity = 0.08f; ///< Mouse look sensitivity multiplier.
    float moveSpeed        = 3.0f;  ///< Base fly camera movement speed in units per second.
    float speedBoostScale  = 3.0f;  ///< Movement speed multiplier when holding sprint key.
    float speedSlowScale   = 0.35f; ///< Movement speed multiplier when holding precision slow key.
    float wheelSpeedScale  = 0.12f; ///< Speed adjustment increment from mouse scroll wheel.
    float minMoveSpeed     = 0.25f; ///< Minimum clamp on adjustable camera movement speed.
    float maxMoveSpeed     = 80.0f; ///< Maximum clamp on adjustable camera movement speed.

    float fixedDeltaSeconds           = 1.0f / 60.0f; ///< Target fixed delta time step in seconds.
    bool useFixedTimestep             = false;        ///< Enforce fixed time step rather than elapsed wall-clock delta.
    bool stepSimulation               = true;         ///< Step the physics simulation on each tick.
    std::uint64_t maxFrames           = 0;            ///< Maximum frames to execute before auto-quitting (0 runs indefinitely).
    bool showStats                    = true;         ///< Display performance metrics overlay.
    bool enableDebugParticles         = false;        ///< Enable particle physics debug visual overlays.
    std::uint32_t statsIntervalFrames = 120;          ///< Interval in frames for logging performance statistics.

    std::string captureVideoPath{};                 ///< Output MP4 path to encode and save video stream via FFmpeg.
    std::uint32_t captureFps                  = 30; ///< Video recording output frame rate.
    std::uint32_t simulationFps               = 60; ///< Simulation stepping rate during offline video capture.
    std::uint32_t captureSwitchIntervalFrames = 0;  ///< Interval in captured frames to rotate cameras (0 disables auto-rotation).

    DebugViewerKeymap keymap{}; ///< Keyboard and mouse button configuration.
};

/// @brief Camera entity binding and movement tuning parameters for the viewer controller.
struct DebugViewerCameraBinding
{
    common::EntityId cameraEntity = common::kInvalidEntityId; ///< Entity ID of camera driven by fly controls.
    float moveSpeed               = 0.0f;                     ///< Initial movement speed override (0 uses app default).
    float inputSensitivity        = 0.0f;                     ///< Mouse look sensitivity override (0 uses app default).
    float speedBoostScale         = 0.0f;                     ///< Sprint boost scale override (0 uses app default).
    float speedSlowScale          = 0.0f;                     ///< Precision slow scale override (0 uses app default).
};

/// @brief Frame event callbacks invoked before and after each simulation tick.
struct DebugViewerCallbacks
{
    std::function<void(const common::FrameContext &, engine::Runtime &)> beforeTick{}; ///< Invoked before physics stepping and systems tick.
    std::function<void(const common::FrameContext &, engine::Runtime &)> afterTick{};  ///< Invoked after physics stepping and render pass dispatch.
};

/// @brief Standalone desktop application hosting GLFW windowing, user input, camera controllers, and main rendering loop.
class CRESSIM_NEO_VIEWER_API DebugViewerApp
{
public:
    /// @brief Default constructor.
    DebugViewerApp();
    /// @brief Destructor releasing window and viewer resources.
    ~DebugViewerApp();

    DebugViewerApp(const DebugViewerApp &)            = delete;
    DebugViewerApp &operator=(const DebugViewerApp &) = delete;
    /// @brief Move constructor.
    /// @param other Instance to move from.
    DebugViewerApp(DebugViewerApp &&other) noexcept;
    /// @brief Move assignment operator.
    /// @param other Instance to move from.
    /// @return Reference to this.
    DebugViewerApp &operator=(DebugViewerApp &&other) noexcept;

    /// @brief Initializes desktop window, input handlers, and configures runtime presentation targets.
    /// @param desc Viewer application configuration settings.
    /// @param inOutRuntimeConfig Engine runtime configuration updated with native window handles.
    /// @return True on success.
    bool initialize(DebugViewerAppDesc desc, engine::RuntimeConfig &inOutRuntimeConfig);
    /// @brief Enters the application main loop, processing events, ticking simulation, and rendering frames until exit.
    /// @param runtime Active engine runtime instance.
    /// @param camera Driven camera entity and movement speed settings.
    /// @param callbacks Optional pre-tick and post-tick hook callbacks.
    /// @return True if execution completed cleanly without errors.
    bool run(engine::Runtime &runtime, DebugViewerCameraBinding camera,
             DebugViewerCallbacks callbacks = {});
    /// @brief Requests the application main loop to terminate after the current frame.
    void requestExit();
    /// @brief Destroys the desktop window and releases viewer subsystems.
    void shutdown();

private:
    class CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::viewer

#endif // CRESSIM_NEO_VIEWER_DEBUG_VIEWER_APP_H
