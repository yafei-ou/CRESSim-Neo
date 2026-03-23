#include "viewer/debug_viewer_app.h"

#include "common/logger.h"
#include "common/math_utils_runtime.h"
#include "engine/components.h"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace cressim::neo::viewer
{

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::TransformComponent;

std::vector<common::EntityId> sortedCameraEntities(const cressim::neo::engine::World& world)
{
    std::vector<graphics::CameraData> cameras;
    cameras.reserve(world.cameras().size());
    for (const graphics::CameraData& camera : world.cameras())
    {
        if (camera.entityId == common::kInvalidEntityId || camera.cameraSlot == 0xffffffffu)
        {
            continue;
        }
        cameras.push_back(camera);
    }

    std::sort(cameras.begin(), cameras.end(),
              [](const graphics::CameraData& lhs, const graphics::CameraData& rhs)
              {
                  if (lhs.renderOrder != rhs.renderOrder)
                  {
                      return lhs.renderOrder < rhs.renderOrder;
                  }
                  return lhs.entityId < rhs.entityId;
              });

    std::vector<common::EntityId> entities;
    entities.reserve(cameras.size());
    for (const graphics::CameraData& camera : cameras)
    {
        entities.push_back(camera.entityId);
    }
    return entities;
}

common::EntityId cyclePresentedCamera(const cressim::neo::engine::World& world,
                                      common::EntityId currentCameraEntity, int direction)
{
    const std::vector<common::EntityId> cameras = sortedCameraEntities(world);
    if (cameras.empty())
    {
        return common::kInvalidEntityId;
    }

    const auto currentIt = std::find(cameras.begin(), cameras.end(), currentCameraEntity);
    if (currentIt == cameras.end())
    {
        return cameras.front();
    }

    const std::ptrdiff_t count        = static_cast<std::ptrdiff_t>(cameras.size());
    const std::ptrdiff_t currentIndex = std::distance(cameras.begin(), currentIt);
    const std::ptrdiff_t wrappedIndex =
        (currentIndex + static_cast<std::ptrdiff_t>(direction) + count) % count;
    return cameras[static_cast<std::size_t>(wrappedIndex)];
}

float clampSpeed(float speed, float minSpeed, float maxSpeed)
{
    const float clampedMin = std::max(minSpeed, 0.001f);
    const float clampedMax = std::max(maxSpeed, clampedMin);
    return std::max(clampedMin, std::min(speed, clampedMax));
}

float clampPitch(float pitchDegrees)
{
    return std::max(-89.0f, std::min(89.0f, pitchDegrees));
}

Diligent::QuaternionF cameraOrientationFromYawPitch(float yawDegrees, float pitchDegrees)
{
    return common::runtime_math::quaternionFromEulerDegrees(yawDegrees, 0.0f, -pitchDegrees);
}

Diligent::float3 rotateVector(const Diligent::QuaternionF& rotation, const Diligent::float3& vector)
{
    return rotation.RotateVector(vector);
}

void yawPitchFromRotation(const Diligent::QuaternionF& rotation, float& outYawDegrees,
                          float& outPitchDegrees)
{
    const Diligent::float3 forward = common::runtime_math::safeNormalize(
        rotateVector(rotation, {0.0f, 0.0f, 1.0f}), Diligent::float3{0.0f, 0.0f, 1.0f});
    outYawDegrees   = common::runtime_math::radiansToDegrees(std::atan2(forward.x, forward.z));
    outPitchDegrees = common::runtime_math::radiansToDegrees(
        std::asin(std::max(-1.0f, std::min(forward.y, 1.0f))));
}

bool isPressed(GLFWwindow* window, int key)
{
    return key >= 0 && glfwGetKey(window, key) == GLFW_PRESS;
}

gpu::GpuRenderTargetDesc resolveDefaultPresentationTargetDesc(engine::Runtime& runtime,
                                                              std::uint32_t requestedWidth,
                                                              std::uint32_t requestedHeight)
{
    gpu::GpuRenderTargetDesc resolvedDesc{};
    gpu::GpuDevice* const device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        resolvedDesc.width  = requestedWidth;
        resolvedDesc.height = requestedHeight;
        return resolvedDesc;
    }

    gpu::GpuRenderTargetSystem& renderTargets      = device->renderTargetSystem();
    const gpu::GpuRenderTargetHandle defaultTarget = renderTargets.defaultRenderTarget();
    if (renderTargets.isValidRenderTarget(defaultTarget))
    {
        (void)renderTargets.resizeRenderTarget(defaultTarget, requestedWidth, requestedHeight);
        if (renderTargets.tryGetRenderTargetDesc(defaultTarget, resolvedDesc))
        {
            return resolvedDesc;
        }
    }

    resolvedDesc.width  = requestedWidth;
    resolvedDesc.height = requestedHeight;
    return resolvedDesc;
}

} // namespace

class DebugViewerApp::Impl
{
public:
    struct CameraState
    {
        Diligent::float3 position{};
        float yawDegrees       = 0.0f;
        float pitchDegrees     = 0.0f;
        float moveSpeed        = 3.0f;
        float inputSensitivity = 0.08f;
        float speedBoostScale  = 3.0f;
        float speedSlowScale   = 0.35f;
    };

    struct CameraOutputSettings
    {
        gpu::CameraOutputBinding output{};
        std::uint32_t outputWidth  = 0u;
        std::uint32_t outputHeight = 0u;
    };

    bool initialize(DebugViewerAppDesc desc, engine::RuntimeConfig& inOutRuntimeConfig)
    {
        shutdown();

        mDesc           = desc;
        mDesc.width     = std::max(mDesc.width, 1u);
        mDesc.height    = std::max(mDesc.height, 1u);
        mDesc.moveSpeed = clampSpeed(mDesc.moveSpeed, mDesc.minMoveSpeed, mDesc.maxMoveSpeed);
        mDesc.inputSensitivity = common::runtime_math::clampPositive(mDesc.inputSensitivity, 0.08f);
        mDesc.speedBoostScale  = common::runtime_math::clampPositive(mDesc.speedBoostScale, 1.0f);
        mDesc.speedSlowScale   = common::runtime_math::clampPositive(mDesc.speedSlowScale, 0.35f);
        mDesc.fixedDeltaSeconds =
            common::runtime_math::clampPositive(mDesc.fixedDeltaSeconds, 1.0f / 60.0f);
        mShowStats = mDesc.showStats;

        if (mDesc.startFullscreen && mDesc.startFullscreenWindowed)
        {
            CRESSIM_LOG_ERROR(
                "DebugViewerApp: fullscreen and fullscreen-windowed cannot both be enabled.");
            return false;
        }

        std::uint32_t effectiveWidth  = mDesc.width;
        std::uint32_t effectiveHeight = mDesc.height;

        inOutRuntimeConfig.gpuDeviceDesc.defaultRenderTargetDesc.colorFormat =
            Diligent::TEX_FORMAT_UNKNOWN;
        inOutRuntimeConfig.gpuDeviceDesc.presentation.enabled      = mDesc.windowEnabled;
        inOutRuntimeConfig.gpuDeviceDesc.presentation.syncInterval = mDesc.vSync ? 1u : 0u;
        inOutRuntimeConfig.gpuDeviceDesc.presentation.preferredColorFormat =
            Diligent::TEX_FORMAT_UNKNOWN;
        inOutRuntimeConfig.gpuDeviceDesc.presentation.nativeWindow     = nullptr;
        inOutRuntimeConfig.gpuDeviceDesc.presentation.nativeWindowId   = 0;
        inOutRuntimeConfig.gpuDeviceDesc.presentation.nativeDisplay    = nullptr;
        inOutRuntimeConfig.gpuDeviceDesc.presentation.nativeConnection = nullptr;

        if (!mDesc.windowEnabled)
        {
            if (mDesc.startFullscreen || mDesc.startFullscreenWindowed)
            {
                CRESSIM_LOG_WARNING(
                    "DebugViewerApp: fullscreen options require windowEnabled=true; ignoring.");
            }
            inOutRuntimeConfig.gpuDeviceDesc.defaultRenderTargetDesc.width  = effectiveWidth;
            inOutRuntimeConfig.gpuDeviceDesc.defaultRenderTargetDesc.height = effectiveHeight;
            mInitialized                                                    = true;
            mExitRequested.store(false);
            return true;
        }

        if (glfwInit() != GLFW_TRUE)
        {
            CRESSIM_LOG_ERROR("DebugViewerApp: failed to initialize GLFW.");
            return false;
        }
        mGlfwInitialized = true;

        GLFWmonitor* targetMonitor = nullptr;
        int monitorPosX            = 0;
        int monitorPosY            = 0;
        if (mDesc.startFullscreen || mDesc.startFullscreenWindowed)
        {
            GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
            if (primaryMonitor == nullptr)
            {
                CRESSIM_LOG_ERROR(
                    "DebugViewerApp: failed to resolve primary monitor for fullscreen.");
                shutdown();
                return false;
            }

            const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);
            if (mode == nullptr)
            {
                CRESSIM_LOG_ERROR("DebugViewerApp: failed to query monitor mode for fullscreen.");
                shutdown();
                return false;
            }

            effectiveWidth  = static_cast<std::uint32_t>(std::max(mode->width, 1));
            effectiveHeight = static_cast<std::uint32_t>(std::max(mode->height, 1));
            if (mDesc.startFullscreen)
            {
                targetMonitor = primaryMonitor;
            }
            else
            {
                glfwGetMonitorPos(primaryMonitor, &monitorPosX, &monitorPosY);
            }
        }

        mDesc.width                                                     = effectiveWidth;
        mDesc.height                                                    = effectiveHeight;
        inOutRuntimeConfig.gpuDeviceDesc.defaultRenderTargetDesc.width  = effectiveWidth;
        inOutRuntimeConfig.gpuDeviceDesc.defaultRenderTargetDesc.height = effectiveHeight;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, mDesc.windowVisible ? GLFW_TRUE : GLFW_FALSE);
        if (mDesc.startFullscreenWindowed)
        {
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        }
        mWindow = glfwCreateWindow(static_cast<int>(mDesc.width), static_cast<int>(mDesc.height),
                                   mDesc.windowTitle.c_str(), targetMonitor, nullptr);
        if (mWindow == nullptr)
        {
            CRESSIM_LOG_ERROR("DebugViewerApp: failed to create window.");
            shutdown();
            return false;
        }

        if (mDesc.startFullscreenWindowed)
        {
            glfwSetWindowPos(mWindow, monitorPosX, monitorPosY);
            glfwSetWindowSize(mWindow, static_cast<int>(mDesc.width),
                              static_cast<int>(mDesc.height));
        }

        glfwSetWindowUserPointer(mWindow, this);
        glfwSetScrollCallback(mWindow, &Impl::scrollCallback);

#if defined(_WIN32)
        inOutRuntimeConfig.gpuDeviceDesc.presentation.nativeWindow = glfwGetWin32Window(mWindow);
#elif defined(__linux__)
        inOutRuntimeConfig.gpuDeviceDesc.presentation.nativeWindowId =
            static_cast<std::uint64_t>(glfwGetX11Window(mWindow));
        inOutRuntimeConfig.gpuDeviceDesc.presentation.nativeDisplay = glfwGetX11Display();
#elif defined(__APPLE__)
        inOutRuntimeConfig.gpuDeviceDesc.presentation.nativeWindow = glfwGetCocoaWindow(mWindow);
#endif

        mInitialized = true;
        mExitRequested.store(false);
        return true;
    }

    bool run(engine::Runtime& runtime, DebugViewerCameraBinding cameraBinding,
             DebugViewerCallbacks callbacks)
    {
        if (!mInitialized)
        {
            return false;
        }
        if (cameraBinding.cameraEntity == common::kInvalidEntityId)
        {
            CRESSIM_LOG_ERROR("DebugViewerApp: camera binding requires a valid entity id.");
            return false;
        }
        if (!mDesc.windowEnabled && mDesc.maxFrames == 0)
        {
            CRESSIM_LOG_ERROR("DebugViewerApp: maxFrames must be > 0 when window is disabled.");
            return false;
        }

        if (runtime.getGpuDevice() == nullptr)
        {
            CRESSIM_LOG_ERROR("DebugViewerApp: runtime has no graphics device.");
            return false;
        }

        auto& world = runtime.getWorld();
        if (!world.isAlive(cameraBinding.cameraEntity))
        {
            CRESSIM_LOG_ERROR("DebugViewerApp: bound camera entity does not exist.");
            return false;
        }

        const std::optional<CameraComponent> existingCamera =
            world.tryGetCamera(cameraBinding.cameraEntity);
        if (!existingCamera)
        {
            CRESSIM_LOG_ERROR("DebugViewerApp: bound entity has no CameraComponent.");
            return false;
        }

        TransformComponent transform{};
        const std::optional<TransformComponent> existingTransform =
            world.tryGetTransform(cameraBinding.cameraEntity);
        if (existingTransform)
        {
            transform = *existingTransform;
        }
        else
        {
            world.setTransform(cameraBinding.cameraEntity, transform);
        }

        CameraComponent camera = *existingCamera;
        CameraState cameraState{};
        cameraState.position = transform.worldTransform.position;
        yawPitchFromRotation(transform.worldTransform.rotation, cameraState.yawDegrees,
                             cameraState.pitchDegrees);
        cameraState.pitchDegrees = clampPitch(cameraState.pitchDegrees);
        cameraState.moveSpeed =
            clampSpeed(cameraBinding.moveSpeed > 0.0f ? cameraBinding.moveSpeed : mDesc.moveSpeed,
                       mDesc.minMoveSpeed, mDesc.maxMoveSpeed);
        cameraState.inputSensitivity = common::runtime_math::clampPositive(
            cameraBinding.inputSensitivity > 0.0f ? cameraBinding.inputSensitivity
                                                  : mDesc.inputSensitivity,
            0.08f);
        cameraState.speedBoostScale = common::runtime_math::clampPositive(
            cameraBinding.speedBoostScale > 0.0f ? cameraBinding.speedBoostScale
                                                 : mDesc.speedBoostScale,
            1.0f);
        cameraState.speedSlowScale = common::runtime_math::clampPositive(
            cameraBinding.speedSlowScale > 0.0f ? cameraBinding.speedSlowScale
                                                : mDesc.speedSlowScale,
            0.35f);

        const CameraState initialCameraState = cameraState;
        mLookActive                          = false;
        mAccumulatedScrollY                  = 0.0;
        mKeyIsDown.clear();
        mExitRequested.store(false);

        mLastTickTime = std::chrono::steady_clock::now();
        common::FrameContext frame{};
        frame.deltaSeconds                          = mDesc.fixedDeltaSeconds;
        common::EntityId presentedCameraEntity      = cameraBinding.cameraEntity;
        common::EntityId outputOverrideCameraEntity = common::kInvalidEntityId;
        runtime.setRenderFrameOptions(graphics::RenderFrameOptions{presentedCameraEntity});

        while (!mExitRequested.load())
        {
            if (mDesc.windowEnabled && mWindow != nullptr)
            {
                glfwPollEvents();
                if (glfwWindowShouldClose(mWindow))
                {
                    break;
                }
                if (consumeKeyPress(mDesc.keymap.quit))
                {
                    break;
                }
            }

            const float deltaSeconds = computeDeltaSeconds();
            frame.deltaSeconds       = deltaSeconds;
            frame.timeSeconds += static_cast<double>(deltaSeconds);

            if (consumeKeyPress(mDesc.keymap.toggleStats))
            {
                mShowStats = !mShowStats;
            }

            if (consumeKeyPress(mDesc.keymap.resetCamera))
            {
                cameraState = initialCameraState;
            }
            if (consumeKeyPress(mDesc.keymap.cyclePresentedCameraPrevious))
            {
                const common::EntityId nextEntity =
                    cyclePresentedCamera(world, presentedCameraEntity, -1);
                if (nextEntity != presentedCameraEntity)
                {
                    presentedCameraEntity = nextEntity;
                    CRESSIM_LOG_INFO("viewer presenting camera entity=", presentedCameraEntity);
                }
            }
            if (consumeKeyPress(mDesc.keymap.cyclePresentedCameraNext))
            {
                const common::EntityId nextEntity =
                    cyclePresentedCamera(world, presentedCameraEntity, 1);
                if (nextEntity != presentedCameraEntity)
                {
                    presentedCameraEntity = nextEntity;
                    CRESSIM_LOG_INFO("viewer presenting camera entity=", presentedCameraEntity);
                }
            }
            if (!world.isAlive(presentedCameraEntity))
            {
                presentedCameraEntity = cameraBinding.cameraEntity;
            }

            const InputState input = sampleInput();
            applyInputToCamera(cameraState, input, deltaSeconds);

            transform.worldTransform.position = cameraState.position;
            transform.worldTransform.rotation =
                cameraOrientationFromYawPitch(cameraState.yawDegrees, cameraState.pitchDegrees);
            world.setTransform(cameraBinding.cameraEntity, transform);

            int outputWidth  = static_cast<int>(mDesc.width);
            int outputHeight = static_cast<int>(mDesc.height);
            if (mDesc.windowEnabled && mWindow != nullptr)
            {
                glfwGetFramebufferSize(mWindow, &outputWidth, &outputHeight);
            }
            world.setCamera(cameraBinding.cameraEntity, camera);

            gpu::GpuRenderTargetDesc presentationTargetDesc{};
            presentationTargetDesc.width  = static_cast<std::uint32_t>(std::max(outputWidth, 1));
            presentationTargetDesc.height = static_cast<std::uint32_t>(std::max(outputHeight, 1));
            if (mDesc.windowEnabled)
            {
                presentationTargetDesc = resolveDefaultPresentationTargetDesc(
                    runtime, presentationTargetDesc.width, presentationTargetDesc.height);

                if (outputOverrideCameraEntity != common::kInvalidEntityId &&
                    outputOverrideCameraEntity != presentedCameraEntity)
                {
                    restorePresentedCameraOutput(world, outputOverrideCameraEntity);
                    outputOverrideCameraEntity = common::kInvalidEntityId;
                }

                applyPresentedCameraOutput(world, presentedCameraEntity,
                                           presentationTargetDesc.width,
                                           presentationTargetDesc.height);
                outputOverrideCameraEntity = presentedCameraEntity;
            }

            runtime.setRenderFrameOptions(graphics::RenderFrameOptions{presentedCameraEntity});

            frame.frameIndex += 1u;

            if (callbacks.beforeTick)
            {
                callbacks.beforeTick(frame, runtime);
            }

            runtime.tick(frame);

            if (callbacks.afterTick)
            {
                callbacks.afterTick(frame, runtime);
            }

            updateWindowTitle(frame.deltaSeconds);

            if (mShowStats && mDesc.statsIntervalFrames > 0 &&
                frame.frameIndex % mDesc.statsIntervalFrames == 0)
            {
                const float fps = frame.deltaSeconds > 0.0f ? (1.0f / frame.deltaSeconds) : 0.0f;
                CRESSIM_LOG_INFO("viewer frame=", frame.frameIndex, " fps=", fps, " camPos=(",
                                 cameraState.position.x, ", ", cameraState.position.y, ", ",
                                 cameraState.position.z, ") yaw=", cameraState.yawDegrees,
                                 " pitch=", cameraState.pitchDegrees,
                                 " speed=", cameraState.moveSpeed);
            }

            if (mDesc.maxFrames > 0 && frame.frameIndex >= mDesc.maxFrames)
            {
                break;
            }
        }

        if (mWindow != nullptr)
        {
            if (mWindowTitleStatsActive)
            {
                glfwSetWindowTitle(mWindow, mDesc.windowTitle.c_str());
            }
            glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        restorePresentedCameraOutput(world, outputOverrideCameraEntity);
        runtime.setRenderFrameOptions(graphics::RenderFrameOptions{});
        mLookActive = false;
        return true;
    }

    void requestExit()
    {
        mExitRequested.store(true);
    }

    void shutdown()
    {
        if (mWindow != nullptr)
        {
            glfwDestroyWindow(mWindow);
            mWindow = nullptr;
        }
        if (mGlfwInitialized)
        {
            glfwTerminate();
            mGlfwInitialized = false;
        }
        mInitialized                   = false;
        mLookActive                    = false;
        mAccumulatedScrollY            = 0.0;
        mWindowTitleStatsActive        = false;
        mWindowTitleAccumulatedSeconds = 0.0f;
        mWindowTitleAccumulatedFrames  = 0u;
        mPresentedCameraOverrides.clear();
        mKeyIsDown.clear();
    }

private:
    struct InputState
    {
        Diligent::float3 moveDirection{};
        float mouseDeltaX = 0.0f;
        float mouseDeltaY = 0.0f;
        float scrollDelta = 0.0f;
        bool boost        = false;
        bool slow         = false;
    };

    void updateWindowTitle(float deltaSeconds)
    {
        if (!mDesc.windowEnabled || mWindow == nullptr)
        {
            return;
        }

        if (!mShowStats)
        {
            if (mWindowTitleStatsActive)
            {
                glfwSetWindowTitle(mWindow, mDesc.windowTitle.c_str());
                mWindowTitleStatsActive = false;
            }
            return;
        }

        mWindowTitleAccumulatedSeconds += deltaSeconds;
        mWindowTitleAccumulatedFrames += 1u;
        if (mWindowTitleAccumulatedSeconds < 0.25f)
        {
            return;
        }

        const float averageFrameSeconds =
            mWindowTitleAccumulatedSeconds /
            static_cast<float>(std::max(mWindowTitleAccumulatedFrames, 1u));
        const float fps = averageFrameSeconds > 0.0f ? (1.0f / averageFrameSeconds) : 0.0f;
        const float frameMilliseconds = averageFrameSeconds * 1000.0f;

        std::ostringstream title;
        title << mDesc.windowTitle << " | " << std::fixed << std::setprecision(1) << fps
              << " FPS | " << std::setprecision(2) << frameMilliseconds << " ms";
        glfwSetWindowTitle(mWindow, title.str().c_str());

        mWindowTitleStatsActive        = true;
        mWindowTitleAccumulatedSeconds = 0.0f;
        mWindowTitleAccumulatedFrames  = 0u;
    }

    void applyPresentedCameraOutput(engine::World& world, common::EntityId cameraEntity,
                                    std::uint32_t width, std::uint32_t height)
    {
        if (cameraEntity == common::kInvalidEntityId)
        {
            return;
        }

        const std::optional<CameraComponent> existing = world.tryGetCamera(cameraEntity);
        if (!existing)
        {
            mPresentedCameraOverrides.erase(cameraEntity);
            return;
        }

        if (mPresentedCameraOverrides.find(cameraEntity) == mPresentedCameraOverrides.end())
        {
            mPresentedCameraOverrides.emplace(
                cameraEntity, CameraOutputSettings{existing->output, existing->outputWidth,
                                                   existing->outputHeight});
        }

        CameraComponent updated = *existing;
        updated.outputWidth     = width;
        updated.outputHeight    = height;
        updated.output.mode     = gpu::CameraOutputMode::ManagedPrimary;
        updated.output.binding  = {};
        world.setCamera(cameraEntity, updated);
    }

    void restorePresentedCameraOutput(engine::World& world, common::EntityId cameraEntity)
    {
        if (cameraEntity == common::kInvalidEntityId)
        {
            return;
        }

        const auto it = mPresentedCameraOverrides.find(cameraEntity);
        if (it == mPresentedCameraOverrides.end())
        {
            return;
        }

        const std::optional<CameraComponent> existing = world.tryGetCamera(cameraEntity);
        if (existing)
        {
            CameraComponent restored = *existing;
            restored.output          = it->second.output;
            restored.outputWidth     = it->second.outputWidth;
            restored.outputHeight    = it->second.outputHeight;
            world.setCamera(cameraEntity, restored);
        }

        mPresentedCameraOverrides.erase(it);
    }

    static void scrollCallback(GLFWwindow* window, double, double yOffset)
    {
        if (window == nullptr)
        {
            return;
        }
        auto* self = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (self != nullptr)
        {
            self->mAccumulatedScrollY += yOffset;
        }
    }

    bool consumeKeyPress(int key)
    {
        if (!mDesc.windowEnabled || mWindow == nullptr || key < 0)
        {
            return false;
        }

        const bool down    = glfwGetKey(mWindow, key) == GLFW_PRESS;
        const bool wasDown = mKeyIsDown[key];
        mKeyIsDown[key]    = down;
        return down && !wasDown;
    }

    bool isKeyDown(int primary, int secondary = -1) const
    {
        if (!mDesc.windowEnabled || mWindow == nullptr)
        {
            return false;
        }
        if (isPressed(mWindow, primary))
        {
            return true;
        }
        return secondary >= 0 && isPressed(mWindow, secondary);
    }

    float computeDeltaSeconds()
    {
        if (!mDesc.windowEnabled)
        {
            return std::max(1.0f / 240.0f, std::min(mDesc.fixedDeltaSeconds, 0.1f));
        }

        const auto now                              = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = now - mLastTickTime;
        mLastTickTime                               = now;
        const float dt                              = static_cast<float>(elapsed.count());
        return std::max(1.0f / 240.0f, std::min(dt, 0.1f));
    }

    InputState sampleInput()
    {
        InputState out{};

        if (!mDesc.windowEnabled || mWindow == nullptr)
        {
            return out;
        }

        out.moveDirection.x += isKeyDown(mDesc.keymap.moveRight) ? 1.0f : 0.0f;
        out.moveDirection.x -= isKeyDown(mDesc.keymap.moveLeft) ? 1.0f : 0.0f;
        out.moveDirection.y += isKeyDown(mDesc.keymap.moveUp) ? 1.0f : 0.0f;
        out.moveDirection.y -= isKeyDown(mDesc.keymap.moveDown) ? 1.0f : 0.0f;
        out.moveDirection.z += isKeyDown(mDesc.keymap.moveForward) ? 1.0f : 0.0f;
        out.moveDirection.z -= isKeyDown(mDesc.keymap.moveBackward) ? 1.0f : 0.0f;

        out.boost = isKeyDown(mDesc.keymap.speedBoostPrimary, mDesc.keymap.speedBoostSecondary);
        out.slow  = isKeyDown(mDesc.keymap.speedSlowPrimary, mDesc.keymap.speedSlowSecondary);

        const bool lookDown = (glfwGetMouseButton(mWindow, mDesc.keymap.lookButton) == GLFW_PRESS);
        if (lookDown && !mLookActive)
        {
            mLookActive = true;
            glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            if (glfwRawMouseMotionSupported() == GLFW_TRUE)
            {
                glfwSetInputMode(mWindow, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
            }
            glfwGetCursorPos(mWindow, &mLastCursorX, &mLastCursorY);
        }
        else if (!lookDown && mLookActive)
        {
            mLookActive = false;
            glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            if (glfwRawMouseMotionSupported() == GLFW_TRUE)
            {
                glfwSetInputMode(mWindow, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
            }
        }

        if (mLookActive)
        {
            double cursorX = mLastCursorX;
            double cursorY = mLastCursorY;
            glfwGetCursorPos(mWindow, &cursorX, &cursorY);
            out.mouseDeltaX = static_cast<float>(cursorX - mLastCursorX);
            out.mouseDeltaY = static_cast<float>(cursorY - mLastCursorY);
            mLastCursorX    = cursorX;
            mLastCursorY    = cursorY;
        }

        out.scrollDelta     = static_cast<float>(mAccumulatedScrollY);
        mAccumulatedScrollY = 0.0;
        return out;
    }

    void applyInputToCamera(CameraState& camera, const InputState& input, float deltaSeconds) const
    {
        if (input.scrollDelta != 0.0f)
        {
            const float factor = std::max(0.1f, 1.0f + input.scrollDelta * mDesc.wheelSpeedScale);
            camera.moveSpeed =
                clampSpeed(camera.moveSpeed * factor, mDesc.minMoveSpeed, mDesc.maxMoveSpeed);
        }

        camera.yawDegrees += input.mouseDeltaX * camera.inputSensitivity;
        camera.pitchDegrees -= input.mouseDeltaY * camera.inputSensitivity;
        camera.pitchDegrees = clampPitch(camera.pitchDegrees);

        const Diligent::QuaternionF orientation =
            cameraOrientationFromYawPitch(camera.yawDegrees, camera.pitchDegrees);
        const Diligent::float3 forward = common::runtime_math::safeNormalize(
            rotateVector(orientation, {0.0f, 0.0f, 1.0f}), Diligent::float3{0.0f, 0.0f, 1.0f});
        const Diligent::float3 right = common::runtime_math::safeNormalize(
            rotateVector(orientation, {1.0f, 0.0f, 0.0f}), Diligent::float3{1.0f, 0.0f, 0.0f});
        const Diligent::float3 worldUp{0.0f, 1.0f, 0.0f};

        Diligent::float3 worldDirection = right * input.moveDirection.x +
                                          worldUp * input.moveDirection.y +
                                          forward * input.moveDirection.z;
        worldDirection                  = common::runtime_math::safeNormalize(worldDirection);

        float movementSpeed = camera.moveSpeed;
        if (input.boost)
        {
            movementSpeed *= camera.speedBoostScale;
        }
        if (input.slow)
        {
            movementSpeed *= camera.speedSlowScale;
        }

        camera.position = camera.position + worldDirection * (movementSpeed * deltaSeconds);
    }

private:
    DebugViewerAppDesc mDesc{};
    GLFWwindow* mWindow                         = nullptr;
    bool mInitialized                           = false;
    bool mGlfwInitialized                       = false;
    bool mShowStats                             = true;
    bool mWindowTitleStatsActive                = false;
    bool mLookActive                            = false;
    double mLastCursorX                         = 0.0;
    double mLastCursorY                         = 0.0;
    double mAccumulatedScrollY                  = 0.0;
    float mWindowTitleAccumulatedSeconds        = 0.0f;
    std::uint32_t mWindowTitleAccumulatedFrames = 0u;
    std::unordered_map<common::EntityId, CameraOutputSettings> mPresentedCameraOverrides;
    std::unordered_map<int, bool> mKeyIsDown;
    std::chrono::steady_clock::time_point mLastTickTime{};
    std::atomic<bool> mExitRequested{false};
};

DebugViewerApp::DebugViewerApp() : mImpl(std::make_unique<Impl>()) {}

DebugViewerApp::~DebugViewerApp() = default;

DebugViewerApp::DebugViewerApp(DebugViewerApp&&) noexcept            = default;
DebugViewerApp& DebugViewerApp::operator=(DebugViewerApp&&) noexcept = default;

bool DebugViewerApp::initialize(DebugViewerAppDesc desc, engine::RuntimeConfig& inOutRuntimeConfig)
{
    return mImpl->initialize(std::move(desc), inOutRuntimeConfig);
}

bool DebugViewerApp::run(engine::Runtime& runtime, DebugViewerCameraBinding camera,
                         DebugViewerCallbacks callbacks)
{
    return mImpl->run(runtime, camera, std::move(callbacks));
}

void DebugViewerApp::requestExit()
{
    mImpl->requestExit();
}

void DebugViewerApp::shutdown()
{
    mImpl->shutdown();
}

} // namespace cressim::neo::viewer
