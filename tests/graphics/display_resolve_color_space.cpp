#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuDevice;
using cressim::neo::gpu::GpuRenderTargetBinding;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::graphics::RenderFrameOptions;

bool isValidReadback(const GpuRenderTargetReadbackEvent &event)
{
    if (event.width == 0u || event.height == 0u || event.rowStrideBytes < event.width * 4u)
    {
        return false;
    }
    return event.colorBytes.size() >=
           static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height);
}

GpuRenderTargetReadbackEvent renderAndReadback(Runtime &runtime, GpuDevice &graphicsDevice,
                                               const GpuRenderTargetHandle target,
                                               FrameContext &frame)
{
    const GpuRenderTargetReadbackRequest request =
        graphicsDevice.renderTargetSystem().requestRenderTargetReadback(
            GpuRenderTargetBinding{target, 0u, 1u});
    runtime.tick(frame);

    GpuRenderTargetReadbackEvent event{};
    if (request.id == 0u || !graphicsDevice.renderTargetSystem().tryGetRenderTargetReadback(request, event))
    {
        return {};
    }
    return event;
}

float toneMapReinhard(float value)
{
    return value / (1.0f + value);
}

float linearToSrgb(float value)
{
    if (value <= 0.0031308f)
    {
        return value * 12.92f;
    }
    return 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

std::uint8_t encodeDisplayByte(float linear)
{
    const float encoded = std::clamp(linearToSrgb(toneMapReinhard(linear)), 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(encoded * 255.0f));
}

bool isNear(std::uint8_t value, std::uint8_t expected, std::uint8_t tolerance)
{
    const int diff = static_cast<int>(value) - static_cast<int>(expected);
    return diff >= -static_cast<int>(tolerance) && diff <= static_cast<int>(tolerance);
}

} // namespace

int main()
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        return 1;
    }

    GpuDevice *graphicsDevice = runtime.getGpuDevice();
    if (graphicsDevice == nullptr)
    {
        CRESSIM_LOG_ERROR("Graphics device not available.\n");
        runtime.shutdown();
        return 1;
    }

    auto &world = runtime.getWorld();
    const auto cameraEntity = world.createEntity();

    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};

    CameraComponent camera{};
    camera.output.mode = cressim::neo::gpu::CameraOutputMode::ManagedPrimary;
    camera.clearColor = true;
    camera.clearDepth = true;
    camera.clearColorValue = {0.18f, 0.50f, 0.75f, 1.0f};
    camera.renderOrder = 0u;

    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, camera);
    runtime.setRenderFrameOptions(RenderFrameOptions{cameraEntity});

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex = 0u;
    frame.timeSeconds = 0.0;
    runtime.tick(frame);

    frame.frameIndex = 1u;
    frame.timeSeconds = static_cast<double>(frame.deltaSeconds);
    const GpuRenderTargetHandle defaultTarget = graphicsDevice->renderTargetSystem().defaultRenderTarget();
    const GpuRenderTargetReadbackEvent event =
        renderAndReadback(runtime, *graphicsDevice, defaultTarget, frame);
    if (!isValidReadback(event))
    {
        CRESSIM_LOG_ERROR("Expected valid readback for default presentation target.\n");
        runtime.shutdown();
        return 1;
    }

    const std::array<std::uint8_t, 3u> expected{
        encodeDisplayByte(camera.clearColorValue.x),
        encodeDisplayByte(camera.clearColorValue.y),
        encodeDisplayByte(camera.clearColorValue.z)};

    const std::size_t offset =
        static_cast<std::size_t>(event.height / 2u) * event.rowStrideBytes +
        static_cast<std::size_t>(event.width / 2u) * 4u;
    const std::array<std::uint8_t, 4u> pixel{
        event.colorBytes[offset + 0u],
        event.colorBytes[offset + 1u],
        event.colorBytes[offset + 2u],
        event.colorBytes[offset + 3u]};

    constexpr std::uint8_t kTolerance = 3u;
    if (!isNear(pixel[0], expected[0], kTolerance) || !isNear(pixel[1], expected[1], kTolerance) ||
        !isNear(pixel[2], expected[2], kTolerance))
    {
        CRESSIM_LOG_ERROR("Unexpected display-resolve color. Expected RGB approximately (",
                          static_cast<int>(expected[0]), ", ", static_cast<int>(expected[1]), ", ",
                          static_cast<int>(expected[2]), "), got (", static_cast<int>(pixel[0]), ", ",
                          static_cast<int>(pixel[1]), ", ", static_cast<int>(pixel[2]), ").\n");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO("Display resolve color-space checks passed.\n");
    return 0;
}
