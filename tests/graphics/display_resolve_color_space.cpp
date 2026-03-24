#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#if PLATFORM_WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#elif PLATFORM_LINUX
#define GLFW_EXPOSE_NATIVE_X11
#elif PLATFORM_MACOS
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
using cressim::neo::gpu::GpuPresentationReadbackEvent;
using cressim::neo::gpu::GpuPresentationReadbackRequest;
using cressim::neo::gpu::GpuPresentationTargetDesc;
using cressim::neo::graphics::RenderFrameOptions;

bool isValidReadback(const GpuPresentationReadbackEvent &event)
{
    if (event.width == 0u || event.height == 0u)
    {
        return false;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.colorFormat);
    if (formatAttribs.Format == Diligent::TEX_FORMAT_UNKNOWN || formatAttribs.IsTypeless ||
        formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_COMPRESSED)
    {
        return false;
    }
    const std::uint32_t minStride = event.width * formatAttribs.GetElementSize();
    if (event.rowStrideBytes < minStride)
    {
        return false;
    }

    return event.colorBytes.size() >=
           static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height);
}

GpuPresentationReadbackEvent renderAndReadback(Runtime &runtime, GpuDevice &graphicsDevice,
                                               FrameContext &frame)
{
    const GpuPresentationReadbackRequest request = graphicsDevice.requestPresentationReadback();
    runtime.tick(frame);

    GpuPresentationReadbackEvent event{};
    if (request.id == 0u || !graphicsDevice.tryGetPresentationReadback(request, event))
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

float halfToFloat(std::uint16_t value)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16u;
    std::uint32_t exponent   = (value >> 10u) & 0x1fu;
    std::uint32_t mantissa   = value & 0x03ffu;

    std::uint32_t bits = 0u;
    if (exponent == 0u)
    {
        if (mantissa != 0u)
        {
            exponent = 113u;
            while ((mantissa & 0x0400u) == 0u)
            {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign | (exponent << 23u) | (mantissa << 13u);
        }
        else
        {
            bits = sign;
        }
    }
    else if (exponent == 0x1fu)
    {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    }
    else
    {
        bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace

int main()
{
    if (glfwInit() != GLFW_TRUE)
    {
        CRESSIM_LOG_WARNING("Skipping display-resolve color-space test because GLFW init failed.\n");
        return 0;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(320, 240, "display_resolve_color_space", nullptr, nullptr);
    if (window == nullptr)
    {
        CRESSIM_LOG_WARNING(
            "Skipping display-resolve color-space test because a hidden GLFW window could not be created.\n");
        glfwTerminate();
        return 0;
    }

    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.presentation.enabled = true;
#if PLATFORM_WIN32
    config.gpuDeviceDesc.presentation.nativeWindow = glfwGetWin32Window(window);
#elif PLATFORM_LINUX
    config.gpuDeviceDesc.presentation.nativeWindowId =
        static_cast<std::uint64_t>(glfwGetX11Window(window));
    config.gpuDeviceDesc.presentation.nativeDisplay = glfwGetX11Display();
#elif PLATFORM_MACOS
    config.gpuDeviceDesc.presentation.nativeWindow = glfwGetCocoaWindow(window);
#endif

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    GpuDevice *graphicsDevice = runtime.getGpuDevice();
    if (graphicsDevice == nullptr)
    {
        CRESSIM_LOG_ERROR("Graphics device not available.\n");
        runtime.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    GpuPresentationTargetDesc presentationDesc{};
    if (!graphicsDevice->tryGetPresentationTargetDesc(presentationDesc))
    {
        CRESSIM_LOG_ERROR("Presentation target not available for display-resolve test.\n");
        runtime.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
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
    runtime.setRenderFrameOptions(RenderFrameOptions{cameraEntity, presentationDesc});

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex = 0u;
    frame.timeSeconds = 0.0;
    runtime.tick(frame);

    frame.frameIndex = 1u;
    frame.timeSeconds = static_cast<double>(frame.deltaSeconds);
    const GpuPresentationReadbackEvent event = renderAndReadback(runtime, *graphicsDevice, frame);
    if (!isValidReadback(event))
    {
        CRESSIM_LOG_ERROR("Expected valid readback for presentation target.\n");
        runtime.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const std::size_t offset =
        static_cast<std::size_t>(event.height / 2u) * event.rowStrideBytes +
        static_cast<std::size_t>(event.width / 2u) *
            Diligent::GetTextureFormatAttribs(event.colorFormat).GetElementSize();

    bool colorMatches = false;
    if (event.colorFormat == Diligent::TEX_FORMAT_RGBA16_FLOAT)
    {
        const std::uint16_t pixelR =
            static_cast<std::uint16_t>(event.colorBytes[offset + 0u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 1u]) << 8u);
        const std::uint16_t pixelG =
            static_cast<std::uint16_t>(event.colorBytes[offset + 2u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 3u]) << 8u);
        const std::uint16_t pixelB =
            static_cast<std::uint16_t>(event.colorBytes[offset + 4u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 5u]) << 8u);

        constexpr float kFloatTolerance = 0.02f;
        colorMatches = std::fabs(halfToFloat(pixelR) - camera.clearColorValue.x) <= kFloatTolerance &&
                       std::fabs(halfToFloat(pixelG) - camera.clearColorValue.y) <= kFloatTolerance &&
                       std::fabs(halfToFloat(pixelB) - camera.clearColorValue.z) <= kFloatTolerance;
    }
    else
    {
        const std::array<std::uint8_t, 3u> expected{
            encodeDisplayByte(camera.clearColorValue.x),
            encodeDisplayByte(camera.clearColorValue.y),
            encodeDisplayByte(camera.clearColorValue.z)};
        const std::array<std::uint8_t, 4u> pixel{
            event.colorBytes[offset + 0u],
            event.colorBytes[offset + 1u],
            event.colorBytes[offset + 2u],
            event.colorBytes[offset + 3u]};

        constexpr std::uint8_t kTolerance = 3u;
        colorMatches = isNear(pixel[0], expected[0], kTolerance) &&
                       isNear(pixel[1], expected[1], kTolerance) &&
                       isNear(pixel[2], expected[2], kTolerance);
    }

    if (!colorMatches)
    {
        CRESSIM_LOG_ERROR("Unexpected display-resolve color for presentation format=",
                          static_cast<std::uint32_t>(event.colorFormat), ".\n");
        runtime.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    runtime.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    CRESSIM_LOG_INFO("Display resolve color-space checks passed.\n");
    return 0;
}
