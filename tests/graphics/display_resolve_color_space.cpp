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
using cressim::neo::graphics::ToneMapper;

struct ReadbackPixel
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

struct RenderScenarioResult
{
    GpuPresentationReadbackEvent event{};
    ReadbackPixel pixel{};
};

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

float toneMapFilmic(float value)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    const float numerator = value * (a * value + b);
    const float denominator = value * (c * value + d) + e;
    return std::clamp(numerator / denominator, 0.0f, 1.0f);
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

std::size_t centerPixelOffset(const GpuPresentationReadbackEvent &event)
{
    return static_cast<std::size_t>(event.height / 2u) * event.rowStrideBytes +
           static_cast<std::size_t>(event.width / 2u) *
               Diligent::GetTextureFormatAttribs(event.colorFormat).GetElementSize();
}

ReadbackPixel decodeCenterPixel(const GpuPresentationReadbackEvent &event)
{
    const std::size_t offset = centerPixelOffset(event);
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
        return {halfToFloat(pixelR), halfToFloat(pixelG), halfToFloat(pixelB)};
    }

    if (event.colorFormat == Diligent::TEX_FORMAT_BGRA8_UNORM ||
        event.colorFormat == Diligent::TEX_FORMAT_BGRA8_UNORM_SRGB)
    {
        return {static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f,
                static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f,
                static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f};
    }

    return {static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f,
            static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f,
            static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f};
}

bool isHdrFormat(Diligent::TEXTURE_FORMAT format)
{
    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(format);
    return formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_FLOAT ||
           formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_COMPOUND;
}

struct ScenarioRunner
{
    Runtime runtime{};
    GpuDevice *graphicsDevice = nullptr;
    cressim::neo::common::EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
    FrameContext frame{};
    GpuPresentationTargetDesc presentationDesc{};

    bool initialize(GLFWwindow *window, Diligent::TEXTURE_FORMAT preferredColorFormat)
    {
        RuntimeConfig config{};
        config.gpuDeviceDesc.preferredBackend                  = GpuBackend::Vulkan;
        config.gpuDeviceDesc.presentation.enabled              = true;
        config.gpuDeviceDesc.presentation.preferredColorFormat = preferredColorFormat;
#if PLATFORM_WIN32
        config.gpuDeviceDesc.presentation.nativeWindow = glfwGetWin32Window(window);
#elif PLATFORM_LINUX
        config.gpuDeviceDesc.presentation.nativeWindowId =
            static_cast<std::uint64_t>(glfwGetX11Window(window));
        config.gpuDeviceDesc.presentation.nativeDisplay = glfwGetX11Display();
#elif PLATFORM_MACOS
        config.gpuDeviceDesc.presentation.nativeWindow = glfwGetCocoaWindow(window);
#endif

        if (!runtime.initialize(config))
        {
            return false;
        }

        graphicsDevice = runtime.getGpuDevice();
        if (graphicsDevice == nullptr ||
            !graphicsDevice->tryGetPresentationTargetDesc(presentationDesc))
        {
            runtime.shutdown();
            graphicsDevice = nullptr;
            return false;
        }

        auto &world = runtime.getWorld();
        cameraEntity = world.createEntity();

        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};

        CameraComponent camera{};
        camera.output.mode = cressim::neo::gpu::RenderOutputMode::ManagedPrimary;
        camera.clearColor  = true;
        camera.clearDepth  = true;
        camera.renderOrder = 0u;

        world.setTransform(cameraEntity, cameraTransform);
        world.setCamera(cameraEntity, camera);

        frame.deltaSeconds = 1.0f / 60.0f;
        frame.frameIndex   = 0u;
        frame.timeSeconds  = 0.0;
        return true;
    }

    RenderScenarioResult runScenario(ToneMapper toneMapper, float exposure,
                                     const Diligent::float4 &clearColor)
    {
        RenderScenarioResult result{};
        if (graphicsDevice == nullptr || cameraEntity == cressim::neo::common::kInvalidEntityId)
        {
            return result;
        }

        auto &world                       = runtime.getWorld();
        std::optional<CameraComponent> camera = world.tryGetCamera(cameraEntity);
        if (!camera.has_value())
        {
            return result;
        }
        camera->clearColorValue = clearColor;
        world.setCamera(cameraEntity, *camera);
        runtime.setRenderFrameOptions(
            RenderFrameOptions{cameraEntity, presentationDesc, toneMapper, exposure});

        auto tickFrame = [&]()
        {
            runtime.tick(frame);
            ++frame.frameIndex;
            frame.timeSeconds =
                static_cast<double>(frame.frameIndex) * static_cast<double>(frame.deltaSeconds);
        };

        tickFrame();
        tickFrame();

        const GpuPresentationReadbackEvent event =
            renderAndReadback(runtime, *graphicsDevice, frame);
        ++frame.frameIndex;
        frame.timeSeconds =
            static_cast<double>(frame.frameIndex) * static_cast<double>(frame.deltaSeconds);

        result.event = event;
        if (isValidReadback(event))
        {
            result.pixel = decodeCenterPixel(event);
        }
        return result;
    }

    void shutdown()
    {
        runtime.shutdown();
        graphicsDevice   = nullptr;
        cameraEntity     = cressim::neo::common::kInvalidEntityId;
        presentationDesc = {};
        frame            = {};
    }
};

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

    const RenderFrameOptions defaultOptions{};
    if (defaultOptions.toneMapper != ToneMapper::Reinhard ||
        std::fabs(defaultOptions.exposure - 1.0f) > 1.0e-6f)
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    ScenarioRunner sdrRunner;
    if (!sdrRunner.initialize(window, Diligent::TEX_FORMAT_UNKNOWN))
    {
        CRESSIM_LOG_WARNING(
            "Skipping display-resolve color-space test because Vulkan presentation runtime initialization failed.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    }

    constexpr Diligent::float4 kSdrClearColor = {0.18f, 0.50f, 0.75f, 1.0f};
    const RenderScenarioResult defaultSdr =
        sdrRunner.runScenario(ToneMapper::Reinhard, 1.0f, kSdrClearColor);
    if (!isValidReadback(defaultSdr.event))
    {
        CRESSIM_LOG_ERROR("Expected valid default presentation readback.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    if (isHdrFormat(defaultSdr.event.colorFormat))
    {
        constexpr float kFloatTolerance = 0.02f;
        if (std::fabs(defaultSdr.pixel.r - kSdrClearColor.x) > kFloatTolerance ||
            std::fabs(defaultSdr.pixel.g - kSdrClearColor.y) > kFloatTolerance ||
            std::fabs(defaultSdr.pixel.b - kSdrClearColor.z) > kFloatTolerance)
        {
            CRESSIM_LOG_ERROR("Unexpected default HDR passthrough color.\n");
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }
    }
    else
    {
        constexpr float kFloatTolerance = 3.0f / 255.0f;
        const ReadbackPixel expected{
            static_cast<float>(encodeDisplayByte(kSdrClearColor.x)) / 255.0f,
            static_cast<float>(encodeDisplayByte(kSdrClearColor.y)) / 255.0f,
            static_cast<float>(encodeDisplayByte(kSdrClearColor.z)) / 255.0f};
        if (std::fabs(defaultSdr.pixel.r - expected.r) > kFloatTolerance ||
            std::fabs(defaultSdr.pixel.g - expected.g) > kFloatTolerance ||
            std::fabs(defaultSdr.pixel.b - expected.b) > kFloatTolerance)
        {
            CRESSIM_LOG_ERROR("Unexpected default SDR resolve color. format=",
                              static_cast<std::uint32_t>(defaultSdr.event.colorFormat),
                              " actual=(",
                              defaultSdr.pixel.r,
                              ", ",
                              defaultSdr.pixel.g,
                              ", ",
                              defaultSdr.pixel.b,
                              ") expected=(",
                              expected.r,
                              ", ",
                              expected.g,
                              ", ",
                              expected.b,
                              ").\n");
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }

        const RenderScenarioResult exposedSdr =
            sdrRunner.runScenario(ToneMapper::Reinhard, 2.0f, kSdrClearColor);
        const RenderScenarioResult filmicSdr =
            sdrRunner.runScenario(ToneMapper::Filmic, 1.0f,
                                  Diligent::float4{4.0f, 2.0f, 0.75f, 1.0f});
        const RenderScenarioResult noneSdr =
            sdrRunner.runScenario(ToneMapper::Disabled, 1.0f,
                                  Diligent::float4{4.0f, 2.0f, 0.75f, 1.0f});

        if (!isValidReadback(exposedSdr.event) || !isValidReadback(filmicSdr.event) ||
            !isValidReadback(noneSdr.event))
        {
            CRESSIM_LOG_ERROR("Expected valid SDR readbacks for tone-mapping scenarios.\n");
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }

        if (!(exposedSdr.pixel.r > defaultSdr.pixel.r && exposedSdr.pixel.g > defaultSdr.pixel.g &&
              exposedSdr.pixel.b > defaultSdr.pixel.b))
        {
            CRESSIM_LOG_ERROR("Exposure did not brighten the SDR presentation result.\n");
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }

        const float brightSource = 4.0f;
        const float filmicExpected = linearToSrgb(toneMapFilmic(brightSource));
        if (std::fabs(filmicSdr.pixel.r - filmicExpected) > 0.03f)
        {
            CRESSIM_LOG_ERROR("Filmic tone mapping did not match expected highlight compression.\n");
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }
        if (std::fabs(filmicSdr.pixel.r - noneSdr.pixel.r) < 0.005f)
        {
            CRESSIM_LOG_ERROR("Filmic tone mapping was not measurably different from no tone mapping.\n");
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }
        if (!(filmicSdr.pixel.r < noneSdr.pixel.r))
        {
            CRESSIM_LOG_ERROR("Filmic tone mapping did not compress SDR highlights.\n");
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }
    }

    sdrRunner.shutdown();

    ScenarioRunner hdrRunner;
    if (!hdrRunner.initialize(window, Diligent::TEX_FORMAT_RGBA16_FLOAT))
    {
        CRESSIM_LOG_WARNING(
            "Skipping HDR presentation assertions because Vulkan HDR presentation runtime initialization failed.\n");
    }
    else
    {
        const RenderScenarioResult defaultHdr =
            hdrRunner.runScenario(ToneMapper::Reinhard, 1.0f, kSdrClearColor);
        if (!isValidReadback(defaultHdr.event))
        {
            CRESSIM_LOG_ERROR("Expected valid HDR presentation readback.\n");
            hdrRunner.shutdown();
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }
        if (!isHdrFormat(defaultHdr.event.colorFormat))
        {
            CRESSIM_LOG_ERROR("Expected HDR presentation format for HDR resolve scenario.\n");
            hdrRunner.shutdown();
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }

        constexpr float kHdrTolerance = 0.02f;
        if (std::fabs(defaultHdr.pixel.r - kSdrClearColor.x) > kHdrTolerance ||
            std::fabs(defaultHdr.pixel.g - kSdrClearColor.y) > kHdrTolerance ||
            std::fabs(defaultHdr.pixel.b - kSdrClearColor.z) > kHdrTolerance)
        {
            CRESSIM_LOG_ERROR("Unexpected default HDR passthrough color.\n");
            hdrRunner.shutdown();
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }

        hdrRunner.shutdown();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    CRESSIM_LOG_INFO("Display resolve color-space checks passed.\n");
    return 0;
}
