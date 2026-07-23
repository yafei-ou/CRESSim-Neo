#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/environment_ibl_baker.h"
#include "helpers/asset_paths.h"
#include "helpers/example_cli.h"
#include "helpers/readback_image_io.h"
#include "helpers/shape_meshes.h"
#include "helpers/skybox_example.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::graphics::EnvironmentIblBakeOptions;
using cressim::neo::graphics::EnvironmentIblDesc;
using cressim::neo::graphics::IblQualityTier;
using cressim::neo::graphics::MaterialFeatureFlags;
using cressim::neo::graphics::MaterialRenderMode;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::ToneMapper;
using cressim::neo::graphics::TextureColorSpace;
using cressim::neo::graphics::TextureResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;
using cressim::neo::viewer::DebugViewerCallbacks;

enum class SensorProductMode
{
    ColorDepth,
    DepthOnly,
    SegmentationDepth,
    All,
};

struct ExampleOptions
{
    CommonExampleOptions common{};
    SensorProductMode sensorProducts = SensorProductMode::All;
    bool saveExplicitOutputs = false;
};

constexpr const char *kCameraOutputsSkyboxCrossPath =
    "environments/cubemaps/Cubemap/Cubemap_Sky_18-512x512.png";
constexpr std::uint32_t kExplicitOutputWidth  = 960u;
constexpr std::uint32_t kExplicitOutputHeight = 720u;

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--sensor-product colordepth|depth|segmentation|all] [--save-explicit-outputs]",
        true);
}

SensorProductMode parseSensorProductMode(const std::string &value)
{
    if (value == "colordepth" || value == "color-depth")
    {
        return SensorProductMode::ColorDepth;
    }
    if (value == "depth" || value == "depth-only")
    {
        return SensorProductMode::DepthOnly;
    }
    if (value == "segmentation" || value == "segmentation-depth")
    {
        return SensorProductMode::SegmentationDepth;
    }
    if (value == "all" || value == "both")
    {
        return SensorProductMode::All;
    }

    throw std::invalid_argument("Unsupported --sensor-product value: " + value);
}

Diligent::QuaternionF quaternionFromEulerDegrees(float pitchDegrees, float yawDegrees,
                                                 float rollDegrees)
{
    const float pitch = pitchDegrees * 0.017453292519943295769f * 0.5f;
    const float yaw   = yawDegrees * 0.017453292519943295769f * 0.5f;
    const float roll  = rollDegrees * 0.017453292519943295769f * 0.5f;

    const float sinPitch = std::sin(pitch);
    const float cosPitch = std::cos(pitch);
    const float sinYaw   = std::sin(yaw);
    const float cosYaw   = std::cos(yaw);
    const float sinRoll  = std::sin(roll);
    const float cosRoll  = std::cos(roll);

    return Diligent::QuaternionF{
        sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
        cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
        cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
        cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw};
}

EnvironmentIblDesc loadCameraOutputsSkyboxIbl(
    cressim::neo::graphics::RenderResourceManager &resources)
{
    const std::filesystem::path crossPath =
        cressim::neo::examples::helpers::assetPath(kCameraOutputsSkyboxCrossPath);

    EnvironmentIblBakeOptions options{};
    options.irradianceSize = 16u;
    options.specularSize = 128u;
    options.specularMipCount = 7u;
    options.irradianceSampleCount = 256u;
    options.specularSampleCount = 128u;
    options.intensity = 0.20f;
    options.backgroundIntensity = 1.15f;
    return cressim::neo::examples::helpers::createEnvironmentIblFromHorizontalCross(
        resources, crossPath, options);
}

void spawnRenderable(cressim::neo::engine::World &world, cressim::neo::graphics::MeshHandle mesh,
                     std::uint32_t envIndex,
                     cressim::neo::graphics::MaterialHandle material,
                     std::uint32_t segmentationId,
                     const Diligent::float3 &position, const Diligent::float3 &scale,
                     const Diligent::QuaternionF &rotation = Diligent::QuaternionF{})
{
    const auto entity = world.createEntity(envIndex);
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale    = scale;
    transform.worldTransform.rotation = rotation;
    world.setTransform(entity, transform);

    MeshRendererComponent renderer{};
    renderer.mesh           = mesh;
    renderer.material       = material;
    renderer.segmentationId = segmentationId;
    world.setMeshRenderer(entity, renderer);
}

bool wantsColorDepthSensor(SensorProductMode mode)
{
    return mode == SensorProductMode::ColorDepth || mode == SensorProductMode::All;
}

bool wantsDepthSensor(SensorProductMode mode)
{
    return mode == SensorProductMode::DepthOnly || mode == SensorProductMode::All;
}

bool wantsSegmentationSensor(SensorProductMode mode)
{
    return mode == SensorProductMode::SegmentationDepth || mode == SensorProductMode::All;
}

TextureResourceDesc makeTextureDesc(const char *debugName, std::uint32_t width,
                                    std::uint32_t height, TextureColorSpace colorSpace,
                                    std::initializer_list<std::uint8_t> pixels)
{
    TextureResourceDesc desc{};
    desc.debugName  = debugName;
    desc.width      = width;
    desc.height     = height;
    desc.colorSpace = colorSpace;
    desc.pixelData.assign(pixels.begin(), pixels.end());
    return desc;
}

TextureResourceDesc makeCutoutCheckerTexture()
{
    constexpr std::uint32_t kSize = 8u;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4u, 0u);

    for (std::uint32_t y = 0u; y < kSize; ++y)
    {
        for (std::uint32_t x = 0u; x < kSize; ++x)
        {
            const bool solid = ((x / 4u) + (y / 4u)) % 2u == 0u;
            const std::size_t offset =
                (static_cast<std::size_t>(y) * kSize + static_cast<std::size_t>(x)) * 4u;
            pixels[offset + 0u] = 255u;
            pixels[offset + 1u] = 255u;
            pixels[offset + 2u] = 255u;
            pixels[offset + 3u] = solid ? 255u : 0u;
        }
    }

    TextureResourceDesc desc{};
    desc.debugName  = "CameraOutputsExample.CutoutTexture";
    desc.width      = kSize;
    desc.height     = kSize;
    desc.colorSpace = TextureColorSpace::Srgb;
    desc.pixelData  = std::move(pixels);
    return desc;
}

void syncSensorCamera(Runtime &runtime, cressim::neo::common::EntityId sourceCameraEntity,
                      cressim::neo::common::EntityId sensorCameraEntity)
{
    auto &world = runtime.getWorld();
    const std::optional<TransformComponent> sourceTransform =
        world.tryGetTransform(sourceCameraEntity);
    const std::optional<CameraComponent> sourceCamera = world.tryGetCamera(sourceCameraEntity);
    const std::optional<CameraComponent> sensorCamera = world.tryGetCamera(sensorCameraEntity);
    if (!sourceTransform.has_value() || !sourceCamera.has_value() || !sensorCamera.has_value())
    {
        return;
    }

    world.setTransform(sensorCameraEntity, *sourceTransform);

    CameraComponent updated = *sensorCamera;
    updated.verticalFovDegrees = sourceCamera->verticalFovDegrees;
    updated.nearClip           = sourceCamera->nearClip;
    updated.farClip            = sourceCamera->farClip;
    world.setCamera(sensorCameraEntity, updated);
}

void syncAllSensorCameras(
    Runtime &runtime, const std::vector<cressim::neo::common::EntityId> &viewerCameraEntities,
    const std::vector<cressim::neo::common::EntityId> &colorDepthSensorEntities,
    const std::vector<cressim::neo::common::EntityId> &depthSensorEntities,
    const std::vector<cressim::neo::common::EntityId> &segmentationSensorEntities)
{
    const std::size_t colorDepthCount =
        std::min(viewerCameraEntities.size(), colorDepthSensorEntities.size());
    for (std::size_t index = 0u; index < colorDepthCount; ++index)
    {
        syncSensorCamera(runtime, viewerCameraEntities[index], colorDepthSensorEntities[index]);
    }

    const std::size_t depthCount = std::min(viewerCameraEntities.size(), depthSensorEntities.size());
    for (std::size_t index = 0u; index < depthCount; ++index)
    {
        syncSensorCamera(runtime, viewerCameraEntities[index], depthSensorEntities[index]);
    }

    const std::size_t segmentationCount =
        std::min(viewerCameraEntities.size(), segmentationSensorEntities.size());
    for (std::size_t index = 0u; index < segmentationCount; ++index)
    {
        syncSensorCamera(runtime, viewerCameraEntities[index], segmentationSensorEntities[index]);
    }
}

bool isValidColorReadback(const GpuRenderTargetReadbackEvent &event)
{
    if (event.colorWidth == 0u || event.colorHeight == 0u)
    {
        return false;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.colorFormat);
    if (formatAttribs.Format == Diligent::TEX_FORMAT_UNKNOWN || formatAttribs.IsTypeless ||
        formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_COMPRESSED)
    {
        return false;
    }

    const std::uint32_t minStride = event.colorWidth * formatAttribs.GetElementSize();
    if (event.colorRowStrideBytes < minStride)
    {
        return false;
    }

    return event.colorBytes.size() >= static_cast<std::size_t>(event.colorRowStrideBytes) *
                                          static_cast<std::size_t>(event.colorHeight);
}

bool isValidDepthReadback(const GpuRenderTargetReadbackEvent &event)
{
    if (event.depthWidth == 0u || event.depthHeight == 0u)
    {
        return false;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.depthFormat);
    if (formatAttribs.Format == Diligent::TEX_FORMAT_UNKNOWN || formatAttribs.IsTypeless ||
        formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_COMPRESSED)
    {
        return false;
    }

    const std::uint32_t minStride = event.depthWidth * formatAttribs.GetElementSize();
    if (event.depthRowStrideBytes < minStride)
    {
        return false;
    }

    return event.depthBytes.size() >= static_cast<std::size_t>(event.depthRowStrideBytes) *
                                          static_cast<std::size_t>(event.depthHeight);
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

struct ReadbackPixel
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

ReadbackPixel segmentationColor(std::uint32_t id)
{
    if (id == 0u)
    {
        return {};
    }

    std::uint32_t hash = id;
    hash ^= 2747636419u;
    hash *= 2654435769u;
    hash ^= hash >> 16u;
    hash *= 2654435769u;
    hash ^= hash >> 16u;
    hash *= 2654435769u;

    const float r = static_cast<float>((hash >> 0u) & 255u) / 255.0f;
    const float g = static_cast<float>((hash >> 8u) & 255u) / 255.0f;
    const float b = static_cast<float>((hash >> 16u) & 255u) / 255.0f;
    return {0.2f + 0.8f * r, 0.2f + 0.8f * g, 0.2f + 0.8f * b};
}

ReadbackPixel decodeColorPixel(const GpuRenderTargetReadbackEvent &event, std::uint32_t x,
                               std::uint32_t y)
{
    ReadbackPixel pixel{};
    if (!isValidColorReadback(event))
    {
        return pixel;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.colorFormat);
    const std::size_t offset =
        static_cast<std::size_t>(y) * event.colorRowStrideBytes +
        static_cast<std::size_t>(x) * formatAttribs.GetElementSize();

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
        pixel.r = halfToFloat(pixelR);
        pixel.g = halfToFloat(pixelG);
        pixel.b = halfToFloat(pixelB);
        return pixel;
    }

    if (event.colorFormat == Diligent::TEX_FORMAT_BGRA8_UNORM ||
        event.colorFormat == Diligent::TEX_FORMAT_BGRA8_UNORM_SRGB)
    {
        pixel.r = static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f;
        pixel.g = static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f;
        pixel.b = static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f;
        return pixel;
    }

    if (event.colorFormat == Diligent::TEX_FORMAT_R32_UINT)
    {
        std::uint32_t id = 0u;
        std::memcpy(&id, event.colorBytes.data() + offset, sizeof(id));
        return segmentationColor(id);
    }

    pixel.r = static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f;
    pixel.g = static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f;
    pixel.b = static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f;
    return pixel;
}

float decodeDepthSample(const GpuRenderTargetReadbackEvent &event, std::uint32_t x, std::uint32_t y)
{
    if (!isValidDepthReadback(event))
    {
        return 1.0f;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.depthFormat);
    const std::size_t offset =
        static_cast<std::size_t>(y) * event.depthRowStrideBytes +
        static_cast<std::size_t>(x) * formatAttribs.GetElementSize();

    if (event.depthFormat == Diligent::TEX_FORMAT_D32_FLOAT ||
        event.depthFormat == Diligent::TEX_FORMAT_R32_FLOAT)
    {
        float depth = 1.0f;
        std::memcpy(&depth, event.depthBytes.data() + offset, sizeof(depth));
        return depth;
    }

    if (event.depthFormat == Diligent::TEX_FORMAT_D16_UNORM)
    {
        std::uint16_t depth = 0u;
        std::memcpy(&depth, event.depthBytes.data() + offset, sizeof(depth));
        return static_cast<float>(depth) / 65535.0f;
    }

    return 1.0f;
}

float depthToLinear01(float depthSample, float nearClip, float farClip)
{
    if (nearClip <= 0.0f || farClip <= nearClip)
    {
        return std::clamp(depthSample, 0.0f, 1.0f);
    }
    const float zNdc = depthSample * 2.0f - 1.0f;
    const float linearDepth = (2.0f * nearClip * farClip) /
                              std::max(farClip + nearClip - zNdc * (farClip - nearClip), 1.0e-6f);
    return std::clamp((linearDepth - nearClip) / std::max(farClip - nearClip, 1.0e-6f), 0.0f, 1.0f);
}

std::uint8_t encodeByte(float value)
{
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

float toneMapReinhard(float value)
{
    const float clamped = std::max(value, 0.0f);
    return clamped / (1.0f + clamped);
}

float toneMapFilmic(float value)
{
    const float clamped = std::max(value, 0.0f);
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return std::clamp((clamped * (a * clamped + b)) / (clamped * (c * clamped + d) + e), 0.0f,
                      1.0f);
}

float linearToSrgb(float value)
{
    const float clamped = std::max(value, 0.0f);
    if (clamped < 0.0031308f)
    {
        return 12.92f * clamped;
    }
    return 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
}

ReadbackPixel toneMapForDisplay(ReadbackPixel pixel, ToneMapper toneMapper, float exposure)
{
    pixel.r = std::max(pixel.r, 0.0f) * std::max(exposure, 0.0f);
    pixel.g = std::max(pixel.g, 0.0f) * std::max(exposure, 0.0f);
    pixel.b = std::max(pixel.b, 0.0f) * std::max(exposure, 0.0f);

    if (toneMapper == ToneMapper::Reinhard)
    {
        pixel.r = toneMapReinhard(pixel.r);
        pixel.g = toneMapReinhard(pixel.g);
        pixel.b = toneMapReinhard(pixel.b);
    }
    else if (toneMapper == ToneMapper::Filmic)
    {
        pixel.r = toneMapFilmic(pixel.r);
        pixel.g = toneMapFilmic(pixel.g);
        pixel.b = toneMapFilmic(pixel.b);
    }

    pixel.r = linearToSrgb(pixel.r);
    pixel.g = linearToSrgb(pixel.g);
    pixel.b = linearToSrgb(pixel.b);
    return pixel;
}

bool writePpm(const std::filesystem::path &path, const GpuRenderTargetReadbackEvent &event,
              ToneMapper toneMapper, float exposure)
{
    if (!isValidColorReadback(event))
    {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
    {
        return false;
    }

    out << "P6\n" << event.colorWidth << " " << event.colorHeight << "\n255\n";
    std::vector<std::uint8_t> rgbRow(static_cast<std::size_t>(event.colorWidth) * 3u);
    for (std::uint32_t y = 0u; y < event.colorHeight; ++y)
    {
        for (std::uint32_t x = 0u; x < event.colorWidth; ++x)
        {
            ReadbackPixel pixel = decodeColorPixel(event, x, y);
            if (event.colorFormat == Diligent::TEX_FORMAT_RGBA16_FLOAT)
            {
                pixel = toneMapForDisplay(pixel, toneMapper, exposure);
            }
            rgbRow[static_cast<std::size_t>(x) * 3u + 0u] = encodeByte(pixel.r);
            rgbRow[static_cast<std::size_t>(x) * 3u + 1u] = encodeByte(pixel.g);
            rgbRow[static_cast<std::size_t>(x) * 3u + 2u] = encodeByte(pixel.b);
        }
        out.write(reinterpret_cast<const char *>(rgbRow.data()),
                  static_cast<std::streamsize>(rgbRow.size()));
    }
    return out.good();
}

bool writeDepthPgm(const std::filesystem::path &path, const GpuRenderTargetReadbackEvent &event,
                   float nearClip, float farClip)
{
    if (!isValidDepthReadback(event))
    {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
    {
        return false;
    }

    out << "P5\n" << event.depthWidth << " " << event.depthHeight << "\n255\n";
    std::vector<std::uint8_t> grayRow(static_cast<std::size_t>(event.depthWidth));
    for (std::uint32_t y = 0u; y < event.depthHeight; ++y)
    {
        for (std::uint32_t x = 0u; x < event.depthWidth; ++x)
        {
            const float depth = decodeDepthSample(event, x, y);
            const float displayValue = 1.0f - depthToLinear01(depth, nearClip, farClip);
            grayRow[static_cast<std::size_t>(x)] = encodeByte(displayValue);
        }
        out.write(reinterpret_cast<const char *>(grayRow.data()),
                  static_cast<std::streamsize>(grayRow.size()));
    }
    return out.good();
}

bool saveExplicitOutputsFirstFrame(
    Runtime &runtime, const std::filesystem::path &outputDir,
    const std::vector<cressim::neo::common::EntityId> &viewerCameraEntities,
    const std::vector<cressim::neo::common::EntityId> &colorDepthSensorEntities,
    const std::vector<cressim::neo::common::EntityId> &depthSensorEntities,
    const std::vector<cressim::neo::common::EntityId> &segmentationSensorEntities)
{
    auto *device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        CRESSIM_LOG_ERROR("Explicit output export failed: no GPU device.\n");
        return false;
    }

    std::filesystem::create_directories(outputDir);
    syncAllSensorCameras(runtime, viewerCameraEntities, colorDepthSensorEntities, depthSensorEntities,
                         segmentationSensorEntities);
    const auto &renderOptions = runtime.renderFrameOptions();

    struct PendingSensorReadback
    {
        std::string label;
        float nearClip = 0.01f;
        float farClip = 1000.0f;
        ToneMapper toneMapper = ToneMapper::Reinhard;
        float exposure = 1.0f;
        GpuRenderTargetReadbackRequest request{};
    };

    std::vector<PendingSensorReadback> pending;
    auto queueSensor = [&](const std::vector<cressim::neo::common::EntityId> &entities,
                           const char *label) -> bool {
        for (std::size_t envIndex = 0u; envIndex < entities.size(); ++envIndex)
        {
            const auto camera = runtime.getWorld().tryGetCamera(entities[envIndex]);
            if (!camera.has_value() || !camera->output.binding.isValid())
            {
                return false;
            }
            const auto request = device->renderTargetSystem().requestRenderTargetReadback(
                camera->output.binding);
            if (request.id == 0u)
            {
                return false;
            }
            pending.push_back(
                {std::string("env") + std::to_string(envIndex) + "_" + label,
                 camera->nearClip,
                 camera->farClip,
                 renderOptions.toneMapper,
                 renderOptions.exposure,
                 request});
        }
        return true;
    };

    if (!queueSensor(colorDepthSensorEntities, "colordepth") ||
        !queueSensor(depthSensorEntities, "depth") ||
        !queueSensor(segmentationSensorEntities, "segmentation"))
    {
        CRESSIM_LOG_ERROR("Explicit output export failed: could not queue readbacks.\n");
        return false;
    }

    cressim::neo::common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    runtime.prepare();
    if (!runtime.uploadWorld() || !runtime.stepPhysics(frame) || !runtime.stepSimulationSensors(frame))
    {
        CRESSIM_LOG_ERROR("Explicit output export failed: staged simulation step failed.\n");
        return false;
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);

    for (const PendingSensorReadback &entry : pending)
    {
        GpuRenderTargetReadbackEvent event{};
        if (!device->renderTargetSystem().tryGetRenderTargetReadback(entry.request, event))
        {
            CRESSIM_LOG_ERROR("Explicit output export failed: incomplete readback for ",
                              entry.label, ".\n");
            return false;
        }

        if (isValidColorReadback(event))
        {
            const std::filesystem::path colorPath = outputDir / (entry.label + "_color.ppm");
            if (!cressim::neo::examples::helpers::writeColorPpm(colorPath, event,
                                                                entry.toneMapper, entry.exposure))
            {
                CRESSIM_LOG_ERROR("Explicit output export failed: could not write ", colorPath,
                                  ".\n");
                return false;
            }
        }

        if (isValidDepthReadback(event))
        {
            const std::filesystem::path depthPath = outputDir / (entry.label + "_depth.pgm");
            if (!cressim::neo::examples::helpers::writeDepthPgm(depthPath, event, entry.nearClip,
                                                                entry.farClip))
            {
                CRESSIM_LOG_ERROR("Explicit output export failed: could not write ", depthPath,
                                  ".\n");
                return false;
            }
        }
    }

    CRESSIM_LOG_INFO("Saved explicit outputs for frame 0 to ", outputDir.string(), ".\n");
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    ExampleOptions options{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options.common, true))
            {
                continue;
            }

            const std::string arg = argv[i];
            if (arg == "--sensor-product")
            {
                options.sensorProducts =
                    parseSensorProductMode(cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--sensor-product"));
                continue;
            }
            if (arg == "--save-explicit-outputs")
            {
                options.saveExplicitOutputs = true;
                continue;
            }

            printUsage(argv[0]);
            return 2;
        }
    }
    catch (const std::invalid_argument &error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        printUsage(argv[0]);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options.common);
    config.rendererDesc.iblQualityTier = IblQualityTier::Full;
    const std::uint32_t environmentCount = std::max(options.common.envCount, 3u);
    config.sceneLayout.envCount          = environmentCount;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Camera Outputs Example";
    viewerDefaults.showStats   = true;
    const auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options.common, viewerDefaults);

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        return 1;
    }

    auto *device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Graphics device unavailable.\n");
        return 1;
    }

    auto &world = runtime.getWorld();

    cressim::neo::gpu::GpuRenderTargetHandle colorDepthTarget{};
    if (wantsColorDepthSensor(options.sensorProducts))
    {
        cressim::neo::gpu::GpuRenderTargetDesc colorDepthTargetDesc{};
        colorDepthTargetDesc.width     = kExplicitOutputWidth;
        colorDepthTargetDesc.height    = kExplicitOutputHeight;
        colorDepthTargetDesc.arraySize = environmentCount;
        colorDepthTargetDesc.layeredRendering = true;
        colorDepthTargetDesc.color     = true;
        colorDepthTargetDesc.depth     = true;
        colorDepthTargetDesc.debugName = "CameraOutputsExample.ColorDepthTarget";
        colorDepthTarget = device->renderTargetSystem().createRenderTarget(colorDepthTargetDesc);
        if (!device->renderTargetSystem().isValidRenderTarget(colorDepthTarget))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("ColorDepth target creation failed.\n");
            return 1;
        }
    }

    cressim::neo::gpu::GpuRenderTargetHandle depthTarget{};
    if (wantsDepthSensor(options.sensorProducts))
    {
        cressim::neo::gpu::GpuRenderTargetDesc depthTargetDesc{};
        depthTargetDesc.width     = kExplicitOutputWidth;
        depthTargetDesc.height    = kExplicitOutputHeight;
        depthTargetDesc.arraySize = environmentCount;
        depthTargetDesc.layeredRendering = true;
        depthTargetDesc.color     = false;
        depthTargetDesc.depth     = true;
        depthTargetDesc.debugName = "CameraOutputsExample.DepthTarget";
        depthTarget = device->renderTargetSystem().createRenderTarget(depthTargetDesc);
        if (!device->renderTargetSystem().isValidRenderTarget(depthTarget))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("Depth target creation failed.\n");
            return 1;
        }
    }

    cressim::neo::gpu::GpuRenderTargetHandle segmentationTarget{};
    if (wantsSegmentationSensor(options.sensorProducts))
    {
        cressim::neo::gpu::GpuRenderTargetDesc segmentationTargetDesc{};
        segmentationTargetDesc.width       = kExplicitOutputWidth;
        segmentationTargetDesc.height      = kExplicitOutputHeight;
        segmentationTargetDesc.arraySize   = environmentCount;
        segmentationTargetDesc.layeredRendering = true;
        segmentationTargetDesc.color       = true;
        segmentationTargetDesc.depth       = true;
        segmentationTargetDesc.colorFormat = Diligent::TEX_FORMAT_R32_UINT;
        segmentationTargetDesc.debugName   = "CameraOutputsExample.SegmentationDepthTarget";
        segmentationTarget =
            device->renderTargetSystem().createRenderTarget(segmentationTargetDesc);
        if (!device->renderTargetSystem().isValidRenderTarget(segmentationTarget))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("SegmentationDepth target creation failed.\n");
            return 1;
        }
    }

    auto &resources = runtime.getResources();
    const auto cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(0.8f, "CameraOutputsExample.CubeMesh"));
    const auto boxMesh = resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
        {0.5f, 0.5f, 0.5f}, "CameraOutputsExample.BoxMesh"));
    const auto sphereMesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        0.7f, 32u, 16u, "CameraOutputsExample.SphereMesh"));
    const auto capsuleMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCapsuleMesh(
            0.38f, 0.72f, 24u, 8u, 4u, "CameraOutputsExample.CapsuleMesh"));
    const auto groundMesh = resources.registerMesh(cressim::neo::examples::helpers::makePlaneMesh(
        14.0f, 14.0f, "CameraOutputsExample.GroundMesh"));
    const auto panelMesh = resources.registerMesh(cressim::neo::examples::helpers::makePlaneMesh(
        1.5f, 4.2f, "CameraOutputsExample.PanelMesh"));
    const auto cutoutTexture = resources.registerTexture(makeCutoutCheckerTexture());
    const Diligent::QuaternionF frontFacingPanelRotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, -Diligent::PI_F * 0.5f);
    const Diligent::QuaternionF backFacingPanelRotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, Diligent::PI_F * 0.5f);

    MaterialResourceDesc groundMaterial{};
    groundMaterial.debugName = "CameraOutputsExample.Ground";
    groundMaterial.baseColor = {0.72f, 0.73f, 0.71f};
    groundMaterial.roughness = 0.92f;
    const auto groundMaterialHandle = resources.registerMaterial(groundMaterial);

    MaterialResourceDesc coralMaterial{};
    coralMaterial.debugName = "CameraOutputsExample.Coral";
    coralMaterial.baseColor = {0.86f, 0.43f, 0.36f};
    coralMaterial.roughness = 0.50f;
    const auto coralMaterialHandle = resources.registerMaterial(coralMaterial);

    MaterialResourceDesc tealMaterial{};
    tealMaterial.debugName = "CameraOutputsExample.Teal";
    tealMaterial.baseColor = {0.24f, 0.60f, 0.70f};
    tealMaterial.roughness = 0.26f;
    tealMaterial.metallic  = 0.18f;
    const auto tealMaterialHandle = resources.registerMaterial(tealMaterial);

    MaterialResourceDesc amberMaterial{};
    amberMaterial.debugName = "CameraOutputsExample.Amber";
    amberMaterial.baseColor = {0.88f, 0.70f, 0.30f};
    amberMaterial.roughness = 0.34f;
    const auto amberMaterialHandle = resources.registerMaterial(amberMaterial);

    MaterialResourceDesc cutoutMaterial{};
    cutoutMaterial.debugName        = "CameraOutputsExample.Cutout";
    cutoutMaterial.baseColor        = {0.92f, 0.84f, 0.30f};
    cutoutMaterial.renderMode       = MaterialRenderMode::Cutout;
    cutoutMaterial.baseColorTexture = cutoutTexture;
    cutoutMaterial.pipeline.alphaCutoff = 0.5f;
    const auto cutoutMaterialHandle = resources.registerMaterial(cutoutMaterial);

    MaterialResourceDesc doubleSidedMaterial{};
    doubleSidedMaterial.debugName = "CameraOutputsExample.DoubleSided";
    doubleSidedMaterial.baseColor = {0.46f, 0.72f, 0.42f};
    doubleSidedMaterial.roughness = 0.66f;
    doubleSidedMaterial.pipeline.featureFlags |= MaterialFeatureFlags::DoubleSided;
    const auto doubleSidedMaterialHandle = resources.registerMaterial(doubleSidedMaterial);

    std::vector<cressim::neo::common::EntityId> viewerCameraEntities;
    std::vector<cressim::neo::common::EntityId> colorDepthSensorEntities;
    std::vector<cressim::neo::common::EntityId> depthSensorEntities;
    std::vector<cressim::neo::common::EntityId> segmentationSensorEntities;
    viewerCameraEntities.reserve(environmentCount);
    colorDepthSensorEntities.reserve(environmentCount);
    depthSensorEntities.reserve(environmentCount);
    segmentationSensorEntities.reserve(environmentCount);

    const auto sharedIbl = loadCameraOutputsSkyboxIbl(resources);
    for (std::uint32_t envIndex = 0u; envIndex < environmentCount; ++envIndex)
    {
        if (!world.setEnvironmentIbl(envIndex, sharedIbl))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("Failed to assign camera outputs skybox IBL.\n");
            return 1;
        }

        const float cameraHeight = 2.35f;
        const float farBias      = static_cast<float>(envIndex) * 1.5f;

        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, cameraHeight, -5.8f};
        cameraTransform.worldTransform.rotation =
            Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.13f);

        CameraComponent viewerCamera{};
        viewerCamera.verticalFovDegrees = 36.0f;
        viewerCamera.nearClip           = 0.05f;
        viewerCamera.farClip            = 18.0f + farBias;
        viewerCamera.renderOrder        = static_cast<int>(envIndex);
        viewerCamera.backgroundMode     = CameraComponent::BackgroundMode::EnvironmentCubemap;

        const auto viewerCameraEntity = world.createEntity(envIndex);
        world.setTransform(viewerCameraEntity, cameraTransform);
        world.setCamera(viewerCameraEntity, viewerCamera);
        viewerCameraEntities.push_back(viewerCameraEntity);

        if (wantsColorDepthSensor(options.sensorProducts))
        {
            const auto sensorEntity = world.createEntity(envIndex);
            world.setTransform(sensorEntity, cameraTransform);

            CameraComponent sensorCamera{};
            sensorCamera.product            = CameraComponent::Product::ColorDepth;
            sensorCamera.verticalFovDegrees = viewerCamera.verticalFovDegrees;
            sensorCamera.nearClip           = viewerCamera.nearClip;
            sensorCamera.farClip            = viewerCamera.farClip;
            sensorCamera.renderOrder        = viewerCamera.renderOrder;
            sensorCamera.output.mode        = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
            sensorCamera.output.binding =
                cressim::neo::gpu::GpuRenderTargetBinding{colorDepthTarget, envIndex, 1u};
            sensorCamera.outputWidth  = kExplicitOutputWidth;
            sensorCamera.outputHeight = kExplicitOutputHeight;
            sensorCamera.clearColor   = true;
            sensorCamera.clearDepth   = true;
            sensorCamera.backgroundMode = CameraComponent::BackgroundMode::EnvironmentCubemap;
            world.setCamera(sensorEntity, sensorCamera);
            colorDepthSensorEntities.push_back(sensorEntity);
        }

        if (wantsDepthSensor(options.sensorProducts))
        {
            const auto sensorEntity = world.createEntity(envIndex);
            world.setTransform(sensorEntity, cameraTransform);

            CameraComponent sensorCamera{};
            sensorCamera.product            = CameraComponent::Product::Depth;
            sensorCamera.verticalFovDegrees = viewerCamera.verticalFovDegrees;
            sensorCamera.nearClip           = viewerCamera.nearClip;
            sensorCamera.farClip            = viewerCamera.farClip;
            sensorCamera.renderOrder        = viewerCamera.renderOrder;
            sensorCamera.output.mode        = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
            sensorCamera.output.binding =
                cressim::neo::gpu::GpuRenderTargetBinding{depthTarget, envIndex, 1u};
            sensorCamera.outputWidth  = kExplicitOutputWidth;
            sensorCamera.outputHeight = kExplicitOutputHeight;
            sensorCamera.clearColor   = false;
            sensorCamera.clearDepth   = true;
            world.setCamera(sensorEntity, sensorCamera);
            depthSensorEntities.push_back(sensorEntity);
        }

        if (wantsSegmentationSensor(options.sensorProducts))
        {
            const auto sensorEntity = world.createEntity(envIndex);
            world.setTransform(sensorEntity, cameraTransform);

            CameraComponent sensorCamera{};
            sensorCamera.product            = CameraComponent::Product::SegmentationDepth;
            sensorCamera.verticalFovDegrees = viewerCamera.verticalFovDegrees;
            sensorCamera.nearClip           = viewerCamera.nearClip;
            sensorCamera.farClip            = viewerCamera.farClip;
            sensorCamera.renderOrder        = viewerCamera.renderOrder;
            sensorCamera.output.mode        = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
            sensorCamera.output.binding =
                cressim::neo::gpu::GpuRenderTargetBinding{segmentationTarget, envIndex, 1u};
            sensorCamera.outputWidth  = kExplicitOutputWidth;
            sensorCamera.outputHeight = kExplicitOutputHeight;
            sensorCamera.clearColor   = true;
            sensorCamera.clearDepth   = true;
            world.setCamera(sensorEntity, sensorCamera);
            segmentationSensorEntities.push_back(sensorEntity);
        }

        const auto lightEntity = world.createEntity(envIndex);
        DirectionalLightComponent light{};
        light.direction = {-0.48f + 0.04f * static_cast<float>(envIndex), -1.0f,
                           0.30f - 0.02f * static_cast<float>(envIndex)};
        light.color = {1.0f, 0.98f, 0.96f};
        light.intensity = 5.4f + 0.2f * static_cast<float>(envIndex);
        world.setDirectionalLight(lightEntity, light);

        spawnRenderable(world, groundMesh, envIndex, groundMaterialHandle, 1u,
                        {0.0f, -1.0f, 4.9f}, {1.0f, 1.0f, 1.0f});
        spawnRenderable(world, boxMesh, envIndex, coralMaterialHandle, 2u,
                        {-1.85f, -0.05f, 4.7f}, {0.85f, 1.85f, 0.85f},
                        quaternionFromEulerDegrees(0.0f, -8.0f, 0.0f));
        spawnRenderable(world, cubeMesh, envIndex, tealMaterialHandle, 3u,
                        {1.15f, 0.05f, 5.95f}, {1.00f, 1.00f, 1.00f},
                        quaternionFromEulerDegrees(8.0f, 18.0f, 0.0f));
        spawnRenderable(world, sphereMesh, envIndex, amberMaterialHandle, 4u,
                        {-0.05f, -0.18f, 3.15f},
                        {1.0f, 1.0f, 1.0f});
        spawnRenderable(world, capsuleMesh, envIndex, tealMaterialHandle, 5u,
                        {1.00f, 0.25f, 3.95f}, {1.0f, 1.0f, 1.0f},
                        quaternionFromEulerDegrees(66.0f, -26.0f, 18.0f));
        spawnRenderable(world, panelMesh, envIndex, amberMaterialHandle, 6u,
                        {-1.95f, 0.62f, 6.30f}, {1.10f, 1.35f, 1.0f},
                        frontFacingPanelRotation);
        spawnRenderable(world, panelMesh, envIndex, cutoutMaterialHandle, 7u,
                        {0.20f, 0.55f, 4.85f}, {1.00f, 1.22f, 1.0f},
                        frontFacingPanelRotation);
        spawnRenderable(world, panelMesh, envIndex, doubleSidedMaterialHandle, 8u,
                        {2.05f, 0.58f, 4.65f}, {0.92f, 1.18f, 1.0f},
                        backFacingPanelRotation);
    }

    const char *modeLabel = options.sensorProducts == SensorProductMode::ColorDepth
                                ? "ColorDepth"
                                : (options.sensorProducts == SensorProductMode::DepthOnly
                                       ? "Depth"
                                       : (options.sensorProducts ==
                                                  SensorProductMode::SegmentationDepth
                                              ? "SegmentationDepth"
                                              : "All"));
    CRESSIM_LOG_INFO("Camera outputs example ready with ", environmentCount,
                     " environments and sensor mode=", modeLabel,
                     ". Camera mode shows the managed-primary viewer camera. "
                     "Press U to switch to explicit sensor outputs, then use , and . to cycle "
                     "between ColorDepth color, ColorDepth depth, Depth-only, "
                     "SegmentationDepth segmentation, and SegmentationDepth depth outputs when present. "
                     "The scene uses a compact centered arrangement of opaque, cutout, and "
                     "double-sided surfaces so the sensor passes visibly exercise alpha-test "
                     "discard, depth, segmentation, and back-face rendering.");

    if (options.saveExplicitOutputs)
    {
        const bool saved = saveExplicitOutputsFirstFrame(
            runtime, "camera_outputs_first_frame", viewerCameraEntities, colorDepthSensorEntities,
            depthSensorEntities, segmentationSensorEntities);
        runtime.shutdown();
        viewer.shutdown();
        return saved ? 0 : 1;
    }

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [viewerCameraEntities, colorDepthSensorEntities, depthSensorEntities,
                            segmentationSensorEntities](
                               const cressim::neo::common::FrameContext &, Runtime &runtimeRef)
    {
        syncAllSensorCameras(runtimeRef, viewerCameraEntities, colorDepthSensorEntities,
                             depthSensorEntities, segmentationSensorEntities);
    };

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{viewerCameraEntities.front()},
                                callbacks);

    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
