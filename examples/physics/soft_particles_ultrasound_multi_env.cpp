#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::engine::UltrasoundAmplitudeRange;
using cressim::neo::engine::UltrasoundProbeComponent;
using cressim::neo::engine::UltrasoundProbeResult;
using cressim::neo::engine::UltrasoundScattererSourceComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::gpu::GpuDevice;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kEnvSpacing              = 6.0f;
constexpr std::uint32_t kDefaultEnvCount = 4u;
constexpr float kPi                      = 3.14159265359f;
constexpr float kProbeHeight             = 0.4f;
constexpr float kProbeBodyHalfHeight     = 0.06f;
constexpr float kProbeBodyDepth          = 0.08f;

enum class ExampleProbeType
{
    Linear,
    Curvilinear,
};

struct SceneMaterials
{
    MaterialHandle ground{};
    MaterialHandle softBody{};
    MaterialHandle probe{};
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", true);
    std::printf("  --debug-particles        Show debug particle rendering.\n");
    std::printf("  --probe-type TYPE        Probe geometry: linear or curvilinear.\n");
    std::printf("  --save-probe-images      Save all probe ultrasound images to the current "
                "working directory and quit.\n");
    std::printf("  --save-probe-delay-seconds N\n");
    std::printf("                           When saving probe images, wait N simulated seconds "
                "before capture. Use 0 for the first frame.\n");
}

bool isValidReadback(const GpuRenderTargetReadbackEvent &event)
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

ReadbackPixel decodePixel(const GpuRenderTargetReadbackEvent &event, std::uint32_t x,
                          std::uint32_t y)
{
    ReadbackPixel pixel{};
    if (!isValidReadback(event))
    {
        return pixel;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.colorFormat);
    const std::size_t offset =
        static_cast<std::size_t>(y) * event.rowStrideBytes +
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

    pixel.r = static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f;
    pixel.g = static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f;
    pixel.b = static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f;
    return pixel;
}

std::uint8_t encodeByte(float value)
{
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

bool writePpm(const std::string &path, const GpuRenderTargetReadbackEvent &event)
{
    if (!isValidReadback(event))
    {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
    {
        return false;
    }

    out << "P6\n" << event.width << " " << event.height << "\n255\n";
    std::vector<std::uint8_t> rgbRow(static_cast<std::size_t>(event.width) * 3u);
    for (std::uint32_t y = 0u; y < event.height; ++y)
    {
        for (std::uint32_t x = 0u; x < event.width; ++x)
        {
            const ReadbackPixel pixel = decodePixel(event, x, y);
            rgbRow[static_cast<std::size_t>(x) * 3u + 0u] = encodeByte(pixel.r);
            rgbRow[static_cast<std::size_t>(x) * 3u + 1u] = encodeByte(pixel.g);
            rgbRow[static_cast<std::size_t>(x) * 3u + 2u] = encodeByte(pixel.b);
        }
        out.write(reinterpret_cast<const char *>(rgbRow.data()),
                  static_cast<std::streamsize>(rgbRow.size()));
    }

    return out.good();
}

float parseNonNegativeFloat(const std::string &value, const char *optionName)
{
    std::size_t parsedLength = 0u;
    const float parsedValue  = std::stof(value, &parsedLength);
    if (parsedLength != value.size() || parsedValue < 0.0f)
    {
        throw std::invalid_argument(std::string("Expected non-negative float for ") + optionName +
                                    ": " + value);
    }
    return parsedValue;
}

Diligent::float3 envOrigin(std::uint32_t envIndex, std::uint32_t envCount)
{
    const std::uint32_t cols = std::max(
        1u, static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<float>(envCount)))));
    const std::uint32_t rows = std::max(1u, (envCount + cols - 1u) / cols);
    const std::uint32_t col  = envIndex % cols;
    const std::uint32_t row  = envIndex / cols;
    const float xCenter      = (static_cast<float>(cols) - 1.0f) * 0.5f;
    const float zCenter      = (static_cast<float>(rows) - 1.0f) * 0.5f;
    return {(static_cast<float>(col) - xCenter) * kEnvSpacing, 0.0f,
            (static_cast<float>(row) - zCenter) * kEnvSpacing};
}

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.metallic  = 0.0f;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

ExampleProbeType parseProbeType(const std::string &value)
{
    if (value == "linear")
    {
        return ExampleProbeType::Linear;
    }
    if (value == "curvilinear" || value == "curved")
    {
        return ExampleProbeType::Curvilinear;
    }

    throw std::invalid_argument("Unsupported probe type: " + value);
}

std::vector<UltrasoundAmplitudeRange> authorAmplitudeRanges(
    const cressim::neo::engine::SoftBodyAuthoringParticles &particles)
{
    if (particles.restPositions.empty())
    {
        return {};
    }

    Diligent::float3 minimum = particles.restPositions.front();
    Diligent::float3 maximum = particles.restPositions.front();
    for (const Diligent::float3 &position : particles.restPositions)
    {
        minimum.x = std::min(minimum.x, position.x);
        minimum.y = std::min(minimum.y, position.y);
        minimum.z = std::min(minimum.z, position.z);
        maximum.x = std::max(maximum.x, position.x);
        maximum.y = std::max(maximum.y, position.y);
        maximum.z = std::max(maximum.z, position.z);
    }

    const Diligent::float3 extent{
        std::max(maximum.x - minimum.x, 1.0e-4f), std::max(maximum.y - minimum.y, 1.0e-4f),
        std::max(maximum.z - minimum.z, 1.0e-4f)};
    const Diligent::float3 center = (minimum + maximum) * 0.5f;

    std::vector<UltrasoundAmplitudeRange> ranges;
    ranges.reserve(particles.restPositions.size());
    for (const Diligent::float3 &position : particles.restPositions)
    {
        const float normalizedHeight = (position.y - minimum.y) / extent.y;
        const Diligent::float3 centered{
            (position.x - center.x) / extent.x, (position.y - center.y) / extent.y,
            (position.z - center.z) / extent.z};
        const float radialDistance = std::sqrt(Diligent::dot(centered, centered));
        const float shellWeight    = std::clamp(1.0f - radialDistance * 1.6f, 0.0f, 1.0f);
        const float baseAmplitude =
            std::clamp(0.15f + 0.55f * normalizedHeight + 0.25f * shellWeight, 0.0f, 1.0f);
        const float minAmplitude = std::clamp(baseAmplitude - 0.10f, 0.0f, 1.0f);
        const float maxAmplitude = std::clamp(baseAmplitude + 0.10f, minAmplitude, 1.0f);
        ranges.push_back(UltrasoundAmplitudeRange{minAmplitude, maxAmplitude});
    }

    return ranges;
}

MeshHandle registerProbeMesh(cressim::neo::graphics::RenderResourceManager &resources,
                             const UltrasoundProbeComponent &probe)
{
    if (probe.geometry == UltrasoundProbeComponent::Geometry::Curvilinear)
    {
        cressim::neo::graphics::MeshResourceDesc mesh{};
        mesh.debugName = "SoftParticlesUltrasoundMultiEnv.CurvilinearProbeMesh";

        const std::uint32_t segments = std::max<std::uint32_t>(24u, probe.numScanlines);
        const float radius = std::max(probe.probeRadius, kProbeBodyDepth);
        const float innerRadius = std::max(radius - kProbeBodyDepth, 1.0e-4f);
        const float outerRadius = radius;
        const float halfAngle = 0.5f * probe.sectorAngleDegrees * (kPi / 180.0f);

        const auto pointOnArc = [&](float radiusValue, float angleValue, float yValue) {
            return Diligent::float3{
                std::sin(angleValue) * radiusValue,
                yValue,
                std::cos(angleValue) * radiusValue - radius,
            };
        };

        const auto appendQuad =
            [&](const Diligent::float3 &a, const Diligent::float3 &b, const Diligent::float3 &c,
                const Diligent::float3 &d) {
                const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
                const Diligent::float3 tangentDir = Diligent::normalize(b - a);
                const Diligent::float3 normal = Diligent::normalize(Diligent::cross(b - a, c - a));
                const Diligent::float4 tangent{tangentDir.x, tangentDir.y, tangentDir.z, 1.0f};
                mesh.vertices.push_back({a, normal, 0.0f, 0.0f, tangent});
                mesh.vertices.push_back({b, normal, 1.0f, 0.0f, tangent});
                mesh.vertices.push_back({c, normal, 1.0f, 1.0f, tangent});
                mesh.vertices.push_back({d, normal, 0.0f, 1.0f, tangent});
                mesh.indices.push_back(base + 0u);
                mesh.indices.push_back(base + 2u);
                mesh.indices.push_back(base + 1u);
                mesh.indices.push_back(base + 0u);
                mesh.indices.push_back(base + 3u);
                mesh.indices.push_back(base + 2u);
            };

        for (std::uint32_t segment = 0u; segment < segments; ++segment)
        {
            const float t0 = static_cast<float>(segment) / static_cast<float>(segments);
            const float t1 = static_cast<float>(segment + 1u) / static_cast<float>(segments);
            const float angle0 = -halfAngle + 2.0f * halfAngle * t0;
            const float angle1 = -halfAngle + 2.0f * halfAngle * t1;

            const Diligent::float3 outerTop0 = pointOnArc(outerRadius, angle0, kProbeBodyHalfHeight);
            const Diligent::float3 outerTop1 = pointOnArc(outerRadius, angle1, kProbeBodyHalfHeight);
            const Diligent::float3 outerBottom0 =
                pointOnArc(outerRadius, angle0, -kProbeBodyHalfHeight);
            const Diligent::float3 outerBottom1 =
                pointOnArc(outerRadius, angle1, -kProbeBodyHalfHeight);
            const Diligent::float3 innerTop0 = pointOnArc(innerRadius, angle0, kProbeBodyHalfHeight);
            const Diligent::float3 innerTop1 = pointOnArc(innerRadius, angle1, kProbeBodyHalfHeight);
            const Diligent::float3 innerBottom0 =
                pointOnArc(innerRadius, angle0, -kProbeBodyHalfHeight);
            const Diligent::float3 innerBottom1 =
                pointOnArc(innerRadius, angle1, -kProbeBodyHalfHeight);

            appendQuad(outerTop0, outerTop1, outerBottom1, outerBottom0);
            appendQuad(innerTop1, innerTop0, innerBottom0, innerBottom1);
            appendQuad(innerTop0, innerTop1, outerTop1, outerTop0);
            appendQuad(outerBottom0, outerBottom1, innerBottom1, innerBottom0);

            if (segment == 0u)
            {
                appendQuad(innerTop0, outerTop0, outerBottom0, innerBottom0);
            }
            if (segment + 1u == segments)
            {
                appendQuad(outerTop1, innerTop1, innerBottom1, outerBottom1);
            }
        }

        return resources.registerMesh(mesh);
    }

    const float lateralSpan = std::max(
        0.04f, static_cast<float>(std::max(probe.numScanlines, 1u) - 1u) * probe.scanlineSpacing);
    const Diligent::float3 halfExtents{
        0.5f * (lateralSpan + 0.04f),
        kProbeBodyHalfHeight,
        0.5f * kProbeBodyDepth,
    };
    return resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
        halfExtents, "SoftParticlesUltrasoundMultiEnv.ProbeMesh"));
}

void authorEnvironment(Runtime &runtime, std::uint32_t envIndex, std::uint32_t envCount,
                       MeshHandle planeMesh, MeshHandle boxMesh, MeshHandle probeMesh,
                       const UltrasoundProbeComponent &probeTemplate,
                       const SceneMaterials &materials,
                       cressim::neo::common::EntityId &outCameraEntity,
                       cressim::neo::common::EntityId &outProbeEntity)
{
    auto &world                   = runtime.getWorld();
    const Diligent::float3 origin = envOrigin(envIndex, envCount);
    const float phase             = static_cast<float>(envIndex) * 0.71f;

    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position =
        origin + Diligent::float3{0.0f, 1.6f, -2.8f - 0.15f * static_cast<float>(envIndex % 3u)};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.12f);
    world.setTransform(outCameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 48.0f;
    camera.renderOrder        = envIndex;
    world.setCamera(outCameraEntity, camera);

    const auto lightEntity = world.createEntity(envIndex);
    DirectionalLightComponent light{};
    light.direction = Diligent::normalize(
        Diligent::float3{-0.30f + 0.08f * std::sin(phase), -1.0f, 0.18f + 0.08f * std::cos(phase)});
    light.intensity = 7.5f;
    world.setDirectionalLight(lightEntity, light);

    const auto groundEntity = world.createEntity(envIndex);
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = origin + Diligent::float3{0.0f, -0.15f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{planeMesh, materials.ground, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType    = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType   = ColliderShapeType::Box;
    groundCollider.shapeParams = {2.0f, 0.05f, 2.0f, 0.0f};
    groundCollider.friction    = 0.55f;
    world.addCollider(groundEntity, groundCollider);

    const auto softEntity = world.createEntity(envIndex);
    TransformComponent softTransform{};
    // Spawn heights are derived from scene geometry instead of tuned by eye:
    // - the cube must start above the ground to avoid an explosive initial overlap
    // - the cube must still intersect the downward probe beam on frame 0 so first-frame
    //   ultrasound capture is meaningful
    const float groundTopHeight = -0.15f + 0.05f;
    const float cubeHalfHeight  = 0.5f * 0.45f;
    const float minSpawnHeight  = groundTopHeight + cubeHalfHeight + 0.08f;
    const float maxSpawnHeight  = kProbeHeight + cubeHalfHeight - 0.04f;
    const float spawnT = envCount > 1u
                             ? static_cast<float>(envIndex) / static_cast<float>(envCount - 1u)
                             : 0.0f;
    const float softSpawnHeight =
        minSpawnHeight + (maxSpawnHeight - minSpawnHeight) * spawnT;
    softTransform.worldTransform.position =
        origin + Diligent::float3{0.0f, softSpawnHeight, 0.0f};
    softTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.15f * std::sin(phase));
    world.setTransform(softEntity, softTransform);
    world.setMeshRenderer(softEntity, MeshRendererComponent{boxMesh, materials.softBody, true});

    SoftBodyComponent softBody{};
    softBody.source.kind                             = SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size                  = {0.45f, 0.45f, 0.45f};
    softBody.source.regularGrid.targetParticleSpacing = 0.08f;
    softBody.particleMass         = 0.01f;
    softBody.particleRadius       = 0.04f;
    softBody.edgeCompliance       = 0.0f;
    softBody.volumeCompliance     = 0.0008f;
    softBody.selfCollisionEnabled = true;
    softBody.collisionLayer       = 0x1u;
    softBody.collisionMask        = 0xffffffffu;
    if (!world.setSoftBody(softEntity, softBody))
    {
        throw std::runtime_error("Failed to author ultrasound soft body.");
    }
    UltrasoundScattererSourceComponent scattererSource{};
    scattererSource.density               = 1000000.0f;
    scattererSource.pointDistanceOverride = 0.0f;
    world.setUltrasoundScattererSource(softEntity, scattererSource);

    const auto authoringParticles = world.tryGetSoftBodyAuthoringParticles(softEntity);
    if (!authoringParticles.has_value())
    {
        throw std::runtime_error("Failed to query authored soft-body particles.");
    }

    world.setUltrasoundScattererAmplitudeRanges(softEntity,
                                                authorAmplitudeRanges(*authoringParticles));

    const auto probeEntity = world.createEntity(envIndex);
    TransformComponent probeTransform{};
    probeTransform.worldTransform.position = origin + Diligent::float3{0.0f, kProbeHeight, 0.0f};
    probeTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.5f * kPi);
    world.setTransform(probeEntity, probeTransform);

    UltrasoundProbeComponent probe = probeTemplate;
    world.setUltrasoundProbe(probeEntity, probe);
    outProbeEntity = probeEntity;

    const auto probeVisualEntity = world.createEntity(envIndex);
    TransformComponent probeVisualTransform{};
    probeVisualTransform.worldTransform.position = probeTransform.worldTransform.position;
    probeVisualTransform.worldTransform.rotation = probeTransform.worldTransform.rotation;
    if (probe.geometry == UltrasoundProbeComponent::Geometry::Linear)
    {
        const Diligent::float3 probeDirection =
            probeTransform.worldTransform.rotation.RotateVector(Diligent::float3{0.0f, 0.0f, 1.0f});
        probeVisualTransform.worldTransform.position =
            probeTransform.worldTransform.position - probeDirection * (0.5f * kProbeBodyDepth);
    }
    world.setTransform(probeVisualEntity, probeVisualTransform);
    world.setMeshRenderer(probeVisualEntity,
                          MeshRendererComponent{probeMesh, materials.probe, true});
}

bool saveProbeImagesAndQuit(Runtime &runtime,
                            const std::vector<cressim::neo::common::EntityId> &probeEntities,
                            float captureDelaySeconds)
{
    GpuDevice *graphicsDevice = runtime.getGpuDevice();
    if (graphicsDevice == nullptr)
    {
        CRESSIM_LOG_ERROR("Probe image export failed: no GPU device.\n");
        return false;
    }

    cressim::neo::common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    const std::uint64_t settleFrames = static_cast<std::uint64_t>(
        std::ceil(captureDelaySeconds / std::max(frame.deltaSeconds, 1.0e-6f)));
    for (std::uint64_t i = 0u; i < settleFrames; ++i)
    {
        runtime.prepare();
        if (!runtime.stepPhysics(frame))
        {
            CRESSIM_LOG_ERROR("Probe image export failed: staged physics step failed.\n");
            return false;
        }
        if (!runtime.stepSimulationSensors(frame))
        {
            CRESSIM_LOG_ERROR("Probe image export failed: staged sensor step failed.\n");
            return false;
        }
        runtime.stepVisualSensors(frame);
        runtime.endFrame(frame);
        ++frame.frameIndex;
        frame.timeSeconds += static_cast<double>(frame.deltaSeconds);
    }

    runtime.prepare();

    std::vector<std::pair<cressim::neo::common::EntityId, GpuRenderTargetReadbackRequest>>
        readbackRequests;
    readbackRequests.reserve(probeEntities.size());
    for (const cressim::neo::common::EntityId probeEntity : probeEntities)
    {
        const UltrasoundProbeResult *probeResult =
            runtime.getWorld().tryGetUltrasoundProbeResult(probeEntity);
        if (probeResult == nullptr || !probeResult->prepared)
        {
            CRESSIM_LOG_ERROR("Probe image export failed: probe entity ", probeEntity,
                              " did not expose a valid ultrasound output after prepare().\n");
            return false;
        }

        const GpuRenderTargetReadbackRequest request =
            graphicsDevice->renderTargetSystem().requestRenderTargetReadback(
                cressim::neo::gpu::GpuRenderTargetBinding{probeResult->imageTarget, 0u, 1u});
        if (request.id == 0u)
        {
            CRESSIM_LOG_ERROR("Probe image export failed: could not queue readback for probe entity ",
                              probeEntity, ".\n");
            return false;
        }

        readbackRequests.emplace_back(probeEntity, request);
    }

    runtime.prepare();
    if (!runtime.stepPhysics(frame))
    {
        CRESSIM_LOG_ERROR("Probe image export failed: staged capture physics step failed.\n");
        return false;
    }
    if (!runtime.stepSimulationSensors(frame))
    {
        CRESSIM_LOG_ERROR("Probe image export failed: staged capture sensor step failed.\n");
        return false;
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);

    bool savedAnyImage = false;
    for (const auto &[probeEntity, request] : readbackRequests)
    {
        const UltrasoundProbeResult *probeResult =
            runtime.getWorld().tryGetUltrasoundProbeResult(probeEntity);
        if (probeResult == nullptr || !probeResult->prepared || !probeResult->completed)
        {
            CRESSIM_LOG_ERROR("Probe image export failed: probe entity ", probeEntity,
                              " did not produce a valid ultrasound image for frame 0 capture.\n");
            return false;
        }

        GpuRenderTargetReadbackEvent event{};
        if (!graphicsDevice->renderTargetSystem().tryGetRenderTargetReadback(request, event) ||
            !isValidReadback(event))
        {
            CRESSIM_LOG_ERROR("Probe image export failed: incomplete readback for probe entity ",
                              probeEntity, ".\n");
            return false;
        }

        const std::string outputPath =
            "ultrasound_probe_" + std::to_string(probeEntity) + ".ppm";
        if (!writePpm(outputPath, event))
        {
            CRESSIM_LOG_ERROR("Probe image export failed: could not write ", outputPath, ".\n");
            return false;
        }

        CRESSIM_LOG_INFO("Saved ultrasound image for probe entity=", probeEntity, " to ",
                         outputPath);
        savedAnyImage = true;
    }

    return savedAnyImage;
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    options.envCount = kDefaultEnvCount;
    bool debugParticles = false;
    bool saveProbeImages = false;
    float saveProbeDelaySeconds = 0.0f;
    ExampleProbeType probeType = ExampleProbeType::Linear;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (std::strcmp(argv[i], "--debug-particles") == 0)
            {
                debugParticles = true;
                continue;
            }
            if (std::strcmp(argv[i], "--probe-type") == 0)
            {
                probeType = parseProbeType(cressim::neo::examples::helpers::requireOptionValue(
                    argc, argv, i, "--probe-type"));
                continue;
            }
            if (std::strcmp(argv[i], "--save-probe-images") == 0)
            {
                saveProbeImages = true;
                continue;
            }
            if (std::strcmp(argv[i], "--save-probe-delay-seconds") == 0)
            {
                saveProbeDelaySeconds = parseNonNegativeFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--save-probe-delay-seconds"),
                    "--save-probe-delay-seconds");
                continue;
            }
            if (cressim::neo::examples::helpers::tryParseCommonArgument(argc, argv, i, options,
                                                                        true))
            {
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

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.physicsDesc.softContactIterations  = 60;
    config.physicsDesc.softInternalIterations = 60;
    config.physicsDesc.enableBlockingReadback = false;
    config.sceneLayout.envCount               = options.envCount;

    DebugViewerApp viewer;
    if (!saveProbeImages)
    {
        ViewerExampleDefaults viewerDefaults{};
        viewerDefaults.windowTitle = "CRESSim Neo Ultrasound Soft Cube Viewer";
        viewerDefaults.showStats   = true;
        viewerDefaults.vSync       = false;
        DebugViewerAppDesc viewerDesc =
            cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
        viewerDesc.statsIntervalFrames  = 60u;
        viewerDesc.enableDebugParticles = debugParticles;

        if (!viewer.initialize(viewerDesc, config))
        {
            CRESSIM_LOG_ERROR("Viewer initialization failed.\n");
            return 1;
        }
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        return 1;
    }

    UltrasoundProbeComponent probeDefaults{};
    probeDefaults.numScanlines         = 50u;
    probeDefaults.lineLength           = 1.2f;
    probeDefaults.scanlineSpacing      = 0.01f;
    probeDefaults.worldUnitsPerMeter   = 10.0f;
    probeDefaults.beamSigmaLateral     = 0.001f;
    probeDefaults.beamSigmaElevational = 0.001f;
    probeDefaults.imageBaseHeight      = 0u;
    probeDefaults.imageUseFixedMaxNormalization = false;
    probeDefaults.imageFixedMaxSignal  = 10.0f;
    if (probeType == ExampleProbeType::Curvilinear)
    {
        probeDefaults.geometry = UltrasoundProbeComponent::Geometry::Curvilinear;
        probeDefaults.sectorAngleDegrees = 60.0f;
        probeDefaults.probeRadius = 0.35f;
    }

    auto &resources = runtime.getResources();
    const MeshHandle boxMesh = resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
        {0.225f, 0.225f, 0.225f}, "SoftParticlesUltrasoundMultiEnv.SoftBodyMesh"));
    const MeshHandle probeMesh = registerProbeMesh(resources, probeDefaults);
    const MeshHandle planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(
            2.0f, "SoftParticlesUltrasoundMultiEnv.PlaneMesh"));

    SceneMaterials materials{};
    materials.ground = registerMaterial(resources, "SoftParticlesUltrasoundMultiEnv.Ground",
                                        {0.72f, 0.75f, 0.79f}, 0.90f);
    materials.softBody = registerMaterial(resources, "SoftParticlesUltrasoundMultiEnv.SoftBody",
                                          {0.86f, 0.54f, 0.44f}, 0.72f);
    materials.probe = registerMaterial(resources, "SoftParticlesUltrasoundMultiEnv.Probe",
                                       {0.24f, 0.28f, 0.33f}, 0.30f);

    cressim::neo::common::EntityId primaryCamera = cressim::neo::common::kInvalidEntityId;
    std::vector<cressim::neo::common::EntityId> probeEntities;
    probeEntities.reserve(options.envCount);
    for (std::uint32_t envIndex = 0u; envIndex < options.envCount; ++envIndex)
    {
        cressim::neo::common::EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        cressim::neo::common::EntityId probeEntity  = cressim::neo::common::kInvalidEntityId;
        authorEnvironment(runtime, envIndex, options.envCount, planeMesh, boxMesh, probeMesh,
                          probeDefaults, materials, cameraEntity, probeEntity);
        if (envIndex == 0u)
        {
            primaryCamera = cameraEntity;
        }
        probeEntities.push_back(probeEntity);
    }

    if (saveProbeImages)
    {
        const bool saved = saveProbeImagesAndQuit(runtime, probeEntities, saveProbeDelaySeconds);
        runtime.shutdown();
        return saved ? 0 : 1;
    }

    CRESSIM_LOG_INFO("Viewer controls: press U to toggle ultrasound image presentation, "
                     "',/.' to cycle probes, and [/] to cycle cameras.");

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{primaryCamera});

    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
