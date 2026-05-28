#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

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
using cressim::neo::engine::UltrasoundScattererSourceComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
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
                       cressim::neo::common::EntityId &outCameraEntity)
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
    softTransform.worldTransform.position =
        origin + Diligent::float3{0.0f, 2.28f + 0.02f * std::sin(phase), 0.0f};
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

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    options.envCount = kDefaultEnvCount;
    bool debugParticles = false;
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
    probeDefaults.beamSigmaLateral     = 0.01f;
    probeDefaults.beamSigmaElevational = 0.01f;
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
    for (std::uint32_t envIndex = 0u; envIndex < options.envCount; ++envIndex)
    {
        cressim::neo::common::EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        authorEnvironment(runtime, envIndex, options.envCount, planeMesh, boxMesh, probeMesh,
                          probeDefaults, materials, cameraEntity);
        if (envIndex == 0u)
        {
            primaryCamera = cameraEntity;
        }
    }

    CRESSIM_LOG_INFO("Viewer controls: press U to toggle ultrasound image presentation, "
                     "',/.' to cycle probes, and [/] to cycle cameras.");

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{primaryCamera});

    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
