#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

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

struct SceneMaterials
{
    MaterialHandle ground{};
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", true);
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

void authorEnvironment(Runtime &runtime, std::uint32_t envIndex, std::uint32_t envCount,
                       MeshHandle planeMesh, const SceneMaterials &materials,
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
        origin + Diligent::float3{0.0f, 0.28f + 0.02f * std::sin(phase), 0.0f};
    softTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.15f * std::sin(phase));
    world.setTransform(softEntity, softTransform);

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

    const auto probeEntity = world.createEntity(envIndex);
    TransformComponent probeTransform{};
    probeTransform.worldTransform.position = origin + Diligent::float3{0.0f, 0.85f, 0.0f};
    probeTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.5f * kPi);
    world.setTransform(probeEntity, probeTransform);

    UltrasoundProbeComponent probe{};
    probe.numScanlines                  = 50u;
    probe.lineLength                    = 1.2f;
    probe.scanlineSpacing               = 0.01f;
    probe.worldUnitsPerMeter            = 10.0f;
    probe.beamSigmaLateral              = 0.01f;
    probe.beamSigmaElevational          = 0.01f;
    probe.imageBaseHeight               = 0u;
    probe.imageUseFixedMaxNormalization = false;
    probe.imageFixedMaxSignal           = 10.0f;
    world.setUltrasoundProbe(probeEntity, probe);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    options.envCount = kDefaultEnvCount;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
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
    viewerDesc.enableDebugParticles = true;

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

    auto &resources = runtime.getResources();
    const MeshHandle planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(
            2.0f, "SoftParticlesUltrasoundMultiEnv.PlaneMesh"));

    SceneMaterials materials{};
    materials.ground = registerMaterial(resources, "SoftParticlesUltrasoundMultiEnv.Ground",
                                        {0.72f, 0.75f, 0.79f}, 0.90f);

    cressim::neo::common::EntityId primaryCamera = cressim::neo::common::kInvalidEntityId;
    for (std::uint32_t envIndex = 0u; envIndex < options.envCount; ++envIndex)
    {
        cressim::neo::common::EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        authorEnvironment(runtime, envIndex, options.envCount, planeMesh, materials, cameraEntity);
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
