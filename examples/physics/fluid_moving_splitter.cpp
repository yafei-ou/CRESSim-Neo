#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/environment_ibl_baker.h"
#include "helpers/asset_paths.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/skybox_example.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::FluidComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::EnvironmentIblBakeOptions;
using cressim::neo::graphics::EnvironmentIblDesc;
using cressim::neo::graphics::IblQualityTier;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::FluidSourceKind;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;
using cressim::neo::viewer::DebugViewerCallbacks;

constexpr float kEnvSpacing               = 22.0f;
constexpr std::uint32_t kDefaultEnvCount  = 4u;
constexpr float kFluidParticleSpacing     = 0.18f;
constexpr float kFluidParticleRadius      = 0.09f;
constexpr float kDefaultFluidHeight       = 2.8f;
constexpr float kFluidBottomY             = 4.0f;
constexpr float kContainerHalfWidth       = 3.0f;
constexpr float kContainerHalfDepth       = 2.35f;
constexpr float kContainerWallHalfHeight  = 4.5f;
constexpr float kContainerCenterY         = 3.5f;
constexpr float kSplitterHalfWidth        = 0.22f;
constexpr float kSplitterHalfHeight       = 2.6f;
constexpr const char *kFluidSplitterSkyboxCrossPath =
    "environments/cubemaps/Cubemap/Cubemap_Sky_16-512x512.png";

struct ExampleOptions
{
    CommonExampleOptions common{};
    float fluidHeight = kDefaultFluidHeight;
};

struct SplitterState
{
    EntityId entityId = cressim::neo::common::kInvalidEntityId;
    Diligent::float3 basePosition{};
    float travel = 1.5f;
    float speed = 1.5f;
    float phaseOffset = 0.0f;
};

float parsePositiveFloat(const std::string &value, const char *optionName)
{
    const char *begin = value.c_str();
    char *end = nullptr;
    const float parsed = std::strtof(begin, &end);
    if (end == begin || *end != '\0' || parsed <= 0.0f)
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }

    return parsed;
}

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.metallic = 0.0f;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

EnvironmentIblDesc loadFluidSplitterSkyboxIbl(
    cressim::neo::graphics::RenderResourceManager &resources)
{
    const std::filesystem::path crossPath =
        cressim::neo::examples::helpers::assetPath(kFluidSplitterSkyboxCrossPath);

    EnvironmentIblBakeOptions options{};
    options.irradianceSize = 16u;
    options.specularSize = 128u;
    options.specularMipCount = 7u;
    options.irradianceSampleCount = 256u;
    options.specularSampleCount = 128u;
    options.intensity = 0.24f;
    options.backgroundIntensity = 1.00f;
    return cressim::neo::examples::helpers::createEnvironmentIblFromHorizontalCross(
        resources, crossPath, options);
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

void spawnStaticBox(cressim::neo::engine::World &world, std::uint32_t envIndex, MeshHandle mesh,
                    MaterialHandle material, const Diligent::float3 &position,
                    const Diligent::float3 &halfExtents)
{
    const auto entity = world.createEntity(envIndex);

    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale = halfExtents;
    world.setTransform(entity, transform);

    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(entity, renderer);

    RigidBodyComponent body{};
    body.bodyType = RigidBodyType::Static;
    body.inverseMass = 0.0f;
    body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(entity, body);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Box;
    collider.shapeParams = {1.0f, 1.0f, 1.0f, 0.0f};
    world.addCollider(entity, collider);
}

void spawnStaticCollisionBox(cressim::neo::engine::World &world, std::uint32_t envIndex,
                             const Diligent::float3 &position,
                             const Diligent::float3 &halfExtents)
{
    const auto entity = world.createEntity(envIndex);

    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale = halfExtents;
    world.setTransform(entity, transform);

    RigidBodyComponent body{};
    body.bodyType = RigidBodyType::Static;
    body.inverseMass = 0.0f;
    body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(entity, body);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Box;
    collider.shapeParams = {1.0f, 1.0f, 1.0f, 0.0f};
    world.addCollider(entity, collider);
}

bool spawnFluid(cressim::neo::engine::World &world, std::uint32_t envIndex,
                const Diligent::float3 &position, const Diligent::float3 &size,
                const cressim::neo::physics::FluidMaterialDesc &material,
                const Diligent::float4 &visualColor)
{
    const auto entity = world.createEntity(envIndex);

    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    FluidComponent fluid{};
    fluid.source.kind = FluidSourceKind::RegularGrid;
    fluid.source.regularGrid.size = size;
    fluid.source.regularGrid.targetParticleSpacing = kFluidParticleSpacing;
    fluid.particleRadius = kFluidParticleRadius;
    fluid.material = material;
    const float particleDiameter = 2.0f * fluid.particleRadius;
    fluid.particleMass =
        1.0f * particleDiameter * particleDiameter * particleDiameter * 1000.0f;
    fluid.visualColor = visualColor;
    fluid.collisionLayer = 0x1u;
    fluid.collisionMask = 0xffffffffu;
    return world.setFluid(entity, fluid);
}

Diligent::float4 fluidColorForEnv(std::uint32_t envIndex)
{
    switch (envIndex % 4u)
    {
    case 0u:
        return {0.16f, 0.68f, 0.96f, 0.70f};
    case 1u:
        return {0.16f, 0.84f, 0.50f, 0.72f};
    case 2u:
        return {0.96f, 0.56f, 0.18f, 0.74f};
    default:
        return {0.82f, 0.34f, 0.94f, 0.72f};
    }
}

cressim::neo::physics::FluidMaterialDesc fluidMaterialForEnv(std::uint32_t envIndex)
{
    cressim::neo::physics::ParticleContactMaterialDesc fluidContact{};
    fluidContact.friction = 0.04f;
    fluidContact.staticFriction = 0.06f;
    fluidContact.restitution = 0.0f;
    fluidContact.damping = 0.2f;

    cressim::neo::physics::FluidMaterialDesc material{};
    material.contact = fluidContact;
    material.cflCoefficient = 1.0f;

    switch (envIndex % 4u)
    {
    case 0u:
        material.viscosity = 1.0f;
        material.cohesion = 0.2f;
        material.gravityScale = 0.85f;
        material.vorticityConfinement = 0.2f;
        material.surfaceTension = 0.0f;
        break;
    case 1u:
        material.viscosity = 6.5f;
        material.cohesion = 0.4f;
        material.gravityScale = 0.65f;
        material.vorticityConfinement = 0.35f;
        material.surfaceTension = 1.0f;
        break;
    case 2u:
        material.viscosity = 2.5f;
        material.cohesion = 0.8f;
        material.gravityScale = 0.55f;
        material.vorticityConfinement = 0.1f;
        material.surfaceTension = 2.0f;
        break;
    default:
        material.viscosity = 0.35f;
        material.cohesion = 1.6f;
        material.gravityScale = 1.05f;
        material.vorticityConfinement = 0.8f;
        material.surfaceTension = 4.0f;
        break;
    }

    const float phase = static_cast<float>(envIndex) * 0.57f;
    material.viscosity *= 0.92f + 0.22f * (0.5f + 0.5f * std::sin(phase));
    material.cohesion *= 0.90f + 0.20f * (0.5f + 0.5f * std::cos(phase * 1.3f));
    material.gravityScale *= 0.94f + 0.10f * (0.5f + 0.5f * std::sin(phase * 0.8f));
    material.surfaceTension *= 0.88f + 0.24f * (0.5f + 0.5f * std::cos(phase * 1.1f));
    return material;
}

void authorEnvironment(Runtime &runtime, std::uint32_t envIndex, std::uint32_t envCount,
                       MeshHandle boxMesh, MaterialHandle floorMaterial,
                       MaterialHandle wallMaterial, MaterialHandle splitterMaterial,
                       float fluidHeight, EntityId &outCameraEntity,
                       std::vector<SplitterState> &outSplitters)
{
    auto &world = runtime.getWorld();
    const Diligent::float3 origin = envOrigin(envIndex, envCount);
    const float phase = static_cast<float>(envIndex);

    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = origin + Diligent::float3{0.0f, 5.4f, -13.5f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.30f);
    world.setTransform(outCameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 36.0f;
    camera.renderOrder        = static_cast<int>(envIndex);
    camera.backgroundMode     = CameraComponent::BackgroundMode::EnvironmentCubemap;
    world.setCamera(outCameraEntity, camera);

    const auto lightEntity = world.createEntity(envIndex);
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 0.98f, 0.95f};
    light.intensity = 5.6f;
    world.setDirectionalLight(lightEntity, light);

    spawnStaticBox(world, envIndex, boxMesh, floorMaterial, origin + Diligent::float3{0.0f, -1.0f, 0.0f},
                   {kContainerHalfWidth, 0.15f, kContainerHalfDepth});
    spawnStaticBox(world, envIndex, boxMesh, wallMaterial,
                   origin + Diligent::float3{-kContainerHalfWidth, kContainerCenterY, 0.0f},
                   {0.15f, kContainerWallHalfHeight, kContainerHalfDepth});
    spawnStaticBox(world, envIndex, boxMesh, wallMaterial,
                   origin + Diligent::float3{kContainerHalfWidth, kContainerCenterY, 0.0f},
                   {0.15f, kContainerWallHalfHeight, kContainerHalfDepth});
    spawnStaticBox(world, envIndex, boxMesh, wallMaterial,
                   origin + Diligent::float3{0.0f, kContainerCenterY, kContainerHalfDepth},
                   {kContainerHalfWidth, kContainerWallHalfHeight, 0.15f});
    spawnStaticCollisionBox(world, envIndex,
                            origin + Diligent::float3{0.0f, kContainerCenterY, -kContainerHalfDepth},
                            {kContainerHalfWidth, kContainerWallHalfHeight, 0.15f});

    const Diligent::float3 splitterBasePosition =
        origin + Diligent::float3{0.55f + 0.35f * std::cos(phase * 0.8f), 1.55f, 0.0f};
    const auto splitterEntity = world.createEntity(envIndex);
    TransformComponent splitterTransform{};
    splitterTransform.worldTransform.position = splitterBasePosition;
    splitterTransform.worldTransform.scale =
        {kSplitterHalfWidth, kSplitterHalfHeight, kContainerHalfDepth};
    world.setTransform(splitterEntity, splitterTransform);

    MeshRendererComponent splitterRenderer{};
    splitterRenderer.mesh = boxMesh;
    splitterRenderer.material = splitterMaterial;
    splitterRenderer.visible = true;
    world.setMeshRenderer(splitterEntity, splitterRenderer);

    RigidBodyComponent splitterBody{};
    splitterBody.bodyType = RigidBodyType::Kinematic;
    splitterBody.inverseMass = 0.0f;
    splitterBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    splitterBody.kinematicTargetPosition = splitterBasePosition;
    splitterBody.kinematicTargetRotation = splitterTransform.worldTransform.rotation;
    splitterBody.kinematicTargetEnabled = true;
    world.setRigidBody(splitterEntity, splitterBody);

    ColliderComponent splitterCollider{};
    splitterCollider.shapeType = ColliderShapeType::Box;
    splitterCollider.shapeParams = {1.0f, 1.0f, 1.0f, 0.0f};
    world.addCollider(splitterEntity, splitterCollider);

    const Diligent::float3 fluidSize = {1.35f, fluidHeight, 2.4f};
    const Diligent::float3 fluidCenter =
        origin + Diligent::float3{-1.45f, kFluidBottomY + 0.5f * fluidSize.y, 0.0f};
    if (!spawnFluid(world, envIndex, fluidCenter, fluidSize, fluidMaterialForEnv(envIndex),
                    fluidColorForEnv(envIndex)))
    {
        throw std::runtime_error("Failed to author fluid body.");
    }

    outSplitters.push_back(SplitterState{
        splitterEntity,
        splitterBasePosition,
        1.2f + 0.35f * (0.5f + 0.5f * std::sin(phase * 0.7f)),
        1.2f + 0.5f * (0.5f + 0.5f * std::cos(phase * 1.1f)),
        0.75f * phase});
}

} // namespace

int main(int argc, char **argv)
{
    ExampleOptions options{};
    options.common.envCount = kDefaultEnvCount;
    bool debugParticles = false;

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
            if (arg == "--fluid-height")
            {
                options.fluidHeight = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--fluid-height"),
                    "fluid height");
                continue;
            }

            if (arg == "--debug-particles")
            {
                debugParticles = true;
                continue;
            }

            cressim::neo::examples::helpers::printUsage(
                argv[0], " [--fluid-height H] [--debug-particles]", true);
            return 2;
        }
    }
    catch (const std::invalid_argument &error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        cressim::neo::examples::helpers::printUsage(
            argv[0], " [--fluid-height H] [--debug-particles]", true);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options.common);
    config.rendererDesc.iblQualityTier = IblQualityTier::Full;
    config.sceneLayout.envCount = options.common.envCount;
    config.physicsDesc.substeps = 1u;
    config.physicsDesc.defaultIterations = 10u;
    config.physicsDesc.fluidIterations = 10u;
    config.physicsDesc.softContactIterations = 0u;
    config.physicsDesc.softInternalIterations = 0u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Fluid Moving Splitter Multi-Env";
    viewerDefaults.showStats = true;
    viewerDefaults.vSync = false;
    auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options.common, viewerDefaults);
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

    auto &resources = runtime.getResources();
    const MeshHandle boxMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "SingleSourceFluid.BoxMesh"));
    const MaterialHandle floorMaterial =
        registerMaterial(resources, "SingleSourceFluid.Floor", {0.78f, 0.80f, 0.84f}, 0.92f);
    const MaterialHandle wallMaterial =
        registerMaterial(resources, "SingleSourceFluid.Wall", {0.16f, 0.42f, 0.82f}, 0.58f);
    const MaterialHandle splitterMaterial =
        registerMaterial(resources, "SingleSourceFluid.Splitter", {0.78f, 0.50f, 0.16f}, 0.38f);
    auto &world = runtime.getWorld();
    const auto sharedIbl = loadFluidSplitterSkyboxIbl(resources);
    for (std::uint32_t envIndex = 0u; envIndex < options.common.envCount; ++envIndex)
    {
        if (!world.setEnvironmentIbl(envIndex, sharedIbl))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("Failed to assign fluid moving splitter skybox IBL.\n");
            return 1;
        }
    }

    EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
    std::vector<SplitterState> splitters{};
    splitters.reserve(options.common.envCount);
    try
    {
        for (std::uint32_t envIndex = 0u; envIndex < options.common.envCount; ++envIndex)
        {
            EntityId envCameraEntity = cressim::neo::common::kInvalidEntityId;
            authorEnvironment(runtime, envIndex, options.common.envCount, boxMesh, floorMaterial,
                              wallMaterial, splitterMaterial, options.fluidHeight,
                              envCameraEntity, splitters);
            if (envIndex == 0u)
            {
                cameraEntity = envCameraEntity;
            }
        }
    }
    catch (const std::runtime_error &error)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR(error.what(), "\n");
        return 1;
    }

    runtime.getWorld().physicsWorld().ensureDerivedStateUpToDate();
    const auto &particles = runtime.getWorld().physicsWorld().particles();
    CRESSIM_LOG_INFO("Fluid moving splitter authored ", particles.size(),
                     " particles across ", options.common.envCount,
                     " environments. Fluid bodies=", runtime.getWorld().physicsWorld().fluidCount(),
                     ".\n");

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [splitters](const FrameContext &frame, Runtime &cbRuntime) {
        for (const SplitterState &splitter : splitters)
        {
            auto rigidBody = cbRuntime.getWorld().tryGetRigidBody(splitter.entityId);
            if (!rigidBody.has_value())
            {
                continue;
            }

            rigidBody->bodyType = RigidBodyType::Kinematic;
            rigidBody->inverseMass = 0.0f;
            rigidBody->inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
            rigidBody->kinematicTargetEnabled = true;
            rigidBody->kinematicTargetPosition = splitter.basePosition;
            rigidBody->kinematicTargetPosition.x +=
                splitter.travel *
                std::sin(static_cast<float>(frame.timeSeconds) * splitter.speed +
                         splitter.phaseOffset);
            cbRuntime.getWorld().setRigidBody(splitter.entityId, *rigidBody);
        }
    };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Viewer run failed.\n");
        return 1;
    }

    return 0;
}
