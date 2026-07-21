#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
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
using cressim::neo::engine::FluidComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::FluidSourceKind;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kEnvSpacing              = 20.0f;
constexpr std::uint32_t kDefaultEnvCount = 4u;
constexpr float kFluidParticleSpacing    = 0.18f;
constexpr float kFluidParticleRadius     = 0.09f;
constexpr float kSoftParticleSpacing     = 0.24f;
constexpr float kSoftParticleRadius      = 0.12f;

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

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, " [--debug-particles]", true);
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
                const Diligent::float3 &position,
                const Diligent::float3 &size,
                const cressim::neo::physics::FluidMaterialDesc &material,
                const Diligent::float4 &visualColor,
                float particleMassScale = 0.8f)
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
    fluid.particleMass = particleMassScale * particleDiameter * particleDiameter *
                         particleDiameter * 1000.0f;
    fluid.visualColor = visualColor;
    fluid.collisionLayer = 0x1u;
    fluid.collisionMask = 0xffffffffu;
    return world.setFluid(entity, fluid);
}

Diligent::float4 baselineFluidColor(std::uint32_t envIndex)
{
    switch (envIndex % 4u)
    {
    case 0u:
        return {0.18f, 0.70f, 0.96f, 0.66f};
    case 1u:
        return {0.22f, 0.88f, 0.58f, 0.64f};
    case 2u:
        return {0.98f, 0.62f, 0.20f, 0.68f};
    default:
        return {0.74f, 0.50f, 0.98f, 0.70f};
    }
}

Diligent::float4 thickerFluidColor(std::uint32_t envIndex)
{
    switch (envIndex % 4u)
    {
    case 0u:
        return {0.04f, 0.34f, 0.90f, 0.84f};
    case 1u:
        return {0.00f, 0.56f, 0.28f, 0.82f};
    case 2u:
        return {0.86f, 0.26f, 0.14f, 0.86f};
    default:
        return {0.44f, 0.16f, 0.82f, 0.84f};
    }
}

cressim::neo::graphics::EnvironmentFluidDesc environmentFluidSettings(std::uint32_t envIndex)
{
    cressim::neo::graphics::EnvironmentFluidDesc desc{};
    switch (envIndex % 4u)
    {
    case 0u:
        desc.smoothness = 0.94f;
        desc.specular = {0.34f, 0.42f, 0.50f};
        desc.fresnel = 0.82f;
        desc.depthEdgeThreshold = 0.16f;
        desc.filterRadiusPixels = 3.0f;
        desc.filterWorldRadius = 0.14f;
        desc.filterDepthThreshold = 0.10f;
        desc.enableBackgroundRefraction = true;
        desc.refractionViewThickness = 0.28f;
        break;
    case 1u:
        desc.smoothness = 0.88f;
        desc.specular = {0.26f, 0.34f, 0.30f};
        desc.fresnel = 0.74f;
        desc.depthEdgeThreshold = 0.22f;
        desc.filterRadiusPixels = 4.0f;
        desc.filterWorldRadius = 0.18f;
        desc.filterDepthThreshold = 0.14f;
        desc.enableBackgroundRefraction = true;
        desc.refractionViewThickness = 0.42f;
        break;
    case 2u:
        desc.smoothness = 0.80f;
        desc.specular = {0.40f, 0.30f, 0.18f};
        desc.fresnel = 0.66f;
        desc.depthEdgeThreshold = 0.26f;
        desc.filterRadiusPixels = 5.0f;
        desc.filterWorldRadius = 0.24f;
        desc.filterDepthThreshold = 0.18f;
        desc.enableBackgroundRefraction = false;
        desc.refractionViewThickness = 0.35f;
        break;
    default:
        desc.smoothness = 0.97f;
        desc.specular = {0.42f, 0.34f, 0.52f};
        desc.fresnel = 0.90f;
        desc.depthEdgeThreshold = 0.14f;
        desc.filterRadiusPixels = 2.0f;
        desc.filterWorldRadius = 0.10f;
        desc.filterDepthThreshold = 0.08f;
        desc.enableBackgroundRefraction = true;
        desc.refractionIor = 1.38f;
        desc.refractionViewThickness = 0.24f;
        break;
    }
    return desc;
}

bool spawnSoftBody(cressim::neo::engine::World &world, std::uint32_t envIndex,
                   const Diligent::float3 &position,
                   const cressim::neo::physics::ParticleContactMaterialDesc &contactMaterial)
{
    const auto entity = world.createEntity(envIndex);

    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.3f);
    world.setTransform(entity, transform);

    SoftBodyComponent softBody{};
    softBody.source.kind = SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = {0.9f, 0.9f, 0.9f};
    softBody.source.regularGrid.targetParticleSpacing = kSoftParticleSpacing;
    softBody.particleMass = 0.12f;
    softBody.particleRadius = kSoftParticleRadius;
    softBody.edgeCompliance = 0.0f;
    softBody.volumeCompliance = 0.0008f;
    softBody.material.contact = contactMaterial;
    softBody.selfCollisionEnabled = true;
    softBody.collisionLayer = 0x1u;
    softBody.collisionMask = 0xffffffffu;
    return world.setSoftBody(entity, softBody);
}

void authorEnvironment(cressim::neo::engine::Runtime &runtime, std::uint32_t envIndex,
                       std::uint32_t envCount, MeshHandle boxMesh, MaterialHandle floorMaterial,
                       MaterialHandle wallMaterial, MaterialHandle dividerMaterial,
                       const cressim::neo::physics::FluidMaterialDesc &baselineFluid,
                       const cressim::neo::physics::FluidMaterialDesc &thickerFluid,
                       const cressim::neo::physics::ParticleContactMaterialDesc &softContact,
                       cressim::neo::common::EntityId &outCameraEntity)
{
    auto &world                   = runtime.getWorld();
    const Diligent::float3 origin = envOrigin(envIndex, envCount);
    const float phase             = static_cast<float>(envIndex) * 0.61f;
    cressim::neo::physics::FluidMaterialDesc envBaselineFluid = baselineFluid;
    cressim::neo::physics::FluidMaterialDesc envThickerFluid  = thickerFluid;

    envBaselineFluid.gravityScale *= 0.92f + 0.10f * std::cos(phase);
    envBaselineFluid.viscosity += 0.18f * (0.5f + 0.5f * std::sin(phase * 1.7f));

    envThickerFluid.gravityScale *= 0.94f + 0.08f * std::sin(phase * 0.9f);
    envThickerFluid.viscosity *= 0.85f + 0.30f * (0.5f + 0.5f * std::cos(phase * 1.3f));
    envThickerFluid.cohesion *= 0.88f + 0.24f * (0.5f + 0.5f * std::sin(phase * 1.1f));
    envThickerFluid.surfaceTension *=
        0.86f + 0.28f * (0.5f + 0.5f * std::cos(phase * 0.8f));
    world.setEnvironmentFluid(envIndex, environmentFluidSettings(envIndex));

    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = origin + Diligent::float3{0.0f, 3.4f, -9.0f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.22f);
    world.setTransform(outCameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 42.0f;
    camera.renderOrder        = static_cast<int>(envIndex);
    world.setCamera(outCameraEntity, camera);

    const auto lightEntity = world.createEntity(envIndex);
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 0.98f, 0.95f};
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    spawnStaticCollisionBox(world, envIndex, origin + Diligent::float3{0.0f, -1.0f, 0.0f},
                            {3.2f, 0.15f, 3.0f});
    spawnStaticBox(world, envIndex, boxMesh, wallMaterial,
                   origin + Diligent::float3{-3.05f, -0.15f, 0.0f},
                   {0.15f, 1.0f, 3.0f});
    spawnStaticBox(world, envIndex, boxMesh, wallMaterial,
                   origin + Diligent::float3{3.05f, -0.15f, 0.0f},
                   {0.15f, 1.0f, 3.0f});
    spawnStaticCollisionBox(world, envIndex,
                            origin + Diligent::float3{0.0f, -0.15f, -2.85f},
                            {3.2f, 1.0f, 0.15f});
    spawnStaticBox(world, envIndex, boxMesh, wallMaterial,
                   origin + Diligent::float3{0.0f, -0.15f, 2.85f},
                   {3.2f, 1.0f, 0.15f});
    spawnStaticBox(world, envIndex, boxMesh, dividerMaterial,
                   origin + Diligent::float3{0.0f, -0.15f, 0.0f},
                   {0.25f, 1.0f, 3.0f});
    spawnStaticBox(world, envIndex, boxMesh, floorMaterial,
                   origin + Diligent::float3{0.0f, -1.16f, 0.0f},
                   {3.2f, 0.04f, 3.0f});

    if (!spawnFluid(world, envIndex,
                    origin + Diligent::float3{-1.575f, 1.0f, 0.0f},
                    {2.5f, 2.2f, 5.2f}, envBaselineFluid, baselineFluidColor(envIndex), 1.0f))
    {
        throw std::runtime_error("Failed to author baseline fluid body.");
    }

    if (!spawnFluid(world, envIndex,
                    origin + Diligent::float3{1.575f, 1.0f, 0.0f},
                    {2.5f, 2.2f, 5.2f}, envThickerFluid, thickerFluidColor(envIndex), 1.0f))
    {
        throw std::runtime_error("Failed to author thicker fluid body.");
    }

    if (!spawnSoftBody(world, envIndex,
                       origin + Diligent::float3{0.0f, 5.45f, 0.0f},
                       softContact))
    {
        throw std::runtime_error("Failed to author soft-body particle block.");
    }
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    options.envCount = kDefaultEnvCount;
    bool debugParticles = false;

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, true))
            {
                continue;
            }

            if (std::string{argv[i]} == "--debug-particles")
            {
                debugParticles = true;
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
    config.sceneLayout.envCount = options.envCount;
    config.physicsDesc.substeps = 1u;
    config.physicsDesc.defaultIterations = 10u;
    config.physicsDesc.fluidIterations = 10u;
    config.physicsDesc.softContactIterations = 8u;
    config.physicsDesc.softInternalIterations = 4u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Fluid Material Showcase Multi-Env";
    viewerDefaults.showStats = true;
    viewerDefaults.vSync = false;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
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
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "ParticleMaterialShowcase.BoxMesh"));
    const MaterialHandle floorMaterial =
        registerMaterial(resources, "ParticleMaterialShowcase.Floor", {0.80f, 0.82f, 0.85f}, 0.95f);
    const MaterialHandle wallMaterial =
        registerMaterial(resources, "ParticleMaterialShowcase.Wall", {0.14f, 0.34f, 0.68f}, 0.6f);
    const MaterialHandle dividerMaterial =
        registerMaterial(resources, "ParticleMaterialShowcase.Divider", {0.72f, 0.48f, 0.18f}, 0.4f);

    cressim::neo::physics::ParticleContactMaterialDesc fluidContact{};
    fluidContact.friction = 0.04f;
    fluidContact.staticFriction = 0.06f;
    fluidContact.restitution = 0.0f;
    fluidContact.damping = 0.2f;

    cressim::neo::physics::ParticleContactMaterialDesc softContact{};
    softContact.friction = 0.55f;
    softContact.staticFriction = 0.75f;
    softContact.restitution = 0.05f;
    softContact.damping = 0.8f;

    cressim::neo::physics::FluidMaterialDesc baselineFluid{};
    baselineFluid.contact = fluidContact;
    baselineFluid.viscosity = 0.0f;
    baselineFluid.gravityScale = 0.2f;
    baselineFluid.cflCoefficient = 1.0f;

    // Keep the comparison modest so the example demonstrates material variation
    // rather than turning into a stress test for the less-mature fluid terms.
    cressim::neo::physics::FluidMaterialDesc thickerFluid = baselineFluid;
    thickerFluid.cohesion = 1.0f;
    thickerFluid.surfaceTension = 1.0f;
    // thickerFluid.vorticityConfinement = 10.0f;
    // thickerFluid.cflCoefficient = 10.0f;
    thickerFluid.viscosity = 1.0f;
    thickerFluid.gravityScale = 0.2f;

    cressim::neo::common::EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
    try
    {
        for (std::uint32_t envIndex = 0u; envIndex < options.envCount; ++envIndex)
        {
            cressim::neo::common::EntityId envCameraEntity = cressim::neo::common::kInvalidEntityId;
            authorEnvironment(runtime, envIndex, options.envCount, boxMesh, floorMaterial,
                              wallMaterial, dividerMaterial, baselineFluid, thickerFluid,
                              softContact, envCameraEntity);
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
    CRESSIM_LOG_INFO("Fluid material showcase multi-env authored ", particles.size(),
                     " particles across ", options.envCount, " environments. Fluid bodies=",
                     runtime.getWorld().physicsWorld().fluidCount(), ", soft bodies=",
                     runtime.getWorld().physicsWorld().softBodyCount(), ".\n");

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, {});

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Viewer run failed.\n");
        return 1;
    }

    return 0;
}
