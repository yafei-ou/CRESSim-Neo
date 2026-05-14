#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <cstdlib>
#include <cmath>
#include <string>
#include <stdexcept>

namespace
{

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
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::FluidSourceKind;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;
using cressim::neo::viewer::DebugViewerCallbacks;

constexpr float kFluidParticleSpacing   = 0.18f;
constexpr float kFluidParticleRadius    = 0.09f;
constexpr float kDefaultFluidHeight     = 2.8f;
constexpr float kFluidBottomY           = 4.0f;
constexpr float kContainerHalfWidth     = 3.0f;
constexpr float kContainerHalfDepth     = 2.35f;
constexpr float kContainerWallHalfHeight = 4.5f;
constexpr float kContainerCenterY       = 3.5f;
constexpr float kSplitterHalfWidth      = 0.22f;
constexpr float kSplitterHalfHeight     = 2.6f;
constexpr float kSplitterTravel         = 1.5f;
constexpr float kSplitterSpeed          = 1.5f;

struct ExampleOptions
{
    CommonExampleOptions common{};
    float fluidHeight = kDefaultFluidHeight;
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

void spawnStaticBox(cressim::neo::engine::World &world, MeshHandle mesh, MaterialHandle material,
                    const Diligent::float3 &position, const Diligent::float3 &halfExtents)
{
    const auto entity = world.createEntity();

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
    body.simulated = true;
    body.bodyType = RigidBodyType::Static;
    body.inverseMass = 0.0f;
    body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(entity, body);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Box;
    collider.shapeParams = {1.0f, 1.0f, 1.0f, 0.0f};
    world.addCollider(entity, collider);
}

void spawnStaticCollisionBox(cressim::neo::engine::World &world,
                             const Diligent::float3 &position,
                             const Diligent::float3 &halfExtents)
{
    const auto entity = world.createEntity();

    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale = halfExtents;
    world.setTransform(entity, transform);

    RigidBodyComponent body{};
    body.simulated = true;
    body.bodyType = RigidBodyType::Static;
    body.inverseMass = 0.0f;
    body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(entity, body);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Box;
    collider.shapeParams = {1.0f, 1.0f, 1.0f, 0.0f};
    world.addCollider(entity, collider);
}

bool spawnFluid(cressim::neo::engine::World &world, const Diligent::float3 &position,
                const Diligent::float3 &size,
                const cressim::neo::physics::FluidMaterialDesc &material)
{
    const auto entity = world.createEntity();

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
    fluid.collisionLayer = 0x1u;
    fluid.collisionMask = 0xffffffffu;
    return world.setFluid(entity, fluid);
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
                    argc, argv, i, options.common, false))
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

            cressim::neo::examples::helpers::printUsage(argv[0], " [--fluid-height H]", false);
            return 2;
        }
    }
    catch (const std::invalid_argument &error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        cressim::neo::examples::helpers::printUsage(argv[0], " [--fluid-height H]", false);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options.common);
    config.physicsDesc.substeps = 1u;
    config.physicsDesc.defaultIterations = 10u;
    config.physicsDesc.fluidIterations = 10u;
    config.physicsDesc.softContactIterations = 0u;
    config.physicsDesc.softInternalIterations = 0u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Fluid Single Source Moving Splitter";
    viewerDefaults.showStats = true;
    viewerDefaults.vSync = false;
    auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options.common, viewerDefaults);
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

    auto &world = runtime.getWorld();
    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 5.4f, -13.5f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.30f);
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 36.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 0.98f, 0.95f};
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const MeshHandle boxMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "SingleSourceFluid.BoxMesh"));
    const MaterialHandle floorMaterial =
        registerMaterial(resources, "SingleSourceFluid.Floor", {0.78f, 0.80f, 0.84f}, 0.92f);
    const MaterialHandle wallMaterial =
        registerMaterial(resources, "SingleSourceFluid.Wall", {0.16f, 0.42f, 0.82f}, 0.58f);
    const MaterialHandle splitterMaterial =
        registerMaterial(resources, "SingleSourceFluid.Splitter", {0.78f, 0.50f, 0.16f}, 0.38f);

    spawnStaticBox(world, boxMesh, floorMaterial, {0.0f, -1.0f, 0.0f},
                   {kContainerHalfWidth, 0.15f, kContainerHalfDepth});
    spawnStaticBox(world, boxMesh, wallMaterial, {-kContainerHalfWidth, kContainerCenterY, 0.0f},
                   {0.15f, kContainerWallHalfHeight, kContainerHalfDepth});
    spawnStaticBox(world, boxMesh, wallMaterial, {kContainerHalfWidth, kContainerCenterY, 0.0f},
                   {0.15f, kContainerWallHalfHeight, kContainerHalfDepth});
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, kContainerCenterY, kContainerHalfDepth},
                   {kContainerHalfWidth, kContainerWallHalfHeight, 0.15f});
    spawnStaticCollisionBox(world, {0.0f, kContainerCenterY, -kContainerHalfDepth},
                            {kContainerHalfWidth, kContainerWallHalfHeight, 0.15f});

    const Diligent::float3 splitterBasePosition = {1.0f, 1.55f, 0.0f};
    const auto splitterEntity = world.createEntity();
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
    splitterBody.simulated = true;
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

    cressim::neo::physics::ParticleContactMaterialDesc fluidContact{};
    fluidContact.friction = 0.04f;
    fluidContact.staticFriction = 0.06f;
    fluidContact.restitution = 0.0f;
    fluidContact.damping = 0.2f;

    cressim::neo::physics::FluidMaterialDesc fluidMaterial{};
    fluidMaterial.contact = fluidContact;
    fluidMaterial.viscosity = 5.0f;
    fluidMaterial.cohesion = 15.0f;
    fluidMaterial.gravityScale = 0.7f;
    fluidMaterial.cflCoefficient = 1.0f;
    fluidMaterial.vorticityConfinement = 0.5f;
    fluidMaterial.surfaceTension = 10.0f;

    const Diligent::float3 fluidSize = {1.35f, options.fluidHeight, 2.4f};
    const Diligent::float3 fluidCenter = {-1.45f, kFluidBottomY + 0.5f * fluidSize.y, 0.0f};
    if (!spawnFluid(world, fluidCenter, fluidSize, fluidMaterial))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author single-source fluid body.\n");
        return 1;
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    const auto &particles = world.physicsWorld().particles();
    CRESSIM_LOG_INFO("Fluid single-source moving splitter authored ", particles.size(),
                     " particles. Fluid size=(", fluidSize.x, ", ", fluidSize.y, ", ",
                     fluidSize.z, "), fluid bodies=", world.physicsWorld().fluidCount(), ".\n");

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [splitterEntity, splitterBasePosition](const FrameContext &frame,
                                                                  Runtime &cbRuntime) {
        auto rigidBody = cbRuntime.getWorld().tryGetRigidBody(splitterEntity);
        if (!rigidBody.has_value())
        {
            return;
        }

        rigidBody->bodyType = RigidBodyType::Kinematic;
        rigidBody->inverseMass = 0.0f;
        rigidBody->inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
        rigidBody->kinematicTargetEnabled = true;
        rigidBody->kinematicTargetPosition = splitterBasePosition;
        rigidBody->kinematicTargetPosition.x +=
            kSplitterTravel * std::sin(static_cast<float>(frame.timeSeconds) * kSplitterSpeed);
        cbRuntime.getWorld().setRigidBody(splitterEntity, *rigidBody);
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
