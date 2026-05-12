#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <stdexcept>
#include <string>

namespace
{

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

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
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

void spawnStaticBox(cressim::neo::engine::World &world, cressim::neo::common::EntityId entity,
                    MeshHandle mesh, MaterialHandle material, const Diligent::float3 &position,
                    const Diligent::float3 &halfExtents)
{
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale = halfExtents;
    world.setTransform(entity, transform);

    world.setMeshRenderer(entity, MeshRendererComponent{mesh, material, true});

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

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, false))
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
    config.physicsDesc.substeps = 4u;
    config.physicsDesc.defaultIterations = 10u;
    config.physicsDesc.fluidIterations = 5u;
    config.physicsDesc.softContactIterations = 8u;
    config.physicsDesc.softInternalIterations = 0u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Fluid Debug Particles";
    viewerDefaults.showStats = true;
    viewerDefaults.vSync = true;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
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
    cameraTransform.worldTransform.position = {0.0f, 2.8f, -8.5f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.18f);
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 42.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 0.98f, 0.95f};
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const MeshHandle boxMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "FluidDebug.BoxMesh"));
    const MaterialHandle floorMaterial =
        registerMaterial(resources, "FluidDebug.Floor", {0.78f, 0.80f, 0.84f}, 0.92f);
    const MaterialHandle wallMaterial =
        registerMaterial(resources, "FluidDebug.Wall", {0.16f, 0.42f, 0.82f}, 0.58f);
    const MaterialHandle obstacleMaterial =
        registerMaterial(resources, "FluidDebug.Obstacle", {0.88f, 0.36f, 0.18f}, 0.35f);

    spawnStaticBox(world, boxMesh, floorMaterial, {0.0f, -1.0f, 0.0f}, {4.5f, 0.15f, 4.5f});
    spawnStaticBox(world, boxMesh, wallMaterial, {-4.35f, 0.75f, 0.0f}, {0.15f, 1.9f, 4.5f});
    spawnStaticBox(world, boxMesh, wallMaterial, {4.35f, 0.75f, 0.0f}, {0.15f, 1.9f, 4.5f});
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, 0.75f, -4.35f}, {4.5f, 1.9f, 0.15f});
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, 0.75f, 4.35f}, {4.5f, 1.9f, 0.15f});

    const auto centerObstacle = world.createEntity();
    spawnStaticBox(world, centerObstacle, boxMesh, obstacleMaterial, {0.0f, -0.15f, 0.4f},
                   {0.85f, 0.65f, 0.85f});

    const auto fluidEntity = world.createEntity();
    TransformComponent fluidTransform{};
    fluidTransform.worldTransform.position = {0.0f, 2.45f, 0.0f};
    world.setTransform(fluidEntity, fluidTransform);

    FluidComponent fluid{};
    fluid.source.kind = FluidSourceKind::RegularGrid;
    fluid.source.regularGrid.size = {3.6f, 0.9f, 1.4f};
    fluid.source.regularGrid.targetParticleSpacing = 0.24f;
    fluid.particleRadius = 0.12f;
    const float particleDiameter = 2.0f * fluid.particleRadius;
    fluid.particleMass =
        0.8f * particleDiameter * particleDiameter * particleDiameter * 1000.0f;
    fluid.material.viscosity = 0.02f;
    fluid.collisionLayer = 0x1u;
    fluid.collisionMask = 0xffffffffu;
    if (!world.setFluid(fluidEntity, fluid))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author fluid body.\n");
        return 1;
    }

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
