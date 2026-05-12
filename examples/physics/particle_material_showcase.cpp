#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

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
    cressim::neo::examples::helpers::printUsage(appName, "", false);
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
                const cressim::neo::physics::FluidMaterialDesc &material,
                float particleMassScale = 0.8f)
{
    const auto entity = world.createEntity();

    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    FluidComponent fluid{};
    fluid.source.kind = FluidSourceKind::RegularGrid;
    fluid.source.regularGrid.size = size;
    fluid.source.regularGrid.targetParticleSpacing = 0.22f;
    fluid.particleRadius = 0.11f;
    fluid.material = material;
    const float particleDiameter = 2.0f * fluid.particleRadius;
    fluid.particleMass = particleMassScale * particleDiameter * particleDiameter *
                         particleDiameter * 1000.0f;
    fluid.collisionLayer = 0x1u;
    fluid.collisionMask = 0xffffffffu;
    return world.setFluid(entity, fluid);
}

bool spawnSoftBody(cressim::neo::engine::World &world, const Diligent::float3 &position,
                   const cressim::neo::physics::ParticleContactMaterialDesc &contactMaterial)
{
    const auto entity = world.createEntity();

    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.3f);
    world.setTransform(entity, transform);

    SoftBodyComponent softBody{};
    softBody.source.kind = SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = {0.9f, 0.9f, 0.9f};
    softBody.source.regularGrid.targetParticleSpacing = 0.3f;
    softBody.particleMass = 0.12f;
    softBody.particleRadius = 0.15f;
    softBody.edgeCompliance = 0.0f;
    softBody.volumeCompliance = 0.0008f;
    softBody.material.contact = contactMaterial;
    softBody.selfCollisionEnabled = true;
    softBody.collisionLayer = 0x1u;
    softBody.collisionMask = 0xffffffffu;
    return world.setSoftBody(entity, softBody);
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
    config.physicsDesc.softInternalIterations = 4u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Particle Material Showcase";
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
    cameraTransform.worldTransform.position = {0.0f, 3.4f, -9.0f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.22f);
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
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "ParticleMaterialShowcase.BoxMesh"));
    const MaterialHandle floorMaterial =
        registerMaterial(resources, "ParticleMaterialShowcase.Floor", {0.80f, 0.82f, 0.85f}, 0.95f);
    const MaterialHandle wallMaterial =
        registerMaterial(resources, "ParticleMaterialShowcase.Wall", {0.14f, 0.34f, 0.68f}, 0.6f);
    const MaterialHandle dividerMaterial =
        registerMaterial(resources, "ParticleMaterialShowcase.Divider", {0.72f, 0.48f, 0.18f}, 0.4f);

    spawnStaticCollisionBox(world, {0.0f, -1.0f, 0.0f}, {3.2f, 0.15f, 3.0f});
    spawnStaticBox(world, boxMesh, wallMaterial, {-3.05f, -0.15f, 0.0f}, {0.15f, 1.0f, 3.0f});
    spawnStaticBox(world, boxMesh, wallMaterial, {3.05f, -0.15f, 0.0f}, {0.15f, 1.0f, 3.0f});
    spawnStaticCollisionBox(world, {0.0f, -0.15f, -2.85f}, {3.2f, 1.0f, 0.15f});
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, -0.15f, 2.85f}, {3.2f, 1.0f, 0.15f});
    spawnStaticBox(world, boxMesh, dividerMaterial, {0.0f, -0.15f, 0.0f}, {0.25f, 1.0f, 3.0f});

    cressim::neo::physics::ParticleContactMaterialDesc fluidContact{};
    fluidContact.friction = 0.04f;
    fluidContact.staticFriction = 0.06f;
    fluidContact.restitution = 0.0f;
    fluidContact.damping = 0.02f;

    cressim::neo::physics::ParticleContactMaterialDesc softContact{};
    softContact.friction = 0.55f;
    softContact.staticFriction = 0.75f;
    softContact.restitution = 0.05f;
    softContact.damping = 0.08f;

    cressim::neo::physics::FluidMaterialDesc baselineFluid{};
    baselineFluid.contact = fluidContact;
    baselineFluid.viscosity = 0.0f;
    baselineFluid.gravityScale = 0.2f;
    baselineFluid.cflCoefficient = 1.0f;

    // Keep the comparison modest so the example demonstrates material variation
    // rather than turning into a stress test for the less-mature fluid terms.
    cressim::neo::physics::FluidMaterialDesc thickerFluid = baselineFluid;
    thickerFluid.cohesion = 10.0f;
    thickerFluid.surfaceTension = 20.0f;
    // thickerFluid.vorticityConfinement = 10.0f;
    // thickerFluid.cflCoefficient = 10.0f;
    thickerFluid.viscosity = 5.0f;
    thickerFluid.gravityScale = 0.2f;

    if (!spawnFluid(world, {-1.575f, 1.0f, 0.0f}, {2.5f, 2.2f, 5.2f}, baselineFluid, 1.0f))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author baseline fluid body.\n");
        return 1;
    }

    if (!spawnFluid(world, {1.575f, 1.0f, 0.0f}, {2.5f, 2.2f, 5.2f}, thickerFluid, 1.0f))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author thicker fluid body.\n");
        return 1;
    }

    if (!spawnSoftBody(world, {0.0f, 5.45f, 0.0f}, softContact))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author soft-body particle block.\n");
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
