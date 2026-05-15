#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::FluidComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::PointLightComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::SpotLightComponent;
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

constexpr float kFluidParticleSpacing = 0.18f;
constexpr float kFluidParticleRadius  = 0.09f;

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

void addPointLight(cressim::neo::engine::World &world, const Diligent::float3 &position,
                   const Diligent::float3 &color, float intensity, float range)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    PointLightComponent light{};
    light.color = color;
    light.intensity = intensity;
    light.range = range;
    world.setPointLight(entity, light);
}

void addSpotLight(cressim::neo::engine::World &world, const Diligent::float3 &position,
                  const Diligent::float3 &direction, const Diligent::float3 &color,
                  float intensity, float range, float innerConeAngle, float outerConeAngle)
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    SpotLightComponent light{};
    light.direction = Diligent::normalize(direction);
    light.color = color;
    light.intensity = intensity;
    light.range = range;
    light.innerConeAngle = innerConeAngle;
    light.outerConeAngle = outerConeAngle;
    world.setSpotLight(entity, light);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    bool debugParticles = false;

    for (int i = 1; i < argc; ++i)
    {
        if (cressim::neo::examples::helpers::tryParseCommonArgument(argc, argv, i, options, false))
        {
            continue;
        }

        const std::string arg = argv[i];
        if (arg == "--debug-particles")
        {
            debugParticles = true;
            continue;
        }

        cressim::neo::examples::helpers::printUsage(argv[0], " [--debug-particles]", false);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.physicsDesc.substeps = 1u;
    config.physicsDesc.defaultIterations = 10u;
    config.physicsDesc.fluidIterations = 10u;
    config.physicsDesc.softContactIterations = 0u;
    config.physicsDesc.softInternalIterations = 0u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Fluid Lighting Showcase";
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

    auto &world = runtime.getWorld();
    auto &resources = runtime.getResources();

    const MeshHandle boxMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "FluidLightingShowcase.BoxMesh"));

    const MaterialHandle floorMaterial =
        registerMaterial(resources, "FluidLightingShowcase.Floor", {0.82f, 0.84f, 0.88f}, 0.95f);
    const MaterialHandle wallMaterial =
        registerMaterial(resources, "FluidLightingShowcase.Wall", {0.26f, 0.31f, 0.38f}, 0.70f);
    const MaterialHandle rimMaterial =
        registerMaterial(resources, "FluidLightingShowcase.Rim", {0.58f, 0.62f, 0.68f}, 0.45f);

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 5.8f, -14.0f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.32f);
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 36.0f;
    camera.clearColorValue = {0.03f, 0.035f, 0.05f, 1.0f};
    world.setCamera(cameraEntity, camera);

    const auto dirLightEntity = world.createEntity();
    DirectionalLightComponent directionalLight{};
    directionalLight.direction = Diligent::normalize(Diligent::float3{-0.45f, -1.0f, 0.25f});
    directionalLight.color = {1.0f, 0.95f, 0.90f};
    directionalLight.intensity = 5.5f;
    directionalLight.castsShadows = true;
    world.setDirectionalLight(dirLightEntity, directionalLight);

    addPointLight(world, {-4.0f, 4.5f, -1.5f}, {1.0f, 0.28f, 0.24f}, 24.0f, 9.0f);
    addPointLight(world, {4.2f, 4.0f, 1.2f}, {0.20f, 0.55f, 1.0f}, 22.0f, 9.5f);
    addSpotLight(world, {0.0f, 7.5f, -5.0f}, {0.0f, -0.9f, 0.4f},
                 {0.95f, 0.85f, 0.32f}, 36.0f, 18.0f, 16.0f, 26.0f);
    addSpotLight(world, {-5.5f, 5.0f, 4.0f}, {0.75f, -0.45f, -0.55f},
                 {0.45f, 1.0f, 0.72f}, 28.0f, 14.0f, 20.0f, 32.0f);

    spawnStaticCollisionBox(world, {0.0f, -1.0f, 0.0f}, {5.2f, 0.15f, 4.6f});
    spawnStaticBox(world, boxMesh, floorMaterial, {0.0f, -1.16f, 0.0f}, {5.2f, 0.04f, 4.6f});
    spawnStaticBox(world, boxMesh, wallMaterial, {-5.05f, 0.9f, 0.0f}, {0.15f, 2.05f, 4.6f});
    spawnStaticBox(world, boxMesh, wallMaterial, {5.05f, 0.9f, 0.0f}, {0.15f, 2.05f, 4.6f});
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, 0.9f, -4.45f}, {5.2f, 2.05f, 0.15f});
    spawnStaticBox(world, boxMesh, wallMaterial, {0.0f, 0.9f, 4.45f}, {5.2f, 2.05f, 0.15f});
    spawnStaticBox(world, boxMesh, rimMaterial, {0.0f, 3.05f, -4.45f}, {5.2f, 0.08f, 0.18f});
    spawnStaticBox(world, boxMesh, rimMaterial, {0.0f, 3.05f, 4.45f}, {5.2f, 0.08f, 0.18f});
    spawnStaticBox(world, boxMesh, rimMaterial, {-5.05f, 3.05f, 0.0f}, {0.18f, 0.08f, 4.6f});
    spawnStaticBox(world, boxMesh, rimMaterial, {5.05f, 3.05f, 0.0f}, {0.18f, 0.08f, 4.6f});

    cressim::neo::physics::ParticleContactMaterialDesc fluidContact{};
    fluidContact.friction = 0.04f;
    fluidContact.staticFriction = 0.06f;
    fluidContact.restitution = 0.0f;
    fluidContact.damping = 0.2f;

    cressim::neo::physics::FluidMaterialDesc fluidMaterial{};
    fluidMaterial.contact = fluidContact;
    fluidMaterial.viscosity = 1.5f;
    fluidMaterial.cohesion = 0.8f;
    fluidMaterial.gravityScale = 0.65f;
    fluidMaterial.cflCoefficient = 1.0f;
    fluidMaterial.vorticityConfinement = 0.25f;
    fluidMaterial.surfaceTension = 1.5f;

    if (!spawnFluid(world, {0.0f, 0.95f, 0.0f}, {8.2f, 2.4f, 7.2f}, fluidMaterial))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author fluid body.\n");
        return 1;
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    const auto &particles = world.physicsWorld().particles();
    CRESSIM_LOG_INFO("Fluid lighting showcase authored ", particles.size(),
                     " particles. Fluid bodies=", world.physicsWorld().fluidCount(), ".\n");

    auto renderOptions = runtime.renderFrameOptions();
    renderOptions.fluid.tint.a = 0.8f;
    renderOptions.fluid.enableBackgroundRefraction = true;
    runtime.setRenderFrameOptions(renderOptions);

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
