#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::MeshfreeSoftBodyComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
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
    desc.metallic  = 0.0f;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

std::vector<Diligent::float3> makeParticleBlock(std::uint32_t sideCount, float spacing)
{
    std::vector<Diligent::float3> particles;
    particles.reserve(sideCount * sideCount * sideCount);

    const float centerOffset = 0.5f * spacing * static_cast<float>(sideCount - 1u);
    for (std::uint32_t z = 0u; z < sideCount; ++z)
    {
        for (std::uint32_t y = 0u; y < sideCount; ++y)
        {
            for (std::uint32_t x = 0u; x < sideCount; ++x)
            {
                particles.push_back({static_cast<float>(x) * spacing - centerOffset,
                                     static_cast<float>(y) * spacing - centerOffset,
                                     static_cast<float>(z) * spacing - centerOffset});
            }
        }
    }

    return particles;
}

MeshResourceDesc makeMeshfreeAnchorMesh(const std::vector<Diligent::float3> &particles)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "MeshfreeDebug.RestParticleAnchorMesh";
    mesh.vertices.reserve(particles.size());
    for (const Diligent::float3 &particle : particles)
    {
        MeshResourceDesc::Vertex vertex{};
        vertex.position = particle;
        mesh.vertices.push_back(vertex);
    }

    return mesh;
}

void applyDebugParticleGraphOptions(Runtime &runtime)
{
    cressim::neo::graphics::RenderFrameOptions renderOptions = runtime.renderFrameOptions();
    renderOptions.debugParticles.enabled                  = true;
    renderOptions.debugParticles.drawConstraintEdges      = true;
    renderOptions.debugParticles.highlightStaticParticles = true;
    renderOptions.debugParticles.useParticleRadii         = true;
    renderOptions.debugParticles.color                    = {0.18f, 0.74f, 1.0f, 1.0f};
    renderOptions.debugParticles.staticColor              = {1.0f, 0.22f, 0.12f, 1.0f};
    renderOptions.debugParticles.edgeColor                = {1.0f, 0.86f, 0.18f, 1.0f};
    renderOptions.debugParticles.fallbackRadius           = 0.045f;
    runtime.setRenderFrameOptions(renderOptions);
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
    config.physicsDesc.substeps                    = 4u;
    config.physicsDesc.defaultIterations           = 16u;
    config.physicsDesc.softInternalIterations      = 32u;
    config.physicsDesc.softContactIterations       = 12u;
    config.physicsDesc.rigidRigidContactIterations = 0u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Meshfree Debug Graph";
    viewerDefaults.width       = 960u;
    viewerDefaults.height      = 640u;
    viewerDefaults.showStats   = true;
    viewerDefaults.vSync       = true;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
    viewerDesc.enableDebugParticles = true;

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Meshfree debug graph viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Meshfree debug graph runtime initialization failed.\n");
        return 1;
    }

    auto &world = runtime.getWorld();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.85f, -4.2f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.10f);
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 42.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color     = {1.0f, 0.98f, 0.94f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const MeshHandle groundMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(4.0f, "MeshfreeDebug.GroundMesh"));
    const MaterialHandle groundMaterial =
        registerMaterial(resources, "MeshfreeDebug.Ground", {0.18f, 0.20f, 0.22f}, 0.86f);
    const MaterialHandle anchorMaterial =
        registerMaterial(resources, "MeshfreeDebug.Anchor", {0.20f, 0.65f, 0.92f}, 0.62f);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -0.72f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity,
                          MeshRendererComponent{groundMesh, groundMaterial, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType            = RigidBodyType::Static;
    groundBody.inverseMass         = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType      = ColliderShapeType::Box;
    groundCollider.shapeParams    = {4.0f, 0.05f, 4.0f, 0.0f};
    groundCollider.friction       = 0.55f;
    groundCollider.staticFriction = 0.75f;
    groundCollider.collisionLayer = 0x1u;
    groundCollider.collisionMask  = 0x2u;
    world.addCollider(groundEntity, groundCollider);

    constexpr std::uint32_t kSideCount = 3u;
    constexpr float kParticleSpacing   = 0.34f;
    std::vector<Diligent::float3> particles = makeParticleBlock(kSideCount, kParticleSpacing);
    const MeshHandle anchorMesh = resources.registerMesh(makeMeshfreeAnchorMesh(particles));

    const auto softEntity = world.createEntity();
    TransformComponent softTransform{};
    softTransform.worldTransform.position = {0.0f, 1.05f, 0.0f};
    world.setTransform(softEntity, softTransform);
    world.setMeshRenderer(softEntity,
                          MeshRendererComponent{anchorMesh, anchorMaterial, true});

    MeshfreeSoftBodyComponent softBody{};
    softBody.particles                       = std::move(particles);
    softBody.neighbourCount                  = 12u;
    softBody.particleRadius                  = 0.06f;
    softBody.particleMass                    = 0.04f;
    softBody.compliance                      = 2.0e-5f;
    softBody.material.contact.friction       = 0.45f;
    softBody.material.contact.staticFriction = 0.60f;
    softBody.material.contact.damping        = 0.60f;
    softBody.selfCollisionEnabled            = false;
    softBody.collisionLayer                  = 0x2u;
    softBody.collisionMask                   = 0x1u;
    if (!world.setMeshfreeSoftBody(softEntity, softBody))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author meshfree debug soft body.\n");
        return 1;
    }

    applyDebugParticleGraphOptions(runtime);

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [](const cressim::neo::common::FrameContext &, Runtime &callbackRuntime)
    {
        applyDebugParticleGraphOptions(callbackRuntime);
    };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk     = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Meshfree debug graph viewer run failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Meshfree debug graph viewer finished.\n");
    return 0;
}
