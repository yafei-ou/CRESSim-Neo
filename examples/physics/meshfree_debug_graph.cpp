#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "physics/load_particle_cloud.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
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
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

struct MeshfreeDebugOptions
{
    CommonExampleOptions common{};
    bool useCube = false;
    bool drawConstraintEdges = false;
    bool vSync = false;
    std::filesystem::path cloudPath =
        std::filesystem::path(__FILE__).parent_path() / "fixtures" / "gallbladder_particles.bin";
    std::uint32_t neighbourCount = 12u;
    float cloudScale = 0.035f;
    float particleRadius = 0.035f;
};

struct ParticleBounds
{
    Diligent::float3 min{};
    Diligent::float3 max{};
    Diligent::float3 center{};
    Diligent::float3 extent{};
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--cube] [--cloud PATH] [--cloud-scale S] [--neighbours N] [--particle-radius R]"
        " [--draw-edges] [--vsync]",
        false);
}

std::uint32_t parsePositiveUint32(const char *value, const char *optionName)
{
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0ul ||
        parsed > static_cast<unsigned long>(UINT32_MAX))
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

float parsePositiveFloat(const char *value, const char *optionName)
{
    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || !(parsed > 0.0f))
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

ParticleBounds computeParticleBounds(const std::vector<Diligent::float3> &particles)
{
    ParticleBounds bounds{};
    if (particles.empty())
    {
        return bounds;
    }

    bounds.min = particles.front();
    bounds.max = particles.front();
    for (const Diligent::float3 &particle : particles)
    {
        bounds.min.x = std::min(bounds.min.x, particle.x);
        bounds.min.y = std::min(bounds.min.y, particle.y);
        bounds.min.z = std::min(bounds.min.z, particle.z);
        bounds.max.x = std::max(bounds.max.x, particle.x);
        bounds.max.y = std::max(bounds.max.y, particle.y);
        bounds.max.z = std::max(bounds.max.z, particle.z);
    }

    bounds.center = {(bounds.min.x + bounds.max.x) * 0.5f,
                     (bounds.min.y + bounds.max.y) * 0.5f,
                     (bounds.min.z + bounds.max.z) * 0.5f};
    bounds.extent = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                     bounds.max.z - bounds.min.z};
    return bounds;
}

void centerAndScaleParticles(std::vector<Diligent::float3> &particles, const float scale)
{
    const ParticleBounds bounds = computeParticleBounds(particles);
    for (Diligent::float3 &particle : particles)
    {
        particle = (particle - bounds.center) * scale;
    }
}

std::vector<Diligent::float3> loadConfiguredParticles(const MeshfreeDebugOptions &options)
{
    if (options.useCube)
    {
        constexpr std::uint32_t kSideCount = 3u;
        constexpr float kParticleSpacing   = 0.35f;
        return makeParticleBlock(kSideCount, kParticleSpacing);
    }

    std::vector<Diligent::float3> particles;
    std::string errorMessage;
    if (!cressim::neo::physics::readParticleCloudBin(options.cloudPath, particles, errorMessage))
    {
        CRESSIM_LOG_ERROR(errorMessage, "\n");
        return {};
    }

    centerAndScaleParticles(particles, options.cloudScale);
    return particles;
}

void applyDebugParticleGraphOptions(Runtime &runtime, const bool drawConstraintEdges,
                                    const float particleRadius)
{
    cressim::neo::graphics::RenderFrameOptions renderOptions = runtime.renderFrameOptions();
    renderOptions.debugParticles.enabled                  = true;
    renderOptions.debugParticles.drawConstraintEdges      = drawConstraintEdges;
    renderOptions.debugParticles.highlightStaticParticles = true;
    renderOptions.debugParticles.useParticleRadii         = true;
    renderOptions.debugParticles.color                    = {0.18f, 0.74f, 1.0f, 1.0f};
    renderOptions.debugParticles.staticColor              = {1.0f, 0.22f, 0.12f, 1.0f};
    renderOptions.debugParticles.edgeColor                = {1.0f, 0.86f, 0.18f, 1.0f};
    renderOptions.debugParticles.fallbackRadius           = particleRadius;
    runtime.setRenderFrameOptions(renderOptions);
}

} // namespace

int main(int argc, char **argv)
{
    MeshfreeDebugOptions options{};
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
            if (arg == "--cube")
            {
                options.useCube = true;
                continue;
            }
            if (arg == "--cloud")
            {
                options.cloudPath = cressim::neo::examples::helpers::requireOptionValue(
                    argc, argv, i, "--cloud");
                options.useCube = false;
                continue;
            }
            if (arg == "--cloud-scale")
            {
                options.cloudScale = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--cloud-scale"),
                    "--cloud-scale");
                continue;
            }
            if (arg == "--neighbours")
            {
                options.neighbourCount = parsePositiveUint32(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--neighbours"),
                    "--neighbours");
                continue;
            }
            if (arg == "--particle-radius")
            {
                options.particleRadius = parsePositiveFloat(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--particle-radius"),
                    "--particle-radius");
                continue;
            }
            if (arg == "--draw-edges")
            {
                options.drawConstraintEdges = true;
                continue;
            }
            if (arg == "--vsync")
            {
                options.vSync = true;
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

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options.common);
    config.physicsDesc.substeps                    = options.useCube ? 4u : 6u;
    config.physicsDesc.defaultIterations           = 16u;
    config.physicsDesc.softInternalIterations      = options.useCube ? 32u : 16u;
    config.physicsDesc.softContactIterations       = options.useCube ? 12u : 16u;
    config.physicsDesc.rigidRigidContactIterations = 0u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = options.useCube ? "CRESSim Neo Meshfree Cube Debug Graph"
                                                 : "CRESSim Neo Gallbladder Particle Cloud";
    viewerDefaults.width       = 960u;
    viewerDefaults.height      = 640u;
    viewerDefaults.showStats   = true;
    viewerDefaults.vSync       = options.vSync;
    auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options.common, viewerDefaults);
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

    constexpr float kGroundSurfaceY       = -0.72f;
    constexpr float kGroundColliderHalfY  = 0.35f;
    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, kGroundSurfaceY, 0.0f};
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
    groundCollider.shapeParams    = {4.0f, kGroundColliderHalfY, 4.0f, 0.0f};
    groundCollider.localPosition  = {0.0f, -kGroundColliderHalfY, 0.0f};
    groundCollider.friction       = 0.55f;
    groundCollider.staticFriction = 0.75f;
    groundCollider.collisionLayer = 0x1u;
    groundCollider.collisionMask  = 0x2u;
    world.addCollider(groundEntity, groundCollider);

    std::vector<Diligent::float3> particles = loadConfiguredParticles(options);
    if (particles.empty())
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("No meshfree particles were loaded.\n");
        return 1;
    }
    const ParticleBounds particleBounds = computeParticleBounds(particles);
    CRESSIM_LOG_INFO("Meshfree debug source: ", particles.size(), " particles, bounds min=(",
                     particleBounds.min.x, ", ", particleBounds.min.y, ", ",
                     particleBounds.min.z, "), max=(", particleBounds.max.x, ", ",
                     particleBounds.max.y, ", ", particleBounds.max.z, ").\n");

    const auto softEntity = world.createEntity();
    TransformComponent softTransform{};
    softTransform.worldTransform.position = {0.0f, 1.05f, 0.0f};
    world.setTransform(softEntity, softTransform);

    MeshfreeSoftBodyComponent softBody{};
    softBody.particles                       = std::move(particles);
    softBody.neighbourCount                  = options.neighbourCount;
    softBody.particleRadius                  = options.useCube ? 0.06f : options.particleRadius;
    softBody.particleMass                    = options.useCube ? 0.04f : 0.0002f;
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

    world.physicsWorld().ensureDerivedStateUpToDate();
    if (const cressim::neo::physics::SoftBodyState *softState =
            world.physicsWorld().tryGetSoftBody(softEntity))
    {
        CRESSIM_LOG_INFO("Meshfree XPBD graph: ", softState->edgeCount, " distance constraints",
                         options.drawConstraintEdges ? " (edge debug draw enabled).\n"
                                                     : " (edge debug draw disabled).\n");
    }

    applyDebugParticleGraphOptions(runtime, options.drawConstraintEdges,
                                   options.useCube ? 0.045f : options.particleRadius);

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&options](const cressim::neo::common::FrameContext &,
                                      Runtime &callbackRuntime)
    {
        applyDebugParticleGraphOptions(callbackRuntime, options.drawConstraintEdges,
                                       options.useCube ? 0.045f : options.particleRadius);
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
