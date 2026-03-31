#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "viewer/debug_viewer_app.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::PointLightComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::SpotLightComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;
constexpr std::uint32_t kDefaultEnvCount = 4u;
constexpr std::uint32_t kGridWidth = 3u;
constexpr std::uint32_t kGridDepth = 3u;
constexpr float kObjectSpacing = 2.2f;
constexpr float kEnvSpacing = 28.0f;
constexpr std::uint32_t kObjectsPerEnvBudget = 16u;
constexpr std::uint32_t kLightsPerEnvBudget = 6u;

GpuBackend parseBackend(const std::string &value)
{
    if (value == "null")
    {
        return GpuBackend::Null;
    }
    if (value == "vulkan")
    {
        return GpuBackend::Vulkan;
    }
    throw std::invalid_argument("Unsupported backend: " + value);
}

void printUsage(const char *appName)
{
    CRESSIM_LOG_ERROR("Usage: ", appName,
                      " [--backend vulkan|null] [--frames N] [--envs N]\n");
}

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "LightMatrix.Cube";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3 &normal, const Diligent::float3 &v0,
                             const Diligent::float3 &v1, const Diligent::float3 &v2,
                             const Diligent::float3 &v3)
    {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f});
        mesh.vertices.push_back({v1, normal, 1.0f, 0.0f});
        mesh.vertices.push_back({v2, normal, 1.0f, 1.0f});
        mesh.vertices.push_back({v3, normal, 0.0f, 1.0f});

        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 1u);
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 3u);
        mesh.indices.push_back(base + 2u);
    };

    const float h = halfExtent;
    addFace({0.0f, 0.0f, 1.0f}, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h});
    addFace({0.0f, 0.0f, -1.0f}, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h});
    addFace({-1.0f, 0.0f, 0.0f}, {-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h});
    addFace({1.0f, 0.0f, 0.0f}, {h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h});
    addFace({0.0f, 1.0f, 0.0f}, {-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h});
    addFace({0.0f, -1.0f, 0.0f}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h});
    return mesh;
}

MeshResourceDesc makePlaneMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "LightMatrix.Plane";
    const float h = halfExtent;
    mesh.vertices = {
        {{-h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f},
        {{h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 1.0f, 0.0f},
        {{h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 1.0f, 1.0f},
        {{-h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f}};
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    return mesh;
}

MeshResourceDesc makeSphereMesh(float radius, std::uint32_t slices, std::uint32_t stacks)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "LightMatrix.Sphere";
    mesh.vertices.reserve((stacks + 1u) * (slices + 1u));
    mesh.indices.reserve(stacks * slices * 6u);

    for (std::uint32_t stack = 0u; stack <= stacks; ++stack)
    {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * kPi;
        const float y = std::cos(phi);
        const float ringRadius = std::sin(phi);

        for (std::uint32_t slice = 0u; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * (2.0f * kPi);
            const float x = ringRadius * std::cos(theta);
            const float z = ringRadius * std::sin(theta);
            const Diligent::float3 normal{x, y, z};
            mesh.vertices.push_back({normal * radius, normal, u, v});
        }
    }

    const std::uint32_t ring = slices + 1u;
    for (std::uint32_t stack = 0u; stack < stacks; ++stack)
    {
        for (std::uint32_t slice = 0u; slice < slices; ++slice)
        {
            const std::uint32_t i0 = stack * ring + slice;
            const std::uint32_t i1 = i0 + 1u;
            const std::uint32_t i2 = i0 + ring;
            const std::uint32_t i3 = i2 + 1u;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}

Diligent::float3 envOrigin(std::uint32_t envIndex, std::uint32_t envCount)
{
    const std::uint32_t cols = std::max(
        1u, static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<float>(envCount)))));
    const std::uint32_t rows = std::max(1u, (envCount + cols - 1u) / cols);
    const std::uint32_t col = envIndex % cols;
    const std::uint32_t row = envIndex / cols;
    const float xCenter = (static_cast<float>(cols) - 1.0f) * 0.5f;
    const float zCenter = (static_cast<float>(rows) - 1.0f) * 0.5f;
    return {(static_cast<float>(col) - xCenter) * kEnvSpacing, 0.0f,
            (static_cast<float>(row) - zCenter) * kEnvSpacing};
}

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float roughness, float metallic = 0.0f)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.roughness = roughness;
    desc.metallic = metallic;
    return resources.registerMaterial(desc);
}

void authorSharedObjectLayout(cressim::neo::engine::World &world, std::uint32_t envIndex,
                              const Diligent::float3 &origin, MeshHandle cubeMesh,
                              MeshHandle sphereMesh, MeshHandle planeMesh,
                              MaterialHandle groundMaterial, MaterialHandle boxMaterial,
                              MaterialHandle sphereMaterial)
{
    const auto groundEntity = world.createEntity(envIndex);
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = origin + Diligent::float3{0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent ground{};
    ground.mesh = planeMesh;
    ground.material = groundMaterial;
    world.setMeshRenderer(groundEntity, ground);

    const float xOrigin = -0.5f * static_cast<float>(kGridWidth - 1u) * kObjectSpacing;
    const float zOrigin = -0.5f * static_cast<float>(kGridDepth - 1u) * kObjectSpacing;
    for (std::uint32_t z = 0u; z < kGridDepth; ++z)
    {
        for (std::uint32_t x = 0u; x < kGridWidth; ++x)
        {
            const auto entity = world.createEntity(envIndex);
            TransformComponent transform{};
            transform.worldTransform.position =
                origin + Diligent::float3{xOrigin + static_cast<float>(x) * kObjectSpacing,
                                          ((x + z) % 2u == 0u) ? 0.3f : 0.75f,
                                          zOrigin + static_cast<float>(z) * kObjectSpacing};
            transform.worldTransform.scale =
                ((x + z) % 2u == 0u) ? Diligent::float3{0.9f, 0.9f, 0.9f}
                                     : Diligent::float3{0.75f, 0.75f, 0.75f};
            world.setTransform(entity, transform);

            MeshRendererComponent renderer{};
            renderer.mesh = ((x + z) % 2u == 0u) ? cubeMesh : sphereMesh;
            renderer.material = ((x + z) % 2u == 0u) ? boxMaterial : sphereMaterial;
            world.setMeshRenderer(entity, renderer);
        }
    }
}

void authorLightSetup(cressim::neo::engine::World &world, std::uint32_t envIndex,
                      const Diligent::float3 &origin)
{
    const auto mainLightEntity = world.createEntity(envIndex);
    DirectionalLightComponent mainLight{};
    mainLight.direction =
        Diligent::normalize(Diligent::float3{-0.55f + 0.12f * static_cast<float>(envIndex),
                                             -1.0f,
                                             0.35f - 0.08f * static_cast<float>(envIndex)});
    mainLight.color = {1.0f, 0.95f, 0.88f};
    mainLight.intensity = 7.0f;
    mainLight.castsShadows = (envIndex % 4u != 2u);
    mainLight.shadowBias = 0.0015;
    world.setDirectionalLight(mainLightEntity, mainLight);

    if (envIndex % 4u == 0u || envIndex % 4u == 2u)
    {
        const auto fillDirEntity = world.createEntity(envIndex);
        DirectionalLightComponent fillDir{};
        fillDir.direction = Diligent::normalize(Diligent::float3{0.65f, -0.4f, -0.2f});
        fillDir.color = {0.55f, 0.72f, 1.0f};
        fillDir.intensity = 1.8f;
        fillDir.castsShadows = true;
        world.setDirectionalLight(fillDirEntity, fillDir);
    }

    {
        const auto pointEntity = world.createEntity(envIndex);
        TransformComponent pointTransform{};
        pointTransform.worldTransform.position =
            origin + Diligent::float3{-4.5f + 1.5f * static_cast<float>(envIndex % 2u),
                                      4.0f,
                                      -2.5f + static_cast<float>(envIndex)};
        world.setTransform(pointEntity, pointTransform);
        PointLightComponent point{};
        point.color = (envIndex % 2u == 0u) ? Diligent::float3{1.0f, 0.45f, 0.32f}
                                            : Diligent::float3{0.35f, 0.65f, 1.0f};
        point.intensity = (envIndex % 4u == 1u) ? 18.0f : 28.0f;
        point.range = 11.0f;
        point.shadowBias = 0.0015;
        point.castsShadows = true;// (envIndex % 4u == 0u || envIndex % 4u == 2u);
        world.setPointLight(pointEntity, point);
    }

    {
        const auto spotEntity = world.createEntity(envIndex);
        TransformComponent spotTransform{};
        spotTransform.worldTransform.position =
            origin + Diligent::float3{4.0f, 6.0f, 3.5f - 1.5f * static_cast<float>(envIndex % 3u)};
        world.setTransform(spotEntity, spotTransform);
        SpotLightComponent spot{};
        spot.direction = Diligent::normalize(Diligent::float3{-0.45f, -1.0f, -0.25f});
        spot.color = (envIndex % 4u == 3u) ? Diligent::float3{0.45f, 1.0f, 0.6f}
                                           : Diligent::float3{1.0f, 0.92f, 0.55f};
        spot.intensity = 22.0f;
        spot.range = 16.0f;
        spot.innerConeAngle = 18.0f;
        spot.outerConeAngle = 28.0f;
        spot.castsShadows = (envIndex % 4u == 0u || envIndex % 4u == 3u);
        world.setSpotLight(spotEntity, spot);
    }

    if (envIndex % 4u == 1u)
    {
        const auto extraPointEntity = world.createEntity(envIndex);
        TransformComponent extraPointTransform{};
        extraPointTransform.worldTransform.position = origin + Diligent::float3{0.0f, 3.5f, 4.5f};
        world.setTransform(extraPointEntity, extraPointTransform);
        PointLightComponent extraPoint{};
        extraPoint.color = {0.85f, 0.35f, 1.0f};
        extraPoint.intensity = 16.0f;
        extraPoint.range = 9.0f;
        extraPoint.castsShadows = false;
        world.setPointLight(extraPointEntity, extraPoint);
    }
}

void authorEnvironment(cressim::neo::engine::World &world, std::uint32_t envIndex,
                       std::uint32_t envCount, MeshHandle cubeMesh, MeshHandle sphereMesh,
                       MeshHandle planeMesh, MaterialHandle groundMaterial,
                       MaterialHandle boxMaterial, MaterialHandle sphereMaterial,
                       EntityId &outCameraEntity)
{
    const Diligent::float3 origin = envOrigin(envIndex, envCount);

    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = origin + Diligent::float3{0.0f, 6.5f, -13.0f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f},
                                                     12.0f * (kPi / 180.0f));
    world.setTransform(outCameraEntity, cameraTransform);

    CameraComponent camera{};
    camera.verticalFovDegrees = 52.0f;
    camera.renderOrder = envIndex;
    world.setCamera(outCameraEntity, camera);

    authorSharedObjectLayout(world, envIndex, origin, cubeMesh, sphereMesh, planeMesh,
                             groundMaterial, boxMaterial, sphereMaterial);
    authorLightSetup(world, envIndex, origin);
}

} // namespace

int main(int argc, char **argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    std::uint64_t numFrames = 0u;
    std::uint32_t envCount = kDefaultEnvCount;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--backend")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            config.gpuDeviceDesc.preferredBackend = parseBackend(argv[++i]);
            continue;
        }
        if (arg == "--frames")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            numFrames = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
            continue;
        }
        if (arg == "--envs")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            envCount = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            if (envCount == 0u)
            {
                CRESSIM_LOG_ERROR("--envs must be greater than zero.\n");
                return 2;
            }
            continue;
        }

        printUsage(argv[0]);
        return 2;
    }

    config.sceneLayout.envCount = envCount;
    config.sceneLayout.maxRenderableObjectsPerEnv = kObjectsPerEnvBudget;
    config.sceneLayout.maxLightsPerEnv = kLightsPerEnvBudget;
    config.sceneLayout.maxCamerasPerEnv = 1u;

    DebugViewerApp viewer;
    DebugViewerAppDesc viewerDesc{};
    const bool windowEnabled = (config.gpuDeviceDesc.preferredBackend != GpuBackend::Null);
    viewerDesc.windowEnabled = windowEnabled;
    viewerDesc.windowVisible = windowEnabled;
    viewerDesc.startFullscreenWindowed = false;
    viewerDesc.maxFrames = numFrames;
    viewerDesc.showStats = true;
    viewerDesc.vSync = false;
    viewerDesc.width = 960;
    viewerDesc.height = 540;
    viewerDesc.windowTitle = "CRESSim Neo Multi Env Light Matrix";

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

    const MeshHandle cubeMesh = resources.registerMesh(makeCubeMesh(0.65f));
    const MeshHandle sphereMesh = resources.registerMesh(makeSphereMesh(0.6f, 20u, 12u));
    const MeshHandle planeMesh = resources.registerMesh(makePlaneMesh(9.0f));

    const MaterialHandle groundMaterial =
        registerMaterial(resources, "LightMatrix.Ground", {0.68f, 0.70f, 0.74f}, 0.92f);
    const MaterialHandle boxMaterial =
        registerMaterial(resources, "LightMatrix.Box", {0.86f, 0.36f, 0.22f}, 0.48f);
    const MaterialHandle sphereMaterial =
        registerMaterial(resources, "LightMatrix.Sphere", {0.18f, 0.46f, 0.96f}, 0.28f, 0.08f);

    EntityId presentedCamera = cressim::neo::common::kInvalidEntityId;
    for (std::uint32_t envIndex = 0u; envIndex < envCount; ++envIndex)
    {
        EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        authorEnvironment(world, envIndex, envCount, cubeMesh, sphereMesh, planeMesh,
                          groundMaterial, boxMaterial, sphereMaterial, cameraEntity);
        if (envIndex == 0u)
        {
            presentedCamera = cameraEntity;
        }
    }

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = presentedCamera;

    const bool runOk = viewer.run(runtime, binding, {});

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Multi-env light matrix viewer run failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Multi-env light matrix viewer finished. Envs=", envCount,
                     " Frames=", viewerDesc.maxFrames, '\n');
    return 0;
}
