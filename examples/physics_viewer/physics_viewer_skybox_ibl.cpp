#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/environment_ibl_baker.h"
#include "helpers/example_cli.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::MeshRendererComponent;
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
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
}

MeshResourceDesc makePlaneMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "ViewerIntegration.SkyboxIbl.Plane";
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
    mesh.debugName = "ViewerIntegration.SkyboxIbl.Sphere";
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

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float metallic, float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.metallic = metallic;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

EnvironmentIblDesc loadSkyboxIbl(cressim::neo::graphics::RenderResourceManager &resources)
{
    const std::filesystem::path skyboxDir =
        std::filesystem::path(CRESSIM_NEO_PROJECT_SOURCE_DIR) / "examples/physics_viewer/skybox";
    const std::array<std::filesystem::path, 6u> facePaths = {
        skyboxDir / "posx.jpg", skyboxDir / "negx.jpg", skyboxDir / "posy.jpg",
        skyboxDir / "negy.jpg", skyboxDir / "posz.jpg", skyboxDir / "negz.jpg"};

    EnvironmentIblBakeOptions options{};
    options.irradianceSize = 16u;
    options.specularSize = 128u;
    options.specularMipCount = 7u;
    options.irradianceSampleCount = 256u;
    options.specularSampleCount = 128u;
    options.intensity = 1.0f;
    return cressim::neo::graphics::createEnvironmentIblFromCubemapFiles(resources, facePaths,
                                                                        options);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.rendererDesc.iblQualityTier = IblQualityTier::Full;
    config.sceneLayout.envCount = 1u;
    config.sceneLayout.maxRenderableObjectsPerEnv = 4u;
    config.sceneLayout.maxLightsPerEnv = 1u;
    config.sceneLayout.maxCamerasPerEnv = 1u;

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

    config.gpuDeviceDesc.preferredBackend = options.backend;

    DebugViewerApp viewer;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(
        options, ViewerExampleDefaults{
                     .windowTitle = "CRESSim Neo Physics Viewer Skybox IBL",
                     .width = 1280u,
                     .height = 720u,
                     .showStats = true,
                     .vSync = true,
                     .startFullscreen = false,
                     .startFullscreenWindowed = false,
                 });

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

    try
    {
        auto &world = runtime.getWorld();
        auto &resources = runtime.getResources();

        const auto cameraEntity = world.createEntity();
        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, 1.2f, -4.0f};
        world.setTransform(cameraEntity, cameraTransform);
        CameraComponent camera{};
        camera.backgroundMode = CameraComponent::BackgroundMode::EnvironmentCubemap;
        world.setCamera(cameraEntity, camera);

        if (!world.setEnvironmentIbl(0u, loadSkyboxIbl(resources)))
        {
            throw std::runtime_error("Failed to assign skybox IBL to environment 0.");
        }

        const MeshHandle planeMesh = resources.registerMesh(makePlaneMesh(8.0f));
        const MeshHandle sphereMesh = resources.registerMesh(makeSphereMesh(1.0f, 48u, 24u));

        const MaterialHandle shinySphereMaterial = registerMaterial(
            resources, "ViewerIntegration.SkyboxIbl.ShinySphere", {0.98f, 0.98f, 0.98f}, 1.0f,
            0.15f);

        const auto sphereEntity = world.createEntity();
        TransformComponent sphereTransform{};
        sphereTransform.worldTransform.position = {0.0f, 0.0f, 0.8f};
        world.setTransform(sphereEntity, sphereTransform);
        MeshRendererComponent sphere{};
        sphere.mesh = sphereMesh;
        sphere.material = shinySphereMaterial;
        sphere.visible = true;
        world.setMeshRenderer(sphereEntity, sphere);

        std::uint64_t beforeCalls = 0u;
        std::uint64_t afterCalls = 0u;
        DebugViewerCallbacks callbacks{};
        callbacks.beforeTick = [&](const FrameContext &, Runtime &) { ++beforeCalls; };
        callbacks.afterTick = [&](const FrameContext &, Runtime &) { ++afterCalls; };

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
        if (viewerDesc.maxFrames > 0 &&
            (beforeCalls != viewerDesc.maxFrames || afterCalls != viewerDesc.maxFrames))
        {
            CRESSIM_LOG_ERROR("Unexpected callback counts. before=", beforeCalls,
                              " after=", afterCalls, " expected=", viewerDesc.maxFrames, '\n');
            return 1;
        }
    }
    catch (const std::exception &ex)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Skybox IBL viewer setup failed: ", ex.what(), '\n');
        return 1;
    }

    CRESSIM_LOG_INFO("Physics viewer skybox IBL passed. Frames=", viewerDesc.maxFrames, '\n');
    return 0;
}
