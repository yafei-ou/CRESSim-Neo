#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include "graphics/render_resource_manager.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::StrandComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

std::uint32_t flattenGridIndex(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                               const Diligent::uint3 &resolution)
{
    return x * resolution.y * resolution.z + y * resolution.z + z;
}

std::vector<std::uint32_t> makeBottomLayerStaticIndices(const Diligent::float3 &size, float spacing)
{
    const float clampedSpacing = std::max(spacing, 1.0e-4f);
    const auto deriveResolution = [clampedSpacing](const float extent) -> std::uint32_t
    {
        return std::max<std::uint32_t>(2u, static_cast<std::uint32_t>(std::ceil(extent / clampedSpacing)) +
                                               1u);
    };

    const Diligent::uint3 resolution{
        deriveResolution(std::max(size.x, 1.0e-4f)),
        deriveResolution(std::max(size.y, 1.0e-4f)),
        deriveResolution(std::max(size.z, 1.0e-4f)),
    };

    std::vector<std::uint32_t> result;
    const std::uint32_t rightPinnedColumns = std::max(1u, (resolution.x + 2u) / 3u);
    const std::uint32_t startX = resolution.x - rightPinnedColumns;
    result.reserve(static_cast<std::size_t>(rightPinnedColumns) * resolution.z);
    for (std::uint32_t x = startX; x < resolution.x; ++x)
    {
        for (std::uint32_t z = 0u; z < resolution.z; ++z)
        {
            result.push_back(flattenGridIndex(x, 0u, z, resolution));
        }
    }
    return result;
}

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
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
    config.physicsDesc.enableBlockingReadback = true;
    config.physicsDesc.substeps               = 2u;
    config.physicsDesc.defaultIterations      = 8u;
    config.physicsDesc.softContactIterations  = 4u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Strand Suturing Path Follow";
    auto viewerDesc            = cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
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

    auto &world     = runtime.getWorld();
    auto &resources = runtime.getResources();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.4f, -7.0f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 48.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.25f, -1.0f, 0.25f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    const auto planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(8.0f, "StrandSuturing.PlaneMesh"));
    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "StrandSuturing.GroundMaterial";
    groundMaterialDesc.baseColor = {0.72f, 0.74f, 0.78f};
    groundMaterialDesc.roughness = 0.95f;
    const auto groundMaterial = resources.registerMaterial(groundMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.3f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(
        groundEntity, cressim::neo::engine::MeshRendererComponent{planeMesh, groundMaterial, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType    = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType   = ColliderShapeType::Box;
    groundCollider.shapeParams = {8.0f, 0.05f, 8.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    const auto softEntity = world.createEntity();
    SoftBodyComponent softBody{};
    softBody.source.kind = SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = {2.2f, 1.1f, 1.1f};
    softBody.source.regularGrid.targetParticleSpacing = 0.3f;
    softBody.source.regularGrid.staticParticleIndices = makeBottomLayerStaticIndices(
        softBody.source.regularGrid.size, softBody.source.regularGrid.targetParticleSpacing);
    softBody.particleMass          = 0.10f;
    softBody.particleRadius        = 0.10f;
    softBody.edgeCompliance        = 0.0f;
    softBody.volumeCompliance      = 0.001f;
    softBody.selfCollisionEnabled  = true;
    softBody.supportsSuturing      = true;
    softBody.material.contact.friction = 0.45f;
    if (!world.setSoftBody(softEntity, softBody))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author suturing soft body example.\n");
        return 1;
    }

    const auto strandEntity = world.createEntity();
    StrandComponent strand{};
    strand.particleMass         = 0.12f;
    strand.particleRadius       = 0.075f;
    strand.distanceCompliance   = 0.000001f;
    strand.bendCompliance       = 0.03f;
    strand.selfCollisionEnabled = false;
    strand.suturingEnabled      = true;
    strand.needleTipParticleIndex = 0u;
    strand.needleTipKinematic   = true;
    strand.pathNodeSpacing      = 0.16f;
    strand.staticParticleIndices = {0u};
    for (std::uint32_t i = 0u; i < 22u; ++i)
    {
        strand.restPositions.push_back(
            {-1.8f - 0.14f * static_cast<float>(i), -0.05f, 0.0f});
    }

    if (!world.setStrand(strandEntity, strand))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to author suturing strand example.\n");
        return 1;
    }

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [strandEntity](const cressim::neo::common::FrameContext &frame,
                                          Runtime &cbRuntime)
    {
        const float t = static_cast<float>(frame.timeSeconds);
        const float cycleDuration = 12.0f;
        const float horizontalPhase = 0.35f;
        const float liftPhase       = 0.30f;
        const float leftPullPhase   = 1.0f - horizontalPhase - liftPhase;
        const float startX          = -1.8f;
        const float exitX           = -0.1f;
        const float endX            = -2.1f;
        const float baseY           = -0.05f;
        const float liftedY         = 1.1f;
        const float cycle = std::fmod(std::max(t, 0.0f), cycleDuration) / cycleDuration;
        float x = startX;
        float y = baseY;
        if (cycle <= horizontalPhase)
        {
            const float u = cycle / horizontalPhase;
            x = startX + (exitX - startX) * u;
        }
        else if (cycle <= horizontalPhase + liftPhase)
        {
            const float u = (cycle - horizontalPhase) / liftPhase;
            const float easedU = u * u * (3.0f - 2.0f * u);
            x = exitX;
            y = baseY + (liftedY - baseY) * easedU;
        }
        else
        {
            const float u = (cycle - horizontalPhase - liftPhase) / leftPullPhase;
            const float easedU = u * u * (3.0f - 2.0f * u);
            x = exitX + (endX - exitX) * easedU;
            y = liftedY;
        }
        cbRuntime.getWorld().physicsWorld().overrideStrandParticlePosition(
            strandEntity, 0u, Diligent::float3{x, y, 0.0f}, true);
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

    CRESSIM_LOG_INFO("Strand suturing path-follow example passed.\n");
    return 0;
}
