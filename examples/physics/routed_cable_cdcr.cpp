#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/inertia.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::ProceduralDeformableCurveRenderComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::StrandComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::AuthoredParticleReference;
using cressim::neo::physics::AuthoredParticleReferenceType;
using cressim::neo::physics::AuthoredParticleSequenceState;
using cressim::neo::physics::AuthoredRigidParticleAttachmentConstraintState;
using cressim::neo::physics::AuthoredRoutedCableConstraintState;
using cressim::neo::physics::AuthoredRoutedCableRoutePoint;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi                  = 3.14159265358979323846f;
constexpr std::uint32_t kGroundLayer = 1u << 0u;
constexpr std::uint32_t kDiskLayer   = 1u << 1u;

struct ExampleOptions
{
    enum class ActuationMode
    {
        TravelingWave,
        SideBend,
    };

    std::uint32_t diskCount   = 9u;
    float pullAmplitude       = 0.72f;
    float drivePeriodSeconds  = 7.0f;
    float diskSpacing         = 0.68f;
    float guideRadius         = 0.34f;
    float backboneCompliance  = 7.5e-7f;
    bool selfCollideDisks     = false;
    ActuationMode actuationMode = ActuationMode::SideBend;
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--disks N] [--pull VALUE] [--period VALUE] [--spacing VALUE]"
        " [--guide-radius VALUE] [--backbone-compliance VALUE]"
        " [--mode traveling|bend] [--self-collide]",
        false);
}

std::uint32_t parseUIntOption(const std::string &value, const char *optionName)
{
    const char *begin = value.c_str();
    char *end         = nullptr;
    const auto parsed = std::strtoul(begin, &end, 10);
    if (end == begin || *end != '\0')
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

float parseFloatOption(const std::string &value, const char *optionName)
{
    const char *begin  = value.c_str();
    char *end          = nullptr;
    const float parsed = std::strtof(begin, &end);
    if (end == begin || *end != '\0')
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }
    return parsed;
}

ExampleOptions::ActuationMode parseActuationMode(const std::string &value)
{
    if (value == "traveling")
    {
        return ExampleOptions::ActuationMode::TravelingWave;
    }
    if (value == "bend")
    {
        return ExampleOptions::ActuationMode::SideBend;
    }

    throw std::invalid_argument(
        "--mode must be one of: traveling, bend.");
}

bool tryParseSceneArgument(int argc, char **argv, int &index, ExampleOptions &options)
{
    const std::string arg = argv[index];
    if (arg == "--disks")
    {
        options.diskCount = parseUIntOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--disks"),
            "disk count");
        if (options.diskCount < 2u)
        {
            throw std::invalid_argument("--disks must be at least 2.");
        }
        return true;
    }

    if (arg == "--pull")
    {
        options.pullAmplitude = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--pull"),
            "pull amount");
        return true;
    }

    if (arg == "--period")
    {
        options.drivePeriodSeconds = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--period"),
            "drive period");
        if (options.drivePeriodSeconds <= 0.0f)
        {
            throw std::invalid_argument("--period must be greater than zero.");
        }
        return true;
    }

    if (arg == "--spacing")
    {
        options.diskSpacing = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--spacing"),
            "disk spacing");
        if (options.diskSpacing <= 0.0f)
        {
            throw std::invalid_argument("--spacing must be greater than zero.");
        }
        return true;
    }

    if (arg == "--guide-radius")
    {
        options.guideRadius = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index,
                                                                 "--guide-radius"),
            "guide radius");
        if (options.guideRadius < 0.0f)
        {
            throw std::invalid_argument("--guide-radius must be non-negative.");
        }
        return true;
    }

    if (arg == "--backbone-compliance")
    {
        options.backboneCompliance =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--backbone-compliance"),
                             "backbone compliance");
        if (options.backboneCompliance < 0.0f)
        {
            throw std::invalid_argument("--backbone-compliance must be non-negative.");
        }
        return true;
    }

    if (arg == "--mode")
    {
        options.actuationMode = parseActuationMode(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--mode"));
        return true;
    }

    if (arg == "--self-collide")
    {
        options.selfCollideDisks = true;
        return true;
    }

    return false;
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

float computeCableLength(const std::vector<Diligent::float3> &positions,
                         const std::vector<Diligent::float3> &offsets)
{
    float total = 0.0f;
    for (std::size_t i = 1; i < positions.size(); ++i)
    {
        total += length((positions[i] + offsets[i]) - (positions[i - 1u] + offsets[i - 1u]));
    }
    return total;
}

float computeActuationWeight(ExampleOptions::ActuationMode mode, std::uint32_t cableIndex,
                             float phase) noexcept
{
    const float cableAngle = (2.0f * kPi * static_cast<float>(cableIndex)) / 3.0f;
    switch (mode)
    {
    case ExampleOptions::ActuationMode::TravelingWave:
        return std::max(0.0f, 0.35f + 0.65f * std::sin(phase));
    case ExampleOptions::ActuationMode::SideBend:
    {
        const float bendDirection = phase;
        const float alignment = std::cos(cableAngle - bendDirection);
        const float activeSide = std::max(0.0f, alignment);
        return activeSide * activeSide;
    }
    }

    return 0.0f;
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    ExampleOptions sceneOptions{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(argc, argv, i, options,
                                                                        false))
            {
                continue;
            }

            if (tryParseSceneArgument(argc, argv, i, sceneOptions))
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
    config.physicsDesc.defaultIterations = 40u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Routed Cable CDCR";
    viewerDefaults.width       = 1280u;
    viewerDefaults.height      = 720u;
    viewerDefaults.vSync       = true;
    const auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);

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

    const auto diskMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "PhysicsRoutedCableCdcr.DiskMesh"));
    const auto groundMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(12.0f, "PhysicsRoutedCableCdcr.GroundMesh"));

    const auto topDiskMaterial =
        registerMaterial(resources, "PhysicsRoutedCableCdcr.TopDisk", {0.88f, 0.82f, 0.72f}, 0.35f);
    const auto diskMaterial =
        registerMaterial(resources, "PhysicsRoutedCableCdcr.Disk", {0.26f, 0.54f, 0.84f}, 0.42f);
    const auto groundMaterial =
        registerMaterial(resources, "PhysicsRoutedCableCdcr.Ground", {0.74f, 0.74f, 0.76f}, 0.9f);

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.9f, -9.2f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.20f};
    light.color     = {1.0f, 0.98f, 0.96f};
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -3.8f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{groundMesh, groundMaterial, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType    = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType      = ColliderShapeType::Box;
    groundCollider.shapeParams    = {12.0f, 0.05f, 12.0f, 0.0f};
    groundCollider.collisionLayer = kGroundLayer;
    groundCollider.collisionMask  = kDiskLayer;
    groundCollider.friction       = 0.8f;
    groundCollider.staticFriction = 0.9f;
    world.addCollider(groundEntity, groundCollider);

    constexpr Diligent::float3 kDiskHalfExtents{0.48f, 0.06f, 0.48f};
    constexpr float kTopY            = 4.75f;
    constexpr float kGuideY          = 0.0f;
    constexpr float kCableCompliance = 2.5e-5f;

    std::vector<EntityId> diskEntities;
    std::vector<Diligent::float3> diskPositions;
    diskEntities.reserve(sceneOptions.diskCount);
    diskPositions.reserve(sceneOptions.diskCount);

    for (std::uint32_t i = 0; i < sceneOptions.diskCount; ++i)
    {
        const auto entity = world.createEntity();
        const float y     = kTopY - static_cast<float>(i) * sceneOptions.diskSpacing;
        diskEntities.push_back(entity);
        diskPositions.push_back({0.0f, y, 0.0f});

        RigidBodyComponent body{};
        body.simulated = true;
        if (i == 0u)
        {
            body.bodyType            = RigidBodyType::Static;
            body.inverseMass         = 0.0f;
            body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
        }
        else
        {
            body.bodyType            = RigidBodyType::Dynamic;
            body.inverseMass         = 1.0f;
            body.inverseInertiaLocal = cressim::neo::examples::helpers::computeBoxInverseInertia(
                kDiskHalfExtents, body.inverseMass);
        }

        TransformComponent transform{};
        transform.worldTransform.position = diskPositions.back();
        transform.worldTransform.scale    = kDiskHalfExtents;
        world.setTransform(entity, transform);
        world.setMeshRenderer(
            entity,
            MeshRendererComponent{diskMesh, i == 0u ? topDiskMaterial : diskMaterial, true});
        world.setRigidBody(entity, body);

        ColliderComponent collider{};
        collider.shapeType   = ColliderShapeType::Box;
        collider.shapeParams = {kDiskHalfExtents.x, kDiskHalfExtents.y, kDiskHalfExtents.z, 0.0f};
        collider.friction    = 0.52f;
        collider.staticFriction = 0.6f;
        collider.collisionLayer = kDiskLayer;
        collider.collisionMask =
            sceneOptions.selfCollideDisks ? (kGroundLayer | kDiskLayer) : kGroundLayer;
        world.addCollider(entity, collider);
    }

    const auto backboneEntity = world.createEntity();
    StrandComponent backbone{};
    backbone.restPositions.reserve(sceneOptions.diskCount);
    backbone.staticParticleIndices.push_back(0u);
    backbone.particleMass       = 0.3f;
    backbone.particleRadius     = 0.075f;
    backbone.stretchShearCompliance =
        std::max(sceneOptions.backboneCompliance * 0.1f, 1.0e-9f);
    backbone.bendCompliance     = sceneOptions.backboneCompliance;
    backbone.selfCollisionEnabled = false;
    backbone.suturingEnabled      = false;
    backbone.collisionLayer       = 0u;
    backbone.collisionMask        = 0u;
    for (const Diligent::float3 &diskPosition : diskPositions)
    {
        backbone.restPositions.push_back(diskPosition);
    }
    if (!world.setStrand(backboneEntity, backbone))
    {
        CRESSIM_LOG_ERROR("Failed to author CDCR backbone strand.\n");
        runtime.shutdown();
        viewer.shutdown();
        return 1;
    }

    AuthoredParticleSequenceState backboneSequence{};
    backboneSequence.entries.reserve(sceneOptions.diskCount);
    for (std::uint32_t i = 0u; i < sceneOptions.diskCount; ++i)
    {
        backboneSequence.entries.push_back(
            AuthoredParticleReference{backboneEntity, AuthoredParticleReferenceType::StrandParticle,
                                      i});
    }
    backboneSequence = world.upsertParticleSequence(backboneSequence);
    world.setProceduralDeformableCurveRender(
        backboneEntity, ProceduralDeformableCurveRenderComponent{backboneSequence.sequenceId, 0.08f,
                                                                 10u, true});

    std::vector<AuthoredRigidParticleAttachmentConstraintState> diskAttachments;
    diskAttachments.reserve(sceneOptions.diskCount);
    for (std::uint32_t i = 0u; i < sceneOptions.diskCount; ++i)
    {
        AuthoredRigidParticleAttachmentConstraintState attachment{};
        attachment.particle = AuthoredParticleReference{
            backboneEntity, AuthoredParticleReferenceType::StrandParticle, i};
        attachment.rigidBodyEntityId = diskEntities[i];
        attachment.localAnchor       = {0.0f, 0.0f, 0.0f};
        attachment.compliance        = std::max(sceneOptions.backboneCompliance * 0.5f, 0.0f);
        attachment.enabled           = true;
        diskAttachments.push_back(world.upsertRigidParticleAttachmentConstraint(attachment));
    }

    std::array<std::vector<Diligent::float3>, 3u> cableOffsets{};
    for (std::uint32_t cableIndex = 0; cableIndex < 3u; ++cableIndex)
    {
        const float angle = (2.0f * kPi * static_cast<float>(cableIndex)) / 3.0f;
        const Diligent::float3 localGuideOffset{sceneOptions.guideRadius * std::cos(angle), kGuideY,
                                                sceneOptions.guideRadius * std::sin(angle)};
        cableOffsets[cableIndex].assign(sceneOptions.diskCount, localGuideOffset);
    }

    std::array<AuthoredRoutedCableConstraintState, 3u> cables{};
    std::array<float, 3u> restLengths{};
    for (std::uint32_t cableIndex = 0; cableIndex < 3u; ++cableIndex)
    {
        auto &cable = cables[cableIndex];
        cable.routePoints.reserve(sceneOptions.diskCount);
        for (std::uint32_t i = 0; i < sceneOptions.diskCount; ++i)
        {
            cable.routePoints.push_back(
                AuthoredRoutedCableRoutePoint{diskEntities[i], cableOffsets[cableIndex][i]});
        }
        cable.compliance        = kCableCompliance;
        cable.tensionOnly       = true;
        cable.enabled           = true;
        restLengths[cableIndex] = computeCableLength(diskPositions, cableOffsets[cableIndex]);
        cable.targetLength      = restLengths[cableIndex];
        cable                   = world.upsertRoutedCableConstraint(cable);
    }

    auto renderOptions                      = runtime.renderFrameOptions();
    renderOptions.debugRoutedCables.enabled = true;
    renderOptions.debugRoutedCables.radius  = 0.028f;
    renderOptions.debugRoutedCables.opacity = 0.95f;
    runtime.setRenderFrameOptions(renderOptions);

    const float maxPull        = std::max(0.0f, sceneOptions.pullAmplitude);
    const float driveFrequency = (2.0f * kPi) / std::max(sceneOptions.drivePeriodSeconds, 0.1f);
    const ExampleOptions::ActuationMode actuationMode = sceneOptions.actuationMode;

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [cables, restLengths, maxPull, driveFrequency,
                            actuationMode](const FrameContext &frame, Runtime &cbRuntime) mutable
    {
        const float t             = static_cast<float>(frame.timeSeconds);
        const float settle        = std::clamp((t - 1.0f) / 1.5f, 0.0f, 1.0f);
        const float commonTighten =
            actuationMode == ExampleOptions::ActuationMode::SideBend ? 0.0f
                                                                     : 0.08f * maxPull * settle;
        for (std::uint32_t cableIndex = 0; cableIndex < 3u; ++cableIndex)
        {
            const float phase = actuationMode == ExampleOptions::ActuationMode::SideBend
                                    ? driveFrequency * t
                                    : driveFrequency * t +
                                          (2.0f * kPi * static_cast<float>(cableIndex)) / 3.0f;
            const float actuationWeight = computeActuationWeight(actuationMode, cableIndex, phase);
            auto updated            = cables[cableIndex];
            updated.targetLength    = std::max(
                0.0f, restLengths[cableIndex] - commonTighten - maxPull * settle * actuationWeight);
            cables[cableIndex]      = cbRuntime.getWorld().upsertRoutedCableConstraint(updated);
        }
    };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk     = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Viewer run failed.\n");
        return 1;
    }

    return 0;
}
