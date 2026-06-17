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
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::AuthoredRoutedCableConstraintState;
using cressim::neo::physics::AuthoredRoutedCableRoutePoint;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::HingeJointState;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::RigidJointDriveMode;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi                     = 3.14159265358979323846f;
constexpr float kEpsilon                = 1.0e-6f;
constexpr std::uint32_t kGroundLayer    = 1u << 0u;
constexpr std::uint32_t kBackboneLayer  = 1u << 1u;
constexpr float kViewerSphereMeshRadius = 0.4f;

struct ExampleOptions
{
    std::uint32_t linkCount      = 7u;
    float pullAmplitude          = 0.95f;
    float drivePeriodSeconds     = 25.5f;
    float cableCompliance        = 1.0e-9f;
    float driveCompliance        = 8.0e-4f;
    float hingeLimitRadians      = 0.65f;
    bool suppressNeighborCollide = true;
};

struct LinkVisual
{
    EntityId entity = cressim::neo::common::kInvalidEntityId;
    Diligent::float3 center{};
};

Diligent::QuaternionF quaternionFromBasis(const Diligent::float3 &x, const Diligent::float3 &y,
                                          const Diligent::float3 &z)
{
    const float m00 = x.x, m01 = y.x, m02 = z.x;
    const float m10 = x.y, m11 = y.y, m12 = z.y;
    const float m20 = x.z, m21 = y.z, m22 = z.z;
    const float trace = m00 + m11 + m22;

    Diligent::QuaternionF q{};
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.q.w = 0.25f * s;
        q.q.x = (m21 - m12) / s;
        q.q.y = (m02 - m20) / s;
        q.q.z = (m10 - m01) / s;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.q.w = (m21 - m12) / s;
        q.q.x = 0.25f * s;
        q.q.y = (m01 + m10) / s;
        q.q.z = (m02 + m20) / s;
    }
    else if (m11 > m22)
    {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.q.w = (m02 - m20) / s;
        q.q.x = (m01 + m10) / s;
        q.q.y = 0.25f * s;
        q.q.z = (m12 + m21) / s;
    }
    else
    {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.q.w = (m10 - m01) / s;
        q.q.x = (m02 + m20) / s;
        q.q.y = (m12 + m21) / s;
        q.q.z = 0.25f * s;
    }

    const float lengthSq = Diligent::dot(q.q, q.q);
    if (lengthSq <= kEpsilon)
    {
        return Diligent::QuaternionF{0.0f, 0.0f, 0.0f, 1.0f};
    }
    return Diligent::normalize(q);
}

Diligent::QuaternionF makeJointFrameRotation(const Diligent::float3 &axisX)
{
    const float lengthSq = Diligent::dot(axisX, axisX);
    const Diligent::float3 x = lengthSq <= kEpsilon ? Diligent::float3{1.0f, 0.0f, 0.0f}
                                                    : axisX * (1.0f / std::sqrt(lengthSq));
    Diligent::float3 reference{1.0f, 0.0f, 0.0f};
    if (std::abs(Diligent::dot(reference, x)) > 0.99f)
    {
        reference = {0.0f, 1.0f, 0.0f};
    }
    Diligent::float3 y = Diligent::cross(x, reference);
    const float yLengthSq = Diligent::dot(y, y);
    y = yLengthSq <= kEpsilon ? Diligent::float3{0.0f, 1.0f, 0.0f}
                              : y * (1.0f / std::sqrt(yLengthSq));
    Diligent::float3 z = Diligent::cross(x, y);
    const float zLengthSq = Diligent::dot(z, z);
    z = zLengthSq <= kEpsilon ? Diligent::float3{0.0f, 0.0f, 1.0f}
                              : z * (1.0f / std::sqrt(zLengthSq));
    return quaternionFromBasis(x, y, z);
}

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--links N] [--pull VALUE] [--period VALUE] [--cable-compliance VALUE]"
        " [--drive-compliance VALUE] [--hinge-limit VALUE] [--keep-neighbor-collisions]",
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

bool tryParseSceneArgument(int argc, char **argv, int &index, ExampleOptions &options)
{
    const std::string arg = argv[index];
    if (arg == "--links")
    {
        options.linkCount = parseUIntOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--links"),
            "link count");
        if (options.linkCount < 2u)
        {
            throw std::invalid_argument("--links must be at least 2.");
        }
        return true;
    }
    if (arg == "--pull")
    {
        options.pullAmplitude = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--pull"),
            "pull amplitude");
        return true;
    }
    if (arg == "--period")
    {
        options.drivePeriodSeconds = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--period"),
            "period");
        if (options.drivePeriodSeconds <= 0.0f)
        {
            throw std::invalid_argument("--period must be greater than zero.");
        }
        return true;
    }
    if (arg == "--cable-compliance")
    {
        options.cableCompliance =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--cable-compliance"),
                             "cable compliance");
        if (options.cableCompliance < 0.0f)
        {
            throw std::invalid_argument("--cable-compliance must be non-negative.");
        }
        return true;
    }
    if (arg == "--drive-compliance")
    {
        options.driveCompliance =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--drive-compliance"),
                             "drive compliance");
        if (options.driveCompliance < 0.0f)
        {
            throw std::invalid_argument("--drive-compliance must be non-negative.");
        }
        return true;
    }
    if (arg == "--hinge-limit")
    {
        options.hingeLimitRadians =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--hinge-limit"),
                             "hinge limit");
        if (options.hingeLimitRadians <= 0.0f)
        {
            throw std::invalid_argument("--hinge-limit must be greater than zero.");
        }
        return true;
    }
    if (arg == "--keep-neighbor-collisions")
    {
        options.suppressNeighborCollide = false;
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

void setVisibleRigidBody(Runtime &runtime, EntityId entityId, cressim::neo::graphics::MeshHandle mesh,
                         MaterialHandle material, const Diligent::float3 &position,
                         const Diligent::float3 &scale, const RigidBodyComponent &body,
                         const ColliderComponent &collider,
                         const Diligent::QuaternionF &rotation = Diligent::QuaternionF{})
{
    auto &world = runtime.getWorld();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.rotation = rotation;
    transform.worldTransform.scale    = scale;
    world.setTransform(entityId, transform);
    world.setMeshRenderer(entityId, MeshRendererComponent{mesh, material, true});
    world.setRigidBody(entityId, body);
    world.addCollider(entityId, collider);
}

cressim::neo::physics::RigidBodyId requireRigidBodyId(Runtime &runtime, EntityId entityId)
{
    const auto *body = runtime.getWorld().physicsWorld().tryGetRigidBody(entityId);
    if (body == nullptr)
    {
        throw std::runtime_error("Missing rigid body while authoring planar CDCR example.");
    }
    return body->rigidBodyId;
}

float computeCableLength(const std::vector<Diligent::float3> &points)
{
    float total = 0.0f;
    for (std::size_t i = 1; i < points.size(); ++i)
    {
        total += length(points[i] - points[i - 1u]);
    }
    return total;
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
    config.physicsDesc.defaultIterations = 100u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Planar Hinge CDCR";
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

    const auto cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "PhysicsPlanarHingeCdcr.CubeMesh"));
    const auto sphereMesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        kViewerSphereMeshRadius, 16u, 12u, "PhysicsPlanarHingeCdcr.SphereMesh"));
    const auto groundMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(16.0f, "PhysicsPlanarHingeCdcr.GroundMesh"));

    const auto baseMaterial =
        registerMaterial(resources, "PhysicsPlanarHingeCdcr.Base", {0.90f, 0.83f, 0.72f}, 0.34f);
    const auto linkMaterial =
        registerMaterial(resources, "PhysicsPlanarHingeCdcr.Link", {0.24f, 0.55f, 0.88f}, 0.38f);
    const auto tipMaterial =
        registerMaterial(resources, "PhysicsPlanarHingeCdcr.Tip", {0.92f, 0.47f, 0.32f}, 0.34f);
    const auto anchorMaterial =
        registerMaterial(resources, "PhysicsPlanarHingeCdcr.Anchor", {0.82f, 0.26f, 0.30f}, 0.28f);
    const auto groundMaterial =
        registerMaterial(resources, "PhysicsPlanarHingeCdcr.Ground", {0.74f, 0.75f, 0.78f}, 0.94f);

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.1f, -11.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.42f, -1.0f, 0.24f};
    light.color     = {1.0f, 0.98f, 0.96f};
    light.intensity = 7.5f;
    world.setDirectionalLight(lightEntity, light);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -3.2f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{groundMesh, groundMaterial, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType    = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType      = ColliderShapeType::Box;
    groundCollider.shapeParams    = {16.0f, 0.05f, 16.0f, 0.0f};
    groundCollider.collisionLayer = kGroundLayer;
    groundCollider.collisionMask  = kBackboneLayer;
    groundCollider.friction       = 0.85f;
    groundCollider.staticFriction = 0.95f;
    world.addCollider(groundEntity, groundCollider);

    constexpr Diligent::float3 kBaseHalfExtents{0.60f, 0.30f, 0.55f};
    constexpr Diligent::float3 kLinkHalfExtents{0.18f, 0.48f, 0.18f};
    constexpr Diligent::float3 kAnchorHalfExtents{0.13f, 0.13f, 0.13f};
    constexpr float kGuideX = 0.42f;
    constexpr float kGuideZ = 0.0f;
    constexpr float kTopY   = 4.4f;
    const float linkLength  = 2.0f * kLinkHalfExtents.y;

    const auto baseEntity = world.createEntity();
    RigidBodyComponent baseBody{};
    baseBody.bodyType            = RigidBodyType::Static;
    baseBody.inverseMass         = 0.0f;
    baseBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    ColliderComponent baseCollider{};
    baseCollider.shapeType      = ColliderShapeType::Box;
    baseCollider.shapeParams    = {kBaseHalfExtents.x, kBaseHalfExtents.y, kBaseHalfExtents.z, 0.0f};
    baseCollider.collisionLayer = kBackboneLayer;
    baseCollider.collisionMask  = kGroundLayer;
    setVisibleRigidBody(runtime, baseEntity, cubeMesh, baseMaterial, {0.0f, kTopY, 0.0f},
                        kBaseHalfExtents, baseBody, baseCollider);

    std::vector<LinkVisual> links;
    links.reserve(sceneOptions.linkCount);
    for (std::uint32_t i = 0u; i < sceneOptions.linkCount; ++i)
    {
        const auto entity = world.createEntity();
        const float centerY =
            (kTopY - kBaseHalfExtents.y - kLinkHalfExtents.y) - static_cast<float>(i) * linkLength;

        RigidBodyComponent body{};
        body.simulated            = true;
        body.bodyType             = RigidBodyType::Dynamic;
        body.inverseMass          = (i + 1u == sceneOptions.linkCount) ? 0.65f : 0.95f;
        body.inverseInertiaLocal  = cressim::neo::examples::helpers::computeBoxInverseInertia(
            kLinkHalfExtents, body.inverseMass);

        ColliderComponent collider{};
        collider.shapeType      = ColliderShapeType::Box;
        collider.shapeParams    = {kLinkHalfExtents.x, kLinkHalfExtents.y, kLinkHalfExtents.z, 0.0f};
        collider.collisionLayer = kBackboneLayer;
        collider.collisionMask  = kGroundLayer;
        collider.friction       = 0.42f;
        collider.staticFriction = 0.55f;

        setVisibleRigidBody(runtime, entity, cubeMesh,
                            (i + 1u == sceneOptions.linkCount) ? tipMaterial : linkMaterial,
                            {0.0f, centerY, 0.0f}, kLinkHalfExtents, body, collider);
        links.push_back(LinkVisual{entity, {0.0f, centerY, 0.0f}});
    }

    for (std::uint32_t i = 0u; i < sceneOptions.linkCount; ++i)
    {
        HingeJointState hinge{};
        hinge.bodyA = requireRigidBodyId(runtime, i == 0u ? baseEntity : links[i - 1u].entity);
        hinge.bodyB = requireRigidBodyId(runtime, links[i].entity);
        hinge.suppressConnectedBodyCollisions = sceneOptions.suppressNeighborCollide;
        hinge.localAnchorA = i == 0u ? Diligent::float3{0.0f, -kBaseHalfExtents.y, 0.0f}
                                     : Diligent::float3{0.0f, -kLinkHalfExtents.y, 0.0f};
        hinge.localAnchorB = {0.0f, kLinkHalfExtents.y, 0.0f};
        hinge.localRotationA = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
        hinge.localRotationB = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
        hinge.limitEnabled   = true;
        hinge.limitMin       = -sceneOptions.hingeLimitRadians;
        hinge.limitMax       = sceneOptions.hingeLimitRadians;
        hinge.constraintCompliance = 0.0f;
        hinge.driveMode            = RigidJointDriveMode::TargetPosition;
        hinge.driveTargetAngle     = 0.0f;
        hinge.driveCompliance      = sceneOptions.driveCompliance;
        if (!world.physicsWorld().upsertHingeJoint(hinge))
        {
            throw std::runtime_error("Failed to author planar backbone hinge joint.");
        }
    }

    std::array<EntityId, 2u> anchorEntities{};
    std::array<Diligent::float3, 2u> anchorPositions{
        Diligent::float3{-0.75f, kTopY + 0.35f, 0.0f},
        Diligent::float3{0.75f, kTopY + 0.35f, 0.0f}};
    for (std::uint32_t cableIndex = 0u; cableIndex < 2u; ++cableIndex)
    {
        const auto anchorEntity = world.createEntity();
        anchorEntities[cableIndex] = anchorEntity;
        RigidBodyComponent body{};
        body.bodyType            = RigidBodyType::Static;
        body.inverseMass         = 0.0f;
        body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
        ColliderComponent collider{};
        collider.shapeType      = ColliderShapeType::Box;
        collider.shapeParams    = {kAnchorHalfExtents.x, kAnchorHalfExtents.y, kAnchorHalfExtents.z, 0.0f};
        collider.collisionLayer = 0u;
        collider.collisionMask  = 0u;
        setVisibleRigidBody(runtime, anchorEntity, sphereMesh, anchorMaterial, anchorPositions[cableIndex],
                            {kAnchorHalfExtents.x / kViewerSphereMeshRadius,
                             kAnchorHalfExtents.y / kViewerSphereMeshRadius,
                             kAnchorHalfExtents.z / kViewerSphereMeshRadius},
                            body, collider);
    }

    std::array<std::vector<Diligent::float3>, 2u> guideOffsets{};
    guideOffsets[0].assign(sceneOptions.linkCount, Diligent::float3{-kGuideX, 0.0f, kGuideZ});
    guideOffsets[1].assign(sceneOptions.linkCount, Diligent::float3{kGuideX, 0.0f, kGuideZ});

    std::array<AuthoredRoutedCableConstraintState, 2u> cables{};
    std::array<float, 2u> restLengths{};
    for (std::uint32_t cableIndex = 0u; cableIndex < 2u; ++cableIndex)
    {
        auto &cable = cables[cableIndex];
        cable.routePoints.push_back(
            AuthoredRoutedCableRoutePoint{anchorEntities[cableIndex], {0.0f, 0.0f, 0.0f}});

        std::vector<Diligent::float3> restPoints;
        restPoints.reserve(sceneOptions.linkCount + 1u);
        restPoints.push_back(anchorPositions[cableIndex]);
        for (std::uint32_t i = 0u; i < sceneOptions.linkCount; ++i)
        {
            cable.routePoints.push_back(
                AuthoredRoutedCableRoutePoint{links[i].entity, guideOffsets[cableIndex][i]});
            restPoints.push_back(links[i].center + guideOffsets[cableIndex][i]);
        }

        cable.compliance  = sceneOptions.cableCompliance;
        cable.tensionOnly = true;
        cable.enabled     = true;
        restLengths[cableIndex] = computeCableLength(restPoints);
        cable.targetLength      = restLengths[cableIndex];
        cable                   = world.upsertRoutedCableConstraint(cable);
    }

    auto renderOptions                      = runtime.renderFrameOptions();
    renderOptions.debugRoutedCables.enabled = true;
    renderOptions.debugRoutedCables.radius  = 0.026f;
    renderOptions.debugRoutedCables.opacity = 0.95f;
    runtime.setRenderFrameOptions(renderOptions);

    const float maxPull = std::max(0.0f, sceneOptions.pullAmplitude);
    const float driveFrequency = (2.0f * kPi) / std::max(sceneOptions.drivePeriodSeconds, 0.1f);
    const float releaseGain = 1.35f;
    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [cables, restLengths, maxPull, driveFrequency, releaseGain](
                               const FrameContext &frame, Runtime &cbRuntime) mutable
    {
        const float t      = static_cast<float>(frame.timeSeconds);
        const float settle = std::clamp((t - 1.0f) / 1.25f, 0.0f, 1.0f);
        const float phase  = std::sin(driveFrequency * t);
        const float shortenLeft  = std::max(0.0f, phase) * maxPull * settle;
        const float shortenRight = std::max(0.0f, -phase) * maxPull * settle;
        const std::array<float, 2u> shortenings{shortenLeft, shortenRight};
        const std::array<float, 2u> releases{shortenRight * releaseGain,
                                             shortenLeft * releaseGain};

        for (std::uint32_t cableIndex = 0u; cableIndex < 2u; ++cableIndex)
        {
            auto updated         = cables[cableIndex];
            updated.targetLength = std::max(
                0.0f, restLengths[cableIndex] - shortenings[cableIndex] + releases[cableIndex]);
            cables[cableIndex]   = cbRuntime.getWorld().upsertRoutedCableConstraint(updated);
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
