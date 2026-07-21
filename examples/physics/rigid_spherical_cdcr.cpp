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
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SphericalJointState;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi                  = 3.14159265358979323846f;
constexpr float kEpsilon             = 1.0e-6f;
constexpr std::uint32_t kGroundLayer = 1u << 0u;
constexpr std::uint32_t kDiskLayer   = 1u << 1u;

struct ExampleOptions
{
    std::uint32_t diskCount   = 9u;
    float pullAmplitude       = 0.65f;
    float drivePeriodSeconds  = 7.0f;
    float diskSpacing         = 0.68f;
    float guideRadius         = 0.34f;
    float cableCompliance     = 2.5e-5f;
    float swingCompliance     = 8.0e-4f;
    float twistCompliance     = 2.0e-3f;
    float swingLimitRadians   = 0.42f;
    float twistLimitRadians   = 0.35f;
    bool selfCollideDisks     = false;
};

struct DiskVisual
{
    EntityId entity = cressim::neo::common::kInvalidEntityId;
    Diligent::float3 center{};
};

cressim::neo::graphics::MeshResourceDesc translateMesh(
    const cressim::neo::graphics::MeshResourceDesc &source, const Diligent::float3 &offset,
    const std::string &debugName)
{
    cressim::neo::graphics::MeshResourceDesc translated = source;
    translated.debugName                                = debugName;
    for (auto &vertex : translated.vertices)
    {
        vertex.position += offset;
    }
    return translated;
}

void appendMesh(cressim::neo::graphics::MeshResourceDesc &target,
                const cressim::neo::graphics::MeshResourceDesc &source)
{
    const std::uint32_t baseVertex = static_cast<std::uint32_t>(target.vertices.size());
    target.vertices.insert(target.vertices.end(), source.vertices.begin(), source.vertices.end());
    target.indices.reserve(target.indices.size() + source.indices.size());
    for (const std::uint32_t index : source.indices)
    {
        target.indices.push_back(baseVertex + index);
    }
}

cressim::neo::graphics::MeshResourceDesc makeDiskBackboneMesh(
    float diskRadius, float diskHalfHeight, float diskSpacing, float backboneRadius,
    bool includeUpperConnector, bool includeLowerConnector, const std::string &debugName)
{
    cressim::neo::graphics::MeshResourceDesc merged =
        cressim::neo::examples::helpers::makeCylinderMesh(
        diskRadius, diskHalfHeight, 48u, debugName);

    const float connectorHalfSpan = 0.25f * diskSpacing - 0.5f * diskHalfHeight;
    const float connectorHalfHeight = std::max(0.01f, connectorHalfSpan - backboneRadius);

    if (includeUpperConnector)
    {
        appendMesh(merged, translateMesh(cressim::neo::examples::helpers::makeCapsuleMesh(
                                             backboneRadius, connectorHalfHeight, 24u, 8u, 1u,
                                             debugName + ".Upper"),
                                         {0.0f, 0.25f * diskSpacing + 0.5f * diskHalfHeight, 0.0f},
                                         debugName + ".UpperTranslated"));
    }

    if (includeLowerConnector)
    {
        appendMesh(merged, translateMesh(cressim::neo::examples::helpers::makeCapsuleMesh(
                                             backboneRadius, connectorHalfHeight, 24u, 8u, 1u,
                                             debugName + ".Lower"),
                                         {0.0f, -0.25f * diskSpacing - 0.5f * diskHalfHeight, 0.0f},
                                         debugName + ".LowerTranslated"));
    }

    return merged;
}

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
        " [--disks N] [--pull VALUE] [--period VALUE] [--spacing VALUE]"
        " [--guide-radius VALUE] [--cable-compliance VALUE]"
        " [--swing-compliance VALUE] [--twist-compliance VALUE]"
        " [--swing-limit VALUE] [--twist-limit VALUE] [--self-collide]",
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
        return true;
    }
    if (arg == "--spacing")
    {
        options.diskSpacing = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index, "--spacing"),
            "disk spacing");
        return true;
    }
    if (arg == "--guide-radius")
    {
        options.guideRadius = parseFloatOption(
            cressim::neo::examples::helpers::requireOptionValue(argc, argv, index,
                                                                 "--guide-radius"),
            "guide radius");
        return true;
    }
    if (arg == "--cable-compliance")
    {
        options.cableCompliance =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--cable-compliance"),
                             "cable compliance");
        return true;
    }
    if (arg == "--swing-compliance")
    {
        options.swingCompliance =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--swing-compliance"),
                             "swing compliance");
        return true;
    }
    if (arg == "--twist-compliance")
    {
        options.twistCompliance =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--twist-compliance"),
                             "twist compliance");
        return true;
    }
    if (arg == "--swing-limit")
    {
        options.swingLimitRadians =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--swing-limit"),
                             "swing limit");
        return true;
    }
    if (arg == "--twist-limit")
    {
        options.twistLimitRadians =
            parseFloatOption(cressim::neo::examples::helpers::requireOptionValue(
                                 argc, argv, index, "--twist-limit"),
                             "twist limit");
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

void setVisibleRigidBody(Runtime &runtime, EntityId entityId, cressim::neo::graphics::MeshHandle mesh,
                         MaterialHandle material, const Diligent::float3 &position,
                         const Diligent::float3 &scale, const RigidBodyComponent &body,
                         const ColliderComponent &collider)
{
    auto &world = runtime.getWorld();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale    = scale;
    world.setTransform(entityId, transform);
    world.setMeshRenderer(entityId, MeshRendererComponent{mesh, material, true});
    world.setRigidBody(entityId, body);
    world.addCollider(entityId, collider);
}

void setVisibleRigidBody(Runtime &runtime, EntityId entityId, cressim::neo::graphics::MeshHandle mesh,
                         MaterialHandle material, const Diligent::float3 &position,
                         const Diligent::float3 &scale, const RigidBodyComponent &body)
{
    auto &world = runtime.getWorld();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale    = scale;
    world.setTransform(entityId, transform);
    world.setMeshRenderer(entityId, MeshRendererComponent{mesh, material, true});
    world.setRigidBody(entityId, body);
}

cressim::neo::physics::RigidBodyId requireRigidBodyId(Runtime &runtime, EntityId entityId)
{
    const auto *body = runtime.getWorld().physicsWorld().tryGetRigidBody(entityId);
    if (body == nullptr)
    {
        throw std::runtime_error("Missing rigid body while authoring rigid spherical CDCR example.");
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
    config.physicsDesc.defaultIterations = 80u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Rigid Spherical CDCR";
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

    constexpr float kDiskRadius = 0.48f;
    constexpr float kDiskHalfHeight = 0.06f;
    constexpr float kBackboneRadius = 0.055f;
    const auto topDiskMesh = resources.registerMesh(makeDiskBackboneMesh(
        kDiskRadius, kDiskHalfHeight, sceneOptions.diskSpacing, kBackboneRadius, false, true,
        "PhysicsRigidSphericalCdcr.TopDiskMesh"));
    const auto middleDiskMesh = resources.registerMesh(makeDiskBackboneMesh(
        kDiskRadius, kDiskHalfHeight, sceneOptions.diskSpacing, kBackboneRadius, true, true,
        "PhysicsRigidSphericalCdcr.MiddleDiskMesh"));
    const auto bottomDiskMesh = resources.registerMesh(makeDiskBackboneMesh(
        kDiskRadius, kDiskHalfHeight, sceneOptions.diskSpacing, kBackboneRadius, true, false,
        "PhysicsRigidSphericalCdcr.BottomDiskMesh"));
    const auto groundMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(12.0f, "PhysicsRigidSphericalCdcr.Ground"));
    const auto topMaterial =
        registerMaterial(resources, "PhysicsRigidSphericalCdcr.Top", {0.88f, 0.82f, 0.72f}, 0.35f);
    const auto diskMaterial =
        registerMaterial(resources, "PhysicsRigidSphericalCdcr.Disk", {0.22f, 0.56f, 0.82f}, 0.4f);
    const auto groundMaterial =
        registerMaterial(resources, "PhysicsRigidSphericalCdcr.Ground", {0.74f, 0.74f, 0.76f}, 0.9f);

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.8f, -9.4f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.clearColorValue = {0.87f, 0.91f, 0.97f, 1.0f};
    world.setCamera(cameraEntity, camera);

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

    constexpr float kTopY = 4.75f;

    std::vector<DiskVisual> disks;
    disks.reserve(sceneOptions.diskCount);
    for (std::uint32_t i = 0u; i < sceneOptions.diskCount; ++i)
    {
        const auto entity = world.createEntity();
        const float y     = kTopY - static_cast<float>(i) * sceneOptions.diskSpacing;
        RigidBodyComponent body{};
        if (i == 0u)
        {
            body.bodyType            = RigidBodyType::Static;
            body.inverseMass         = 0.0f;
            body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
        }
        else
        {
            body.bodyType            = RigidBodyType::Dynamic;
            body.inverseMass         = 0.95f;
            body.inverseInertiaLocal =
                cressim::neo::examples::helpers::computeCylinderInverseInertia(
                    kDiskRadius, kDiskHalfHeight, body.inverseMass);
        }

        const auto diskMesh = i == 0u ? topDiskMesh
                                      : (i + 1u == sceneOptions.diskCount ? bottomDiskMesh
                                                                           : middleDiskMesh);
        setVisibleRigidBody(runtime, entity, diskMesh, i == 0u ? topMaterial : diskMaterial,
                            {0.0f, y, 0.0f}, {1.0f, 1.0f, 1.0f}, body);
        disks.push_back(DiskVisual{entity, {0.0f, y, 0.0f}});
    }

    for (std::uint32_t i = 1u; i < sceneOptions.diskCount; ++i)
    {
        SphericalJointState joint{};
        joint.bodyA = requireRigidBodyId(runtime, disks[i - 1u].entity);
        joint.bodyB = requireRigidBodyId(runtime, disks[i].entity);
        joint.suppressConnectedBodyCollisions = true;
        joint.localAnchorA = {0.0f, -0.5f * sceneOptions.diskSpacing, 0.0f};
        joint.localAnchorB = {0.0f, 0.5f * sceneOptions.diskSpacing, 0.0f};
        joint.localRotationA = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
        joint.localRotationB = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
        joint.limitEnabled = true;
        joint.swingLimitY  = sceneOptions.swingLimitRadians;
        joint.swingLimitZ  = sceneOptions.swingLimitRadians;
        joint.twistLimitMin = -sceneOptions.twistLimitRadians;
        joint.twistLimitMax = sceneOptions.twistLimitRadians;
        joint.constraintCompliance = 0.0f;
        joint.swingCompliance      = sceneOptions.swingCompliance;
        joint.twistCompliance      = sceneOptions.twistCompliance;
        if (!world.physicsWorld().upsertSphericalJoint(joint))
        {
            throw std::runtime_error("Failed to author rigid spherical backbone joint.");
        }
    }

    std::array<std::vector<Diligent::float3>, 3u> cableOffsets{};
    for (std::uint32_t cableIndex = 0u; cableIndex < 3u; ++cableIndex)
    {
        const float angle = (2.0f * kPi * static_cast<float>(cableIndex)) / 3.0f;
        const Diligent::float3 localGuideOffset{sceneOptions.guideRadius * std::cos(angle), 0.0f,
                                                sceneOptions.guideRadius * std::sin(angle)};
        cableOffsets[cableIndex].assign(sceneOptions.diskCount, localGuideOffset);
    }

    std::array<AuthoredRoutedCableConstraintState, 3u> cables{};
    std::array<float, 3u> restLengths{};
    for (std::uint32_t cableIndex = 0u; cableIndex < 3u; ++cableIndex)
    {
        auto &cable = cables[cableIndex];
        std::vector<Diligent::float3> restPoints;
        restPoints.reserve(sceneOptions.diskCount);
        for (std::uint32_t i = 0u; i < sceneOptions.diskCount; ++i)
        {
            cable.routePoints.push_back(
                AuthoredRoutedCableRoutePoint{disks[i].entity, cableOffsets[cableIndex][i]});
            restPoints.push_back(disks[i].center + cableOffsets[cableIndex][i]);
        }
        cable.compliance        = sceneOptions.cableCompliance;
        cable.tensionOnly       = true;
        cable.enabled           = true;
        restLengths[cableIndex] = computeCableLength(restPoints);
        cable.targetLength      = restLengths[cableIndex];
        world.upsertRoutedCableConstraint(cable, &cable);
    }

    auto renderOptions                      = runtime.renderFrameOptions();
    renderOptions.debugRoutedCables.enabled = true;
    renderOptions.debugRoutedCables.radius  = 0.028f;
    renderOptions.debugRoutedCables.opacity = 0.95f;
    runtime.setRenderFrameOptions(renderOptions);

    const float maxPull        = std::max(0.0f, sceneOptions.pullAmplitude);
    const float driveFrequency = (2.0f * kPi) / std::max(sceneOptions.drivePeriodSeconds, 0.1f);
    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [cables, restLengths, maxPull, driveFrequency](
                               const FrameContext &frame, Runtime &cbRuntime) mutable
    {
        const float t      = static_cast<float>(frame.timeSeconds);
        const float settle = std::clamp((t - 1.0f) / 1.5f, 0.0f, 1.0f);
        const float bendDirection = driveFrequency * t;
        for (std::uint32_t cableIndex = 0u; cableIndex < 3u; ++cableIndex)
        {
            const float cableAngle =
                (2.0f * kPi * static_cast<float>(cableIndex)) / 3.0f;
            const float alignment = std::cos(cableAngle - bendDirection);
            const float activeSide = std::max(0.0f, alignment);
            const float actuationWeight = activeSide * activeSide;
            auto updated         = cables[cableIndex];
            updated.targetLength =
                std::max(0.0f, restLengths[cableIndex] - maxPull * settle * actuationWeight);
            cbRuntime.getWorld().upsertRoutedCableConstraint(updated, &cables[cableIndex]);
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
