#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/shape_meshes.h"
#include "viewer/debug_viewer_app.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::BallJointState;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::HingeJointState;
using cressim::neo::physics::RigidJointDriveMode;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SliderJointState;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kEpsilon = 1.0e-6f;
constexpr std::uint32_t kGroundCollisionLayer = 1u << 0u;
constexpr std::uint32_t kBallClusterLayer = 1u << 1u;
constexpr std::uint32_t kHingeClusterLayer = 1u << 2u;
constexpr std::uint32_t kSliderClusterLayer = 1u << 3u;
constexpr std::uint32_t kBallDropLayer = 1u << 4u;
constexpr std::uint32_t kHingeDropLayer = 1u << 5u;
constexpr std::uint32_t kSliderDropLayer = 1u << 6u;
constexpr float kViewerSphereMeshRadius = 0.4f;

struct ViewerJointOptions
{
    bool enablePositionDriveTargets = false;
    bool enableVelocityDriveTargets = false;
    bool suppressConnectedBodyCollisions = false;
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
    y = yLengthSq <= kEpsilon ? Diligent::float3{0.0f, 1.0f, 0.0f} : y * (1.0f / std::sqrt(yLengthSq));
    Diligent::float3 z = Diligent::cross(x, y);
    const float zLengthSq = Diligent::dot(z, z);
    z = zLengthSq <= kEpsilon ? Diligent::float3{0.0f, 0.0f, 1.0f} : z * (1.0f / std::sqrt(zLengthSq));
    return quaternionFromBasis(x, y, z);
}

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
                      " [--backend vulkan|null] [--frames N] [--joint-drive]"
                      " [--joint-velocity-drive] [--suppress-connected-collisions]\n");
}

Diligent::float3 computeBoxInverseInertia(const Diligent::float3 &halfExtents, float inverseMass)
{
    if (inverseMass <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float ix = mass * (halfExtents.y * halfExtents.y + halfExtents.z * halfExtents.z) / 3.0f;
    const float iy = mass * (halfExtents.x * halfExtents.x + halfExtents.z * halfExtents.z) / 3.0f;
    const float iz = mass * (halfExtents.x * halfExtents.x + halfExtents.y * halfExtents.y) / 3.0f;
    return {ix > 0.0f ? 1.0f / ix : 0.0f, iy > 0.0f ? 1.0f / iy : 0.0f,
            iz > 0.0f ? 1.0f / iz : 0.0f};
}

Diligent::float3 computeSphereInverseInertia(float radius, float inverseMass)
{
    if (inverseMass <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float inertia = 0.4f * mass * radius * radius;
    const float inv = inertia > 0.0f ? 1.0f / inertia : 0.0f;
    return {inv, inv, inv};
}

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

void setVisibleRigidBody(Runtime &runtime, cressim::neo::common::EntityId entityId,
                         MeshHandle mesh, MaterialHandle material,
                         const Diligent::float3 &position, const Diligent::float3 &scale,
                         const RigidBodyComponent &body, const ColliderComponent &collider,
                         const Diligent::QuaternionF &rotation = Diligent::QuaternionF{})
{
    auto &world = runtime.getWorld();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.rotation = rotation;
    transform.worldTransform.scale = scale;
    world.setTransform(entityId, transform);
    world.setMeshRenderer(entityId, MeshRendererComponent{mesh, material, true});
    world.setRigidBody(entityId, body);
    world.addCollider(entityId, collider);
}

cressim::neo::physics::RigidBodyId requireRigidBodyId(Runtime &runtime,
                                                      cressim::neo::common::EntityId entityId)
{
    const auto *body = runtime.getWorld().physicsWorld().tryGetRigidBody(entityId);
    if (body == nullptr)
    {
        throw std::runtime_error("Missing rigid body while authoring joint viewer scene.");
    }
    return body->rigidBodyId;
}

void authorBallJointCluster(Runtime &runtime, MeshHandle anchorMesh, MeshHandle sphereMesh,
                            MaterialHandle anchorMaterial, MaterialHandle bodyMaterial,
                            const ViewerJointOptions &options)
{
    auto &world = runtime.getWorld();
    constexpr float kAnchorHalfExtent = 0.35f;
    constexpr float kBobRadius = 0.4f;
    constexpr float kLinkSpacing = 2.0f * kBobRadius;

    const auto anchorEntity = world.createEntity();
    RigidBodyComponent anchorBody{};
    anchorBody.bodyType = RigidBodyType::Static;
    anchorBody.inverseMass = 0.0f;
    ColliderComponent anchorCollider{};
    anchorCollider.shapeType = ColliderShapeType::Box;
    anchorCollider.shapeParams = {0.35f, 0.35f, 0.35f, 0.0f};
    anchorCollider.collisionLayer = kBallClusterLayer;
    anchorCollider.collisionMask = kBallClusterLayer | kGroundCollisionLayer | kBallDropLayer;
    setVisibleRigidBody(runtime, anchorEntity, anchorMesh, anchorMaterial, {-7.0f, 6.0f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, anchorBody, anchorCollider);

    const float anchorPointX = -7.0f;
    const float anchorPointY = 6.0f - kAnchorHalfExtent;
    std::vector<cressim::neo::common::EntityId> bobs;
    for (std::uint32_t i = 0; i < 3u; ++i)
    {
        const auto entity = world.createEntity();
        RigidBodyComponent body{};
        body.inverseMass = 1.0f;
        body.inverseInertiaLocal = computeSphereInverseInertia(kViewerSphereMeshRadius, body.inverseMass);
        if (i == 2u)
        {
            body.linearVelocity = {0.0f, 0.0f, 2.2f};
        }

        ColliderComponent collider{};
        collider.shapeType = ColliderShapeType::Sphere;
        collider.shapeParams = {0.4f, 0.0f, 0.0f, 0.0f};
        collider.collisionLayer = kBallClusterLayer;
        collider.collisionMask = kBallClusterLayer | kGroundCollisionLayer | kBallDropLayer;
        const float bobCenterX =
            anchorPointX + kBobRadius + (kLinkSpacing * static_cast<float>(i));
        setVisibleRigidBody(runtime, entity, sphereMesh, bodyMaterial,
                            {bobCenterX, anchorPointY, 0.0f},
                            {1.0f, 1.0f, 1.0f}, body, collider);
        bobs.push_back(entity);
    }

    BallJointState root{};
    root.bodyA = requireRigidBodyId(runtime, anchorEntity);
    root.bodyB = requireRigidBodyId(runtime, bobs[0]);
    root.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
    root.localAnchorA = {0.0f, -kAnchorHalfExtent, 0.0f};
    root.localAnchorB = {-kBobRadius, 0.0f, 0.0f};
    if (!world.physicsWorld().upsertBallJoint(root))
    {
        throw std::runtime_error("Failed to author root ball joint.");
    }

    for (std::size_t i = 0; i + 1u < bobs.size(); ++i)
    {
        BallJointState joint{};
        joint.bodyA = requireRigidBodyId(runtime, bobs[i]);
        joint.bodyB = requireRigidBodyId(runtime, bobs[i + 1u]);
        joint.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
        joint.localAnchorA = {kBobRadius, 0.0f, 0.0f};
        joint.localAnchorB = {-kBobRadius, 0.0f, 0.0f};
        if (!world.physicsWorld().upsertBallJoint(joint))
        {
            throw std::runtime_error("Failed to author chain ball joint.");
        }
    }
}

void authorHingeJointCluster(Runtime &runtime, MeshHandle baseMesh, MeshHandle linkMesh,
                             MaterialHandle anchorMaterial, MaterialHandle bodyMaterial,
                             const ViewerJointOptions &options)
{
    auto &world = runtime.getWorld();
    constexpr float kLinkHalfX = 0.25f;
    constexpr float kLinkHalfY = 1.1f;

    const auto baseEntity = world.createEntity();
    RigidBodyComponent baseBody{};
    baseBody.bodyType = RigidBodyType::Static;
    baseBody.inverseMass = 0.0f;
    ColliderComponent baseCollider{};
    baseCollider.shapeType = ColliderShapeType::Box;
    baseCollider.shapeParams = {0.45f, 0.35f, 0.35f, 0.0f};
    baseCollider.collisionLayer = kHingeClusterLayer;
    baseCollider.collisionMask = kHingeClusterLayer | kGroundCollisionLayer | kHingeDropLayer;
    setVisibleRigidBody(runtime, baseEntity, baseMesh, anchorMaterial, {0.0f, 4.3f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, baseBody, baseCollider);

    const Diligent::QuaternionF swingRotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, -Diligent::PI_F * 0.5f);
    const Diligent::float3 topAnchorOffset =
        swingRotation.RotateVector(Diligent::float3{0.0f, kLinkHalfY, 0.0f});
    const Diligent::float3 bottomAnchorOffset =
        swingRotation.RotateVector(Diligent::float3{0.0f, -kLinkHalfY, 0.0f});
    const Diligent::float3 baseAnchor = {0.0f, 4.3f - 0.35f, 0.0f};
    const Diligent::float3 firstCenter = baseAnchor - topAnchorOffset;
    const Diligent::float3 lowerAnchor = firstCenter + bottomAnchorOffset;
    const Diligent::float3 secondCenter = lowerAnchor - topAnchorOffset;

    std::vector<cressim::neo::common::EntityId> links;
    for (std::uint32_t i = 0; i < 2u; ++i)
    {
        const auto entity = world.createEntity();
        RigidBodyComponent body{};
        body.inverseMass = 0.75f;
        body.inverseInertiaLocal =
            computeBoxInverseInertia({kLinkHalfX, kLinkHalfY, kLinkHalfX}, body.inverseMass);

        ColliderComponent collider{};
        collider.shapeType = ColliderShapeType::Box;
        collider.shapeParams = {kLinkHalfX, kLinkHalfY, kLinkHalfX, 0.0f};
        collider.collisionLayer = kHingeClusterLayer;
        collider.collisionMask = kHingeClusterLayer | kGroundCollisionLayer | kHingeDropLayer;
        const Diligent::float3 center = (i == 0u) ? firstCenter : secondCenter;
        setVisibleRigidBody(runtime, entity, linkMesh, bodyMaterial,
                            center, {1.0f, 1.0f, 1.0f}, body, collider,
                            swingRotation);
        links.push_back(entity);
    }

    HingeJointState upper{};
    upper.bodyA = requireRigidBodyId(runtime, baseEntity);
    upper.bodyB = requireRigidBodyId(runtime, links[0]);
    upper.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
    upper.localAnchorA = {0.0f, -0.35f, 0.0f};
    upper.localAnchorB = {0.0f, 1.1f, 0.0f};
    upper.localRotationA = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    upper.localRotationB = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    upper.limitEnabled = true;
    upper.limitMin = -2.55f;
    upper.limitMax = 0.55f;
    upper.driveMode = options.enableVelocityDriveTargets
                          ? RigidJointDriveMode::TargetVelocity
                          : (options.enablePositionDriveTargets
                                 ? RigidJointDriveMode::TargetPosition
                                 : RigidJointDriveMode::None);
    upper.driveTargetAngle = -1.37f;
    upper.driveTargetAngularVelocity = -1.1f;
    if (!world.physicsWorld().upsertHingeJoint(upper))
    {
        throw std::runtime_error("Failed to author upper hinge joint.");
    }

    HingeJointState lower{};
    lower.bodyA = requireRigidBodyId(runtime, links[0]);
    lower.bodyB = requireRigidBodyId(runtime, links[1]);
    lower.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
    lower.localAnchorA = {0.0f, -1.1f, 0.0f};
    lower.localAnchorB = {0.0f, 1.1f, 0.0f};
    lower.localRotationA = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    lower.localRotationB = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    lower.limitEnabled = true;
    lower.limitMin = -1.8f;
    lower.limitMax = 1.2f;
    lower.driveMode = options.enableVelocityDriveTargets
                          ? RigidJointDriveMode::TargetVelocity
                          : (options.enablePositionDriveTargets
                                 ? RigidJointDriveMode::TargetPosition
                                 : RigidJointDriveMode::None);
    lower.driveTargetAngle = 0.1f;
    lower.driveTargetAngularVelocity = 0.9f;
    if (!world.physicsWorld().upsertHingeJoint(lower))
    {
        throw std::runtime_error("Failed to author lower hinge joint.");
    }
}

void authorSliderJointCluster(Runtime &runtime, MeshHandle guideMesh, MeshHandle sliderMesh,
                              MaterialHandle guideMaterial, MaterialHandle sliderMaterial,
                              const ViewerJointOptions &options)
{
    auto &world = runtime.getWorld();
    constexpr Diligent::float3 kGuideHalfExtents = {3.8f, 0.45f, 0.45f};
    constexpr Diligent::float3 kSliderHalfExtents = {0.7f, 0.7f, 0.7f};

    const auto guideEntity = world.createEntity();
    RigidBodyComponent guideBody{};
    guideBody.bodyType = RigidBodyType::Static;
    guideBody.inverseMass = 0.0f;
    ColliderComponent guideCollider{};
    guideCollider.shapeType = ColliderShapeType::Box;
    guideCollider.shapeParams = {kGuideHalfExtents.x, kGuideHalfExtents.y, kGuideHalfExtents.z, 0.0f};
    guideCollider.collisionLayer = kSliderClusterLayer;
    guideCollider.collisionMask = kSliderClusterLayer | kGroundCollisionLayer | kSliderDropLayer;
    setVisibleRigidBody(runtime, guideEntity, guideMesh, guideMaterial, {7.0f, 2.0f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, guideBody, guideCollider);

    const auto sliderEntity = world.createEntity();
    RigidBodyComponent sliderBody{};
    sliderBody.inverseMass = 1.0f;
    sliderBody.inverseInertiaLocal =
        computeBoxInverseInertia(kSliderHalfExtents, sliderBody.inverseMass);
    sliderBody.linearVelocity = {1.5f, 0.0f, 0.0f};
    sliderBody.angularVelocity = {0.0f, 0.0f, 0.0f};
    ColliderComponent sliderCollider{};
    sliderCollider.shapeType = ColliderShapeType::Box;
    sliderCollider.shapeParams = {kSliderHalfExtents.x, kSliderHalfExtents.y, kSliderHalfExtents.z, 0.0f};
    sliderCollider.collisionLayer = kSliderClusterLayer;
    sliderCollider.collisionMask = kSliderClusterLayer | kGroundCollisionLayer | kSliderDropLayer;
    setVisibleRigidBody(runtime, sliderEntity, sliderMesh, sliderMaterial, {5.4f, 2.0f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, sliderBody, sliderCollider);

    SliderJointState horizontalSlider{};
    horizontalSlider.bodyA = requireRigidBodyId(runtime, guideEntity);
    horizontalSlider.bodyB = requireRigidBodyId(runtime, sliderEntity);
    horizontalSlider.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
    horizontalSlider.localRotationA = makeJointFrameRotation({1.0f, 0.0f, 0.0f});
    horizontalSlider.localRotationB = makeJointFrameRotation({1.0f, 0.0f, 0.0f});
    horizontalSlider.limitEnabled = true;
    horizontalSlider.limitMin = -0.6f;
    horizontalSlider.limitMax = 1.8f;
    horizontalSlider.driveMode = options.enableVelocityDriveTargets
                                     ? RigidJointDriveMode::TargetVelocity
                                     : (options.enablePositionDriveTargets
                                            ? RigidJointDriveMode::TargetPosition
                                            : RigidJointDriveMode::None);
    horizontalSlider.driveTargetPosition = 0.20f;
    horizontalSlider.driveTargetVelocity = 1.4f;
    if (!world.physicsWorld().upsertSliderJoint(horizontalSlider))
    {
        throw std::runtime_error("Failed to author slider joint.");
    }

    const auto stageEntity = world.createEntity();
    RigidBodyComponent stageBody{};
    stageBody.inverseMass = 0.85f;
    stageBody.inverseInertiaLocal =
        computeBoxInverseInertia(kSliderHalfExtents, stageBody.inverseMass);
    ColliderComponent stageCollider{};
    stageCollider.shapeType = ColliderShapeType::Box;
    stageCollider.shapeParams = {kSliderHalfExtents.x, kSliderHalfExtents.y, kSliderHalfExtents.z, 0.0f};
    stageCollider.collisionLayer = kSliderClusterLayer;
    stageCollider.collisionMask = kSliderClusterLayer | kGroundCollisionLayer | kSliderDropLayer;
    setVisibleRigidBody(runtime, stageEntity, sliderMesh, sliderMaterial, {5.4f, 3.20f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, stageBody, stageCollider);

    SliderJointState verticalSlider{};
    verticalSlider.bodyA = requireRigidBodyId(runtime, sliderEntity);
    verticalSlider.bodyB = requireRigidBodyId(runtime, stageEntity);
    verticalSlider.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
    verticalSlider.localRotationA = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
    verticalSlider.localRotationB = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
    verticalSlider.limitEnabled = true;
    verticalSlider.limitMin = -0.35f;
    verticalSlider.limitMax = 1.0f;
    verticalSlider.driveMode = options.enableVelocityDriveTargets
                                   ? RigidJointDriveMode::TargetVelocity
                                   : (options.enablePositionDriveTargets
                                          ? RigidJointDriveMode::TargetPosition
                                          : RigidJointDriveMode::None);
    verticalSlider.driveTargetPosition = 0.50f;
    verticalSlider.driveTargetVelocity = 0.8f;
    if (!world.physicsWorld().upsertSliderJoint(verticalSlider))
    {
        throw std::runtime_error("Failed to author vertical slider joint.");
    }
}

void authorDropDisturbers(Runtime &runtime, MeshHandle hingeDropMesh, MeshHandle sphereMesh,
                          MaterialHandle ballMaterial, MaterialHandle hingeMaterial,
                          MaterialHandle sliderMaterial)
{
    auto &world = runtime.getWorld();

    constexpr float kDropSphereRadius = 0.45f;
    const float kDropSphereVisualScale = kDropSphereRadius / kViewerSphereMeshRadius;

    const auto ballDrop = world.createEntity();
    RigidBodyComponent ballDropBody{};
    ballDropBody.inverseMass = 0.75f;
    ballDropBody.inverseInertiaLocal = computeSphereInverseInertia(kDropSphereRadius, ballDropBody.inverseMass);
    ColliderComponent ballDropCollider{};
    ballDropCollider.shapeType = ColliderShapeType::Sphere;
    ballDropCollider.shapeParams = {kDropSphereRadius, 0.0f, 0.0f, 0.0f};
    ballDropCollider.collisionLayer = kBallDropLayer;
    ballDropCollider.collisionMask = kBallClusterLayer | kGroundCollisionLayer;
    setVisibleRigidBody(runtime, ballDrop, sphereMesh, ballMaterial, {-8.0f, 8.8f, 0.2f},
                        {kDropSphereVisualScale, kDropSphereVisualScale, kDropSphereVisualScale},
                        ballDropBody, ballDropCollider);

    const auto hingeDrop = world.createEntity();
    RigidBodyComponent hingeDropBody{};
    hingeDropBody.inverseMass = 0.85f;
    hingeDropBody.inverseInertiaLocal =
        computeBoxInverseInertia({0.4f, 0.4f, 0.4f}, hingeDropBody.inverseMass);
    hingeDropBody.angularVelocity = {0.8f, 0.2f, -0.4f};
    ColliderComponent hingeDropCollider{};
    hingeDropCollider.shapeType = ColliderShapeType::Box;
    hingeDropCollider.shapeParams = {0.4f, 0.4f, 0.4f, 0.0f};
    hingeDropCollider.collisionLayer = kHingeDropLayer;
    hingeDropCollider.collisionMask = kHingeClusterLayer | kGroundCollisionLayer;
    setVisibleRigidBody(runtime, hingeDrop, hingeDropMesh, hingeMaterial, {0.5f, 8.4f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, hingeDropBody, hingeDropCollider,
                        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, 0.35f));

    const auto sliderDrop = world.createEntity();
    RigidBodyComponent sliderDropBody{};
    sliderDropBody.inverseMass = 0.8f;
    sliderDropBody.inverseInertiaLocal = computeSphereInverseInertia(kDropSphereRadius, sliderDropBody.inverseMass);
    sliderDropBody.angularVelocity = {0.0f, 0.0f, 0.0f};
    ColliderComponent sliderDropCollider{};
    sliderDropCollider.shapeType = ColliderShapeType::Sphere;
    sliderDropCollider.shapeParams = {kDropSphereRadius, 0.0f, 0.0f, 0.0f};
    sliderDropCollider.collisionLayer = kSliderDropLayer;
    sliderDropCollider.collisionMask = kSliderClusterLayer | kGroundCollisionLayer;
    setVisibleRigidBody(runtime, sliderDrop, sphereMesh, sliderMaterial, {6.7f, 6.6f, 0.0f},
                        {kDropSphereVisualScale, kDropSphereVisualScale, kDropSphereVisualScale},
                        sliderDropBody, sliderDropCollider);
}

} // namespace

int main(int argc, char **argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.physicsDesc.substeps = 2u;
    config.physicsDesc.rigidRigidContactIterations = 16u;
    config.physicsDesc.rigidJointIterations = 48u;
    std::uint64_t numFrames = 0;
    ViewerJointOptions jointOptions{};

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
        if (arg == "--joint-drive")
        {
            jointOptions.enablePositionDriveTargets = true;
            jointOptions.enableVelocityDriveTargets = false;
            continue;
        }
        if (arg == "--joint-velocity-drive")
        {
            jointOptions.enableVelocityDriveTargets = true;
            jointOptions.enablePositionDriveTargets = false;
            continue;
        }
        if (arg == "--suppress-connected-collisions")
        {
            jointOptions.suppressConnectedBodyCollisions = true;
            continue;
        }

        printUsage(argv[0]);
        return 2;
    }

    DebugViewerApp viewer;
    DebugViewerAppDesc viewerDesc{};
    const bool windowEnabled = (config.gpuDeviceDesc.preferredBackend != GpuBackend::Null);
    viewerDesc.windowEnabled = windowEnabled;
    viewerDesc.windowVisible = windowEnabled;
    viewerDesc.startFullscreenWindowed = false;
    viewerDesc.maxFrames = numFrames;
    viewerDesc.showStats = true;
    viewerDesc.statsIntervalFrames = 60u;
    viewerDesc.width = 1440;
    viewerDesc.height = 900;
    viewerDesc.windowTitle = "CRESSim Neo Rigid Joint Viewer";

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Rigid joint viewer initialization failed.");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Rigid joint viewer runtime initialization failed.");
        return 1;
    }

    try
    {
        auto &world = runtime.getWorld();

        const auto cameraEntity = world.createEntity();
        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, 5.5f, -18.0f};
        cameraTransform.worldTransform.rotation =
            Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.18f);
        world.setTransform(cameraEntity, cameraTransform);
        CameraComponent camera{};
        camera.verticalFovDegrees = 42.0f;
        world.setCamera(cameraEntity, camera);

        const auto lightEntity = world.createEntity();
        DirectionalLightComponent light{};
        light.direction = Diligent::normalize(Diligent::float3{-0.4f, -1.0f, 0.25f});
        light.intensity = 6.5f;
        world.setDirectionalLight(lightEntity, light);

        auto &resources = runtime.getResources();
        const MeshHandle planeMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makePlaneMesh(24.0f, "RigidJointViewer.PlaneMesh"));
        const MeshHandle sphereMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeSphereMesh(
                0.4f, 24u, 16u, "RigidJointViewer.SphereMesh"));
        const MeshHandle ballAnchorMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
                {0.35f, 0.35f, 0.35f}, "RigidJointViewer.BoxMesh"));
        const MeshHandle hingeBaseMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
                {0.45f, 0.35f, 0.35f}, "RigidJointViewer.BoxMesh"));
        const MeshHandle hingeLinkMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
                {0.25f, 1.1f, 0.25f}, "RigidJointViewer.BoxMesh"));
        const MeshHandle hingeDropMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
                {0.4f, 0.4f, 0.4f}, "RigidJointViewer.BoxMesh"));
        const MeshHandle sliderGuideMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
                {3.8f, 0.45f, 0.45f}, "RigidJointViewer.BoxMesh"));
        const MeshHandle sliderCarriageMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
                {0.7f, 0.7f, 0.7f}, "RigidJointViewer.BoxMesh"));

        const MaterialHandle groundMaterial =
            registerMaterial(resources, "RigidJointViewer.Ground", {0.70f, 0.72f, 0.76f}, 0.88f);
        const MaterialHandle anchorMaterial =
            registerMaterial(resources, "RigidJointViewer.Anchor", {0.90f, 0.72f, 0.24f}, 0.52f);
        const MaterialHandle ballMaterial =
            registerMaterial(resources, "RigidJointViewer.Ball", {0.20f, 0.70f, 0.96f}, 0.38f);
        const MaterialHandle hingeMaterial =
            registerMaterial(resources, "RigidJointViewer.Hinge", {0.93f, 0.38f, 0.29f}, 0.42f);
        const MaterialHandle sliderMaterial =
            registerMaterial(resources, "RigidJointViewer.Slider", {0.30f, 0.88f, 0.50f}, 0.34f);
        const MaterialHandle guideMaterial =
            registerMaterial(resources, "RigidJointViewer.Guide", {0.25f, 0.34f, 0.36f}, 0.74f);

        const auto groundEntity = world.createEntity();
        TransformComponent groundTransform{};
        groundTransform.worldTransform.position = {0.0f, -1.0f, 0.0f};
        world.setTransform(groundEntity, groundTransform);
        world.setMeshRenderer(groundEntity, MeshRendererComponent{planeMesh, groundMaterial, true});
        RigidBodyComponent groundBody{};
        groundBody.bodyType = RigidBodyType::Static;
        groundBody.inverseMass = 0.0f;
        world.setRigidBody(groundEntity, groundBody);
        ColliderComponent groundCollider{};
        groundCollider.shapeType = ColliderShapeType::Box;
        groundCollider.shapeParams = {24.0f, 0.05f, 24.0f, 0.0f};
        groundCollider.friction = 0.8f;
        groundCollider.staticFriction = 0.9f;
        groundCollider.collisionLayer = kGroundCollisionLayer;
        groundCollider.collisionMask =
            kGroundCollisionLayer | kBallClusterLayer | kHingeClusterLayer | kSliderClusterLayer |
            kBallDropLayer | kHingeDropLayer | kSliderDropLayer;
        world.addCollider(groundEntity, groundCollider);

        authorBallJointCluster(runtime, ballAnchorMesh, sphereMesh, anchorMaterial, ballMaterial,
                               jointOptions);
        authorHingeJointCluster(runtime, hingeBaseMesh, hingeLinkMesh, anchorMaterial,
                                hingeMaterial, jointOptions);
        authorSliderJointCluster(runtime, sliderGuideMesh, sliderCarriageMesh, guideMaterial,
                                 sliderMaterial, jointOptions);
        authorDropDisturbers(runtime, hingeDropMesh, sphereMesh, ballMaterial, hingeMaterial,
                             sliderMaterial);

        const bool runOk = viewer.run(runtime, DebugViewerCameraBinding{cameraEntity});
        viewer.shutdown();
        if (!runOk)
        {
            CRESSIM_LOG_ERROR("Rigid joint viewer run failed.");
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Rigid joint viewer scene setup failed: ", e.what());
        return 1;
    }

    CRESSIM_LOG_INFO("Rigid joint viewer finished. Frames=", viewerDesc.maxFrames);
    return 0;
}
