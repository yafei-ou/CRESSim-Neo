#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/environment_ibl_baker.h"
#include "helpers/viewer_example.h"
#include "helpers/inertia.h"
#include "helpers/shape_meshes.h"
#include "helpers/skybox_example.h"
#include "viewer/debug_viewer_app.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
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
using cressim::neo::physics::BallJointState;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::HingeJointState;
using cressim::neo::physics::RigidJointDriveMode;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SphericalJointState;
using cressim::neo::physics::SliderJointState;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kEpsilon = 1.0e-6f;
constexpr std::uint32_t kGroundCollisionLayer = 1u << 0u;
constexpr std::uint32_t kBallClusterLayer = 1u << 1u;
constexpr std::uint32_t kHingeClusterLayer = 1u << 2u;
constexpr std::uint32_t kSliderClusterLayer = 1u << 3u;
constexpr std::uint32_t kBallDropLayer = 1u << 4u;
constexpr std::uint32_t kHingeDropLayer = 1u << 5u;
constexpr std::uint32_t kSliderDropLayer = 1u << 6u;
constexpr std::uint32_t kSphericalClusterLayer = 1u << 7u;
constexpr std::uint32_t kSphericalDropLayer = 1u << 8u;
constexpr float kViewerSphereMeshRadius = 0.4f;
constexpr float kAnchorHalfExtent = 0.35f;
constexpr float kHingeBaseHalfX = 0.45f;
constexpr float kHingeBaseHalfY = 0.35f;
constexpr float kHingeBaseHalfZ = 0.35f;
constexpr float kExampleHingeDriveCompliance = 2.5e-6f;
constexpr float kExampleHingeDriveDamping = 0.0f;
constexpr float kExampleHingeDriveMaxAngularVelocity = 0.9f;
constexpr float kExampleSliderDriveCompliance = 3.0e-4f;
constexpr float kExampleSphericalDriveCompliance = 8.0e-5f;
constexpr float kExampleSphericalFreeCompliance = 1.0e6f;
constexpr float kUpperHingeRestAngle = -Diligent::PI_F * 0.5f;
constexpr float kLowerHingeRestAngle = 0.0f;
constexpr float kHingeDriveHalfRange = 2.0f * Diligent::PI_F;
constexpr float kHingeDriveRange = 2.0f * kHingeDriveHalfRange;
constexpr float kLowerHingeDriveHalfRange = 0.5f * Diligent::PI_F;
constexpr float kLowerHingeDriveRange = 2.0f * kLowerHingeDriveHalfRange;
constexpr float kHorizontalSliderDriveCenter = 0.12f;
constexpr float kVerticalSliderDriveCenter = 0.32f;
constexpr float kOverviewAreaOffsetX = 2.0f;
constexpr const char *kRigidJointsSkyboxCrossPath =
    "examples/cubemaps/Cubemap/Cubemap_Sky_05-512x512.png";

struct AuthoredHingeJointIds
{
    cressim::neo::physics::HingeJointId upper = cressim::neo::physics::kInvalidHingeJointId;
    cressim::neo::physics::HingeJointId lower = cressim::neo::physics::kInvalidHingeJointId;
};

struct AuthoredSliderJointIds
{
    cressim::neo::physics::SliderJointId horizontal = cressim::neo::physics::kInvalidSliderJointId;
    cressim::neo::physics::SliderJointId vertical   = cressim::neo::physics::kInvalidSliderJointId;
};

struct AuthoredSphericalJointIds
{
    std::vector<cressim::neo::physics::SphericalJointId> chain{};
};

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

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName,
        " [--joint-drive] [--joint-velocity-drive] [--suppress-connected-collisions]",
        false);
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

EnvironmentIblDesc loadRigidJointsSkyboxIbl(
    cressim::neo::graphics::RenderResourceManager &resources)
{
    const std::filesystem::path crossPath =
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
        kRigidJointsSkyboxCrossPath;

    EnvironmentIblBakeOptions options{};
    options.irradianceSize = 16u;
    options.specularSize = 128u;
    options.specularMipCount = 7u;
    options.irradianceSampleCount = 256u;
    options.specularSampleCount = 128u;
    options.intensity = 0.18f;
    options.backgroundIntensity = 1.00f;
    return cressim::neo::examples::helpers::createEnvironmentIblFromHorizontalCross(
        resources, crossPath, options);
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

void authorBallJointCluster(Runtime &runtime, MeshHandle anchorMesh, MeshHandle linkMesh,
                            MaterialHandle anchorMaterial, MaterialHandle bodyMaterial,
                            const ViewerJointOptions &options)
{
    auto &world = runtime.getWorld();
    constexpr float kAnchorHalfExtent = 0.35f;
    constexpr float kLinkRadius = 0.22f;
    constexpr float kLinkHalfHeight = 0.46f;
    constexpr float kLinkHalfExtentX = kLinkHalfHeight + kLinkRadius;
    constexpr float kLinkSpacing = 2.0f * kLinkHalfExtentX;

    const auto anchorEntity = world.createEntity();
    RigidBodyComponent anchorBody{};
    anchorBody.bodyType = RigidBodyType::Static;
    anchorBody.inverseMass = 0.0f;
    ColliderComponent anchorCollider{};
    anchorCollider.shapeType = ColliderShapeType::Box;
    anchorCollider.shapeParams = {0.35f, 0.35f, 0.35f, 0.0f};
    anchorCollider.collisionLayer = kBallClusterLayer;
    anchorCollider.collisionMask = kBallClusterLayer | kGroundCollisionLayer | kBallDropLayer;
    setVisibleRigidBody(runtime, anchorEntity, anchorMesh, anchorMaterial,
                        {-7.0f + kOverviewAreaOffsetX, 6.0f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, anchorBody, anchorCollider);

    const float anchorPointX = -7.0f + kOverviewAreaOffsetX;
    const float anchorPointY = 6.0f - kAnchorHalfExtent;
    std::vector<cressim::neo::common::EntityId> bobs;
    for (std::uint32_t i = 0; i < 3u; ++i)
    {
        const auto entity = world.createEntity();
        RigidBodyComponent body{};
        body.inverseMass = 1.0f;
        body.inverseInertiaLocal =
            cressim::neo::examples::helpers::computeApproximateCapsuleInverseInertia(
                kLinkRadius, kLinkHalfHeight, body.inverseMass);
        if (i == 2u)
        {
            body.linearVelocity = {0.0f, 0.0f, 2.2f};
        }

        ColliderComponent collider{};
        collider.shapeType = ColliderShapeType::Capsule;
        collider.shapeParams = {kLinkRadius, kLinkHalfHeight, 0.0f, 0.0f};
        collider.collisionLayer = kBallClusterLayer;
        collider.collisionMask = kBallClusterLayer | kGroundCollisionLayer | kBallDropLayer;
        const float bobCenterX =
            anchorPointX + kLinkHalfExtentX + (kLinkSpacing * static_cast<float>(i));
        setVisibleRigidBody(runtime, entity, linkMesh, bodyMaterial,
                            {bobCenterX, anchorPointY, 0.0f},
                            {1.0f, 1.0f, 1.0f}, body, collider,
                            Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f},
                                                                         Diligent::PI_F * 0.5f));
        bobs.push_back(entity);
    }

    BallJointState root{};
    root.bodyA = requireRigidBodyId(runtime, anchorEntity);
    root.bodyB = requireRigidBodyId(runtime, bobs[0]);
    root.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
    root.localAnchorA = {0.0f, -kAnchorHalfExtent, 0.0f};
    root.localAnchorB = {0.0f, kLinkHalfExtentX, 0.0f};
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
        joint.localAnchorA = {0.0f, -kLinkHalfExtentX, 0.0f};
        joint.localAnchorB = {0.0f, kLinkHalfExtentX, 0.0f};
        if (!world.physicsWorld().upsertBallJoint(joint))
        {
            throw std::runtime_error("Failed to author chain ball joint.");
        }
    }
}

AuthoredHingeJointIds authorHingeJointCluster(Runtime &runtime, MeshHandle baseMesh,
                                              MeshHandle linkMesh,
                                              MaterialHandle anchorMaterial,
                                              MaterialHandle bodyMaterial,
                                              const ViewerJointOptions &options)
{
    auto &world = runtime.getWorld();
    AuthoredHingeJointIds jointIds{};
    constexpr float kLinkHalfX = 0.25f;
    constexpr float kLinkHalfY = 1.1f;

    const auto baseEntity = world.createEntity();
    RigidBodyComponent baseBody{};
    baseBody.bodyType = RigidBodyType::Static;
    baseBody.inverseMass = 0.0f;
    ColliderComponent baseCollider{};
    baseCollider.shapeType = ColliderShapeType::Box;
    baseCollider.shapeParams = {kHingeBaseHalfX, kHingeBaseHalfY, kHingeBaseHalfZ, 0.0f};
    baseCollider.collisionLayer = kHingeClusterLayer;
    // Let the anchored links swing through the support body instead of being blocked by the
    // static base collider, while still allowing environment/disturber contacts.
    baseCollider.collisionMask = kGroundCollisionLayer | kHingeDropLayer;
    setVisibleRigidBody(runtime, baseEntity, baseMesh, anchorMaterial,
                        {0.0f + kOverviewAreaOffsetX, 4.3f, 0.0f},
                        {2.0f * kHingeBaseHalfX, 2.0f * kHingeBaseHalfY, 2.0f * kHingeBaseHalfZ},
                        baseBody, baseCollider);

    const Diligent::QuaternionF swingRotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, -Diligent::PI_F * 0.5f);
    const Diligent::float3 topAnchorOffset =
        swingRotation.RotateVector(Diligent::float3{0.0f, kLinkHalfY, 0.0f});
    const Diligent::float3 bottomAnchorOffset =
        swingRotation.RotateVector(Diligent::float3{0.0f, -kLinkHalfY, 0.0f});
    const Diligent::float3 baseAnchor = {0.0f + kOverviewAreaOffsetX, 4.3f - kHingeBaseHalfY, 0.0f};
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
            cressim::neo::examples::helpers::computeBoxInverseInertia(
                {kLinkHalfX, kLinkHalfY, kLinkHalfX}, body.inverseMass);

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
    upper.localAnchorA = {0.0f, -kHingeBaseHalfY, 0.0f};
    upper.localAnchorB = {0.0f, 1.1f, 0.0f};
    upper.localRotationA = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    upper.localRotationB = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    upper.limitEnabled = true;
    upper.limitMin = kUpperHingeRestAngle - 0.5f * kHingeDriveRange;
    upper.limitMax = kUpperHingeRestAngle + 0.5f * kHingeDriveRange;
    upper.driveMode = options.enableVelocityDriveTargets
                          ? RigidJointDriveMode::TargetVelocity
                          : (options.enablePositionDriveTargets
                                 ? RigidJointDriveMode::TargetPosition
                                 : RigidJointDriveMode::None);
    upper.driveTargetAngle = kUpperHingeRestAngle;
    upper.driveTargetAngularVelocity = 0.0f;
    upper.driveCompliance = kExampleHingeDriveCompliance;
    upper.driveDamping = kExampleHingeDriveDamping;
    upper.driveMaxAngularVelocity = kExampleHingeDriveMaxAngularVelocity;
    if (!world.physicsWorld().upsertHingeJoint(upper))
    {
        throw std::runtime_error("Failed to author upper hinge joint.");
    }
    jointIds.upper = world.physicsWorld().hingeJointSnapshot().back().jointId;

    HingeJointState lower{};
    lower.bodyA = requireRigidBodyId(runtime, links[0]);
    lower.bodyB = requireRigidBodyId(runtime, links[1]);
    lower.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
    lower.localAnchorA = {0.0f, -1.1f, 0.0f};
    lower.localAnchorB = {0.0f, 1.1f, 0.0f};
    lower.localRotationA = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    lower.localRotationB = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    lower.limitEnabled = true;
    lower.limitMin = kLowerHingeRestAngle - 0.5f * kLowerHingeDriveRange;
    lower.limitMax = kLowerHingeRestAngle + 0.5f * kLowerHingeDriveRange;
    lower.driveMode = options.enableVelocityDriveTargets
                          ? RigidJointDriveMode::TargetVelocity
                          : (options.enablePositionDriveTargets
                                 ? RigidJointDriveMode::TargetPosition
                                 : RigidJointDriveMode::None);
    lower.driveTargetAngle = kLowerHingeRestAngle;
    lower.driveTargetAngularVelocity = 0.0f;
    lower.driveCompliance = kExampleHingeDriveCompliance;
    lower.driveDamping = kExampleHingeDriveDamping;
    lower.driveMaxAngularVelocity = kExampleHingeDriveMaxAngularVelocity;
    if (!world.physicsWorld().upsertHingeJoint(lower))
    {
        throw std::runtime_error("Failed to author lower hinge joint.");
    }
    jointIds.lower = world.physicsWorld().hingeJointSnapshot().back().jointId;
    return jointIds;
}

AuthoredSliderJointIds authorSliderJointCluster(Runtime &runtime, MeshHandle guideMesh,
                                                MeshHandle sliderMesh,
                                                MaterialHandle guideMaterial,
                                                MaterialHandle sliderMaterial,
                                                const ViewerJointOptions &options)
{
    auto &world = runtime.getWorld();
    AuthoredSliderJointIds jointIds{};
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
    setVisibleRigidBody(runtime, guideEntity, guideMesh, guideMaterial,
                        {7.0f + kOverviewAreaOffsetX, 2.0f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, guideBody, guideCollider);

    const auto sliderEntity = world.createEntity();
    RigidBodyComponent sliderBody{};
    sliderBody.inverseMass = 1.0f;
    sliderBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(
            kSliderHalfExtents, sliderBody.inverseMass);
    sliderBody.linearVelocity = {0.35f, 0.0f, 0.0f};
    sliderBody.angularVelocity = {0.0f, 0.0f, 0.0f};
    ColliderComponent sliderCollider{};
    sliderCollider.shapeType = ColliderShapeType::Box;
    sliderCollider.shapeParams = {kSliderHalfExtents.x, kSliderHalfExtents.y, kSliderHalfExtents.z, 0.0f};
    sliderCollider.collisionLayer = kSliderClusterLayer;
    sliderCollider.collisionMask = kSliderClusterLayer | kGroundCollisionLayer | kSliderDropLayer;
    setVisibleRigidBody(runtime, sliderEntity, sliderMesh, sliderMaterial,
                        {5.4f + kOverviewAreaOffsetX, 2.0f, 0.0f},
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
    horizontalSlider.driveCompliance = kExampleSliderDriveCompliance;
    horizontalSlider.driveTargetPosition = kHorizontalSliderDriveCenter;
    horizontalSlider.driveTargetVelocity = 0.0f;
    if (!world.physicsWorld().upsertSliderJoint(horizontalSlider))
    {
        throw std::runtime_error("Failed to author slider joint.");
    }
    jointIds.horizontal = world.physicsWorld().sliderJointSnapshot().back().jointId;

    const auto stageEntity = world.createEntity();
    RigidBodyComponent stageBody{};
    stageBody.inverseMass = 0.85f;
    stageBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(
            kSliderHalfExtents, stageBody.inverseMass);
    ColliderComponent stageCollider{};
    stageCollider.shapeType = ColliderShapeType::Box;
    stageCollider.shapeParams = {kSliderHalfExtents.x, kSliderHalfExtents.y, kSliderHalfExtents.z, 0.0f};
    stageCollider.collisionLayer = kSliderClusterLayer;
    stageCollider.collisionMask = kSliderClusterLayer | kGroundCollisionLayer | kSliderDropLayer;
    setVisibleRigidBody(runtime, stageEntity, sliderMesh, sliderMaterial,
                        {5.4f + kOverviewAreaOffsetX, 3.20f, 0.0f},
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
    verticalSlider.driveCompliance = kExampleSliderDriveCompliance;
    verticalSlider.driveTargetPosition = kVerticalSliderDriveCenter;
    verticalSlider.driveTargetVelocity = 0.0f;
    if (!world.physicsWorld().upsertSliderJoint(verticalSlider))
    {
        throw std::runtime_error("Failed to author vertical slider joint.");
    }
    jointIds.vertical = world.physicsWorld().sliderJointSnapshot().back().jointId;
    return jointIds;
}

AuthoredSphericalJointIds authorSphericalJointCluster(Runtime &runtime, MeshHandle baseMesh,
                                                      MeshHandle linkMesh,
                                                      MaterialHandle anchorMaterial,
                                                      MaterialHandle bodyMaterial,
                                                      const ViewerJointOptions &options)
{
    auto &world = runtime.getWorld();
    AuthoredSphericalJointIds jointIds{};
    constexpr Diligent::float3 kDiskHalfExtents = {0.75f, 0.18f, 0.75f};
    constexpr float kDiskSpacing = 1.45f;
    constexpr Diligent::float3 kClusterOrigin = {-14.0f + kOverviewAreaOffsetX, 7.2f, 0.0f};
    constexpr std::uint32_t kDiskCount = 3u;

    std::vector<cressim::neo::common::EntityId> disks;
    disks.reserve(kDiskCount);
    for (std::uint32_t i = 0u; i < kDiskCount; ++i)
    {
        const auto entity = world.createEntity();
        RigidBodyComponent body{};
        Diligent::float3 visualScale = {2.0f * kDiskHalfExtents.x, 2.0f * kDiskHalfExtents.y,
                                        2.0f * kDiskHalfExtents.z};
        if (i == 0u)
        {
            body.bodyType = RigidBodyType::Static;
            body.inverseMass = 0.0f;
            visualScale = {2.0f * kAnchorHalfExtent, 2.0f * kAnchorHalfExtent,
                           2.0f * kAnchorHalfExtent};
        }
        else
        {
            body.inverseMass = 0.85f;
            body.inverseInertiaLocal =
                cressim::neo::examples::helpers::computeBoxInverseInertia(
                    kDiskHalfExtents, body.inverseMass);
        }

        ColliderComponent collider{};
        collider.shapeType = ColliderShapeType::Box;
        collider.shapeParams = i == 0u
                                   ? Diligent::float4{kAnchorHalfExtent, kAnchorHalfExtent,
                                                      kAnchorHalfExtent, 0.0f}
                                   : Diligent::float4{kDiskHalfExtents.x, kDiskHalfExtents.y,
                                                      kDiskHalfExtents.z, 0.0f};
        collider.collisionLayer = kSphericalClusterLayer;
        collider.collisionMask = kGroundCollisionLayer | kSphericalDropLayer;
        setVisibleRigidBody(runtime, entity, i == 0u ? baseMesh : linkMesh,
                            i == 0u ? anchorMaterial : bodyMaterial,
                            kClusterOrigin - Diligent::float3{0.0f, kDiskSpacing * static_cast<float>(i), 0.0f},
                            visualScale, body, collider);
        disks.push_back(entity);
    }

    for (std::uint32_t i = 1u; i < kDiskCount; ++i)
    {
        SphericalJointState joint{};
        joint.bodyA = requireRigidBodyId(runtime, disks[i - 1u]);
        joint.bodyB = requireRigidBodyId(runtime, disks[i]);
        joint.suppressConnectedBodyCollisions = options.suppressConnectedBodyCollisions;
        joint.localAnchorA = {0.0f, -0.5f * kDiskSpacing, 0.0f};
        joint.localAnchorB = {0.0f, 0.5f * kDiskSpacing, 0.0f};
        joint.localRotationA = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
        joint.localRotationB = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
        joint.constraintCompliance = 0.0f;
        if (options.enablePositionDriveTargets || options.enableVelocityDriveTargets)
        {
            joint.limitEnabled = true;
            joint.swingLimitY = 0.65f;
            joint.swingLimitZ = 0.65f;
            joint.twistLimitMin = -0.45f;
            joint.twistLimitMax = 0.45f;
            joint.swingCompliance = 5.0e-5f;
            joint.twistCompliance = 8.0e-5f;
            joint.driveMode = RigidJointDriveMode::TargetOrientation;
            joint.driveCompliance = kExampleSphericalDriveCompliance;
            joint.driveTargetOrientation =
                Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f},
                                                             i == 1u ? 0.42f : -0.28f) *
                Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f},
                                                             i == 1u ? 0.10f : -0.08f);
        }
        else
        {
            joint.limitEnabled = false;
            joint.swingCompliance = kExampleSphericalFreeCompliance;
            joint.twistCompliance = kExampleSphericalFreeCompliance;
        }
        if (!world.physicsWorld().upsertSphericalJoint(joint))
        {
            throw std::runtime_error("Failed to author spherical joint cluster.");
        }
        jointIds.chain.push_back(world.physicsWorld().sphericalJointSnapshot().back().jointId);
    }
    return jointIds;
}

void authorDropDisturbers(Runtime &runtime, MeshHandle hingeDropMesh, MeshHandle sphereMesh,
                          MaterialHandle ballMaterial, MaterialHandle hingeMaterial,
                          MaterialHandle sliderMaterial, MaterialHandle sphericalMaterial)
{
    auto &world = runtime.getWorld();

    constexpr float kDropSphereRadius = 0.45f;
    const float kDropSphereVisualScale = kDropSphereRadius / kViewerSphereMeshRadius;

    const auto ballDrop = world.createEntity();
    RigidBodyComponent ballDropBody{};
    ballDropBody.inverseMass = 0.75f;
    ballDropBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeSphereInverseInertia(
            kDropSphereRadius, ballDropBody.inverseMass);
    ColliderComponent ballDropCollider{};
    ballDropCollider.shapeType = ColliderShapeType::Sphere;
    ballDropCollider.shapeParams = {kDropSphereRadius, 0.0f, 0.0f, 0.0f};
    ballDropCollider.collisionLayer = kBallDropLayer;
    ballDropCollider.collisionMask = kBallClusterLayer | kGroundCollisionLayer;
    setVisibleRigidBody(runtime, ballDrop, sphereMesh, ballMaterial,
                        {-8.0f + kOverviewAreaOffsetX, 8.8f, 0.2f},
                        {kDropSphereVisualScale, kDropSphereVisualScale, kDropSphereVisualScale},
                        ballDropBody, ballDropCollider);

    const auto hingeDrop = world.createEntity();
    RigidBodyComponent hingeDropBody{};
    hingeDropBody.inverseMass = 0.85f;
    hingeDropBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(
            {0.4f, 0.4f, 0.4f}, hingeDropBody.inverseMass);
    hingeDropBody.angularVelocity = {0.8f, 0.2f, -0.4f};
    ColliderComponent hingeDropCollider{};
    hingeDropCollider.shapeType = ColliderShapeType::Box;
    hingeDropCollider.shapeParams = {0.4f, 0.4f, 0.4f, 0.0f};
    hingeDropCollider.collisionLayer = kHingeDropLayer;
    hingeDropCollider.collisionMask = kHingeClusterLayer | kGroundCollisionLayer;
    setVisibleRigidBody(runtime, hingeDrop, hingeDropMesh, hingeMaterial,
                        {0.5f + kOverviewAreaOffsetX, 8.4f, 0.0f},
                        {1.0f, 1.0f, 1.0f}, hingeDropBody, hingeDropCollider,
                        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, 0.35f));

    const auto sliderDrop = world.createEntity();
    RigidBodyComponent sliderDropBody{};
    sliderDropBody.inverseMass = 0.8f;
    sliderDropBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeSphereInverseInertia(
            kDropSphereRadius, sliderDropBody.inverseMass);
    sliderDropBody.angularVelocity = {0.0f, 0.0f, 0.0f};
    ColliderComponent sliderDropCollider{};
    sliderDropCollider.shapeType = ColliderShapeType::Sphere;
    sliderDropCollider.shapeParams = {kDropSphereRadius, 0.0f, 0.0f, 0.0f};
    sliderDropCollider.collisionLayer = kSliderDropLayer;
    sliderDropCollider.collisionMask = kSliderClusterLayer | kGroundCollisionLayer;
    setVisibleRigidBody(runtime, sliderDrop, sphereMesh, sliderMaterial,
                        {6.7f + kOverviewAreaOffsetX, 6.6f, 0.0f},
                        {kDropSphereVisualScale, kDropSphereVisualScale, kDropSphereVisualScale},
                        sliderDropBody, sliderDropCollider);

    const auto sphericalDrop = world.createEntity();
    RigidBodyComponent sphericalDropBody{};
    sphericalDropBody.inverseMass = 0.72f;
    sphericalDropBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeSphereInverseInertia(
            kDropSphereRadius, sphericalDropBody.inverseMass);
    sphericalDropBody.linearVelocity = {-0.2f, 0.0f, -0.6f};
    ColliderComponent sphericalDropCollider{};
    sphericalDropCollider.shapeType = ColliderShapeType::Sphere;
    sphericalDropCollider.shapeParams = {kDropSphereRadius, 0.0f, 0.0f, 0.0f};
    sphericalDropCollider.collisionLayer = kSphericalDropLayer;
    sphericalDropCollider.collisionMask = kSphericalClusterLayer | kGroundCollisionLayer;
    setVisibleRigidBody(runtime, sphericalDrop, sphereMesh, sphericalMaterial,
                        {-12.8f + kOverviewAreaOffsetX, 9.8f, 0.3f},
                        {kDropSphereVisualScale, kDropSphereVisualScale, kDropSphereVisualScale},
                        sphericalDropBody, sphericalDropCollider);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    ViewerJointOptions jointOptions{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, false))
            {
                continue;
            }
            const std::string arg = argv[i];
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
    }
    catch (const std::invalid_argument& error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        printUsage(argv[0]);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.rendererDesc.iblQualityTier = IblQualityTier::Full;
    config.physicsDesc.substeps = 2u;
    config.physicsDesc.rigidRigidContactIterations = 16u;
    config.physicsDesc.rigidJointIterations = 48u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Rigid Joint Viewer";
    viewerDefaults.showStats = true;
    viewerDefaults.vSync = true;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
    viewerDesc.useFixedTimestep = true;
    viewerDesc.statsIntervalFrames = 60u;

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
        cameraTransform.worldTransform.position = {0.0f, 5.7f, -26.0f};
        cameraTransform.worldTransform.rotation =
            Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.18f);
        world.setTransform(cameraEntity, cameraTransform);
        CameraComponent camera{};
        camera.verticalFovDegrees = 48.0f;
        camera.backgroundMode = CameraComponent::BackgroundMode::EnvironmentCubemap;
        world.setCamera(cameraEntity, camera);

        const auto lightEntity = world.createEntity();
        DirectionalLightComponent light{};
        light.direction = Diligent::normalize(Diligent::float3{-0.4f, -1.0f, 0.25f});
        light.intensity = 4.2f;
        world.setDirectionalLight(lightEntity, light);

        auto &resources = runtime.getResources();
        if (!world.setEnvironmentIbl(0u, loadRigidJointsSkyboxIbl(resources)))
        {
            throw std::runtime_error("Failed to assign rigid joints skybox IBL.");
        }

        const MeshHandle planeMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makePlaneMesh(24.0f, "RigidJointViewer.PlaneMesh"));
        const MeshHandle sphereMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeSphereMesh(
                0.4f, 24u, 16u, "RigidJointViewer.SphereMesh"));
        const MeshHandle ballLinkMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeCapsuleMesh(
                0.22f, 0.46f, 24u, 8u, 4u, "RigidJointViewer.BallLinkCapsuleMesh"));
        const MeshHandle ballAnchorMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeBoxMesh(
                {0.35f, 0.35f, 0.35f}, "RigidJointViewer.BoxMesh"));
        const MeshHandle hingeBaseMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeCubeMesh(
                0.5f, "RigidJointViewer.UnitCubeMesh"));
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
        const MeshHandle sphericalDiskMesh =
            resources.registerMesh(cressim::neo::examples::helpers::makeCubeMesh(
                0.5f, "RigidJointViewer.SphericalUnitCubeMesh"));

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
        const MaterialHandle sphericalMaterial =
            registerMaterial(resources, "RigidJointViewer.Spherical", {0.94f, 0.56f, 0.22f}, 0.36f);
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
            kSphericalClusterLayer | kBallDropLayer | kHingeDropLayer | kSliderDropLayer |
            kSphericalDropLayer;
        world.addCollider(groundEntity, groundCollider);

        authorBallJointCluster(runtime, ballAnchorMesh, ballLinkMesh, anchorMaterial, ballMaterial,
                               jointOptions);
        const AuthoredHingeJointIds hingeJointIds =
            authorHingeJointCluster(runtime, hingeBaseMesh, hingeLinkMesh, anchorMaterial,
                                    hingeMaterial, jointOptions);
        const AuthoredSliderJointIds sliderJointIds =
            authorSliderJointCluster(runtime, sliderGuideMesh, sliderCarriageMesh, guideMaterial,
                                     sliderMaterial, jointOptions);
        const AuthoredSphericalJointIds sphericalJointIds =
            authorSphericalJointCluster(runtime, ballAnchorMesh, sphericalDiskMesh,
                                        anchorMaterial, sphericalMaterial, jointOptions);
        authorDropDisturbers(runtime, hingeDropMesh, sphereMesh, ballMaterial, hingeMaterial,
                             sliderMaterial, sphericalMaterial);

        DebugViewerCallbacks callbacks{};
        if (jointOptions.enablePositionDriveTargets || jointOptions.enableVelocityDriveTargets)
        {
            callbacks.beforeTick =
                [jointOptions, hingeJointIds, sliderJointIds, sphericalJointIds](
                    const cressim::neo::common::FrameContext &frame, Runtime &cbRuntime)
            {
                auto &physicsWorld = cbRuntime.getWorld().physicsWorld();
                const float t = static_cast<float>(frame.timeSeconds);
                const float settle = std::clamp(t / 1.5f, 0.0f, 1.0f);

                auto driveWave = [t, settle](float frequencyHz, float amplitude, float phase)
                {
                    const float angle = (2.0f * Diligent::PI_F * frequencyHz * t) + phase;
                    return amplitude * settle * std::sin(angle);
                };
                auto velocityWave = [t, settle](float frequencyHz, float amplitude, float phase)
                {
                    const float angle = (2.0f * Diligent::PI_F * frequencyHz * t) + phase;
                    return amplitude * settle * std::cos(angle);
                };

                if (auto *upper = physicsWorld.tryGetHingeJoint(hingeJointIds.upper))
                {
                    auto updated = *upper;
                    if (jointOptions.enableVelocityDriveTargets)
                    {
                        updated.driveTargetAngularVelocity =
                            velocityWave(0.08f, 0.22f, 0.0f);
                    }
                    else
                    {
                        updated.driveMode = RigidJointDriveMode::TargetPosition;
                        updated.driveTargetAngle =
                            kUpperHingeRestAngle +
                            driveWave(0.02f, kHingeDriveHalfRange, 0.0f);
                    }
                    physicsWorld.upsertHingeJoint(updated);
                }

                if (auto *lower = physicsWorld.tryGetHingeJoint(hingeJointIds.lower))
                {
                    auto updated = *lower;
                    if (jointOptions.enableVelocityDriveTargets)
                    {
                        updated.driveTargetAngularVelocity =
                            velocityWave(0.06f, 0.12f, 0.8f);
                    }
                    else
                    {
                        updated.driveMode = RigidJointDriveMode::TargetPosition;
                        updated.driveTargetAngle =
                            kLowerHingeRestAngle +
                            driveWave(0.035f, kLowerHingeDriveHalfRange, 0.8f);
                    }
                    physicsWorld.upsertHingeJoint(updated);
                }

                if (auto *horizontal = physicsWorld.tryGetSliderJoint(sliderJointIds.horizontal))
                {
                    auto updated = *horizontal;
                    if (jointOptions.enableVelocityDriveTargets)
                    {
                        updated.driveTargetVelocity = velocityWave(0.24f, 0.60f, 0.0f);
                    }
                    else
                    {
                        updated.driveTargetPosition =
                            kHorizontalSliderDriveCenter + driveWave(0.19f, 0.45f, 0.0f);
                    }
                    physicsWorld.upsertSliderJoint(updated);
                }

                if (auto *vertical = physicsWorld.tryGetSliderJoint(sliderJointIds.vertical))
                {
                    auto updated = *vertical;
                    if (jointOptions.enableVelocityDriveTargets)
                    {
                        updated.driveTargetVelocity = velocityWave(0.29f, 0.40f,
                                                                   Diligent::PI_F * 0.45f);
                    }
                    else
                    {
                        updated.driveTargetPosition =
                            kVerticalSliderDriveCenter +
                            driveWave(0.23f, 0.28f, Diligent::PI_F * 0.45f);
                    }
                    physicsWorld.upsertSliderJoint(updated);
                }

                if (jointOptions.enablePositionDriveTargets || jointOptions.enableVelocityDriveTargets)
                {
                    for (std::size_t i = 0; i < sphericalJointIds.chain.size(); ++i)
                    {
                        if (auto *joint = physicsWorld.tryGetSphericalJoint(
                                sphericalJointIds.chain[i]))
                        {
                            auto updated = *joint;
                            updated.driveMode = RigidJointDriveMode::TargetOrientation;
                            const float primary =
                                driveWave(0.17f + 0.03f * static_cast<float>(i), 0.32f,
                                          static_cast<float>(i) * Diligent::PI_F * 0.5f);
                            const float secondary =
                                driveWave(0.17f + 0.03f * static_cast<float>(i), 0.10f,
                                          static_cast<float>(i) * Diligent::PI_F * 0.5f +
                                              Diligent::PI_F * 0.25f);
                            updated.driveTargetOrientation =
                                Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f},
                                                                             primary) *
                                Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f},
                                                                             secondary);
                            physicsWorld.upsertSphericalJoint(updated);
                        }
                    }
                }
            };
        }

        const bool runOk = viewer.run(runtime, DebugViewerCameraBinding{cameraEntity}, callbacks);
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
