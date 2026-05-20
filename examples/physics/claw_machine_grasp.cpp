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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

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
using cressim::neo::graphics::MeshHandle;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::HingeJointId;
using cressim::neo::physics::HingeJointState;
using cressim::neo::physics::RigidBodyId;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::RigidJointDriveMode;
using cressim::neo::physics::SliderJointId;
using cressim::neo::physics::SliderJointState;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1.0e-6f;

constexpr std::uint32_t kWorldLayer = 1u << 0u;
constexpr std::uint32_t kRobotLayer = 1u << 1u;
constexpr std::uint32_t kObjectLayer = 1u << 2u;
constexpr float kRobotBaseZ = 1.80f;
constexpr float kXLimitMin = -1.45f;
constexpr float kXLimitMax = 1.45f;
constexpr float kZLimitMin = -1.10f;
constexpr float kZLimitMax = 1.60f;
constexpr float kYLimitMin = 0.0f;
constexpr float kYLimitMax = 1.25f;
constexpr float kLeftFingerLimitMin = 0.0f;
constexpr float kLeftFingerLimitMax = 0.42f;
constexpr float kRightFingerLimitMin = -0.42f;
constexpr float kRightFingerLimitMax = 0.0f;
constexpr float kFingerStartCenterOffsetX = 0.34f;
constexpr float kFingerHalfExtentX = 0.08f;
constexpr Diligent::float3 kPickupObjectHalfExtents{0.22f, 0.22f, 0.22f};
constexpr float kGripSqueezeMargin = 0.2f;
constexpr float kFingerJointConstraintCompliance = 1.0e-6f;
constexpr float kFingerJointDriveCompliance = 8.0e-4f;
constexpr float kFingerHingeLimit = 0.95f;
constexpr float kFingerHingePalmAnchorOffsetX = 0.18f;
constexpr float kFingerHingePalmAnchorOffsetY = -0.42f;
constexpr float kFingerHingeBodyAnchorOffsetY = 0.52f;
constexpr float kZCarriageStartY = 3.20f;
constexpr float kPalmStartY = 2.10f;
constexpr float kFingerStartY = 1.16f;

enum class FingerJointKind : std::uint8_t
{
    Slider,
    Hinge,
};

struct RobotScene
{
    EntityId xRail = cressim::neo::common::kInvalidEntityId;
    EntityId xCarriage = cressim::neo::common::kInvalidEntityId;
    EntityId zCarriage = cressim::neo::common::kInvalidEntityId;
    EntityId palm = cressim::neo::common::kInvalidEntityId;
    EntityId leftFinger = cressim::neo::common::kInvalidEntityId;
    EntityId rightFinger = cressim::neo::common::kInvalidEntityId;

    SliderJointId xJoint = cressim::neo::physics::kInvalidSliderJointId;
    SliderJointId zJoint = cressim::neo::physics::kInvalidSliderJointId;
    SliderJointId yJoint = cressim::neo::physics::kInvalidSliderJointId;
    SliderJointId leftFingerJoint = cressim::neo::physics::kInvalidSliderJointId;
    SliderJointId rightFingerJoint = cressim::neo::physics::kInvalidSliderJointId;
    HingeJointId leftFingerHinge = cressim::neo::physics::kInvalidHingeJointId;
    HingeJointId rightFingerHinge = cressim::neo::physics::kInvalidHingeJointId;
    FingerJointKind fingerJointKind = FingerJointKind::Slider;
};

struct PoseTargets
{
    float x = 0.0f;
    float z = 0.0f;
    float y = 0.0f;
    float grip = 0.0f;
};

constexpr Diligent::float3 kPickupObjectPosition{0.95f, -0.05f, 0.65f};
constexpr Diligent::float3 kDropObjectPosition{-1.15f, -0.12f, 0.20f};

Diligent::QuaternionF quaternionFromBasis(const Diligent::float3 &x, const Diligent::float3 &y,
                                          const Diligent::float3 &z)
{
    const float m00 = x.x;
    const float m01 = y.x;
    const float m02 = z.x;
    const float m10 = x.y;
    const float m11 = y.y;
    const float m12 = z.y;
    const float m20 = x.z;
    const float m21 = y.z;
    const float m22 = z.z;
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

Diligent::float3 rotateVector(const Diligent::QuaternionF &q, const Diligent::float3 &v)
{
    const Diligent::float3 qv{q.q.x, q.q.y, q.q.z};
    const Diligent::float3 t = 2.0f * Diligent::cross(qv, v);
    return v + q.q.w * t + Diligent::cross(qv, t);
}

float saturate(float x)
{
    return std::clamp(x, 0.0f, 1.0f);
}

float smoothStep(float x)
{
    const float t = saturate(x);
    return t * t * (3.0f - 2.0f * t);
}

float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

float clampToRange(float value, float minValue, float maxValue)
{
    return std::clamp(value, minValue, maxValue);
}

float computePickupGripCloseAmount()
{
    const float initialFingerGap = 2.0f * (kFingerStartCenterOffsetX - kFingerHalfExtentX);
    const float objectGripWidth = 2.0f * kPickupObjectHalfExtents.x;
    const float desiredClosedGap = std::max(objectGripWidth - (2.0f * kGripSqueezeMargin), 0.0f);
    return clampToRange(0.5f * std::max(initialFingerGap - desiredClosedGap, 0.0f),
                        kLeftFingerLimitMin, kLeftFingerLimitMax);
}

float computeFingerHingeOpenAngle()
{
    const float horizontalOffset = kFingerStartCenterOffsetX - kFingerHingePalmAnchorOffsetX;
    return std::asin(std::clamp(horizontalOffset / kFingerHingeBodyAnchorOffsetY, -1.0f, 1.0f));
}

struct FingerPose
{
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF rotation{};
};

FingerPose computeHingeFingerPose(float signedAngle)
{
    const float sideSign = signedAngle < 0.0f ? -1.0f : 1.0f;
    const Diligent::float3 anchorWorld{sideSign * kFingerHingePalmAnchorOffsetX,
                                       kPalmStartY + kFingerHingePalmAnchorOffsetY,
                                       kRobotBaseZ};
    const Diligent::QuaternionF rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 0.0f, 1.0f}, signedAngle);
    const Diligent::float3 localAnchorB{0.0f, kFingerHingeBodyAnchorOffsetY, 0.0f};
    const Diligent::float3 center = anchorWorld - rotateVector(rotation, localAnchorB);
    return FingerPose{center, rotation};
}

float evalSegment(float time, float startTime, float endTime, float startValue, float endValue)
{
    if (time <= startTime)
    {
        return startValue;
    }
    if (time >= endTime)
    {
        return endValue;
    }
    return lerp(startValue, endValue, smoothStep((time - startTime) / (endTime - startTime)));
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

void setVisibleRigidBody(Runtime &runtime, EntityId entityId, MeshHandle mesh, MaterialHandle material,
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

RigidBodyId requireRigidBodyId(Runtime &runtime, EntityId entityId)
{
    const auto *body = runtime.getWorld().physicsWorld().tryGetRigidBody(entityId);
    if (body == nullptr)
    {
        throw std::runtime_error("Missing rigid body while authoring claw grasp example.");
    }
    return body->rigidBodyId;
}

SliderJointId authorDrivenSlider(Runtime &runtime, RigidBodyId bodyA, RigidBodyId bodyB,
                                 const Diligent::float3 &axis, const Diligent::float3 &anchorA,
                                 const Diligent::float3 &anchorB, float limitMin, float limitMax,
                                 float targetPosition, float targetVelocity,
                                 bool suppressConnectedCollisions = true,
                                 float constraintCompliance = 0.0f,
                                 float driveCompliance = 0.0f)
{
    SliderJointState joint{};
    joint.bodyA = bodyA;
    joint.bodyB = bodyB;
    joint.suppressConnectedBodyCollisions = suppressConnectedCollisions;
    joint.localAnchorA = anchorA;
    joint.localAnchorB = anchorB;
    joint.localRotationA = makeJointFrameRotation(axis);
    joint.localRotationB = makeJointFrameRotation(axis);
    joint.limitEnabled = true;
    joint.limitMin = limitMin;
    joint.limitMax = limitMax;
    joint.constraintCompliance = constraintCompliance;
    joint.driveCompliance = driveCompliance;
    joint.driveMode = RigidJointDriveMode::None;
    joint.driveTargetPosition = targetPosition;
    joint.driveTargetVelocity = targetVelocity;
    if (!runtime.getWorld().physicsWorld().upsertSliderJoint(joint))
    {
        throw std::runtime_error("Failed to author driven slider joint.");
    }

    const auto &snapshot = runtime.getWorld().physicsWorld().sliderJointSnapshot();
    if (snapshot.empty())
    {
        throw std::runtime_error("Slider joint authored but missing from snapshot.");
    }
    return snapshot.back().jointId;
}

HingeJointId authorDrivenHinge(Runtime &runtime, RigidBodyId bodyA, RigidBodyId bodyB,
                               const Diligent::float3 &axis, const Diligent::float3 &anchorA,
                               const Diligent::float3 &anchorB, float limitMin, float limitMax,
                               float targetAngle, float targetAngularVelocity,
                               bool suppressConnectedCollisions = true,
                               float constraintCompliance = 0.0f,
                               float driveCompliance = 0.0f)
{
    HingeJointState joint{};
    joint.bodyA = bodyA;
    joint.bodyB = bodyB;
    joint.suppressConnectedBodyCollisions = suppressConnectedCollisions;
    joint.localAnchorA = anchorA;
    joint.localAnchorB = anchorB;
    joint.localRotationA = makeJointFrameRotation(axis);
    joint.localRotationB = makeJointFrameRotation(axis);
    joint.limitEnabled = true;
    joint.limitMin = limitMin;
    joint.limitMax = limitMax;
    joint.constraintCompliance = constraintCompliance;
    joint.driveCompliance = driveCompliance;
    joint.driveMode = RigidJointDriveMode::None;
    joint.driveTargetAngle = targetAngle;
    joint.driveTargetAngularVelocity = targetAngularVelocity;
    if (!runtime.getWorld().physicsWorld().upsertHingeJoint(joint))
    {
        throw std::runtime_error("Failed to author driven hinge joint.");
    }

    const auto &snapshot = runtime.getWorld().physicsWorld().hingeJointSnapshot();
    if (snapshot.empty())
    {
        throw std::runtime_error("Hinge joint authored but missing from snapshot.");
    }
    return snapshot.back().jointId;
}

void setJointTarget(Runtime &runtime, SliderJointId jointId, float targetPosition)
{
    const SliderJointState *existing = runtime.getWorld().physicsWorld().tryGetSliderJoint(jointId);
    if (existing == nullptr)
    {
        return;
    }

    SliderJointState updated = *existing;
    updated.driveMode = RigidJointDriveMode::TargetPosition;
    updated.driveTargetPosition = targetPosition;
    runtime.getWorld().physicsWorld().upsertSliderJoint(updated);
}

void setHingeJointTarget(Runtime &runtime, HingeJointId jointId, float targetAngle)
{
    const HingeJointState *existing = runtime.getWorld().physicsWorld().tryGetHingeJoint(jointId);
    if (existing == nullptr)
    {
        return;
    }

    HingeJointState updated = *existing;
    updated.driveMode = RigidJointDriveMode::TargetPosition;
    updated.driveTargetAngle = targetAngle;
    runtime.getWorld().physicsWorld().upsertHingeJoint(updated);
}

float gripToFingerHingeAngle(float gripCloseAmount)
{
    const float maxGripCloseAmount = std::max(computePickupGripCloseAmount(), kEpsilon);
    const float closeT = saturate(gripCloseAmount / maxGripCloseAmount);
    return std::clamp(lerp(computeFingerHingeOpenAngle(), 0.0f, closeT), 0.0f, kFingerHingeLimit);
}

PoseTargets evaluateTargets(double timeSeconds)
{
    constexpr float kPeriod = 10.0f;
    const float t = std::fmod(static_cast<float>(timeSeconds), kPeriod);
    const float pickupTargetX = clampToRange(kPickupObjectPosition.x, kXLimitMin, kXLimitMax);
    const float dropTargetX = clampToRange(kDropObjectPosition.x, kXLimitMin, kXLimitMax);
    const float pickupTargetZ =
        clampToRange(kPickupObjectPosition.z - kRobotBaseZ, kZLimitMin, kZLimitMax);
    const float dropTargetZ =
        clampToRange(kDropObjectPosition.z - kRobotBaseZ, kZLimitMin, kZLimitMax);
    const float descendTargetY = clampToRange(1.05f, kYLimitMin, kYLimitMax);
    const float liftTargetY = clampToRange(0.10f, kYLimitMin, kYLimitMax);
    const float releaseTargetY = clampToRange(0.85f, kYLimitMin, kYLimitMax);
    const float gripCloseAmount = computePickupGripCloseAmount();
    const PoseTargets kHome{0.0f, 0.0f, kYLimitMin, 0.0f};
    const PoseTargets kApproach{pickupTargetX, pickupTargetZ, kYLimitMin, 0.0f};
    const PoseTargets kDescend{pickupTargetX, pickupTargetZ, descendTargetY, 0.0f};
    const PoseTargets kGrip{pickupTargetX, pickupTargetZ, descendTargetY, gripCloseAmount};
    const PoseTargets kLift{pickupTargetX, pickupTargetZ, liftTargetY, gripCloseAmount};
    const PoseTargets kCarry{dropTargetX, dropTargetZ, liftTargetY, gripCloseAmount};
    const PoseTargets kRelease{dropTargetX, dropTargetZ, releaseTargetY, 0.0f};

    PoseTargets out = kHome;
    if (t < 1.5f)
    {
        out.x = evalSegment(t, 0.0f, 1.5f, kHome.x, kApproach.x);
        out.z = evalSegment(t, 0.0f, 1.5f, kHome.z, kApproach.z);
        out.y = kHome.y;
        out.grip = kHome.grip;
    }
    else if (t < 3.0f)
    {
        out = kApproach;
        out.y = evalSegment(t, 1.5f, 3.0f, kApproach.y, kDescend.y);
    }
    else if (t < 4.0f)
    {
        out = kDescend;
        out.grip = evalSegment(t, 3.0f, 4.0f, kDescend.grip, kGrip.grip);
    }
    else if (t < 5.3f)
    {
        out = kGrip;
        out.y = evalSegment(t, 4.0f, 5.3f, kGrip.y, kLift.y);
    }
    else if (t < 7.2f)
    {
        out = kLift;
        out.x = evalSegment(t, 5.3f, 7.2f, kLift.x, kCarry.x);
        out.z = evalSegment(t, 5.3f, 7.2f, kLift.z, kCarry.z);
    }
    else if (t < 8.2f)
    {
        out = kCarry;
        out.y = evalSegment(t, 7.2f, 8.2f, kCarry.y, kRelease.y);
    }
    else if (t < 9.1f)
    {
        out = kRelease;
    }
    else
    {
        out = kRelease;
        out.x = evalSegment(t, 9.1f, 10.0f, kRelease.x, kHome.x);
        out.z = evalSegment(t, 9.1f, 10.0f, kRelease.z, kHome.z);
        out.y = evalSegment(t, 9.1f, 10.0f, kRelease.y, kHome.y);
        out.grip = kRelease.grip;
    }

    return out;
}

void authorGround(Runtime &runtime, MeshHandle planeMesh, MaterialHandle groundMaterial)
{
    auto &world = runtime.getWorld();
    const auto groundEntity = world.createEntity();

    TransformComponent transform{};
    transform.worldTransform.position = {0.0f, -0.65f, 0.0f};
    world.setTransform(groundEntity, transform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{planeMesh, groundMaterial, true});

    RigidBodyComponent body{};
    body.bodyType = RigidBodyType::Static;
    body.inverseMass = 0.0f;
    body.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, body);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Box;
    collider.shapeParams = {9.0f, 0.10f, 9.0f, 0.0f};
    collider.localPosition = {0.0f, -collider.shapeParams.y, 0.0f};
    collider.friction = 0.9f;
    collider.staticFriction = 1.1f;
    collider.collisionLayer = kWorldLayer;
    collider.collisionMask = kRobotLayer | kObjectLayer;
    world.addCollider(groundEntity, collider);
}

RobotScene authorRobot(Runtime &runtime, MeshHandle beamMesh, MeshHandle carriageMesh,
                       MeshHandle palmMesh, MeshHandle fingerMesh, MaterialHandle railMaterial,
                       MaterialHandle carriageMaterial, MaterialHandle fingerMaterial,
                       FingerJointKind fingerJointKind)
{
    auto &world = runtime.getWorld();
    RobotScene robot{};

    constexpr Diligent::float3 kRailHalfExtents = {2.6f, 0.18f, 0.18f};
    constexpr Diligent::float3 kZCarriageHalfExtents = {0.55f, 0.22f, 0.40f};
    constexpr Diligent::float3 kPalmHalfExtents = {0.28f, 0.18f, 0.28f};
    constexpr Diligent::float3 kFingerHalfExtents = {0.08f, 0.52f, 0.14f};

    robot.xRail = world.createEntity();
    RigidBodyComponent railBody{};
    railBody.bodyType = RigidBodyType::Static;
    railBody.inverseMass = 0.0f;
    railBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    ColliderComponent railCollider{};
    railCollider.shapeType = ColliderShapeType::Box;
    railCollider.shapeParams = {kRailHalfExtents.x, kRailHalfExtents.y, kRailHalfExtents.z, 0.0f};
    railCollider.friction = 0.7f;
    railCollider.staticFriction = 0.9f;
    railCollider.collisionLayer = kRobotLayer;
    railCollider.collisionMask = kWorldLayer | kObjectLayer;
    setVisibleRigidBody(runtime, robot.xRail, beamMesh, railMaterial, {0.0f, kZCarriageStartY, kRobotBaseZ},
                        {1.0f, 1.0f, 1.0f}, railBody, railCollider);

    robot.xCarriage = world.createEntity();
    RigidBodyComponent xBody{};
    xBody.inverseMass = 0.35f;
    xBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(kZCarriageHalfExtents,
                                                                  xBody.inverseMass);
    ColliderComponent xCollider{};
    xCollider.shapeType = ColliderShapeType::Box;
    xCollider.shapeParams = {kZCarriageHalfExtents.x, kZCarriageHalfExtents.y,
                             kZCarriageHalfExtents.z, 0.0f};
    xCollider.friction = 0.75f;
    xCollider.staticFriction = 0.9f;
    xCollider.collisionLayer = kRobotLayer;
    xCollider.collisionMask = kWorldLayer | kObjectLayer;
    setVisibleRigidBody(runtime, robot.xCarriage, carriageMesh, carriageMaterial,
                        {0.0f, kZCarriageStartY, kRobotBaseZ}, {1.0f, 1.0f, 1.0f}, xBody, xCollider);

    robot.zCarriage = world.createEntity();
    RigidBodyComponent zBody{};
    zBody.inverseMass = 0.45f;
    zBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(kZCarriageHalfExtents,
                                                                  zBody.inverseMass);
    ColliderComponent zCollider = xCollider;
    setVisibleRigidBody(runtime, robot.zCarriage, carriageMesh, carriageMaterial,
                        {0.0f, kZCarriageStartY, kRobotBaseZ}, {1.0f, 1.0f, 1.0f}, zBody, zCollider);

    robot.palm = world.createEntity();
    RigidBodyComponent palmBody{};
    palmBody.inverseMass = 0.7f;
    palmBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(kPalmHalfExtents,
                                                                  palmBody.inverseMass);
    ColliderComponent palmCollider{};
    palmCollider.shapeType = ColliderShapeType::Box;
    palmCollider.shapeParams = {kPalmHalfExtents.x, kPalmHalfExtents.y, kPalmHalfExtents.z, 0.0f};
    palmCollider.friction = 0.8f;
    palmCollider.staticFriction = 1.0f;
    palmCollider.collisionLayer = kRobotLayer;
    palmCollider.collisionMask = kWorldLayer | kObjectLayer;
    setVisibleRigidBody(runtime, robot.palm, palmMesh, carriageMaterial,
                        {0.0f, kPalmStartY, kRobotBaseZ}, {1.0f, 1.0f, 1.0f}, palmBody, palmCollider);

    robot.leftFinger = world.createEntity();
    RigidBodyComponent leftBody{};
    leftBody.inverseMass = 0.8f;
    leftBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(kFingerHalfExtents,
                                                                  leftBody.inverseMass);
    ColliderComponent fingerCollider{};
    fingerCollider.shapeType = ColliderShapeType::Box;
    fingerCollider.shapeParams = {kFingerHalfExtents.x, kFingerHalfExtents.y,
                                  kFingerHalfExtents.z, 0.0f};
    fingerCollider.friction = 0.7f;
    fingerCollider.staticFriction = 0.85f;
    fingerCollider.collisionLayer = kRobotLayer;
    fingerCollider.collisionMask = kWorldLayer | kObjectLayer;
    const float hingeOpenAngle = computeFingerHingeOpenAngle();
    const FingerPose leftFingerPose =
        fingerJointKind == FingerJointKind::Hinge
            ? computeHingeFingerPose(-hingeOpenAngle)
            : FingerPose{{-kFingerStartCenterOffsetX, kFingerStartY, kRobotBaseZ},
                         Diligent::QuaternionF{}};
    setVisibleRigidBody(runtime, robot.leftFinger, fingerMesh, fingerMaterial,
                        leftFingerPose.position, {1.0f, 1.0f, 1.0f}, leftBody, fingerCollider,
                        leftFingerPose.rotation);

    robot.rightFinger = world.createEntity();
    RigidBodyComponent rightBody = leftBody;
    ColliderComponent rightCollider = fingerCollider;
    const FingerPose rightFingerPose =
        fingerJointKind == FingerJointKind::Hinge
            ? computeHingeFingerPose(hingeOpenAngle)
            : FingerPose{{kFingerStartCenterOffsetX, kFingerStartY, kRobotBaseZ},
                         Diligent::QuaternionF{}};
    setVisibleRigidBody(runtime, robot.rightFinger, fingerMesh, fingerMaterial,
                        rightFingerPose.position, {1.0f, 1.0f, 1.0f}, rightBody, rightCollider,
                        rightFingerPose.rotation);
    robot.fingerJointKind = fingerJointKind;

    const RigidBodyId railId = requireRigidBodyId(runtime, robot.xRail);
    const RigidBodyId xId = requireRigidBodyId(runtime, robot.xCarriage);
    const RigidBodyId zId = requireRigidBodyId(runtime, robot.zCarriage);
    const RigidBodyId palmId = requireRigidBodyId(runtime, robot.palm);
    const RigidBodyId leftId = requireRigidBodyId(runtime, robot.leftFinger);
    const RigidBodyId rightId = requireRigidBodyId(runtime, robot.rightFinger);

    robot.xJoint = authorDrivenSlider(runtime, railId, xId, {1.0f, 0.0f, 0.0f},
                                      {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
                                      kXLimitMin, kXLimitMax, 0.0f, 2.0f);
    robot.zJoint = authorDrivenSlider(runtime, xId, zId, {0.0f, 0.0f, 1.0f},
                                      {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
                                      kZLimitMin, kZLimitMax, 0.0f, 1.8f);
    robot.yJoint = authorDrivenSlider(runtime, zId, palmId, {0.0f, -1.0f, 0.0f},
                                      {0.0f, -0.95f, 0.0f}, {0.0f, 0.15f, 0.0f},
                                      kYLimitMin, kYLimitMax, 0.0f, 1.5f);
    if (fingerJointKind == FingerJointKind::Hinge)
    {
        robot.leftFingerHinge = authorDrivenHinge(runtime, palmId, leftId, {0.0f, 0.0f, 1.0f},
                                                  {-kFingerHingePalmAnchorOffsetX,
                                                   kFingerHingePalmAnchorOffsetY, 0.0f},
                                                  {0.0f, kFingerHingeBodyAnchorOffsetY, 0.0f},
                                                  -kFingerHingeLimit, 0.0f, -hingeOpenAngle, 1.2f,
                                                  true,
                                                  kFingerJointConstraintCompliance,
                                                  kFingerJointDriveCompliance);
        robot.rightFingerHinge = authorDrivenHinge(runtime, palmId, rightId, {0.0f, 0.0f, 1.0f},
                                                   {kFingerHingePalmAnchorOffsetX,
                                                    kFingerHingePalmAnchorOffsetY, 0.0f},
                                                   {0.0f, kFingerHingeBodyAnchorOffsetY, 0.0f},
                                                   0.0f, kFingerHingeLimit, hingeOpenAngle, 1.2f,
                                                   true,
                                                   kFingerJointConstraintCompliance,
                                                   kFingerJointDriveCompliance);
    }
    else
    {
        robot.leftFingerJoint = authorDrivenSlider(runtime, palmId, leftId, {1.0f, 0.0f, 0.0f},
                                                   {-0.18f, -0.42f, 0.0f}, {0.0f, 0.52f, 0.0f},
                                                   kLeftFingerLimitMin, kLeftFingerLimitMax, 0.0f,
                                                   1.2f, true, kFingerJointConstraintCompliance,
                                                   kFingerJointDriveCompliance);
        robot.rightFingerJoint = authorDrivenSlider(runtime, palmId, rightId, {1.0f, 0.0f, 0.0f},
                                                    {0.18f, -0.42f, 0.0f}, {0.0f, 0.52f, 0.0f},
                                                    kRightFingerLimitMin, kRightFingerLimitMax, 0.0f,
                                                    1.2f, true, kFingerJointConstraintCompliance,
                                                    kFingerJointDriveCompliance);
    }
    return robot;
}

void authorObject(Runtime &runtime, EntityId entity, MeshHandle mesh, MaterialHandle material,
                  const Diligent::float3 &position, const Diligent::float3 &halfExtents,
                  float inverseMass)
{
    RigidBodyComponent body{};
    body.inverseMass = inverseMass;
    body.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(halfExtents, inverseMass);

    ColliderComponent collider{};
    collider.shapeType = ColliderShapeType::Box;
    collider.shapeParams = {halfExtents.x, halfExtents.y, halfExtents.z, 0.0f};
    collider.friction = 0.85f;
    collider.staticFriction = 1.05f;
    collider.restitution = 0.02f;
    collider.collisionLayer = kObjectLayer;
    collider.collisionMask = kWorldLayer | kRobotLayer | kObjectLayer;

    setVisibleRigidBody(runtime, entity, mesh, material, position, {1.0f, 1.0f, 1.0f}, body, collider);
}

void authorProps(Runtime &runtime, MeshHandle cubeMesh, MeshHandle clutterMesh, MeshHandle tallMesh,
                 MaterialHandle graspMaterial, MaterialHandle clutterMaterial)
{
    auto &world = runtime.getWorld();

    authorObject(runtime, world.createEntity(), cubeMesh, graspMaterial, {0.95f, -0.05f, 0.65f},
                 kPickupObjectHalfExtents, 1.6f);
    authorObject(runtime, world.createEntity(), clutterMesh, clutterMaterial,
                 {-0.60f, -0.12f, 0.20f}, {0.28f, 0.15f, 0.28f}, 1.1f);
    authorObject(runtime, world.createEntity(), tallMesh, clutterMaterial,
                 {0.10f, 0.10f, -0.85f}, {0.18f, 0.32f, 0.18f}, 1.3f);
}

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, " [--hinge-fingers]", false);
    std::fprintf(stdout, "  --hinge-fingers  Use hinge joints for the two claw fingers.\n");
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};
    FingerJointKind fingerJointKind = FingerJointKind::Slider;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(argc, argv, i, options, false))
            {
                continue;
            }

            if (std::string_view{argv[i]} == "--hinge-fingers")
            {
                fingerJointKind = FingerJointKind::Hinge;
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
    config.physicsDesc.defaultIterations = 50;

    DebugViewerApp viewer;
    ViewerExampleDefaults defaults{};
    defaults.windowTitle = "CRESSim Neo Claw Machine Grasp";
    defaults.width = 1400u;
    defaults.height = 900u;
    const auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options, defaults);

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Claw grasp viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Claw grasp runtime initialization failed.\n");
        return 1;
    }

    auto &world = runtime.getWorld();
    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.8f, -7.4f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, -30.0f * kPi);
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.35f, -1.0f, 0.25f};
    light.color = {1.0f, 0.98f, 0.94f};
    light.intensity = 7.5f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const MeshHandle planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(9.0f, "ClawGrasp.Ground", 6.0f));
    const MeshHandle beamMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeBoxMesh({2.6f, 0.18f, 0.18f}, "ClawGrasp.Beam"));
    const MeshHandle carriageMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeBoxMesh({0.55f, 0.22f, 0.40f}, "ClawGrasp.Carriage"));
    const MeshHandle palmMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeBoxMesh({0.28f, 0.18f, 0.28f}, "ClawGrasp.Palm"));
    const MeshHandle fingerMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeBoxMesh({0.08f, 0.52f, 0.14f}, "ClawGrasp.Finger"));
    const MeshHandle cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeBoxMesh({0.22f, 0.22f, 0.22f}, "ClawGrasp.Cube"));
    const MeshHandle clutterMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeBoxMesh({0.28f, 0.15f, 0.28f}, "ClawGrasp.ClutterBox"));
    const MeshHandle tallMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeBoxMesh({0.18f, 0.32f, 0.18f}, "ClawGrasp.TallBox"));

    const MaterialHandle groundMaterial =
        registerMaterial(resources, "ClawGrasp.GroundMaterial", {0.72f, 0.74f, 0.78f}, 0.92f);
    const MaterialHandle railMaterial =
        registerMaterial(resources, "ClawGrasp.RailMaterial", {0.22f, 0.28f, 0.35f}, 0.45f, 0.15f);
    const MaterialHandle carriageMaterial =
        registerMaterial(resources, "ClawGrasp.CarriageMaterial", {0.80f, 0.52f, 0.18f}, 0.38f, 0.05f);
    const MaterialHandle graspMaterial =
        registerMaterial(resources, "ClawGrasp.GraspMaterial", {0.10f, 0.60f, 0.90f}, 0.35f);
    const MaterialHandle fingerMaterial =
        registerMaterial(resources, "ClawGrasp.FingerMaterial", {0.84f, 0.86f, 0.89f}, 0.22f,
                         0.18f);
    const MaterialHandle clutterMaterial =
        registerMaterial(resources, "ClawGrasp.ClutterMaterial", {0.84f, 0.25f, 0.18f}, 0.48f);

    try
    {
        authorGround(runtime, planeMesh, groundMaterial);
        const RobotScene robot =
            authorRobot(runtime, beamMesh, carriageMesh, palmMesh, fingerMesh, railMaterial,
                        carriageMaterial, fingerMaterial, fingerJointKind);
        authorProps(runtime, cubeMesh, clutterMesh, tallMesh, graspMaterial, clutterMaterial);

        DebugViewerCallbacks callbacks{};
        callbacks.beforeTick = [robot](const FrameContext &frame, Runtime &cbRuntime) {
            const PoseTargets targets = evaluateTargets(frame.timeSeconds);
            setJointTarget(cbRuntime, robot.xJoint, targets.x);
            setJointTarget(cbRuntime, robot.zJoint, targets.z);
            setJointTarget(cbRuntime, robot.yJoint, targets.y);
            if (robot.fingerJointKind == FingerJointKind::Hinge)
            {
                const float gripAngle = gripToFingerHingeAngle(targets.grip);
                setHingeJointTarget(cbRuntime, robot.leftFingerHinge, -gripAngle);
                setHingeJointTarget(cbRuntime, robot.rightFingerHinge, gripAngle);
            }
            else
            {
                setJointTarget(cbRuntime, robot.leftFingerJoint, targets.grip);
                setJointTarget(cbRuntime, robot.rightFingerJoint, -targets.grip);
            }
        };

        DebugViewerCameraBinding binding{};
        binding.cameraEntity = cameraEntity;
        const bool runOk = viewer.run(runtime, binding, callbacks);

        runtime.shutdown();
        viewer.shutdown();

        if (!runOk)
        {
            CRESSIM_LOG_ERROR("Claw grasp viewer run failed.\n");
            return 1;
        }
    }
    catch (const std::runtime_error &error)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Claw grasp scene setup failed: ", error.what(), "\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Claw grasp example finished. Frames=", viewerDesc.maxFrames, "\n");
    return 0;
}
