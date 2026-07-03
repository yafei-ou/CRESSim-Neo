#include "physics/physics_world.h"
#include "common/logger.h"

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::BallJointState;
using cressim::neo::physics::HingeJointState;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::RigidJointDriveMode;
using cressim::neo::physics::RigidBodyState;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SliderJointState;
using cressim::neo::physics::SphericalJointState;

constexpr float kEpsilon = 1.0e-6f;

RigidBodyState makeBody(EntityId entityId, std::uint32_t env, float x)
{
    RigidBodyState state{};
    state.entityId = entityId;
    state.environmentIndex = env;
    state.position = {x, 0.0f, 0.0f};
    state.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    state.scale = {1.0f, 1.0f, 1.0f};
    state.inverseMass = 1.0f;
    state.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    state.bodyType = RigidBodyType::Dynamic;
    return state;
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
    y = yLengthSq <= kEpsilon ? Diligent::float3{0.0f, 1.0f, 0.0f} : y * (1.0f / std::sqrt(yLengthSq));
    Diligent::float3 z = Diligent::cross(x, y);
    const float zLengthSq = Diligent::dot(z, z);
    z = zLengthSq <= kEpsilon ? Diligent::float3{0.0f, 0.0f, 1.0f} : z * (1.0f / std::sqrt(zLengthSq));
    return quaternionFromBasis(x, y, z);
}

} // namespace

int main()
{
    PhysicsWorld world;

    world.upsertRigidBody(makeBody(1001u, 0u, 0.0f));
    world.upsertRigidBody(makeBody(1002u, 0u, 1.0f));
    world.upsertRigidBody(makeBody(1003u, 1u, 2.0f));

    const auto *bodyA = world.tryGetRigidBody(1001u);
    const auto *bodyB = world.tryGetRigidBody(1002u);
    const auto *bodyC = world.tryGetRigidBody(1003u);
    if (bodyA == nullptr || bodyB == nullptr || bodyC == nullptr)
    {
        CRESSIM_LOG_ERROR("Failed to create rigid bodies for joint test.");
        return 1;
    }

    BallJointState ball{};
    ball.bodyA = bodyA->rigidBodyId;
    ball.bodyB = bodyB->rigidBodyId;
    ball.suppressConnectedBodyCollisions = true;
    ball.localAnchorA = {0.25f, 0.0f, 0.0f};
    ball.localAnchorB = {-0.25f, 0.0f, 0.0f};
    if (!world.upsertBallJoint(ball))
    {
        CRESSIM_LOG_ERROR("Failed to insert ball joint.");
        return 1;
    }
    ball.jointId = world.ballJointSnapshot().back().jointId;

    SphericalJointState spherical{};
    spherical.bodyA = bodyA->rigidBodyId;
    spherical.bodyB = bodyB->rigidBodyId;
    spherical.suppressConnectedBodyCollisions = true;
    spherical.localRotationA = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
    spherical.localRotationB = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
    spherical.limitEnabled = true;
    spherical.swingLimitY = 0.2f;
    spherical.swingLimitZ = 0.3f;
    spherical.twistLimitMin = -0.4f;
    spherical.twistLimitMax = 0.5f;
    spherical.constraintCompliance = 0.01f;
    spherical.swingCompliance = 0.02f;
    spherical.twistCompliance = 0.03f;
    spherical.driveMode = RigidJointDriveMode::TargetOrientation;
    spherical.driveCompliance = 0.04f;
    spherical.driveTargetOrientation = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    if (!world.upsertSphericalJoint(spherical))
    {
        CRESSIM_LOG_ERROR("Failed to insert spherical joint.");
        return 1;
    }
    spherical.jointId = world.sphericalJointSnapshot().back().jointId;

    HingeJointState hinge{};
    hinge.bodyA = bodyA->rigidBodyId;
    hinge.bodyB = bodyB->rigidBodyId;
    hinge.localAnchorA = {0.0f, 0.0f, 0.0f};
    hinge.localAnchorB = {0.0f, 0.0f, 0.0f};
    hinge.localRotationA = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
    hinge.localRotationB = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
    hinge.limitEnabled = true;
    hinge.limitMin = -0.25f;
    hinge.limitMax = 0.75f;
    hinge.driveMode = RigidJointDriveMode::TargetPosition;
    hinge.driveTargetAngle = 6.0f * Diligent::PI_F;
    hinge.driveDamping = 0.35f;
    hinge.driveMaxAngularVelocity = 4.5f;
    hinge.driveTargetAngularVelocity = 1.25f;
    if (!world.upsertHingeJoint(hinge))
    {
        CRESSIM_LOG_ERROR("Failed to insert hinge joint.");
        return 1;
    }
    hinge.jointId = world.hingeJointSnapshot().back().jointId;

    SliderJointState slider{};
    slider.bodyA = bodyA->rigidBodyId;
    slider.bodyB = bodyB->rigidBodyId;
    slider.localRotationA = makeJointFrameRotation({1.0f, 0.0f, 0.0f});
    slider.localRotationB = makeJointFrameRotation({1.0f, 0.0f, 0.0f});
    slider.limitEnabled = true;
    slider.limitMin = -0.4f;
    slider.limitMax = 1.1f;
    slider.driveMode = RigidJointDriveMode::TargetPosition;
    slider.driveDamping = 0.45f;
    slider.driveMaxVelocity = 1.8f;
    slider.driveTargetPosition = 0.75f;
    slider.driveTargetVelocity = 0.6f;
    if (!world.upsertSliderJoint(slider))
    {
        CRESSIM_LOG_ERROR("Failed to insert slider joint.");
        return 1;
    }
    slider.jointId = world.sliderJointSnapshot().back().jointId;

    BallJointState crossEnvBall{};
    crossEnvBall.bodyA = bodyA->rigidBodyId;
    crossEnvBall.bodyB = bodyC->rigidBodyId;
    if (world.upsertBallJoint(crossEnvBall))
    {
        CRESSIM_LOG_ERROR("Cross-environment ball joint should be rejected.");
        return 1;
    }
    SphericalJointState crossEnvSpherical{};
    crossEnvSpherical.bodyA = bodyA->rigidBodyId;
    crossEnvSpherical.bodyB = bodyC->rigidBodyId;
    if (world.upsertSphericalJoint(crossEnvSpherical))
    {
        CRESSIM_LOG_ERROR("Cross-environment spherical joint should be rejected.");
        return 1;
    }

    const auto &scene = world.rigidJointScene();
    const auto &suppression = world.jointCollisionSuppression();
    if (scene.ball.size() != 1u || scene.spherical.size() != 1u || scene.hinge.size() != 1u ||
        scene.slider.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Unexpected rigid joint scene counts.");
        return 1;
    }

    if (suppression.neighborOffsets.size() != 4u || suppression.neighbors.size() != 2u ||
        suppression.neighborOffsets[0] != 0u || suppression.neighborOffsets[1] != 1u ||
        suppression.neighborOffsets[2] != 2u || suppression.neighborOffsets[3] != 2u ||
        suppression.neighbors[0] != 1u || suppression.neighbors[1] != 0u)
    {
        CRESSIM_LOG_ERROR("Joint collision suppression adjacency was not rebuilt correctly.");
        return 1;
    }

    ball.enabled = false;
    if (!world.upsertBallJoint(ball))
    {
        CRESSIM_LOG_ERROR("Failed to disable ball joint.");
        return 1;
    }
    spherical.enabled = false;
    if (!world.upsertSphericalJoint(spherical))
    {
        CRESSIM_LOG_ERROR("Failed to disable spherical joint.");
        return 1;
    }

    const auto &disabledSuppression = world.jointCollisionSuppression();
    if (disabledSuppression.neighborOffsets.size() != 4u ||
        disabledSuppression.neighbors.size() != 0u ||
        disabledSuppression.neighborOffsets[0] != 0u ||
        disabledSuppression.neighborOffsets[1] != 0u ||
        disabledSuppression.neighborOffsets[2] != 0u ||
        disabledSuppression.neighborOffsets[3] != 0u)
    {
        CRESSIM_LOG_ERROR("Disabled joints should not suppress connected-body collisions.");
        return 1;
    }

    ball.enabled = true;
    if (!world.upsertBallJoint(ball))
    {
        CRESSIM_LOG_ERROR("Failed to re-enable ball joint.");
        return 1;
    }
    spherical.enabled = true;
    if (!world.upsertSphericalJoint(spherical))
    {
        CRESSIM_LOG_ERROR("Failed to re-enable spherical joint.");
        return 1;
    }

    if (scene.ball.bodyIndicesA.front() != 0u || scene.ball.bodyIndicesB.front() != 1u ||
        scene.spherical.bodyIndicesA.front() != 0u || scene.spherical.bodyIndicesB.front() != 1u ||
        scene.hinge.bodyIndicesA.front() != 0u || scene.hinge.bodyIndicesB.front() != 1u ||
        scene.slider.bodyIndicesA.front() != 0u || scene.slider.bodyIndicesB.front() != 1u)
    {
        CRESSIM_LOG_ERROR("Rigid joint scene body indices were not rebuilt correctly.");
        return 1;
    }

    if (scene.spherical.driveModes.front() !=
            static_cast<std::uint32_t>(RigidJointDriveMode::TargetOrientation) ||
        scene.spherical.limitEnabledFlags.front() != 1u ||
        std::fabs(scene.spherical.swingLimitYs.front() - 0.2f) > kEpsilon ||
        std::fabs(scene.spherical.swingLimitZs.front() - 0.3f) > kEpsilon ||
        std::fabs(scene.spherical.twistLimitMins.front() + 0.4f) > kEpsilon ||
        std::fabs(scene.spherical.twistLimitMaxs.front() - 0.5f) > kEpsilon ||
        std::fabs(scene.spherical.constraintCompliances.front() - 0.01f) > kEpsilon ||
        std::fabs(scene.spherical.swingCompliances.front() - 0.02f) > kEpsilon ||
        std::fabs(scene.spherical.twistCompliances.front() - 0.03f) > kEpsilon ||
        std::fabs(scene.spherical.driveCompliances.front() - 0.04f) > kEpsilon)
    {
        CRESSIM_LOG_ERROR("Spherical drive target state was not rebuilt correctly.");
        return 1;
    }

    if (scene.hinge.driveModes.front() != static_cast<std::uint32_t>(RigidJointDriveMode::TargetPosition) ||
        scene.hinge.limitEnabledFlags.front() != 1u ||
        std::fabs(scene.hinge.limitMins.front() + 0.25f) > kEpsilon ||
        std::fabs(scene.hinge.limitMaxs.front() - 0.75f) > kEpsilon ||
        std::fabs(scene.hinge.driveTargetAngles.front() - 6.0f * Diligent::PI_F) > kEpsilon ||
        std::fabs(scene.hinge.driveDampings.front() - 0.35f) > kEpsilon ||
        std::fabs(scene.hinge.driveMaxAngularVelocities.front() - 4.5f) > kEpsilon ||
        std::fabs(scene.hinge.driveTargetAngularVelocities.front() - 1.25f) > kEpsilon)
    {
        CRESSIM_LOG_ERROR("Hinge drive target state was not rebuilt correctly.");
        return 1;
    }

    if (scene.slider.driveModes.front() != static_cast<std::uint32_t>(RigidJointDriveMode::TargetPosition) ||
        scene.slider.limitEnabledFlags.front() != 1u ||
        std::fabs(scene.slider.limitMins.front() + 0.4f) > kEpsilon ||
        std::fabs(scene.slider.limitMaxs.front() - 1.1f) > kEpsilon ||
        std::fabs(scene.slider.driveDampings.front() - 0.45f) > kEpsilon ||
        std::fabs(scene.slider.driveMaxVelocities.front() - 1.8f) > kEpsilon ||
        std::fabs(scene.slider.driveTargetPositions.front() - 0.75f) > kEpsilon ||
        std::fabs(scene.slider.driveTargetVelocities.front() - 0.6f) > kEpsilon)
    {
        CRESSIM_LOG_ERROR("Slider drive target state was not rebuilt correctly.");
        return 1;
    }

    hinge.driveMode = RigidJointDriveMode::TargetVelocity;
    hinge.driveTargetAngularVelocity = -0.8f;
    hinge.limitMin = 3.0f;
    hinge.limitMax = 4.0f;
    if (!world.upsertHingeJoint(hinge))
    {
        CRESSIM_LOG_ERROR("Failed to update hinge joint drive mode.");
        return 1;
    }

    slider.driveMode = RigidJointDriveMode::TargetVelocity;
    slider.driveTargetVelocity = -0.45f;
    if (!world.upsertSliderJoint(slider))
    {
        CRESSIM_LOG_ERROR("Failed to update slider joint drive mode.");
        return 1;
    }

    const auto &sceneAfterVelocityMode = world.rigidJointScene();
    if (sceneAfterVelocityMode.hinge.driveModes.front() !=
            static_cast<std::uint32_t>(RigidJointDriveMode::TargetVelocity) ||
        std::fabs(sceneAfterVelocityMode.hinge.limitMins.front() - 3.0f) > kEpsilon ||
        std::fabs(sceneAfterVelocityMode.hinge.limitMaxs.front() - 4.0f) > kEpsilon ||
        std::fabs(sceneAfterVelocityMode.hinge.driveTargetAngularVelocities.front() + 0.8f) >
            kEpsilon)
    {
        CRESSIM_LOG_ERROR("Hinge velocity drive target state was not rebuilt correctly. mode=",
                          sceneAfterVelocityMode.hinge.driveModes.front(), " limitMin=",
                          sceneAfterVelocityMode.hinge.limitMins.front(), " limitMax=",
                          sceneAfterVelocityMode.hinge.limitMaxs.front(), " targetVel=",
                          sceneAfterVelocityMode.hinge.driveTargetAngularVelocities.front());
        return 1;
    }

    if (sceneAfterVelocityMode.slider.driveModes.front() !=
            static_cast<std::uint32_t>(RigidJointDriveMode::TargetVelocity) ||
        std::fabs(sceneAfterVelocityMode.slider.driveTargetVelocities.front() + 0.45f) >
            kEpsilon)
    {
        CRESSIM_LOG_ERROR("Slider velocity drive target state was not rebuilt correctly. mode=",
                          sceneAfterVelocityMode.slider.driveModes.front(), " targetVel=",
                          sceneAfterVelocityMode.slider.driveTargetVelocities.front());
        return 1;
    }

    world.removeRigidBody(1001u);
    const auto &sceneAfterRemoval = world.rigidJointScene();
    if (!sceneAfterRemoval.ball.empty() || !sceneAfterRemoval.spherical.empty() ||
        !sceneAfterRemoval.hinge.empty() ||
        !sceneAfterRemoval.slider.empty())
    {
        CRESSIM_LOG_ERROR("Joint scene should drop joints referencing removed bodies.");
        return 1;
    }
    if (!world.ballJointSnapshot().empty() || !world.sphericalJointSnapshot().empty() ||
        !world.hingeJointSnapshot().empty() ||
        !world.sliderJointSnapshot().empty())
    {
        CRESSIM_LOG_ERROR("Authored joint snapshots should drop joints referencing removed bodies.");
        return 1;
    }

    CRESSIM_LOG_INFO("Physics world rigid joint checks passed.");
    return 0;
}
