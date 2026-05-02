#include "physics/physics_world.h"
#include "common/logger.h"

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::BallJointState;
using cressim::neo::physics::HingeJointState;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::RigidBodyState;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SliderJointState;

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
    ball.localAnchorA = {0.25f, 0.0f, 0.0f};
    ball.localAnchorB = {-0.25f, 0.0f, 0.0f};
    if (!world.upsertBallJoint(ball))
    {
        CRESSIM_LOG_ERROR("Failed to insert ball joint.");
        return 1;
    }

    HingeJointState hinge{};
    hinge.bodyA = bodyA->rigidBodyId;
    hinge.bodyB = bodyB->rigidBodyId;
    hinge.localAnchorA = {0.0f, 0.0f, 0.0f};
    hinge.localAnchorB = {0.0f, 0.0f, 0.0f};
    hinge.localRotationA = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
    hinge.localRotationB = makeJointFrameRotation({0.0f, 1.0f, 0.0f});
    if (!world.upsertHingeJoint(hinge))
    {
        CRESSIM_LOG_ERROR("Failed to insert hinge joint.");
        return 1;
    }

    SliderJointState slider{};
    slider.bodyA = bodyA->rigidBodyId;
    slider.bodyB = bodyB->rigidBodyId;
    slider.localRotationA = makeJointFrameRotation({1.0f, 0.0f, 0.0f});
    slider.localRotationB = makeJointFrameRotation({1.0f, 0.0f, 0.0f});
    if (!world.upsertSliderJoint(slider))
    {
        CRESSIM_LOG_ERROR("Failed to insert slider joint.");
        return 1;
    }

    BallJointState crossEnvBall{};
    crossEnvBall.bodyA = bodyA->rigidBodyId;
    crossEnvBall.bodyB = bodyC->rigidBodyId;
    if (world.upsertBallJoint(crossEnvBall))
    {
        CRESSIM_LOG_ERROR("Cross-environment ball joint should be rejected.");
        return 1;
    }

    const auto &scene = world.rigidJointScene();
    if (scene.ball.size() != 1u || scene.hinge.size() != 1u || scene.slider.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Unexpected rigid joint scene counts.");
        return 1;
    }

    if (scene.ball.bodyIndicesA.front() != 0u || scene.ball.bodyIndicesB.front() != 1u ||
        scene.hinge.bodyIndicesA.front() != 0u || scene.hinge.bodyIndicesB.front() != 1u ||
        scene.slider.bodyIndicesA.front() != 0u || scene.slider.bodyIndicesB.front() != 1u)
    {
        CRESSIM_LOG_ERROR("Rigid joint scene body indices were not rebuilt correctly.");
        return 1;
    }

    world.removeRigidBody(1001u);
    const auto &sceneAfterRemoval = world.rigidJointScene();
    if (!sceneAfterRemoval.ball.empty() || !sceneAfterRemoval.hinge.empty() ||
        !sceneAfterRemoval.slider.empty())
    {
        CRESSIM_LOG_ERROR("Joint scene should drop joints referencing removed bodies.");
        return 1;
    }

    CRESSIM_LOG_INFO("Physics world rigid joint checks passed.");
    return 0;
}
