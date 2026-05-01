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
    hinge.localAxisA = {0.0f, 1.0f, 0.0f};
    hinge.localAxisB = {0.0f, 1.0f, 0.0f};
    if (!world.upsertHingeJoint(hinge))
    {
        CRESSIM_LOG_ERROR("Failed to insert hinge joint.");
        return 1;
    }

    SliderJointState slider{};
    slider.bodyA = bodyA->rigidBodyId;
    slider.bodyB = bodyB->rigidBodyId;
    slider.localAxisA = {1.0f, 0.0f, 0.0f};
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
