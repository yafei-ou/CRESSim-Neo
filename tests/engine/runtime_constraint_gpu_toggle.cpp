#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/custom_compute.h"
#include "engine/runtime.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

using namespace cressim::neo;

constexpr const char *kRigidDistanceToggleShader = R"(
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer ConstraintToggleConstants
{
    uint enabled;
    float targetY;
    float padding0;
    float padding1;
};

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_KinematicTargetFlags);
CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidDistanceConstraint, g_Constraints);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    GpuRigidDistanceConstraint constraint = CRESSIM_SB_LOAD(g_Constraints, 0u);
    constraint.enabled = enabled;
    CRESSIM_SB_STORE(g_Constraints, 0u, constraint);

    float4 targetPosition = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, 0u);
    targetPosition.y = targetY;
    CRESSIM_SB_STORE(g_KinematicTargetPositions, 0u, targetPosition);
    CRESSIM_SB_STORE(g_KinematicTargetOrientations, 0u,
                     QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, 0u)));
    CRESSIM_SB_STORE(g_KinematicTargetFlags, 0u, kKinematicTargetEnabled);
}
)";

constexpr const char *kRoutedCableToggleShader = R"(
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer ConstraintToggleConstants
{
    uint enabled;
    float targetY;
    float padding0;
    float padding1;
};

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_KinematicTargetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_KinematicTargetFlags);
CRESSIM_RW_STRUCTURED_BUFFER(GpuRoutedCableConstraint, g_Constraints);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    GpuRoutedCableConstraint constraint = CRESSIM_SB_LOAD(g_Constraints, 0u);
    constraint.enabled = enabled;
    CRESSIM_SB_STORE(g_Constraints, 0u, constraint);

    float4 targetPosition = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, 0u);
    targetPosition.y = targetY;
    CRESSIM_SB_STORE(g_KinematicTargetPositions, 0u, targetPosition);
    CRESSIM_SB_STORE(g_KinematicTargetOrientations, 0u,
                     QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, 0u)));
    CRESSIM_SB_STORE(g_KinematicTargetFlags, 0u, kKinematicTargetEnabled);
}
)";

engine::RuntimeConfig makeRuntimeConfig()
{
    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.physicsDesc.substeps = 4u;
    config.physicsDesc.defaultIterations = 24u;
    return config;
}

bool stepRuntime(engine::Runtime &runtime, std::uint64_t &frameCursor, std::uint32_t frameCount)
{
    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    for (std::uint32_t frameOffset = 0; frameOffset < frameCount; ++frameOffset)
    {
        frame.frameIndex = frameCursor;
        frame.timeSeconds = frame.deltaSeconds * static_cast<float>(frameCursor);
        if (!runtime.stepPhysics(frame))
        {
            return false;
        }
        runtime.endFrame(frame);
        ++frameCursor;
    }
    return true;
}

engine::CustomComputePassHandle createTogglePass(
    engine::Runtime &runtime, const char *debugName, const char *shaderSource,
    const char *constantBufferName, std::uint32_t constantBufferSizeBytes,
    const std::vector<engine::CustomComputeResourceBindingDesc> &bindings)
{
    engine::CustomComputePassDesc passDesc{};
    passDesc.debugName = debugName;
    passDesc.shaderSource = shaderSource;
    passDesc.threadGroupSizeX = 1u;
    passDesc.resourceBindings = bindings;
    passDesc.dispatch.mode = engine::CustomComputeDispatchMode::ExplicitGroupCount;
    passDesc.dispatch.groupCountX = 1u;
    passDesc.constantBufferVariableName = constantBufferName;
    passDesc.constantBufferSizeBytes = constantBufferSizeBytes;
    return runtime.createCustomComputePass(passDesc);
}

bool executeToggle(engine::Runtime &runtime, engine::CustomComputePassHandle pass,
                   const std::vector<std::uint8_t> &constantData)
{
    return runtime.updateCustomComputePassConstants(pass, constantData) &&
           runtime.executeCustomComputePass(pass);
}

std::vector<std::uint8_t> makeDrivenToggleConstants(std::uint32_t enabled, float targetY)
{
    struct Constants
    {
        std::uint32_t enabled = 0u;
        float targetY = 0.0f;
        float padding0 = 0.0f;
        float padding1 = 0.0f;
    };
    const Constants constants{enabled, targetY, 0.0f, 0.0f};
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(&constants);
    return std::vector<std::uint8_t>(bytes, bytes + sizeof(constants));
}

bool testRigidDistanceToggle()
{
    using namespace cressim::neo;

    engine::Runtime runtime;
    if (!runtime.initialize(makeRuntimeConfig()))
    {
        CRESSIM_LOG_WARNING("Skipping rigid-distance GPU toggle test because runtime initialization failed.");
        return true;
    }

    auto &world = runtime.getWorld();
    const common::EntityId baseEntity = world.createEntity();
    engine::TransformComponent baseTransform{};
    baseTransform.worldTransform.position = {0.0f, 4.0f, 0.0f};
    world.setTransform(baseEntity, baseTransform);
    engine::RigidBodyComponent baseBody{};
    baseBody.bodyType = physics::RigidBodyType::Kinematic;
    baseBody.inverseMass = 1.0f;
    baseBody.simulated = true;
    baseBody.kinematicTargetEnabled = true;
    baseBody.kinematicTargetPosition = baseTransform.worldTransform.position;
    world.setRigidBody(baseEntity, baseBody);

    const common::EntityId targetEntity = world.createEntity();
    engine::TransformComponent targetTransform{};
    targetTransform.worldTransform.position = {0.0f, 1.0f, 0.0f};
    world.setTransform(targetEntity, targetTransform);
    engine::RigidBodyComponent targetBody{};
    targetBody.bodyType = physics::RigidBodyType::Dynamic;
    targetBody.inverseMass = 1.0f;
    targetBody.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    targetBody.simulated = true;
    world.setRigidBody(targetEntity, targetBody);
    engine::ColliderComponent targetCollider{};
    targetCollider.shapeType = physics::ColliderShapeType::Box;
    targetCollider.shapeParams = {0.12f, 0.12f, 0.12f, 0.0f};
    world.addCollider(targetEntity, targetCollider);

    physics::AuthoredRigidDistanceConstraintState distance{};
    distance.constraintId = 1u;
    distance.entityA = baseEntity;
    distance.entityB = targetEntity;
    distance.restDistance = 0.5f;
    distance.enabled = true;
    if (!world.upsertRigidDistanceConstraint(distance))
    {
        CRESSIM_LOG_ERROR("Failed to author rigid-distance toggle test constraint.");
        runtime.shutdown();
        return false;
    }

    runtime.prepare();
    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload rigid-distance toggle test world.");
        runtime.shutdown();
        return false;
    }

    std::vector<engine::CustomComputeResourceBindingDesc> bindings(6u);
    bindings[0].shaderVariableName = "g_RigidBodyPositionsInvMass";
    bindings[0].resourceKey = "rigid.positions";
    bindings[0].access = engine::CustomComputeResourceAccess::ReadOnly;
    bindings[1].shaderVariableName = "g_RigidBodyOrientations";
    bindings[1].resourceKey = "rigid.orientations";
    bindings[1].access = engine::CustomComputeResourceAccess::ReadOnly;
    bindings[2].shaderVariableName = "g_KinematicTargetPositions";
    bindings[2].resourceKey = "rigid.kinematic_target_positions";
    bindings[2].access = engine::CustomComputeResourceAccess::ReadWrite;
    bindings[3].shaderVariableName = "g_KinematicTargetOrientations";
    bindings[3].resourceKey = "rigid.kinematic_target_orientations";
    bindings[3].access = engine::CustomComputeResourceAccess::ReadWrite;
    bindings[4].shaderVariableName = "g_KinematicTargetFlags";
    bindings[4].resourceKey = "rigid.kinematic_target_flags";
    bindings[4].access = engine::CustomComputeResourceAccess::ReadWrite;
    bindings[5].shaderVariableName = "g_Constraints";
    bindings[5].resourceKey = "constraint.rigid_distance_constraints";
    bindings[5].access = engine::CustomComputeResourceAccess::ReadWrite;
    engine::CustomComputePassHandle pass =
        createTogglePass(runtime, "RigidDistanceConstraintToggle", kRigidDistanceToggleShader,
                         "ConstraintToggleConstants", 16u, bindings);
    if (!pass.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create rigid-distance toggle pass.");
        runtime.shutdown();
        return false;
    }

    std::uint64_t frameCursor = 0u;
    if (!stepRuntime(runtime, frameCursor, 30u))
    {
        CRESSIM_LOG_ERROR("Rigid-distance enabled step failed.");
        runtime.shutdown();
        return false;
    }
    const physics::RigidBodyState *targetState = world.physicsWorld().tryGetRigidBody(targetEntity);
    if (targetState == nullptr)
    {
        CRESSIM_LOG_ERROR("Rigid-distance target state is unavailable.");
        runtime.shutdown();
        return false;
    }
    const float enabledDistance = std::fabs(baseTransform.worldTransform.position.y - targetState->position.y);

    constexpr float kDrivenBaseY = 6.5f;
    if (!executeToggle(runtime, pass, makeDrivenToggleConstants(0u, kDrivenBaseY)) ||
        !stepRuntime(runtime, frameCursor, 30u))
    {
        CRESSIM_LOG_ERROR("Rigid-distance disable toggle path failed.");
        runtime.shutdown();
        return false;
    }
    targetState = world.physicsWorld().tryGetRigidBody(targetEntity);
    const float disabledDistance = std::fabs(kDrivenBaseY - targetState->position.y);

    if (!executeToggle(runtime, pass, makeDrivenToggleConstants(1u, kDrivenBaseY)) ||
        !stepRuntime(runtime, frameCursor, 45u))
    {
        CRESSIM_LOG_ERROR("Rigid-distance re-enable toggle path failed.");
        runtime.shutdown();
        return false;
    }
    targetState = world.physicsWorld().tryGetRigidBody(targetEntity);
    const float reenabledDistance = std::fabs(kDrivenBaseY - targetState->position.y);
    runtime.shutdown();

    if (!std::isfinite(enabledDistance) || !std::isfinite(disabledDistance) ||
        !std::isfinite(reenabledDistance))
    {
        CRESSIM_LOG_ERROR("Rigid-distance toggle produced non-finite distances.");
        return false;
    }
    if (disabledDistance <= enabledDistance + 0.2f)
    {
        CRESSIM_LOG_ERROR("Disabling the rigid-distance constraint did not measurably weaken it.");
        return false;
    }
    if (reenabledDistance >= disabledDistance - 0.2f)
    {
        CRESSIM_LOG_ERROR("Re-enabling the rigid-distance constraint did not restore its effect.");
        return false;
    }
    return true;
}

bool testRoutedCableToggle()
{
    using namespace cressim::neo;

    engine::Runtime runtime;
    if (!runtime.initialize(makeRuntimeConfig()))
    {
        CRESSIM_LOG_WARNING("Skipping routed-cable GPU toggle test because runtime initialization failed.");
        return true;
    }

    auto &world = runtime.getWorld();
    const common::EntityId baseEntity = world.createEntity();
    engine::TransformComponent baseTransform{};
    baseTransform.worldTransform.position = {0.0f, 4.0f, 0.0f};
    world.setTransform(baseEntity, baseTransform);
    engine::RigidBodyComponent baseBody{};
    baseBody.bodyType = physics::RigidBodyType::Kinematic;
    baseBody.inverseMass = 1.0f;
    baseBody.simulated = true;
    baseBody.kinematicTargetEnabled = true;
    baseBody.kinematicTargetPosition = baseTransform.worldTransform.position;
    world.setRigidBody(baseEntity, baseBody);

    const common::EntityId targetEntity = world.createEntity();
    engine::TransformComponent targetTransform{};
    targetTransform.worldTransform.position = {0.0f, 1.0f, 0.0f};
    world.setTransform(targetEntity, targetTransform);
    engine::RigidBodyComponent targetBody{};
    targetBody.bodyType = physics::RigidBodyType::Dynamic;
    targetBody.inverseMass = 1.0f;
    targetBody.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    targetBody.simulated = true;
    world.setRigidBody(targetEntity, targetBody);
    engine::ColliderComponent targetCollider{};
    targetCollider.shapeType = physics::ColliderShapeType::Box;
    targetCollider.shapeParams = {0.12f, 0.12f, 0.12f, 0.0f};
    world.addCollider(targetEntity, targetCollider);

    physics::AuthoredRoutedCableConstraintState cable{};
    cable.constraintId = 1u;
    cable.routePoints.push_back({baseEntity, {0.0f, 0.0f, 0.0f}});
    cable.routePoints.push_back({targetEntity, {0.0f, 0.0f, 0.0f}});
    cable.targetLength = 0.5f;
    cable.enabled = true;
    if (!world.upsertRoutedCableConstraint(cable))
    {
        CRESSIM_LOG_ERROR("Failed to author routed-cable toggle test constraint.");
        runtime.shutdown();
        return false;
    }

    runtime.prepare();
    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload routed-cable toggle test world.");
        runtime.shutdown();
        return false;
    }

    std::vector<engine::CustomComputeResourceBindingDesc> bindings(6u);
    bindings[0].shaderVariableName = "g_RigidBodyPositionsInvMass";
    bindings[0].resourceKey = "rigid.positions";
    bindings[0].access = engine::CustomComputeResourceAccess::ReadOnly;
    bindings[1].shaderVariableName = "g_RigidBodyOrientations";
    bindings[1].resourceKey = "rigid.orientations";
    bindings[1].access = engine::CustomComputeResourceAccess::ReadOnly;
    bindings[2].shaderVariableName = "g_KinematicTargetPositions";
    bindings[2].resourceKey = "rigid.kinematic_target_positions";
    bindings[2].access = engine::CustomComputeResourceAccess::ReadWrite;
    bindings[3].shaderVariableName = "g_KinematicTargetOrientations";
    bindings[3].resourceKey = "rigid.kinematic_target_orientations";
    bindings[3].access = engine::CustomComputeResourceAccess::ReadWrite;
    bindings[4].shaderVariableName = "g_KinematicTargetFlags";
    bindings[4].resourceKey = "rigid.kinematic_target_flags";
    bindings[4].access = engine::CustomComputeResourceAccess::ReadWrite;
    bindings[5].shaderVariableName = "g_Constraints";
    bindings[5].resourceKey = "constraint.routed_cable_descriptors";
    bindings[5].access = engine::CustomComputeResourceAccess::ReadWrite;
    engine::CustomComputePassHandle pass =
        createTogglePass(runtime, "RoutedCableConstraintToggle", kRoutedCableToggleShader,
                         "ConstraintToggleConstants", 16u, bindings);
    if (!pass.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create routed-cable toggle pass.");
        runtime.shutdown();
        return false;
    }

    std::uint64_t frameCursor = 0u;
    if (!stepRuntime(runtime, frameCursor, 30u))
    {
        CRESSIM_LOG_ERROR("Routed-cable enabled step failed.");
        runtime.shutdown();
        return false;
    }
    const physics::RigidBodyState *targetState = world.physicsWorld().tryGetRigidBody(targetEntity);
    if (targetState == nullptr)
    {
        CRESSIM_LOG_ERROR("Routed-cable target state is unavailable.");
        runtime.shutdown();
        return false;
    }
    const float enabledDistance = std::fabs(baseTransform.worldTransform.position.y - targetState->position.y);

    constexpr float kDrivenBaseY = 6.5f;
    if (!executeToggle(runtime, pass, makeDrivenToggleConstants(0u, kDrivenBaseY)) ||
        !stepRuntime(runtime, frameCursor, 30u))
    {
        CRESSIM_LOG_ERROR("Routed-cable disable toggle path failed.");
        runtime.shutdown();
        return false;
    }
    targetState = world.physicsWorld().tryGetRigidBody(targetEntity);
    const float disabledDistance = std::fabs(kDrivenBaseY - targetState->position.y);

    if (!executeToggle(runtime, pass, makeDrivenToggleConstants(1u, kDrivenBaseY)) ||
        !stepRuntime(runtime, frameCursor, 45u))
    {
        CRESSIM_LOG_ERROR("Routed-cable re-enable toggle path failed.");
        runtime.shutdown();
        return false;
    }
    targetState = world.physicsWorld().tryGetRigidBody(targetEntity);
    const float reenabledDistance = std::fabs(kDrivenBaseY - targetState->position.y);
    runtime.shutdown();

    if (!std::isfinite(enabledDistance) || !std::isfinite(disabledDistance) ||
        !std::isfinite(reenabledDistance))
    {
        CRESSIM_LOG_ERROR("Routed-cable toggle produced non-finite distances.");
        return false;
    }
    if (disabledDistance <= enabledDistance + 0.2f)
    {
        CRESSIM_LOG_ERROR("Disabling the routed cable did not measurably weaken it.");
        return false;
    }
    if (reenabledDistance >= disabledDistance - 0.2f)
    {
        CRESSIM_LOG_ERROR("Re-enabling the routed cable did not restore its effect.");
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!testRigidDistanceToggle() || !testRoutedCableToggle())
    {
        return 1;
    }
    return 0;
}
