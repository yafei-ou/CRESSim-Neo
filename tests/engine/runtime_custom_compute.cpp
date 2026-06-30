#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/custom_compute.h"
#include "engine/runtime.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace
{

constexpr const char *kRuntimeCustomComputeShader = R"(
#include "include/structured_buffer_compat.hlsli"
#include "include/physics/core/physics_math.hlsli"
#include "include/physics/rigid/physics_rigid_types.hlsli"

cbuffer CustomRigidLateralShiftConstants
{
    float lateralShift;
    float verticalLift;
    float padding0;
    float padding1;
};

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyKinematicTargetPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidBodyKinematicTargetOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_RigidBodyKinematicTargetFlags);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    uint bodyCount = 0u;
    uint elementStride = 0u;
    g_RigidBodyPositionsInvMass.GetDimensions(bodyCount, elementStride);
    if (idx >= bodyCount)
    {
        return;
    }

    const float4 sourcePosition = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, idx);
    const float4 sourceOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, idx));

    float4 targetPosition = sourcePosition;
    targetPosition.x += lateralShift;
    targetPosition.y += verticalLift;

    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetPositions, idx, targetPosition);
    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetOrientations, idx, sourceOrientation);
    CRESSIM_SB_STORE(g_RigidBodyKinematicTargetFlags, idx, kKinematicTargetEnabled);
}
)";

} // namespace

int main()
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR("Runtime initialization failed.");
        return 1;
    }

    auto &world = runtime.getWorld();

    const common::EntityId bodyEntity = world.createEntity();
    engine::TransformComponent transform{};
    transform.worldTransform.position = {0.0f, 1.0f, 0.0f};
    world.setTransform(bodyEntity, transform);

    engine::RigidBodyComponent body{};
    body.bodyType    = physics::RigidBodyType::Kinematic;
    body.inverseMass = 1.0f;
    body.simulated   = true;
    world.setRigidBody(bodyEntity, body);

    engine::ColliderComponent collider{};
    collider.shapeType   = physics::ColliderShapeType::Box;
    collider.shapeParams = {0.25f, 0.25f, 0.25f, 0.0f};
    world.addCollider(bodyEntity, collider);

    runtime.prepare();
    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload prepared world state.");
        runtime.shutdown();
        return 1;
    }

    const std::vector<engine::CustomComputeResourceDesc> resources =
        runtime.listCustomComputeResources();
    if (resources.empty())
    {
        CRESSIM_LOG_ERROR("Custom compute resource registry is empty.");
        runtime.shutdown();
        return 1;
    }

    engine::CustomComputePassDesc passDesc{};
    passDesc.debugName        = "RuntimeCustomComputeSmoke";
    passDesc.shaderSource     = kRuntimeCustomComputeShader;
    passDesc.threadGroupSizeX = 64u;
    passDesc.resourceBindings.resize(5u);
    passDesc.resourceBindings[0].shaderVariableName = "g_RigidBodyPositionsInvMass";
    passDesc.resourceBindings[0].resourceKey        = "rigid.positions";
    passDesc.resourceBindings[0].access = engine::CustomComputeResourceAccess::ReadOnly;
    passDesc.resourceBindings[1].shaderVariableName = "g_RigidBodyOrientations";
    passDesc.resourceBindings[1].resourceKey        = "rigid.orientations";
    passDesc.resourceBindings[1].access = engine::CustomComputeResourceAccess::ReadOnly;
    passDesc.resourceBindings[2].shaderVariableName = "g_RigidBodyKinematicTargetPositions";
    passDesc.resourceBindings[2].resourceKey        = "rigid.kinematic_target_positions";
    passDesc.resourceBindings[2].access = engine::CustomComputeResourceAccess::ReadWrite;
    passDesc.resourceBindings[3].shaderVariableName = "g_RigidBodyKinematicTargetOrientations";
    passDesc.resourceBindings[3].resourceKey = "rigid.kinematic_target_orientations";
    passDesc.resourceBindings[3].access = engine::CustomComputeResourceAccess::ReadWrite;
    passDesc.resourceBindings[4].shaderVariableName = "g_RigidBodyKinematicTargetFlags";
    passDesc.resourceBindings[4].resourceKey        = "rigid.kinematic_target_flags";
    passDesc.resourceBindings[4].access = engine::CustomComputeResourceAccess::ReadWrite;
    passDesc.dispatch.mode            = engine::CustomComputeDispatchMode::ResourceElementCount;
    passDesc.dispatch.countResourceKey = "rigid.positions";
    passDesc.constantBufferVariableName = "CustomRigidLateralShiftConstants";
    passDesc.constantBufferSizeBytes    = 16u;
    const float constants[4]            = {0.25f, 0.0f, 0.0f, 0.0f};
    const auto *constantBytes = reinterpret_cast<const std::uint8_t *>(constants);
    passDesc.constantData.assign(constantBytes, constantBytes + sizeof(constants));

    const engine::CustomComputePassHandle pass = runtime.createCustomComputePass(passDesc);
    if (!pass.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create runtime custom compute pass.");
        runtime.shutdown();
        return 1;
    }

    const float updatedConstants[4] = {0.5f, 0.0f, 0.0f, 0.0f};
    const auto *updatedConstantBytes =
        reinterpret_cast<const std::uint8_t *>(updatedConstants);
    std::vector<std::uint8_t> updatedConstantData(updatedConstantBytes,
                                                  updatedConstantBytes +
                                                      sizeof(updatedConstants));
    if (!runtime.updateCustomComputePassConstants(pass, updatedConstantData))
    {
        CRESSIM_LOG_ERROR("Failed to update runtime custom compute pass constants.");
        runtime.shutdown();
        return 1;
    }

    if (!runtime.executeCustomComputePass(pass))
    {
        CRESSIM_LOG_ERROR("Failed to execute runtime custom compute pass.");
        runtime.shutdown();
        return 1;
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex   = 0u;
    if (!runtime.stepPhysics(frame))
    {
        CRESSIM_LOG_ERROR("Physics step failed after custom compute dispatch.");
        runtime.shutdown();
        return 1;
    }

    const physics::RigidBodyState *state = world.physicsWorld().tryGetRigidBody(bodyEntity);
    if (state == nullptr)
    {
        CRESSIM_LOG_ERROR("Rigid body disappeared after custom compute test.");
        runtime.shutdown();
        return 1;
    }

    if (std::fabs(state->position.x - 0.5f) > 0.05f)
    {
        CRESSIM_LOG_ERROR("Unexpected kinematic target result. Expected x=0.5, actual=",
                          state->position.x, ".");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    return 0;
}
