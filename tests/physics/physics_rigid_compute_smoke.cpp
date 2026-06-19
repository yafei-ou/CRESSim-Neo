#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "common/logger.h"

#include <cmath>

int main()
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
        return 1;
    }

    auto& world = runtime.getWorld();

    const common::EntityId cameraEntity = world.createEntity();
    engine::TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, engine::CameraComponent{});

    const common::EntityId rigidEntity = world.createEntity();
    engine::TransformComponent rigidTransform{};
    rigidTransform.worldTransform.position = {0.0f, 0.0f, 0.0f};
    world.setTransform(rigidEntity, rigidTransform);

    engine::RigidBodyComponent rigidBody{};
    rigidBody.linearVelocity = {1.5f, 0.0f, 0.0f};
    rigidBody.angularVelocity = {0.0f, 1.0f, 0.0f};
    rigidBody.inverseMass    = 1.0f;
    rigidBody.simulated      = true;
    world.setRigidBody(rigidEntity, rigidBody);
    engine::ColliderComponent rigidCollider{};
    rigidCollider.shapeType = physics::ColliderShapeType::Sphere;
    rigidCollider.shapeParams = {0.5f, 0.0f, 0.0f, 0.0f};
    world.addCollider(rigidEntity, rigidCollider);

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    constexpr std::uint32_t kFrames = 10;
    for (std::uint32_t i = 0; i < kFrames; ++i)
    {
        frame.frameIndex  = i;
        frame.timeSeconds = static_cast<double>(i) * static_cast<double>(frame.deltaSeconds);
        runtime.prepare();
        const bool physicsStepSucceeded = runtime.stepPhysics(frame);
        if (physicsStepSucceeded)
        {
            (void)runtime.stepSimulationSensors(frame);
        }
        runtime.stepVisualSensors(frame);
        runtime.endFrame(frame);
    }

    const physics::RigidBodyState* finalRigidBody = world.physicsWorld().tryGetRigidBody(rigidEntity);
    if (finalRigidBody == nullptr)
    {
        CRESSIM_LOG_ERROR( "Rigid body missing after simulation.\n");
        runtime.shutdown();
        return 1;
    }

    const float expectedX =
        rigidBody.linearVelocity.x * frame.deltaSeconds * static_cast<float>(kFrames);
    const float actualX = finalRigidBody->position.x;
    if (std::fabs(actualX - expectedX) > 0.05f)
    {
        CRESSIM_LOG_ERROR( "Unexpected rigid body integration. expected=" , expectedX
                  , " actual=" , actualX , "\n");
        runtime.shutdown();
        return 1;
    }

    const graphics::RenderStats stats = runtime.lastRenderStats();
    if (stats.renderedCameraCount == 0)
    {
        CRESSIM_LOG_ERROR( "Renderer did not execute camera rendering.\n");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO( "Physics rigid compute smoke checks passed.\n");
    return 0;
}
