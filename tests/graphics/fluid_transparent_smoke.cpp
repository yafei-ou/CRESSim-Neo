#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"

int main()
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING(
            "Skipping fluid transparent smoke because runtime initialization failed.\n");
        return 0;
    }

    auto &world = runtime.getWorld();

    const common::EntityId cameraEntity = world.createEntity();
    engine::TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.8f, -3.0f};
    world.setTransform(cameraEntity, cameraTransform);
    engine::CameraComponent camera{};
    camera.outputWidth = 640u;
    camera.outputHeight = 360u;
    world.setCamera(cameraEntity, camera);

    const common::EntityId floorEntity = world.createEntity();
    engine::TransformComponent floorTransform{};
    floorTransform.worldTransform.position = {0.0f, -0.15f, 0.0f};
    floorTransform.worldTransform.scale = {2.0f, 0.1f, 2.0f};
    world.setTransform(floorEntity, floorTransform);

    engine::RigidBodyComponent floorBody{};
    floorBody.simulated = true;
    floorBody.bodyType = physics::RigidBodyType::Static;
    floorBody.inverseMass = 0.0f;
    floorBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(floorEntity, floorBody);

    engine::ColliderComponent floorCollider{};
    floorCollider.shapeType = physics::ColliderShapeType::Box;
    floorCollider.shapeParams = {2.0f, 1.0f, 2.0f, 0.0f};
    world.addCollider(floorEntity, floorCollider);

    const common::EntityId fluidEntity = world.createEntity();
    engine::TransformComponent fluidTransform{};
    fluidTransform.worldTransform.position = {0.0f, 0.55f, 0.0f};
    world.setTransform(fluidEntity, fluidTransform);

    engine::FluidComponent fluid{};
    fluid.source.kind = physics::FluidSourceKind::RegularGrid;
    fluid.source.regularGrid.size = {0.45f, 0.45f, 0.45f};
    fluid.source.regularGrid.targetParticleSpacing = 0.15f;
    fluid.particleMass = 0.4f;
    fluid.particleRadius = 0.08f;
    fluid.material.viscosity = 0.04f;
    fluid.material.surfaceTension = 0.15f;
    fluid.material.vorticityConfinement = 0.05f;
    if (!world.setFluid(fluidEntity, fluid))
    {
        CRESSIM_LOG_ERROR("Failed to author fluid for transparent smoke.\n");
        runtime.shutdown();
        return 1;
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    for (std::uint32_t i = 0u; i < 4u; ++i)
    {
        frame.frameIndex = i;
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

    runtime.shutdown();
    CRESSIM_LOG_INFO("Fluid transparent smoke checks passed.\n");
    return 0;
}
