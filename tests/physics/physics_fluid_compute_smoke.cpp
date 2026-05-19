#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "common/logger.h"

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
            "Skipping fluid compute smoke because runtime initialization failed.\n");
        return 0;
    }

    auto &world = runtime.getWorld();

    const common::EntityId cameraEntity = world.createEntity();
    engine::TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, -3.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, engine::CameraComponent{});

    const common::EntityId fluidEntity = world.createEntity();
    engine::TransformComponent fluidTransform{};
    fluidTransform.worldTransform.position = {0.0f, 0.45f, 0.0f};
    world.setTransform(fluidEntity, fluidTransform);

    const common::EntityId floorEntity = world.createEntity();
    engine::TransformComponent floorTransform{};
    floorTransform.worldTransform.position = {0.0f, 0.0f, 0.0f};
    floorTransform.worldTransform.scale = {1.0f, 0.1f, 1.0f};
    world.setTransform(floorEntity, floorTransform);

    engine::RigidBodyComponent floorBody{};
    floorBody.simulated = true;
    floorBody.bodyType = physics::RigidBodyType::Static;
    floorBody.inverseMass = 0.0f;
    floorBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(floorEntity, floorBody);

    engine::ColliderComponent floorCollider{};
    floorCollider.shapeType = physics::ColliderShapeType::Box;
    floorCollider.shapeParams = {1.0f, 1.0f, 1.0f, 0.0f};
    world.addCollider(floorEntity, floorCollider);

    engine::FluidComponent fluid{};
    fluid.source.kind = physics::FluidSourceKind::RegularGrid;
    fluid.source.regularGrid.size = {0.2f, 0.2f, 0.2f};
    fluid.source.regularGrid.targetParticleSpacing = 0.2f;
    fluid.particleMass = 0.5f;
    fluid.particleRadius = 0.1f;
    fluid.material.viscosity = 0.05f;
    if (!world.setFluid(fluidEntity, fluid))
    {
        CRESSIM_LOG_ERROR("Failed to author fluid component.\n");
        runtime.shutdown();
        return 1;
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    const physics::FluidState *initialFluid = world.physicsWorld().tryGetFluid(fluidEntity);
    if (initialFluid == nullptr || initialFluid->particleCount == 0u)
    {
        CRESSIM_LOG_ERROR("Fluid authoring did not create any particles.\n");
        runtime.shutdown();
        return 1;
    }

    const std::uint32_t particleIndex = initialFluid->particleOffset;
    const auto &initialParticles = world.physicsWorld().particles();
    const float initialY = initialParticles.positionsInvMass[particleIndex].y;

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    for (std::uint32_t i = 0u; i < 30u; ++i)
    {
        frame.frameIndex = i;
        frame.timeSeconds = static_cast<double>(i) * static_cast<double>(frame.deltaSeconds);
        runtime.tick(frame);
    }

    world.physicsWorld().ensureDerivedStateUpToDate();
    const physics::FluidState *finalFluid = world.physicsWorld().tryGetFluid(fluidEntity);
    const auto &finalParticles = world.physicsWorld().particles();
    if (finalFluid == nullptr || finalFluid->particleCount == 0u)
    {
        CRESSIM_LOG_ERROR("Fluid state disappeared after simulation.\n");
        runtime.shutdown();
        return 1;
    }

    const float finalY = finalParticles.positionsInvMass[finalFluid->particleOffset].y;
    if (!(finalY < initialY))
    {
        CRESSIM_LOG_ERROR("Fluid particle did not advance under gravity.\n");
        runtime.shutdown();
        return 1;
    }
    if (finalY < -0.05f)
    {
        CRESSIM_LOG_ERROR("Fluid particle escaped through the authored static floor.\n");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO("Physics fluid compute smoke checks passed.\n");
    return 0;
}
