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
    try
    {
        if (!runtime.initialize(config))
        {
            CRESSIM_LOG_INFO("Skipping soft GPU scene upload test because Vulkan runtime "
                             "initialization is unavailable on this machine.");
            return 0;
        }
    }
    catch (const std::exception &)
    {
        CRESSIM_LOG_INFO("Skipping soft GPU scene upload test because Vulkan runtime "
                         "initialization is unavailable on this machine.");
        return 0;
    }

    engine::World &world            = runtime.getWorld();
    const common::EntityId softBody = world.createEntity();

    engine::TransformComponent transform{};
    transform.worldTransform.position = {0.0f, 1.0f, 0.0f};
    world.setTransform(softBody, transform);

    engine::SoftBodyComponent soft{};
    soft.source.regularGrid.size                  = {1.0f, 1.0f, 1.0f};
    soft.source.regularGrid.targetParticleSpacing = 0.5f;
    soft.particleMass                             = 1.0f;
    soft.particleRadius                           = 0.1f;
    soft.edgeCompliance                           = 0.0f;
    soft.volumeCompliance = 0.0f;
    soft.simulated         = true;
    soft.selfCollisionEnabled = true;
    soft.collisionLayer       = 0x2u;
    soft.collisionMask        = 0x9u;
    if (!world.setSoftBody(softBody, soft))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body in GPU scene upload test.");
        runtime.shutdown();
        return 1;
    }

    const std::optional<engine::SoftBodyComponent> roundTripped = world.tryGetSoftBody(softBody);
    if (!roundTripped.has_value() || !roundTripped->selfCollisionEnabled ||
        roundTripped->collisionLayer != 0x2u || roundTripped->collisionMask != 0x9u ||
        roundTripped->source.kind != physics::SoftBodySourceKind::RegularGrid)
    {
        CRESSIM_LOG_ERROR("Soft body component round-trip failed in GPU scene upload test.");
        runtime.shutdown();
        return 1;
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    runtime.tick(frame);

    const physics::PhysicsSolver *solver = runtime.getPhysicsSolver();
    if (solver == nullptr)
    {
        CRESSIM_LOG_ERROR("Runtime returned null physics solver.");
        runtime.shutdown();
        return 1;
    }

    const physics::PhysicsGpuSceneView sceneView = solver->gpuSceneView();
    const std::uint32_t expectedParticleCount =
        static_cast<std::uint32_t>(world.physicsWorld().particles().size());
    const std::uint32_t expectedEdgeCount =
        static_cast<std::uint32_t>(world.physicsWorld().softEdges().size());
    const std::uint32_t expectedTetCount =
        static_cast<std::uint32_t>(world.physicsWorld().softTets().size());
    if (sceneView.soft.softBodyCount != 1u || sceneView.soft.particles.count != expectedParticleCount ||
        sceneView.soft.edgeCount != expectedEdgeCount || sceneView.soft.tetCount != expectedTetCount)
    {
        CRESSIM_LOG_ERROR("Unexpected soft GPU scene counts.");
        runtime.shutdown();
        return 1;
    }

    if (sceneView.soft.particles.positionsInvMassBuffer == nullptr ||
        sceneView.soft.particles.previousPositionsBuffer == nullptr ||
        sceneView.soft.particles.velocitiesBuffer == nullptr ||
        sceneView.soft.particles.radiiBuffer == nullptr ||
        sceneView.soft.particles.environmentIndicesBuffer == nullptr ||
        sceneView.soft.particles.owningSoftBodyIndicesBuffer == nullptr ||
        sceneView.soft.particles.phasesBuffer == nullptr ||
        sceneView.soft.particles.adjacencyOffsetsBuffer == nullptr ||
        sceneView.soft.particles.adjacencyCountsBuffer == nullptr ||
        sceneView.soft.particles.adjacencyIndicesBuffer == nullptr ||
        sceneView.soft.edgesBuffer == nullptr || sceneView.soft.tetsBuffer == nullptr)
    {
        CRESSIM_LOG_ERROR("Soft GPU scene view is missing required buffers.");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO("Soft GPU scene upload checks passed.");
    return 0;
}
