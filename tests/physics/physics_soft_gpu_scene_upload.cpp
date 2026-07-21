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
    const common::EntityId strandEntity = world.createEntity();

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
    soft.selfCollisionEnabled = true;
    soft.collisionLayer       = 0x2u;
    soft.collisionMask        = 0x9u;
    if (!world.setSoftBody(softBody, soft))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body in GPU scene upload test.");
        runtime.shutdown();
        return 1;
    }

    engine::StrandComponent strand{};
    strand.restPositions = {
        {-0.2f, 1.5f, 0.0f},
        {0.0f, 1.7f, 0.0f},
        {0.2f, 1.5f, 0.0f},
    };
    strand.particleMass            = 0.5f;
    strand.particleRadius          = 0.05f;
    strand.stretchShearCompliance  = 0.01f;
    strand.bendCompliance          = 0.02f;
    strand.twistCompliance         = 0.03f;
    strand.rootMaterialNormal      = {0.0f, 0.0f, 1.0f};
    strand.selfCollisionEnabled    = true;
    if (!world.setStrand(strandEntity, strand))
    {
        CRESSIM_LOG_ERROR("Failed to author strand in GPU scene upload test.");
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
    runtime.prepare();
    const bool physicsStepSucceeded = runtime.uploadWorld() && runtime.stepPhysics(frame);
    if (physicsStepSucceeded)
    {
        (void)runtime.stepSimulationSensors(frame);
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);

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
    const std::uint32_t expectedStrandSegmentCount =
        static_cast<std::uint32_t>(world.physicsWorld().strandSegments().size());
    const std::uint32_t expectedStrandJointCount =
        static_cast<std::uint32_t>(world.physicsWorld().strandJoints().size());
    if (sceneView.soft.softBodyCount != 1u || sceneView.soft.particles.count != expectedParticleCount ||
        sceneView.soft.edgeCount != expectedEdgeCount || sceneView.soft.tetCount != expectedTetCount ||
        sceneView.soft.strandSegmentCount != expectedStrandSegmentCount ||
        sceneView.soft.strandJointCount != expectedStrandJointCount)
    {
        CRESSIM_LOG_ERROR("Unexpected soft GPU scene counts.");
        runtime.shutdown();
        return 1;
    }

    if (sceneView.soft.particles.positionsInvMassBuffer == nullptr ||
        sceneView.soft.particles.previousPositionsBuffer == nullptr ||
        sceneView.soft.particles.velocitiesBuffer == nullptr ||
        sceneView.soft.particles.radiiBuffer == nullptr ||
        sceneView.soft.particles.particleMaterialIndicesBuffer == nullptr ||
        sceneView.soft.particles.particleContactMaterialsBuffer == nullptr ||
        sceneView.soft.particles.environmentIndicesBuffer == nullptr ||
        sceneView.soft.particles.owningSoftBodyIndicesBuffer == nullptr ||
        sceneView.soft.particles.phasesBuffer == nullptr ||
        sceneView.soft.particles.adjacencyOffsetsBuffer == nullptr ||
        sceneView.soft.particles.adjacencyCountsBuffer == nullptr ||
        sceneView.soft.particles.adjacencyIndicesBuffer == nullptr ||
        sceneView.soft.edgesBuffer == nullptr || sceneView.soft.tetsBuffer == nullptr ||
        sceneView.soft.strandSegmentsBuffer == nullptr || sceneView.soft.strandJointsBuffer == nullptr ||
        sceneView.soft.strandSegmentStatesBuffer == nullptr ||
        sceneView.soft.segmentStrandJointRangesBuffer == nullptr ||
        sceneView.soft.segmentIncidentStrandJointsBuffer == nullptr)
    {
        CRESSIM_LOG_ERROR("Soft GPU scene view is missing required buffers.");
        runtime.shutdown();
        return 1;
    }

    if (sceneView.soft.particles.contactMaterialCount == 0u)
    {
        CRESSIM_LOG_ERROR("Soft GPU scene view did not upload contact material table.");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO("Soft GPU scene upload checks passed.");
    return 0;
}
