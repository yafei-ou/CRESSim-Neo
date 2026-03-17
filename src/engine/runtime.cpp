#include "engine/runtime.h"

#include "engine/world_to_render_world_sync.h"

#include <iostream>
#include <unordered_map>

namespace cressim::neo::engine
{

namespace
{

std::vector<gpu::GpuEntityPoseMappingEntry> buildPhysicsRenderableMappings(
    const World& world, std::unordered_map<common::EntityId, std::uint32_t>& outPoseIndices)
{
    std::vector<gpu::GpuEntityPoseMappingEntry> mappings;
    outPoseIndices.clear();

    const auto& rigidBodies = world.physicsWorld().rigidBodySoA();
    const auto& meshRenderers = world.meshRendererSoA();
    if (rigidBodies.entityIds.empty() || meshRenderers.entityIds.empty())
    {
        return mappings;
    }

    std::unordered_map<common::EntityId, std::uint32_t> rigidBodyIndexByEntity;
    rigidBodyIndexByEntity.reserve(rigidBodies.entityIds.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(rigidBodies.entityIds.size()); ++i)
    {
        rigidBodyIndexByEntity.emplace(rigidBodies.entityIds[i], i);
    }

    mappings.reserve(meshRenderers.entityIds.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(meshRenderers.entityIds.size()); ++i)
    {
        if (meshRenderers.visibleFlags[i] == 0u)
        {
            continue;
        }

        const common::EntityId entityId = meshRenderers.entityIds[i];
        const auto rigidBodyIt = rigidBodyIndexByEntity.find(entityId);
        if (rigidBodyIt == rigidBodyIndexByEntity.end())
        {
            continue;
        }

        gpu::GpuEntityPoseMappingEntry entry{};
        entry.sourcePoseIndex = rigidBodyIt->second;
        entry.entityPoseIndex = static_cast<std::uint32_t>(mappings.size());
        outPoseIndices[entityId] = entry.entityPoseIndex;
        mappings.push_back(entry);
    }

    return mappings;
}

std::vector<gpu::GpuRenderableMetadata> buildRenderableMetadata(
    const graphics::RenderWorld& renderWorld, const graphics::RenderResourceManager& resources,
    const std::unordered_map<common::EntityId, std::uint32_t>& poseIndices)
{
    std::vector<gpu::GpuRenderableMetadata> metadata;
    const auto& renderables = renderWorld.renderables();
    metadata.reserve(renderables.size());

    for (const graphics::RenderableInstance& renderable : renderables)
    {
        gpu::GpuRenderableMetadata entry{};
        const graphics::MaterialResourceDesc* material = resources.tryGetMaterial(renderable.material);
        if (material != nullptr)
        {
            if (material->blendMode == graphics::BlendMode::Transparent)
            {
                entry.flags |= gpu::GpuRenderableFlag_Transparent;
            }
            else
            {
                entry.flags |= gpu::GpuRenderableFlag_Opaque;
            }
            if (material->castsShadows && material->blendMode != graphics::BlendMode::Transparent)
            {
                entry.flags |= gpu::GpuRenderableFlag_ShadowCaster;
            }
        }

        Diligent::float3 localBoundsMin{};
        Diligent::float3 localBoundsMax{};
        if (resources.tryGetMeshLocalBounds(renderable.mesh, localBoundsMin, localBoundsMax))
        {
            entry.localBoundsMin =
                Diligent::float4{localBoundsMin.x, localBoundsMin.y, localBoundsMin.z, 1.0f};
            entry.localBoundsMax =
                Diligent::float4{localBoundsMax.x, localBoundsMax.y, localBoundsMax.z, 1.0f};
        }

        const auto poseIt = poseIndices.find(renderable.entityId);
        if (poseIt != poseIndices.end() && material != nullptr &&
            material->blendMode != graphics::BlendMode::Transparent)
        {
            entry.entityPoseIndex = poseIt->second;
            entry.flags |= gpu::GpuRenderableFlag_UsesGpuPose;
        }

        metadata.push_back(entry);
    }

    return metadata;
}

const char* stageName(physics::PhysicsSolverStage stage)
{
    switch (stage)
    {
    case physics::PhysicsSolverStage::PredictState:
        return "PredictState";
    case physics::PhysicsSolverStage::UpdateWorldAabbs:
        return "UpdateWorldAabbs";
    case physics::PhysicsSolverStage::BuildBroadPhase:
        return "BuildBroadPhase";
    case physics::PhysicsSolverStage::GenerateBroadPhasePairs:
        return "GenerateBroadPhasePairs";
    case physics::PhysicsSolverStage::GenerateContacts:
        return "GenerateContacts";
    case physics::PhysicsSolverStage::SolveConstraints:
        return "SolveConstraints";
    case physics::PhysicsSolverStage::UpdateVelocities:
        return "UpdateVelocities";
    case physics::PhysicsSolverStage::CommitResults:
        return "CommitResults";
    case physics::PhysicsSolverStage::Count:
        break;
    }
    return "Unknown";
}

void logPhysicsStepFailure(const common::FrameContext& frameContext,
                           const physics::PhysicsSolverStageStats& stats)
{
    std::cerr << "Runtime: physics step failed at frame " << frameContext.frameIndex
              << " (dt=" << frameContext.deltaSeconds << "). Executed stages:";
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(physics::PhysicsSolverStage::Count);
         ++i)
    {
        const auto stage = static_cast<physics::PhysicsSolverStage>(i);
        std::cerr << ' ' << stageName(stage) << '=' << (stats.executed[i] ? '1' : '0');
    }
    std::cerr << '\n';
}

} // namespace

bool Runtime::initialize(const RuntimeConfig& config)
{
    if (mInitialized)
    {
        return true;
    }

    mGpuDevice = gpu::createGpuDevice();
    if (!mGpuDevice)
    {
        return false;
    }

    if (!mGpuDevice->initialize(config.gpuDeviceDesc))
    {
        mGpuDevice.reset();
        return false;
    }

    mPhysicsSolver = std::make_unique<physics::PhysicsSolver>(*mGpuDevice, config.physicsDesc);
    if (!mPhysicsSolver || !mPhysicsSolver->initialize())
    {
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mGpuSceneSync = std::make_unique<gpu::GpuSceneSync>(*mGpuDevice);
    if (!mGpuSceneSync || !mGpuSceneSync->initialize())
    {
        mGpuSceneSync.reset();
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mRenderer = std::make_unique<graphics::Renderer>(*mGpuDevice, mResources, config.rendererDesc);
    if (!mRenderer->initialize())
    {
        mRenderer.reset();
        mGpuSceneSync->shutdown();
        mGpuSceneSync.reset();
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mInitialized = true;
    return true;
}

void Runtime::shutdown()
{
    if (!mInitialized)
    {
        return;
    }

    mRenderer.reset();

    if (mGpuSceneSync)
    {
        mGpuSceneSync->shutdown();
        mGpuSceneSync.reset();
    }

    if (mPhysicsSolver)
    {
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
    }

    if (mGpuDevice)
    {
        mGpuDevice->shutdown();
        mGpuDevice.reset();
    }

    mLastRenderStats          = {};
    mLastSyncedRenderRevision = ~0ull;
    mInitialized              = false;
}

void Runtime::tick(const common::FrameContext& frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    bool physicsStepSucceeded = true;
    if (mPhysicsSolver)
    {
        physicsStepSucceeded = mPhysicsSolver->step(frameContext, mWorld.physicsWorld());
        if (!physicsStepSucceeded)
        {
            logPhysicsStepFailure(frameContext, mPhysicsSolver->lastStageStats());
        }
    }
    std::unordered_map<common::EntityId, std::uint32_t> poseIndices;
    bool gpuSceneReady = false;
    if (physicsStepSucceeded && mGpuSceneSync && mPhysicsSolver)
    {
        const std::vector<gpu::GpuEntityPoseMappingEntry> mappings =
            buildPhysicsRenderableMappings(mWorld, poseIndices);
        if (mGpuSceneSync->syncEntityPoses(mPhysicsSolver->gpuSceneView().rigid.poses, mappings))
        {
            gpuSceneReady = true;
            gpu::GpuComputeBackendContext computeBackend{};
            if (mGpuDevice && mGpuDevice->tryGetPhysicsBackendContext(computeBackend) &&
                computeBackend.computeContext != nullptr)
            {
                computeBackend.computeContext->Flush();
            }
        }
    }
    const bool syncSkipped = syncWorldToRenderWorld();

    if (gpuSceneReady && mGpuSceneSync)
    {
        const std::vector<gpu::GpuRenderableMetadata> renderableMetadata =
            buildRenderableMetadata(mRenderWorld, mResources, poseIndices);
        if (mGpuSceneSync->syncRenderableMetadata(renderableMetadata))
        {
            mRenderWorld.setGpuEntityScene(mGpuSceneSync->sceneView(), poseIndices);
        }
        else
        {
            mRenderWorld.setGpuEntityScene({}, {});
        }
    }
    else
    {
        mRenderWorld.setGpuEntityScene({}, {});
    }

    mLastRenderStats                        = mRenderer->render(frameContext, mRenderWorld);
    mLastRenderStats.worldSyncSkippedFrames = syncSkipped ? 1u : 0u;
}

World& Runtime::getWorld() noexcept
{
    return mWorld;
}

const World& Runtime::getWorld() const noexcept
{
    return mWorld;
}

gpu::GpuDevice* Runtime::getGpuDevice() noexcept
{
    return mGpuDevice.get();
}

const gpu::GpuDevice* Runtime::getGpuDevice() const noexcept
{
    return mGpuDevice.get();
}

physics::PhysicsSolver* Runtime::getPhysicsSolver() noexcept
{
    return mPhysicsSolver.get();
}

const physics::PhysicsSolver* Runtime::getPhysicsSolver() const noexcept
{
    return mPhysicsSolver.get();
}

const graphics::RenderStats& Runtime::lastRenderStats() const noexcept
{
    return mLastRenderStats;
}

graphics::RenderResourceManager& Runtime::getResources() noexcept
{
    return mResources;
}

const graphics::RenderResourceManager& Runtime::getResources() const noexcept
{
    return mResources;
}

bool Runtime::syncWorldToRenderWorld()
{
    if (mWorld.renderRevision() == mLastSyncedRenderRevision)
    {
        return true;
    }

    detail::syncWorldToRenderWorld(mWorld, mRenderWorld);
    mWorld.clearRenderDirtyEntities();
    mLastSyncedRenderRevision = mWorld.renderRevision();
    return false;
}

} // namespace cressim::neo::engine
