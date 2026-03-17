#include "engine/runtime.h"

#include "engine/world_to_render_world_sync.h"

#include <iostream>
#include <unordered_map>

namespace cressim::neo::engine
{

namespace
{

struct RenderObjectPoseData
{
    std::vector<Diligent::float4> positions;
    std::vector<Diligent::float4> orientations;
    std::vector<Diligent::float4> scales;
};

std::unordered_map<common::EntityId, std::uint32_t> buildRenderObjectPoseIndices(
    const graphics::RenderWorld& renderWorld, const gpu::GpuSceneLayoutDesc& layout)
{
    std::unordered_map<common::EntityId, std::uint32_t> poseIndices;
    poseIndices.reserve(renderWorld.renderables().size());
    for (const graphics::RenderableInstance& renderable : renderWorld.renderables())
    {
        if (renderable.objectSlot == 0xffffffffu)
        {
            continue;
        }
        const std::uint32_t objectIndex =
            renderable.envIndex * layout.maxObjectsPerEnv + renderable.objectSlot;
        if (objectIndex >= layout.totalObjectCapacity())
        {
            continue;
        }
        poseIndices[renderable.entityId] = objectIndex;
    }
    return poseIndices;
}

RenderObjectPoseData buildRenderObjectPoseData(const graphics::RenderWorld& renderWorld,
                                               const gpu::GpuSceneLayoutDesc& layout)
{
    RenderObjectPoseData poseData{};
    poseData.positions.resize(layout.totalObjectCapacity());
    poseData.orientations.resize(layout.totalObjectCapacity(),
                                 Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f});
    poseData.scales.resize(layout.totalObjectCapacity(), Diligent::float4{1.0f, 1.0f, 1.0f, 0.0f});

    for (const graphics::RenderableInstance& renderable : renderWorld.renderables())
    {
        if (renderable.objectSlot == 0xffffffffu)
        {
            continue;
        }
        const std::uint32_t objectIndex =
            renderable.envIndex * layout.maxObjectsPerEnv + renderable.objectSlot;
        if (objectIndex >= poseData.positions.size())
        {
            continue;
        }

        poseData.positions[objectIndex] = Diligent::float4{
            renderable.worldTransform.position.x, renderable.worldTransform.position.y,
            renderable.worldTransform.position.z, 1.0f};
        poseData.orientations[objectIndex] = Diligent::float4{
            renderable.worldTransform.rotation.q.x, renderable.worldTransform.rotation.q.y,
            renderable.worldTransform.rotation.q.z, renderable.worldTransform.rotation.q.w};
        poseData.scales[objectIndex] =
            Diligent::float4{renderable.worldTransform.scale.x, renderable.worldTransform.scale.y,
                             renderable.worldTransform.scale.z, 0.0f};
    }

    return poseData;
}

std::vector<gpu::GpuEntityPoseMappingEntry> buildPhysicsRenderableMappings(
    const World& world, const graphics::RenderWorld& renderWorld,
    const gpu::GpuSceneLayoutDesc& layout)
{
    std::vector<gpu::GpuEntityPoseMappingEntry> mappings;

    const auto& rigidBodies = world.physicsWorld().rigidBodySoA();
    const auto& renderables = renderWorld.renderables();
    if (rigidBodies.entityIds.empty() || renderables.empty())
    {
        return mappings;
    }

    std::unordered_map<common::EntityId, std::uint32_t> rigidBodyIndexByEntity;
    rigidBodyIndexByEntity.reserve(rigidBodies.entityIds.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(rigidBodies.entityIds.size()); ++i)
    {
        rigidBodyIndexByEntity.emplace(rigidBodies.entityIds[i], i);
    }

    mappings.reserve(renderables.size());
    for (const graphics::RenderableInstance& renderable : renderables)
    {
        if (renderable.objectSlot == 0xffffffffu)
        {
            continue;
        }
        const auto rigidBodyIt = rigidBodyIndexByEntity.find(renderable.entityId);
        if (rigidBodyIt == rigidBodyIndexByEntity.end())
        {
            continue;
        }

        gpu::GpuEntityPoseMappingEntry entry{};
        entry.sourcePoseIndex = rigidBodyIt->second;
        entry.objectIndex = renderable.envIndex * layout.maxObjectsPerEnv + renderable.objectSlot;
        mappings.push_back(entry);
    }

    return mappings;
}

std::vector<gpu::GpuRenderableMetadata> buildRenderableMetadata(
    const graphics::RenderWorld& renderWorld, const graphics::RenderResourceManager& resources,
    const gpu::GpuSceneLayoutDesc& layout)
{
    std::vector<gpu::GpuRenderableMetadata> metadata(layout.totalObjectCapacity());
    const auto& renderables = renderWorld.renderables();

    for (const graphics::RenderableInstance& renderable : renderables)
    {
        if (renderable.objectSlot == 0xffffffffu)
        {
            continue;
        }
        const std::uint32_t objectIndex =
            renderable.envIndex * layout.maxObjectsPerEnv + renderable.objectSlot;
        if (objectIndex >= metadata.size())
        {
            continue;
        }
        gpu::GpuRenderableMetadata entry{};
        entry.objectSlot = renderable.objectSlot;
        entry.envIndex   = renderable.envIndex;
        entry.flags |= gpu::GpuRenderableFlag_Active;
        const graphics::MaterialResourceDesc* material =
            resources.tryGetMaterial(renderable.material);
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

        if (material != nullptr && material->blendMode != graphics::BlendMode::Transparent)
        {
            entry.flags |= gpu::GpuRenderableFlag_UsesGpuPose;
        }

        metadata[objectIndex] = entry;
    }

    return metadata;
}

std::vector<gpu::GpuCameraInput> buildCameraInputs(const graphics::RenderWorld& renderWorld,
                                                   const gpu::GpuSceneLayoutDesc& layout)
{
    std::vector<gpu::GpuCameraInput> inputs(layout.totalCameraCapacity());
    for (const graphics::CameraData& camera : renderWorld.cameras())
    {
        if (camera.cameraSlot == 0xffffffffu)
        {
            continue;
        }
        const std::uint32_t cameraIndex =
            camera.envIndex * layout.maxCamerasPerEnv + camera.cameraSlot;
        if (cameraIndex >= inputs.size())
        {
            continue;
        }
        gpu::GpuCameraInput input{};
        input.position =
            Diligent::float4{camera.worldTransform.position.x, camera.worldTransform.position.y,
                             camera.worldTransform.position.z, 1.0f};
        input.orientation = Diligent::float4{
            camera.worldTransform.rotation.q.x, camera.worldTransform.rotation.q.y,
            camera.worldTransform.rotation.q.z, camera.worldTransform.rotation.q.w};
        float aspect = camera.aspectRatio;
        if (aspect <= 0.0f && camera.outputWidth > 0u && camera.outputHeight > 0u)
        {
            aspect =
                static_cast<float>(camera.outputWidth) / static_cast<float>(camera.outputHeight);
        }
        if (aspect <= 0.0f)
        {
            aspect = 1.0f;
        }
        input.projectionParams =
            Diligent::float4{camera.verticalFovDegrees, aspect, camera.nearClip, camera.farClip};
        input.envIndex      = camera.envIndex;
        input.cameraSlot    = camera.cameraSlot;
        input.active        = 1u;
        inputs[cameraIndex] = input;
    }
    return inputs;
}

std::vector<gpu::GpuDirectionalLightInput> buildLightInputs(
    const graphics::RenderWorld& renderWorld, const gpu::GpuSceneLayoutDesc& layout)
{
    std::vector<gpu::GpuDirectionalLightInput> inputs(layout.totalLightCapacity());
    for (const graphics::DirectionalLightData& light : renderWorld.directionalLights())
    {
        if (light.lightSlot == 0xffffffffu)
        {
            continue;
        }
        const std::uint32_t lightIndex = light.envIndex * layout.maxLightsPerEnv + light.lightSlot;
        if (lightIndex >= inputs.size())
        {
            continue;
        }
        gpu::GpuDirectionalLightInput input{};
        input.directionIntensity = Diligent::float4{light.direction.x, light.direction.y,
                                                    light.direction.z, light.intensity};
        input.color = Diligent::float4{light.color.x, light.color.y, light.color.z, 0.0f};
        input.shadowParams =
            Diligent::float4{light.shadowDistance, light.shadowFadeDistance, 0.0f, 0.0f};
        input.envIndex     = light.envIndex;
        input.lightSlot    = light.lightSlot;
        input.active       = 1u;
        inputs[lightIndex] = input;
    }
    return inputs;
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
    if (!mGpuSceneSync || !mGpuSceneSync->initialize(config.rendererDesc.sceneLayout))
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
    const bool syncSkipped = syncWorldToRenderWorld();

    std::unordered_map<common::EntityId, std::uint32_t> poseIndices;
    bool gpuSceneReady = false;
    if (mGpuSceneSync)
    {
        poseIndices = buildRenderObjectPoseIndices(mRenderWorld, mGpuSceneSync->layout());
        const RenderObjectPoseData poseData =
            buildRenderObjectPoseData(mRenderWorld, mGpuSceneSync->layout());
        gpuSceneReady = mGpuSceneSync->syncEntityPoseData(poseData.positions, poseData.orientations,
                                                          poseData.scales);
    }
    if (gpuSceneReady && physicsStepSucceeded && mGpuSceneSync && mPhysicsSolver)
    {
        const std::vector<gpu::GpuEntityPoseMappingEntry> mappings =
            buildPhysicsRenderableMappings(mWorld, mRenderWorld, mGpuSceneSync->layout());
        if (mGpuSceneSync->syncEntityPoses(mPhysicsSolver->gpuSceneView().rigid.poses, mappings))
        {
            gpu::GpuComputeBackendContext computeBackend{};
            if (mGpuDevice && mGpuDevice->tryGetPhysicsBackendContext(computeBackend) &&
                computeBackend.computeContext != nullptr)
            {
                computeBackend.computeContext->Flush();
            }
        }
        else
        {
            gpuSceneReady = false;
        }
    }
    if (gpuSceneReady && mGpuSceneSync)
    {
        const std::vector<gpu::GpuRenderableMetadata> renderableMetadata =
            buildRenderableMetadata(mRenderWorld, mResources, mGpuSceneSync->layout());
        const std::vector<gpu::GpuCameraInput> cameraInputs =
            buildCameraInputs(mRenderWorld, mGpuSceneSync->layout());
        const std::vector<gpu::GpuDirectionalLightInput> lightInputs =
            buildLightInputs(mRenderWorld, mGpuSceneSync->layout());
        if (mGpuSceneSync->syncRenderableMetadata(renderableMetadata) &&
            mGpuSceneSync->syncCameraInputs(cameraInputs) &&
            mGpuSceneSync->syncLightInputs(lightInputs))
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
