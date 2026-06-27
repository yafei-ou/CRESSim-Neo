#include "engine/runtime.h"

#include "common/logger.h"
#include "engine/custom_compute_service.h"
#include "engine/entity_scene_gpu_state.h"
#include "engine/runtime_internal.h"
#include "engine/shared_buffer_service.h"

#include <algorithm>
#include <unordered_map>

namespace cressim::neo::engine
{

namespace
{

bool hasGraphicsBackendContext(gpu::GpuDevice &device)
{
    gpu::GpuGraphicsBackendContext graphicsContext{};
    return device.tryGetGraphicsBackendContext(graphicsContext) &&
           graphicsContext.renderDevice != nullptr && graphicsContext.graphicsContext != nullptr;
}

bool hasPhysicsBackendContext(gpu::GpuDevice &device)
{
    gpu::GpuComputeBackendContext computeContext{};
    return device.tryGetPhysicsBackendContext(computeContext) &&
           computeContext.renderDevice != nullptr && computeContext.computeContext != nullptr;
}

void ensureDeviceFrameActive(gpu::GpuDevice *device, const common::FrameContext &frameContext,
                             bool &inOutDeviceFrameActive)
{
    if (device == nullptr || inOutDeviceFrameActive)
    {
        return;
    }

    device->beginFrame(frameContext);
    inOutDeviceFrameActive = true;
}

bool syncGpuScene(World &world, gpu::GpuDevice *device, EntitySceneGpuState *entitySceneState,
                  RenderSceneUploader *uploader, physics::PhysicsSolver *physicsSolver,
                  bool usePhysicsPoses, std::uint64_t &lastEntityPoseRevision,
                  std::uint64_t &lastRenderableMetadataRevision,
                  std::uint64_t &lastRenderableQueueInfoRevision,
                  std::uint64_t &lastSoftBodyVertexBindingRevision,
                  std::uint64_t &lastCameraInputRevision, std::uint64_t &lastLightInputRevision,
                  std::uint64_t &lastLocalLightSelectionRevision)
{
    if (entitySceneState == nullptr || uploader == nullptr)
    {
        world.setGpuEntityScene({});
        return false;
    }

    const std::uint64_t entityPoseRevision = world.entityPoseRevision();
    const std::uint64_t renderableMetadataRevision = world.renderableMetadataRevision();
    const std::uint64_t renderableQueueInfoRevision = world.renderableQueueInfoRevision();
    const std::uint64_t softBodyVertexBindingRevision = world.softBodyVertexBindingRevision();
    const std::uint64_t cameraInputRevision = world.cameraInputRevision();
    const std::uint64_t lightInputRevision = world.lightInputRevision();
    const std::uint64_t localLightSelectionRevision = world.localLightSelectionRevision();

    const bool needsEntityPoseUpload = entityPoseRevision != lastEntityPoseRevision;
    const bool needsRenderableMetadataUpload =
        renderableMetadataRevision != lastRenderableMetadataRevision;
    const bool needsRenderableQueueInfoUpload =
        renderableQueueInfoRevision != lastRenderableQueueInfoRevision;
    const bool needsSoftBodyVertexBindingUpload =
        softBodyVertexBindingRevision != lastSoftBodyVertexBindingRevision;
    const bool needsCameraInputUpload = cameraInputRevision != lastCameraInputRevision;
    const bool needsLightInputUpload = lightInputRevision != lastLightInputRevision;
    const bool needsLocalLightSelectionUpload =
        localLightSelectionRevision != lastLocalLightSelectionRevision;

    bool gpuSceneReady = true;
    if (needsEntityPoseUpload)
    {
        gpuSceneReady = entitySceneState->uploadAuthoredEntityPoses(
            world.entityPosePositions(), world.entityPoseOrientations(), world.entityPoseScales());
    }
    if (gpuSceneReady && usePhysicsPoses && physicsSolver != nullptr)
    {
        if (!entitySceneState->applyMappedEntityPoses(physicsSolver->gpuSceneView().rigid.poses,
                                                      world.physicsRenderableMappings()))
        {
            gpuSceneReady = false;
        }
    }

    if (gpuSceneReady && needsRenderableMetadataUpload)
    {
        gpuSceneReady = uploader->uploadRenderableMetadata(world.renderableMetadata());
    }
    if (gpuSceneReady && needsRenderableQueueInfoUpload)
    {
        gpuSceneReady = uploader->uploadRenderableQueueInfo(world.renderableQueueInfo());
    }
    if (gpuSceneReady && needsSoftBodyVertexBindingUpload)
    {
        gpuSceneReady = uploader->uploadSoftBodyVertexBindings(world.softBodyVertexBindings());
    }
    if (gpuSceneReady && needsCameraInputUpload)
    {
        gpuSceneReady = uploader->uploadCameraInputs(world.cameraInputs());
    }
    if (gpuSceneReady && needsLightInputUpload)
    {
        gpuSceneReady = uploader->uploadLightInputs(world.lightInputs());
    }
    if (gpuSceneReady && needsLocalLightSelectionUpload)
    {
        gpuSceneReady = uploader->uploadLocalLightSelections(world.localLightSelections());
    }

    if (gpuSceneReady && (!device || device->waitForPhysicsOnGraphics()))
    {
        world.setGpuEntityScene(
            uploader->sceneView(entitySceneState->poseView(), entitySceneState->entityCount()));
        lastEntityPoseRevision = entityPoseRevision;
        lastRenderableMetadataRevision = renderableMetadataRevision;
        lastRenderableQueueInfoRevision = renderableQueueInfoRevision;
        lastSoftBodyVertexBindingRevision = softBodyVertexBindingRevision;
        lastCameraInputRevision = cameraInputRevision;
        lastLightInputRevision = lightInputRevision;
        lastLocalLightSelectionRevision = localLightSelectionRevision;
        return true;
    }

    CRESSIM_LOG_WARNING("Runtime: GPU scene sync failed.");
    world.setGpuEntityScene({});
    return false;
}

} // namespace

Runtime::Runtime() = default;

Runtime::~Runtime()
{
    shutdown();
}

bool Runtime::initialize(const RuntimeConfig &config)
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

    mUltrasoundSystem = std::make_unique<UltrasoundSystem>(*mGpuDevice, *mPhysicsSolver);
    if (!mUltrasoundSystem || !mUltrasoundSystem->initialize())
    {
        mUltrasoundSystem.reset();
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mWorld.setSceneLayout(config.sceneLayout);

    if (hasGraphicsBackendContext(*mGpuDevice) && hasPhysicsBackendContext(*mGpuDevice))
    {
        mEntitySceneGpuState = std::make_unique<EntitySceneGpuState>(*mGpuDevice);
        mRenderSceneUploader = std::make_unique<RenderSceneUploader>(*mGpuDevice);
        if (!mEntitySceneGpuState || !mEntitySceneGpuState->initialize() ||
            !mRenderSceneUploader || !mRenderSceneUploader->initialize(config.sceneLayout))
        {
            if (mEntitySceneGpuState)
            {
                mEntitySceneGpuState->shutdown();
                mEntitySceneGpuState.reset();
            }
            mRenderSceneUploader.reset();
            mUltrasoundSystem->shutdown();
            mUltrasoundSystem.reset();
            mPhysicsSolver->shutdown();
            mPhysicsSolver.reset();
            mGpuDevice->shutdown();
            mGpuDevice.reset();
            return false;
        }
    }

    mRenderer = std::make_unique<graphics::Renderer>(*mGpuDevice, mResources, config.rendererDesc);
    if (!mRenderer->initialize())
    {
        mRenderer.reset();
        if (mRenderSceneUploader)
        {
            mRenderSceneUploader->shutdown();
            mRenderSceneUploader.reset();
        }
        if (mEntitySceneGpuState)
        {
            mEntitySceneGpuState->shutdown();
            mEntitySceneGpuState.reset();
        }
        if (mUltrasoundSystem)
        {
            mUltrasoundSystem->shutdown();
            mUltrasoundSystem.reset();
        }
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mCustomComputeService = std::make_unique<CustomComputeService>(*mGpuDevice);
    mSharedBufferService  = std::make_unique<SharedBufferService>(*mGpuDevice);
    mInitialized          = true;
    return true;
}

void Runtime::shutdown()
{
    if (!mInitialized)
    {
        return;
    }

    if (mDeviceFrameActive && mGpuDevice)
    {
        mGpuDevice->endFrame(mLastFrameContext);
        mDeviceFrameActive = false;
    }

    mRenderer.reset();
    if (mCustomComputeService)
    {
        mCustomComputeService->clear();
        mCustomComputeService.reset();
    }
    if (mSharedBufferService)
    {
        mSharedBufferService->clear();
        mSharedBufferService.reset();
    }

    if (mRenderSceneUploader)
    {
        mRenderSceneUploader->shutdown();
        mRenderSceneUploader.reset();
    }
    if (mEntitySceneGpuState)
    {
        mEntitySceneGpuState->shutdown();
        mEntitySceneGpuState.reset();
    }
    if (mUltrasoundSystem)
    {
        mUltrasoundSystem->shutdown();
        mUltrasoundSystem.reset();
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

    mLastRenderStats    = {};
    mRenderFrameOptions = {};
    mLastFrameContext   = {};
    mDeviceFrameActive  = false;
    mWorldUploaded      = false;
    mHasPhysicsState    = false;
    mInitialized        = false;
    mLastUploadedEntityPoseRevision = 0u;
    mLastUploadedRenderableMetadataRevision = 0u;
    mLastUploadedRenderableQueueInfoRevision = 0u;
    mLastUploadedSoftBodyVertexBindingRevision = 0u;
    mLastUploadedCameraInputRevision = 0u;
    mLastUploadedLightInputRevision = 0u;
    mLastUploadedLocalLightSelectionRevision = 0u;
}

void Runtime::prepare()
{
    if (!mInitialized)
    {
        return;
    }

    mWorld.ensureRenderStateUpToDate(mResources);
    if (mUltrasoundSystem)
    {
        if (!mUltrasoundSystem->prepare(mWorld))
        {
            CRESSIM_LOG_WARNING("Runtime: ultrasound prepare failed.");
        }
    }
    mWorldUploaded   = false;
    mHasPhysicsState = false;
}

bool Runtime::uploadWorld()
{
    if (!mInitialized)
    {
        return false;
    }

    bool uploaded = true;
    if (mPhysicsSolver && !mPhysicsSolver->syncWorldState(mWorld.physicsWorld()))
    {
        CRESSIM_LOG_ERROR("Runtime: physics world upload failed.");
        uploaded = false;
    }

    if (uploaded && mEntitySceneGpuState && mRenderSceneUploader &&
        !syncGpuScene(mWorld, mGpuDevice.get(), mEntitySceneGpuState.get(),
                      mRenderSceneUploader.get(), mPhysicsSolver.get(), false,
                      mLastUploadedEntityPoseRevision, mLastUploadedRenderableMetadataRevision,
                      mLastUploadedRenderableQueueInfoRevision,
                      mLastUploadedSoftBodyVertexBindingRevision,
                      mLastUploadedCameraInputRevision, mLastUploadedLightInputRevision,
                      mLastUploadedLocalLightSelectionRevision))
    {
        uploaded = false;
    }

    mWorldUploaded   = uploaded;
    mHasPhysicsState = false;
    return uploaded;
}

bool Runtime::stepPhysics(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return false;
    }
    if (!mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: stepPhysics() requires uploadWorld() after prepare() and "
                          "before execution.");
        return false;
    }

    mLastFrameContext = frameContext;
    ensureDeviceFrameActive(mGpuDevice.get(), frameContext, mDeviceFrameActive);

    bool physicsStepSucceeded = true;
    if (mPhysicsSolver)
    {
        physicsStepSucceeded = mPhysicsSolver->step(frameContext, mWorld.physicsWorld());
        if (!physicsStepSucceeded)
        {
            CRESSIM_LOG_ERROR("Runtime: physics step failed at frame ", frameContext.frameIndex,
                              " (dt=", frameContext.deltaSeconds, ").");
        }
    }
    if (physicsStepSucceeded)
    {
        mHasPhysicsState = true;
    }
    return physicsStepSucceeded;
}

bool Runtime::stepSimulationSensors(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return false;
    }

    mLastFrameContext = frameContext;

    if (!mUltrasoundSystem)
    {
        return true;
    }

    ensureDeviceFrameActive(mGpuDevice.get(), frameContext, mDeviceFrameActive);

    const bool succeeded = mUltrasoundSystem->execute(frameContext, mWorld);
    if (!succeeded)
    {
        CRESSIM_LOG_WARNING("Runtime: ultrasound step failed at frame ", frameContext.frameIndex,
                            ".");
    }
    return succeeded;
}

void Runtime::stepVisualSensors(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    (void)syncGpuScene(mWorld, mGpuDevice.get(), mEntitySceneGpuState.get(),
                       mRenderSceneUploader.get(), mPhysicsSolver.get(), mHasPhysicsState,
                       mLastUploadedEntityPoseRevision, mLastUploadedRenderableMetadataRevision,
                       mLastUploadedRenderableQueueInfoRevision,
                       mLastUploadedSoftBodyVertexBindingRevision,
                       mLastUploadedCameraInputRevision, mLastUploadedLightInputRevision,
                       mLastUploadedLocalLightSelectionRevision);

    mLastFrameContext = frameContext;

    ensureDeviceFrameActive(mGpuDevice.get(), frameContext, mDeviceFrameActive);

    physics::PhysicsGpuSceneView physicsSceneView{};
    const physics::PhysicsGpuSceneView *physicsScenePtr = nullptr;
    if (mPhysicsSolver)
    {
        physicsSceneView = mPhysicsSolver->gpuSceneView();
        physicsScenePtr  = &physicsSceneView;
    }

    mLastRenderStats = mRenderer->render(frameContext, mWorld.hostSceneView(), physicsScenePtr,
                                         mRenderFrameOptions);
}

void Runtime::endFrame(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    mLastFrameContext = frameContext;
    if (!mDeviceFrameActive || !mGpuDevice)
    {
        return;
    }

    mGpuDevice->endFrame(frameContext);
    mDeviceFrameActive = false;
}

World &Runtime::getWorld() noexcept
{
    return mWorld;
}

const World &Runtime::getWorld() const noexcept
{
    return mWorld;
}

gpu::GpuDevice *Runtime::getGpuDevice() noexcept
{
    return mGpuDevice.get();
}

const gpu::GpuDevice *Runtime::getGpuDevice() const noexcept
{
    return mGpuDevice.get();
}

physics::PhysicsSolver *Runtime::getPhysicsSolver() noexcept
{
    return mPhysicsSolver.get();
}

const physics::PhysicsSolver *Runtime::getPhysicsSolver() const noexcept
{
    return mPhysicsSolver.get();
}

const graphics::RenderStats &Runtime::lastRenderStats() const noexcept
{
    return mLastRenderStats;
}

void Runtime::setRenderFrameOptions(const graphics::RenderFrameOptions &options) noexcept
{
    mRenderFrameOptions = options;
}

const graphics::RenderFrameOptions &Runtime::renderFrameOptions() const noexcept
{
    return mRenderFrameOptions;
}

graphics::RenderResourceManager &Runtime::getResources() noexcept
{
    return mResources;
}

const graphics::RenderResourceManager &Runtime::getResources() const noexcept
{
    return mResources;
}

SharedBufferHandle Runtime::createSharedBuffer(const SharedBufferDesc &desc)
{
    return mInitialized && mSharedBufferService != nullptr
               ? mSharedBufferService->createBuffer(desc)
               : SharedBufferHandle{};
}

bool Runtime::destroySharedBuffer(const SharedBufferHandle handle)
{
    return mInitialized && mSharedBufferService != nullptr &&
           mSharedBufferService->destroyBuffer(handle);
}

std::vector<SharedBufferInfo> Runtime::listSharedBuffers() const
{
    return mInitialized && mSharedBufferService != nullptr ? mSharedBufferService->listBuffers()
                                                           : std::vector<SharedBufferInfo>{};
}

bool Runtime::tryGetSharedBufferInfo(const SharedBufferHandle handle,
                                     SharedBufferInfo &outInfo) const
{
    return mInitialized && mSharedBufferService != nullptr &&
           mSharedBufferService->tryGetBufferInfo(handle, outInfo);
}

bool Runtime::tryGetSharedBufferCudaView(const SharedBufferHandle handle,
                                         SharedBufferCudaView &outView) const
{
    return mInitialized && mSharedBufferService != nullptr &&
           mSharedBufferService->tryGetCudaView(handle, outView);
}

std::shared_ptr<void> Runtime::retainSharedBuffer(const SharedBufferHandle handle) const
{
    return mInitialized && mSharedBufferService != nullptr
               ? mSharedBufferService->retainBuffer(handle)
               : std::shared_ptr<void>{};
}

std::shared_ptr<void> RuntimeInternalAccess::retainSharedBufferLease(
    const Runtime &runtime, const SharedBufferHandle handle)
{
    return runtime.retainSharedBuffer(handle);
}

bool Runtime::syncSharedBufferToCuda(const SharedBufferHandle handle)
{
    if (!mInitialized || mSharedBufferService == nullptr || mGpuDevice == nullptr)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mGpuDevice->tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        return false;
    }

    return mSharedBufferService->synchronizeToCuda(handle, computeBackend.computeContext);
}

bool Runtime::syncSharedBufferFromCuda(const SharedBufferHandle handle)
{
    if (!mInitialized || mSharedBufferService == nullptr || mGpuDevice == nullptr)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mGpuDevice->tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        return false;
    }

    return mSharedBufferService->synchronizeFromCuda(handle, computeBackend.computeContext);
}

bool Runtime::tryGetPreparedRigidLayoutMapping(RigidLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mInitialized || mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld    = mWorld.physicsWorld();
    const physics::RigidBodySoAHost &rigidBodies = physicsWorld.rigidBodySoA();
    const physics::ColliderSoAHost &colliders    = physicsWorld.colliderSoA();
    const physics::BodyColliderMappingHost &bodyColliderMapping =
        physicsWorld.bodyColliderMapping();
    outMapping.rigidBodyCount              = static_cast<std::uint32_t>(rigidBodies.size());
    outMapping.colliderCount               = static_cast<std::uint32_t>(colliders.size());
    outMapping.layoutRevision              = physicsWorld.rigidBodyTopologyRevision();
    outMapping.rigidBodyIds                = rigidBodies.rigidBodyIds;
    outMapping.rigidBodyEntityIds          = rigidBodies.entityIds;
    outMapping.rigidBodyEnvironmentIndices = rigidBodies.environmentIndices;
    outMapping.colliderIds                 = colliders.colliderIds;
    outMapping.colliderEntityIds           = colliders.entityIds;
    outMapping.colliderOwnerBodyIds        = colliders.ownerRigidBodyIds;
    outMapping.colliderOwnerBodyIndices    = colliders.ownerRigidBodyIndices;
    outMapping.colliderEnvironmentIndices  = colliders.environmentIndices;
    outMapping.colliderShapeTypes          = colliders.shapeTypes;
    outMapping.colliderEnabledFlags        = colliders.enabledFlags;
    outMapping.colliderCollisionLayers     = colliders.collisionLayers;
    outMapping.colliderCollisionMasks      = colliders.collisionMasks;
    outMapping.colliderShapeParams         = colliders.shapeParams;
    outMapping.bodyColliderOffsets         = bodyColliderMapping.colliderOffsets;
    outMapping.bodyColliderCounts          = bodyColliderMapping.colliderCounts;
    outMapping.bodyColliderIndices         = bodyColliderMapping.colliderIndices;
    outMapping.colliderLocalPositions.reserve(colliders.localPositions.size());
    outMapping.colliderLocalRotations.reserve(colliders.localOrientations.size());
    for (const Diligent::float4 &position : colliders.localPositions)
    {
        outMapping.colliderLocalPositions.push_back({position.x, position.y, position.z});
    }
    for (const Diligent::float4 &rotation : colliders.localOrientations)
    {
        outMapping.colliderLocalRotations.push_back(
            Diligent::QuaternionF{rotation.x, rotation.y, rotation.z, rotation.w});
    }
    return true;
}

bool Runtime::tryGetPreparedConstraintLayoutMapping(ConstraintLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mInitialized || mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld    = mWorld.physicsWorld();
    const physics::RigidBodySoAHost &rigidBodies = physicsWorld.rigidBodySoA();
    const auto &rigidParticleAttachments = physicsWorld.rigidParticleAttachmentConstraintSnapshot();
    const auto &rigidDistanceConstraints = physicsWorld.rigidDistanceConstraintSnapshot();
    const auto &routedCables             = physicsWorld.routedCableConstraintSnapshot();

    std::unordered_map<common::EntityId, std::uint32_t> bodyIndexByEntity;
    bodyIndexByEntity.reserve(rigidBodies.entityIds.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(rigidBodies.entityIds.size()); ++i)
    {
        bodyIndexByEntity.emplace(rigidBodies.entityIds[i], i);
    }

    outMapping.layoutRevision = std::max({physicsWorld.rigidParticleAttachmentDefinitionRevision(),
                                          physicsWorld.rigidDistanceConstraintDefinitionRevision(),
                                          physicsWorld.routedCableDefinitionRevision()});

    auto &attachmentMapping = outMapping.rigidParticleAttachments;
    attachmentMapping.count = static_cast<std::uint32_t>(rigidParticleAttachments.size());
    attachmentMapping.constraintIds.reserve(rigidParticleAttachments.size());
    attachmentMapping.environmentIndices.reserve(rigidParticleAttachments.size());
    attachmentMapping.rigidBodyIds.reserve(rigidParticleAttachments.size());
    attachmentMapping.rigidBodyIndices.reserve(rigidParticleAttachments.size());
    attachmentMapping.particleEntityIds.reserve(rigidParticleAttachments.size());
    attachmentMapping.particleReferenceTypes.reserve(rigidParticleAttachments.size());
    attachmentMapping.particleLocalIndices.reserve(rigidParticleAttachments.size());
    attachmentMapping.enabledFlags.reserve(rigidParticleAttachments.size());
    for (const auto &constraint : rigidParticleAttachments)
    {
        const auto bodyIt = bodyIndexByEntity.find(constraint.rigidBodyEntityId);
        const std::uint32_t bodyIndex =
            bodyIt != bodyIndexByEntity.end() ? bodyIt->second : 0xffffffffu;
        attachmentMapping.constraintIds.push_back(constraint.constraintId);
        attachmentMapping.environmentIndices.push_back(
            bodyIndex < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndex]
                : 0u);
        attachmentMapping.rigidBodyIds.push_back(bodyIndex < rigidBodies.rigidBodyIds.size()
                                                     ? rigidBodies.rigidBodyIds[bodyIndex]
                                                     : physics::kInvalidRigidBodyId);
        attachmentMapping.rigidBodyIndices.push_back(bodyIndex);
        attachmentMapping.particleEntityIds.push_back(constraint.particle.entityId);
        attachmentMapping.particleReferenceTypes.push_back(
            static_cast<std::uint32_t>(constraint.particle.type));
        attachmentMapping.particleLocalIndices.push_back(constraint.particle.localParticleIndex);
        attachmentMapping.enabledFlags.push_back(constraint.enabled ? 1u : 0u);
    }

    auto &distanceMapping = outMapping.rigidDistanceConstraints;
    distanceMapping.count = static_cast<std::uint32_t>(rigidDistanceConstraints.size());
    distanceMapping.constraintIds.reserve(rigidDistanceConstraints.size());
    distanceMapping.environmentIndices.reserve(rigidDistanceConstraints.size());
    distanceMapping.rigidBodyIdsA.reserve(rigidDistanceConstraints.size());
    distanceMapping.rigidBodyIdsB.reserve(rigidDistanceConstraints.size());
    distanceMapping.rigidBodyIndicesA.reserve(rigidDistanceConstraints.size());
    distanceMapping.rigidBodyIndicesB.reserve(rigidDistanceConstraints.size());
    distanceMapping.enabledFlags.reserve(rigidDistanceConstraints.size());
    for (const auto &constraint : rigidDistanceConstraints)
    {
        const auto bodyItA = bodyIndexByEntity.find(constraint.entityA);
        const auto bodyItB = bodyIndexByEntity.find(constraint.entityB);
        const std::uint32_t bodyIndexA =
            bodyItA != bodyIndexByEntity.end() ? bodyItA->second : 0xffffffffu;
        const std::uint32_t bodyIndexB =
            bodyItB != bodyIndexByEntity.end() ? bodyItB->second : 0xffffffffu;
        distanceMapping.constraintIds.push_back(constraint.constraintId);
        distanceMapping.environmentIndices.push_back(
            bodyIndexA < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndexA]
                : 0u);
        distanceMapping.rigidBodyIdsA.push_back(bodyIndexA < rigidBodies.rigidBodyIds.size()
                                                    ? rigidBodies.rigidBodyIds[bodyIndexA]
                                                    : physics::kInvalidRigidBodyId);
        distanceMapping.rigidBodyIdsB.push_back(bodyIndexB < rigidBodies.rigidBodyIds.size()
                                                    ? rigidBodies.rigidBodyIds[bodyIndexB]
                                                    : physics::kInvalidRigidBodyId);
        distanceMapping.rigidBodyIndicesA.push_back(bodyIndexA);
        distanceMapping.rigidBodyIndicesB.push_back(bodyIndexB);
        distanceMapping.enabledFlags.push_back(constraint.enabled ? 1u : 0u);
    }

    auto &routedCableMapping = outMapping.routedCables;
    routedCableMapping.count = static_cast<std::uint32_t>(routedCables.size());
    routedCableMapping.constraintIds.reserve(routedCables.size());
    routedCableMapping.environmentIndices.reserve(routedCables.size());
    routedCableMapping.routePointOffsets.reserve(routedCables.size());
    routedCableMapping.routePointCounts.reserve(routedCables.size());
    routedCableMapping.enabledFlags.reserve(routedCables.size());
    std::uint32_t routePointOffset = 0u;
    for (const auto &constraint : routedCables)
    {
        routedCableMapping.constraintIds.push_back(constraint.constraintId);
        routedCableMapping.routePointOffsets.push_back(routePointOffset);
        routedCableMapping.routePointCounts.push_back(
            static_cast<std::uint32_t>(constraint.routePoints.size()));
        routedCableMapping.enabledFlags.push_back(constraint.enabled ? 1u : 0u);

        std::uint32_t envIndex = 0u;
        bool haveEnv           = false;
        for (const auto &routePoint : constraint.routePoints)
        {
            const auto bodyIt = bodyIndexByEntity.find(routePoint.entityId);
            const std::uint32_t bodyIndex =
                bodyIt != bodyIndexByEntity.end() ? bodyIt->second : 0xffffffffu;
            if (!haveEnv && bodyIndex < rigidBodies.environmentIndices.size())
            {
                envIndex = rigidBodies.environmentIndices[bodyIndex];
                haveEnv  = true;
            }
            routedCableMapping.routePointRigidBodyIds.push_back(
                bodyIndex < rigidBodies.rigidBodyIds.size() ? rigidBodies.rigidBodyIds[bodyIndex]
                                                            : physics::kInvalidRigidBodyId);
            routedCableMapping.routePointRigidBodyIndices.push_back(bodyIndex);
            routedCableMapping.routePointLocalGuideOffsets.push_back(routePoint.localGuideOffset);
        }
        routedCableMapping.environmentIndices.push_back(envIndex);
        routePointOffset += static_cast<std::uint32_t>(constraint.routePoints.size());
    }

    return true;
}

bool Runtime::tryGetPreparedParticleLayoutMapping(ParticleLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mInitialized || mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld = mWorld.physicsWorld();
    physicsWorld.ensureDerivedStateUpToDate();

    const physics::ParticleSoAHost &particles = physicsWorld.particles();
    const auto &softBodies                    = physicsWorld.softBodySnapshot();
    const auto &fluids                        = physicsWorld.fluidSnapshot();
    const auto &strands                       = physicsWorld.strandSnapshot();
    outMapping.particleCount                  = static_cast<std::uint32_t>(particles.size());
    outMapping.softBodyCount                  = static_cast<std::uint32_t>(softBodies.size());
    outMapping.fluidCount                     = static_cast<std::uint32_t>(fluids.size());
    outMapping.strandCount                    = static_cast<std::uint32_t>(strands.size());
    outMapping.layoutRevision                 = physicsWorld.softParticleRevision();
    outMapping.environmentIndices             = particles.environmentIndices;
    outMapping.particleKinds                  = particles.particleKinds;
    outMapping.ownerTypes                     = particles.ownerTypes;
    outMapping.ownerIndices                   = particles.ownerIndices;
    outMapping.strandIds                      = particles.strandIds;
    outMapping.strandOrders                   = particles.strandOrders;
    outMapping.strandRoles                    = particles.strandRoles;
    outMapping.owningSoftBodyIndices          = particles.owningSoftBodyIndices;
    outMapping.particleMaterialIndices        = particles.particleMaterialIndices;
    outMapping.fluidMaterialIndices           = particles.fluidMaterialIndices;
    outMapping.phases                         = particles.phases;
    outMapping.collisionLayers                = particles.collisionLayers;
    outMapping.collisionMasks                 = particles.collisionMasks;
    outMapping.adjacencyOffsets               = particles.adjacencyOffsets;
    outMapping.adjacencyCounts                = particles.adjacencyCounts;

    outMapping.softBodyEntityIds.reserve(softBodies.size());
    outMapping.softBodyEnvironmentIndices.reserve(softBodies.size());
    outMapping.softBodyParticleOffsets.reserve(softBodies.size());
    outMapping.softBodyParticleCounts.reserve(softBodies.size());
    for (const auto &softBody : softBodies)
    {
        outMapping.softBodyEntityIds.push_back(softBody.entityId);
        outMapping.softBodyEnvironmentIndices.push_back(softBody.environmentIndex);
        outMapping.softBodyParticleOffsets.push_back(softBody.particleOffset);
        outMapping.softBodyParticleCounts.push_back(softBody.particleCount);
    }

    outMapping.fluidEntityIds.reserve(fluids.size());
    outMapping.fluidEnvironmentIndices.reserve(fluids.size());
    outMapping.fluidParticleOffsets.reserve(fluids.size());
    outMapping.fluidParticleCounts.reserve(fluids.size());
    for (const auto &fluid : fluids)
    {
        outMapping.fluidEntityIds.push_back(fluid.entityId);
        outMapping.fluidEnvironmentIndices.push_back(fluid.environmentIndex);
        outMapping.fluidParticleOffsets.push_back(fluid.particleOffset);
        outMapping.fluidParticleCounts.push_back(fluid.particleCount);
    }

    outMapping.strandEntityIds.reserve(strands.size());
    outMapping.strandEnvironmentIndices.reserve(strands.size());
    outMapping.strandParticleOffsets.reserve(strands.size());
    outMapping.strandParticleCounts.reserve(strands.size());
    for (const auto &strand : strands)
    {
        outMapping.strandEntityIds.push_back(strand.entityId);
        outMapping.strandEnvironmentIndices.push_back(strand.environmentIndex);
        outMapping.strandParticleOffsets.push_back(strand.particleOffset);
        outMapping.strandParticleCounts.push_back(strand.particleCount);
    }

    return true;
}

bool Runtime::tryGetPreparedJointLayoutMapping(JointLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mInitialized || mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld           = mWorld.physicsWorld();
    const physics::RigidBodySoAHost &rigidBodies        = physicsWorld.rigidBodySoA();
    const std::vector<physics::BallJointState> &balls   = physicsWorld.ballJointSnapshot();
    const std::vector<physics::HingeJointState> &hinges = physicsWorld.hingeJointSnapshot();
    const std::vector<physics::SphericalJointState> &spherical =
        physicsWorld.sphericalJointSnapshot();
    const std::vector<physics::SliderJointState> &sliders = physicsWorld.sliderJointSnapshot();
    const physics::RigidJointSceneHost &jointScene        = physicsWorld.rigidJointScene();

    outMapping.ballJointCount      = static_cast<std::uint32_t>(balls.size());
    outMapping.hingeJointCount     = static_cast<std::uint32_t>(hinges.size());
    outMapping.sphericalJointCount = static_cast<std::uint32_t>(spherical.size());
    outMapping.sliderJointCount    = static_cast<std::uint32_t>(sliders.size());
    outMapping.layoutRevision      = physicsWorld.rigidJointTopologyRevision();

    outMapping.ballJointIds.reserve(balls.size());
    outMapping.ballEnvironmentIndices.reserve(balls.size());
    outMapping.ballBodyIdsA.reserve(balls.size());
    outMapping.ballBodyIdsB.reserve(balls.size());
    outMapping.ballBodyIndicesA.reserve(balls.size());
    outMapping.ballBodyIndicesB.reserve(balls.size());
    for (std::size_t i = 0; i < balls.size(); ++i)
    {
        const std::uint32_t bodyIndexA = jointScene.ball.bodyIndicesA[i];
        const std::uint32_t bodyIndexB = jointScene.ball.bodyIndicesB[i];
        outMapping.ballJointIds.push_back(balls[i].jointId);
        outMapping.ballEnvironmentIndices.push_back(bodyIndexA <
                                                            rigidBodies.environmentIndices.size()
                                                        ? rigidBodies.environmentIndices[bodyIndexA]
                                                        : 0u);
        outMapping.ballBodyIdsA.push_back(balls[i].bodyA);
        outMapping.ballBodyIdsB.push_back(balls[i].bodyB);
        outMapping.ballBodyIndicesA.push_back(bodyIndexA);
        outMapping.ballBodyIndicesB.push_back(bodyIndexB);
    }

    outMapping.hingeJointIds.reserve(hinges.size());
    outMapping.hingeEnvironmentIndices.reserve(hinges.size());
    outMapping.hingeBodyIdsA.reserve(hinges.size());
    outMapping.hingeBodyIdsB.reserve(hinges.size());
    outMapping.hingeBodyIndicesA.reserve(hinges.size());
    outMapping.hingeBodyIndicesB.reserve(hinges.size());
    for (std::size_t i = 0; i < hinges.size(); ++i)
    {
        const std::uint32_t bodyIndexA = jointScene.hinge.bodyIndicesA[i];
        const std::uint32_t bodyIndexB = jointScene.hinge.bodyIndicesB[i];
        outMapping.hingeJointIds.push_back(hinges[i].jointId);
        outMapping.hingeEnvironmentIndices.push_back(
            bodyIndexA < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndexA]
                : 0u);
        outMapping.hingeBodyIdsA.push_back(hinges[i].bodyA);
        outMapping.hingeBodyIdsB.push_back(hinges[i].bodyB);
        outMapping.hingeBodyIndicesA.push_back(bodyIndexA);
        outMapping.hingeBodyIndicesB.push_back(bodyIndexB);
    }

    outMapping.sphericalJointIds.reserve(spherical.size());
    outMapping.sphericalEnvironmentIndices.reserve(spherical.size());
    outMapping.sphericalBodyIdsA.reserve(spherical.size());
    outMapping.sphericalBodyIdsB.reserve(spherical.size());
    outMapping.sphericalBodyIndicesA.reserve(spherical.size());
    outMapping.sphericalBodyIndicesB.reserve(spherical.size());
    for (std::size_t i = 0; i < spherical.size(); ++i)
    {
        const std::uint32_t bodyIndexA = jointScene.spherical.bodyIndicesA[i];
        const std::uint32_t bodyIndexB = jointScene.spherical.bodyIndicesB[i];
        outMapping.sphericalJointIds.push_back(spherical[i].jointId);
        outMapping.sphericalEnvironmentIndices.push_back(
            bodyIndexA < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndexA]
                : 0u);
        outMapping.sphericalBodyIdsA.push_back(spherical[i].bodyA);
        outMapping.sphericalBodyIdsB.push_back(spherical[i].bodyB);
        outMapping.sphericalBodyIndicesA.push_back(bodyIndexA);
        outMapping.sphericalBodyIndicesB.push_back(bodyIndexB);
    }

    outMapping.sliderJointIds.reserve(sliders.size());
    outMapping.sliderEnvironmentIndices.reserve(sliders.size());
    outMapping.sliderBodyIdsA.reserve(sliders.size());
    outMapping.sliderBodyIdsB.reserve(sliders.size());
    outMapping.sliderBodyIndicesA.reserve(sliders.size());
    outMapping.sliderBodyIndicesB.reserve(sliders.size());
    for (std::size_t i = 0; i < sliders.size(); ++i)
    {
        const std::uint32_t bodyIndexA = jointScene.slider.bodyIndicesA[i];
        const std::uint32_t bodyIndexB = jointScene.slider.bodyIndicesB[i];
        outMapping.sliderJointIds.push_back(sliders[i].jointId);
        outMapping.sliderEnvironmentIndices.push_back(
            bodyIndexA < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndexA]
                : 0u);
        outMapping.sliderBodyIdsA.push_back(sliders[i].bodyA);
        outMapping.sliderBodyIdsB.push_back(sliders[i].bodyB);
        outMapping.sliderBodyIndicesA.push_back(bodyIndexA);
        outMapping.sliderBodyIndicesB.push_back(bodyIndexB);
    }

    return true;
}

std::vector<CustomComputeResourceDesc> Runtime::listCustomComputeResources()
{
    if (!mInitialized || mCustomComputeService == nullptr || mPhysicsSolver == nullptr)
    {
        return {};
    }
    if (!mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: listCustomComputeResources() requires uploadWorld() after "
                          "prepare() and before execution.");
        return {};
    }
    return mCustomComputeService->listResources(*mPhysicsSolver, mWorld.physicsWorld());
}

CustomComputePassHandle Runtime::createCustomComputePass(const CustomComputePassDesc &desc)
{
    if (!mInitialized || mCustomComputeService == nullptr || mPhysicsSolver == nullptr)
    {
        return {};
    }
    if (!mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: createCustomComputePass() requires uploadWorld() after "
                          "prepare() and before execution.");
        return {};
    }
    ensureDeviceFrameActive(mGpuDevice.get(), mLastFrameContext, mDeviceFrameActive);
    return mCustomComputeService->createPass(*mPhysicsSolver, mWorld.physicsWorld(),
                                             mSharedBufferService.get(), desc);
}

bool Runtime::updateCustomComputePassConstants(CustomComputePassHandle handle,
                                               const std::vector<std::uint8_t> &data)
{
    if (!mInitialized || mCustomComputeService == nullptr)
    {
        return false;
    }
    ensureDeviceFrameActive(mGpuDevice.get(), mLastFrameContext, mDeviceFrameActive);
    return mCustomComputeService->updatePassConstants(handle, data);
}

bool Runtime::executeCustomComputePass(CustomComputePassHandle handle)
{
    if (!mInitialized || mCustomComputeService == nullptr || mPhysicsSolver == nullptr)
    {
        return false;
    }
    if (!mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: executeCustomComputePass() requires uploadWorld() after "
                          "prepare() and before execution.");
        return false;
    }
    ensureDeviceFrameActive(mGpuDevice.get(), mLastFrameContext, mDeviceFrameActive);
    return mCustomComputeService->executePass(*mPhysicsSolver, mWorld.physicsWorld(),
                                              mSharedBufferService.get(), handle);
}

bool Runtime::destroyCustomComputePass(CustomComputePassHandle handle)
{
    return mInitialized && mCustomComputeService != nullptr &&
           mCustomComputeService->destroyPass(handle);
}

} // namespace cressim::neo::engine
