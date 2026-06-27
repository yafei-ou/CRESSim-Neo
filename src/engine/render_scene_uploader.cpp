#include "engine/render_scene_uploader.h"

#include "gpu/gpu_buffer_utils.h"
#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"

#include <array>
#include <cstring>

namespace cressim::neo::engine
{

namespace
{

constexpr std::uint32_t kComputeThreadGroupSize = 64u;

struct GpuEntityPoseSyncConstants
{
    std::uint32_t mappingCount = 0;
    std::uint32_t padding0     = 0;
    std::uint32_t padding1     = 0;
    std::uint32_t padding2     = 0;
};

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + kComputeThreadGroupSize - 1u) / kComputeThreadGroupSize;
}

std::uint64_t combineGenerations(std::uint64_t a, std::uint64_t b) noexcept
{
    return a * 1469598103934665603ull ^ (b + 1099511628211ull);
}

void bumpGeneration(std::uint64_t &generation) noexcept
{
    ++generation;
    if (generation == 0u)
    {
        generation = 1u;
    }
}

bool ensureStructuredBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                            std::uint32_t elementStride, std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage,
                            Diligent::CPU_ACCESS_FLAGS cpuAccess,
                            Diligent::Uint64 immediateContextMask,
                            Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer,
                            std::uint32_t &inOutCapacity, std::uint32_t minimumCapacity)
{
    return gpu::detail::ensureStructuredBufferCapacity(
        renderDevice, name, elementStride, elementCount, minimumCapacity, bindFlags, usage,
        cpuAccess, immediateContextMask, outBuffer, inOutCapacity);
}

const Diligent::ShaderResourceVariableDesc kEntityPoseSyncVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GpuEntityPoseSyncConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SourcePositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SourceOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SourceScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_Mappings", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

const gpu::GpuComputePassDefinition kEntityPoseSyncPassDefinition = {
    "gpu/gpu_entity_pose_sync.cs.hlsl",  "CRESSimNeo.Gpu.EntityPoseSync",
    "CRESSimNeo.Gpu.EntityPoseSync.PSO", kEntityPoseSyncVars,
    std::size(kEntityPoseSyncVars),
};

} // namespace

RenderSceneUploader::RenderSceneUploader(gpu::GpuDevice &device) : mDevice(device) {}

bool RenderSceneUploader::initialize(const common::SceneLayoutDesc &layout)
{
    shutdown();
    mLayout = layout;

    gpu::GpuGraphicsBackendContext graphicsContext{};
    gpu::GpuComputeBackendContext physicsContext{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsContext) ||
        !mDevice.tryGetPhysicsBackendContext(physicsContext) ||
        graphicsContext.renderDevice == nullptr || physicsContext.renderDevice == nullptr)
    {
        return false;
    }
    if (graphicsContext.renderDevice != physicsContext.renderDevice)
    {
        return false;
    }

    mGraphicsContextMask   = gpu::contextMaskForId(graphicsContext.contextId);
    mPhysicsContextMask    = gpu::contextMaskForId(physicsContext.contextId);
    mSharedPoseContextMask = mGraphicsContextMask | mPhysicsContextMask;

    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        return false;
    }

    if (!mEntityPoseSyncPass.initialize(mDevice, streamFactory, mPhysicsContextMask,
                                        kEntityPoseSyncPassDefinition))
    {
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Name                 = "CRESSimNeo.Gpu.EntityPoseSyncConstants";
    constantsDesc.Size                 = sizeof(GpuEntityPoseSyncConstants);
    constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    constantsDesc.ImmediateContextMask = mPhysicsContextMask;
    physicsContext.renderDevice->CreateBuffer(constantsDesc, nullptr, &mConstantsBuffer);
    if (mConstantsBuffer == nullptr)
    {
        return false;
    }

    mInitialized = true;
    return true;
}

void RenderSceneUploader::shutdown()
{
    mLayout                                 = {};
    mInitialized                            = false;
    mPoseCapacity                           = 0;
    mPhysicsSyncCapacity                    = 0;
    mEntityCount                            = 0;
    mRenderableCapacity                     = 0;
    mRenderableCount                        = 0;
    mSoftBodyVertexBindingCapacity          = 0;
    mSoftBodyVertexBindingCount             = 0;
    mCameraCapacity                         = 0;
    mCameraCount                            = 0;
    mLightCapacity                          = 0;
    mLightCount                             = 0;
    mLocalLightSelectionCapacity            = 0;
    mGraphicsContextMask                    = 0;
    mPhysicsContextMask                     = 0;
    mSharedPoseContextMask                  = 0;
    mMappingBuffer                          = nullptr;
    mConstantsBuffer                        = nullptr;
    mEntityPositionsBuffer                  = nullptr;
    mEntityOrientationsBuffer               = nullptr;
    mEntityScalesBuffer                     = nullptr;
    mRenderableMetadataBuffer               = nullptr;
    mRenderableQueueInfoBuffer              = nullptr;
    mRenderableVisibilityFlagsBuffer        = nullptr;
    mRenderableShadowCascadeMasksBuffer     = nullptr;
    mSoftBodyVertexBindingBuffer            = nullptr;
    mCameraInputsBuffer                     = nullptr;
    mPreparedCamerasBuffer                  = nullptr;
    mLightInputsBuffer                      = nullptr;
    mLocalLightSelectionBuffer              = nullptr;
    mEntityPoseSyncPass                     = {};
    mPoseBindingGeneration                  = 1u;
    mPhysicsSyncBindingGeneration           = 1u;
    mSceneBindingGeneration                 = 1u;
    mLastMappedSourcePoseBindingGeneration  = 0u;
    mLastMappedOutputPoseBindingGeneration  = 0u;
    mLastMappedPhysicsSyncBindingGeneration = 0u;
}

bool RenderSceneUploader::ensureSharedPoseCapacity(Diligent::IRenderDevice *renderDevice,
                                                   std::uint32_t entityCount)
{
    if (renderDevice == nullptr || mSharedPoseContextMask == 0)
    {
        return false;
    }

    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(entityCount, 1u);
    if (mPoseCapacity >= requiredCapacity && mEntityPositionsBuffer != nullptr &&
        mEntityOrientationsBuffer != nullptr && mEntityScalesBuffer != nullptr)
    {
        return true;
    }

    const std::uint32_t newCapacity    = std::max<std::uint32_t>(requiredCapacity, 64u);
    Diligent::IBuffer *oldPositions    = mEntityPositionsBuffer;
    Diligent::IBuffer *oldOrientations = mEntityOrientationsBuffer;
    Diligent::IBuffer *oldScales       = mEntityScalesBuffer;
    const bool success =
        ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Gpu.EntityPositions", sizeof(Diligent::float4), newCapacity,
            Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, mSharedPoseContextMask,
            mEntityPositionsBuffer, mPoseCapacity, 64u) &&
        ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Gpu.EntityOrientations", sizeof(Diligent::float4),
            newCapacity, Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, mSharedPoseContextMask,
            mEntityOrientationsBuffer, mPoseCapacity, 64u) &&
        ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.EntityScales",
                               sizeof(Diligent::float4), newCapacity,
                               Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                               Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
                               mSharedPoseContextMask, mEntityScalesBuffer, mPoseCapacity, 64u);
    if (!success)
    {
        return false;
    }
    if (oldPositions != mEntityPositionsBuffer || oldOrientations != mEntityOrientationsBuffer ||
        oldScales != mEntityScalesBuffer)
    {
        bumpGeneration(mPoseBindingGeneration);
        bumpGeneration(mSceneBindingGeneration);
    }
    return true;
}

bool RenderSceneUploader::ensurePhysicsSyncCapacity(Diligent::IRenderDevice *renderDevice,
                                                    std::uint32_t mappingCount)
{
    if (renderDevice == nullptr || mPhysicsContextMask == 0)
    {
        return false;
    }

    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(mappingCount, 1u);
    if (mPhysicsSyncCapacity >= requiredCapacity && mMappingBuffer != nullptr)
    {
        return true;
    }

    const std::uint32_t newCapacity = std::max<std::uint32_t>(requiredCapacity, 64u);
    Diligent::IBuffer *oldMapping   = mMappingBuffer;
    const bool success              = ensureStructuredBuffer(
        renderDevice, "CRESSimNeo.Gpu.EntityPoseMappings", sizeof(EntityPoseMappingEntry),
        newCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
        Diligent::CPU_ACCESS_NONE, mPhysicsContextMask, mMappingBuffer, mPhysicsSyncCapacity, 64u);
    if (!success)
    {
        return false;
    }
    if (oldMapping != mMappingBuffer)
    {
        bumpGeneration(mPhysicsSyncBindingGeneration);
    }
    return true;
}

bool RenderSceneUploader::ensureRenderableCapacity(Diligent::IRenderDevice *renderDevice,
                                                   std::uint32_t renderableCount)
{
    if (renderDevice == nullptr || mGraphicsContextMask == 0)
    {
        return false;
    }

    const std::uint32_t requiredRenderableCapacity = std::max<std::uint32_t>(renderableCount, 1u);
    const std::uint32_t visibilityCapacity         = std::max<std::uint32_t>(
        mLayout.maxRenderableObjectsPerEnv * std::max(mLayout.totalCameraCapacity(), 1u), 1u);
    const std::uint32_t requiredCapacity = std::max(requiredRenderableCapacity, visibilityCapacity);
    if (mRenderableCapacity >= requiredCapacity && mRenderableMetadataBuffer != nullptr &&
        mRenderableQueueInfoBuffer != nullptr && mRenderableVisibilityFlagsBuffer != nullptr &&
        mRenderableShadowCascadeMasksBuffer != nullptr)
    {
        return true;
    }

    Diligent::IBuffer *oldMetadata    = mRenderableMetadataBuffer;
    Diligent::IBuffer *oldQueueInfo   = mRenderableQueueInfoBuffer;
    Diligent::IBuffer *oldVisibility  = mRenderableVisibilityFlagsBuffer;
    Diligent::IBuffer *oldShadowMasks = mRenderableShadowCascadeMasksBuffer;
    const bool success =
        ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.RenderableMetadata",
                               sizeof(graphics::GpuRenderableMetadata), requiredCapacity,
                               Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                               Diligent::CPU_ACCESS_NONE, mGraphicsContextMask,
                               mRenderableMetadataBuffer, mRenderableCapacity, 64u) &&
        ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.RenderableQueueInfo",
                               sizeof(graphics::GpuRenderableQueueInfo), requiredCapacity,
                               Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                               Diligent::CPU_ACCESS_NONE, mGraphicsContextMask,
                               mRenderableQueueInfoBuffer, mRenderableCapacity, 64u) &&
        ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Gpu.RenderableVisibilityFlags", sizeof(std::uint32_t),
            requiredCapacity, Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, mGraphicsContextMask,
            mRenderableVisibilityFlagsBuffer, mRenderableCapacity, requiredCapacity) &&
        ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Gpu.RenderableShadowCascadeMasks", sizeof(std::uint32_t),
            requiredCapacity, Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, mGraphicsContextMask,
            mRenderableShadowCascadeMasksBuffer, mRenderableCapacity, requiredCapacity);
    if (!success)
    {
        return false;
    }
    if (oldMetadata != mRenderableMetadataBuffer || oldQueueInfo != mRenderableQueueInfoBuffer ||
        oldVisibility != mRenderableVisibilityFlagsBuffer ||
        oldShadowMasks != mRenderableShadowCascadeMasksBuffer)
    {
        bumpGeneration(mSceneBindingGeneration);
    }
    return true;
}

bool RenderSceneUploader::ensureCameraCapacity(Diligent::IRenderDevice *renderDevice,
                                               std::uint32_t cameraCount)
{
    if (renderDevice == nullptr || mGraphicsContextMask == 0)
    {
        return false;
    }

    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(cameraCount, 1u);
    if (mCameraCapacity >= requiredCapacity && mCameraInputsBuffer != nullptr &&
        mPreparedCamerasBuffer != nullptr)
    {
        return true;
    }

    Diligent::IBuffer *oldInputs   = mCameraInputsBuffer;
    Diligent::IBuffer *oldPrepared = mPreparedCamerasBuffer;
    const bool success =
        ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.CameraInputs",
                               sizeof(graphics::GpuCameraInput), requiredCapacity,
                               Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                               Diligent::CPU_ACCESS_NONE, mGraphicsContextMask, mCameraInputsBuffer,
                               mCameraCapacity, 1u) &&
        ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.PreparedCameras",
                               sizeof(graphics::GpuPreparedCamera), requiredCapacity,
                               Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                               Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
                               mGraphicsContextMask, mPreparedCamerasBuffer, mCameraCapacity, 1u);
    if (!success)
    {
        return false;
    }
    if (oldInputs != mCameraInputsBuffer || oldPrepared != mPreparedCamerasBuffer)
    {
        bumpGeneration(mSceneBindingGeneration);
    }
    return true;
}

bool RenderSceneUploader::ensureLightCapacity(Diligent::IRenderDevice *renderDevice,
                                              std::uint32_t lightCount)
{
    if (renderDevice == nullptr || mGraphicsContextMask == 0)
    {
        return false;
    }

    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(lightCount, 1u);
    if (mLightCapacity >= requiredCapacity && mLightInputsBuffer != nullptr)
    {
        return true;
    }

    Diligent::IBuffer *oldInputs = mLightInputsBuffer;
    const bool success           = ensureStructuredBuffer(
        renderDevice, "CRESSimNeo.Gpu.LightInputs", sizeof(graphics::GpuLightInput),
        requiredCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
        Diligent::CPU_ACCESS_NONE, mGraphicsContextMask, mLightInputsBuffer, mLightCapacity, 1u);
    if (!success)
    {
        return false;
    }
    if (oldInputs != mLightInputsBuffer)
    {
        bumpGeneration(mSceneBindingGeneration);
    }
    return true;
}

bool RenderSceneUploader::ensureLocalLightSelectionCapacity(Diligent::IRenderDevice *renderDevice,
                                                            std::uint32_t selectionCount)
{
    if (renderDevice == nullptr || mGraphicsContextMask == 0)
    {
        return false;
    }

    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(selectionCount, 1u);
    if (mLocalLightSelectionCapacity >= requiredCapacity && mLocalLightSelectionBuffer != nullptr)
    {
        return true;
    }

    Diligent::IBuffer *oldSelections = mLocalLightSelectionBuffer;
    const bool success               = ensureStructuredBuffer(
        renderDevice, "CRESSimNeo.Gpu.LocalLightSelections",
        sizeof(graphics::GpuLocalLightSelection), requiredCapacity, Diligent::BIND_SHADER_RESOURCE,
        Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, mGraphicsContextMask,
        mLocalLightSelectionBuffer, mLocalLightSelectionCapacity, 1u);
    if (!success)
    {
        return false;
    }
    if (oldSelections != mLocalLightSelectionBuffer)
    {
        bumpGeneration(mSceneBindingGeneration);
    }
    return true;
}

namespace
{
bool ensureSoftBodyBufferCapacity(Diligent::IRenderDevice *renderDevice,
                                  Diligent::Uint64 contextMask, const char *name,
                                  std::uint32_t elementStride, std::uint32_t elementCount,
                                  Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer,
                                  std::uint32_t &inOutCapacity)
{
    if (renderDevice == nullptr || contextMask == 0)
    {
        return false;
    }

    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(elementCount, 1u);
    if (inOutCapacity >= requiredCapacity && outBuffer != nullptr)
    {
        return true;
    }

    return ensureStructuredBuffer(
        renderDevice, name, elementStride, std::max<std::uint32_t>(requiredCapacity, 64u),
        Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
        contextMask, outBuffer, inOutCapacity, 64u);
}
} // namespace

bool RenderSceneUploader::uploadRenderableMetadata(
    const std::vector<graphics::GpuRenderableMetadata> &renderables)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext graphicsContext{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsContext) ||
        graphicsContext.renderDevice == nullptr || graphicsContext.graphicsContext == nullptr)
    {
        return false;
    }

    mRenderableCount = static_cast<std::uint32_t>(renderables.size());
    if (!ensureRenderableCapacity(graphicsContext.renderDevice, mRenderableCount))
    {
        return false;
    }

    if (mRenderableCount == 0u)
    {
        return true;
    }

    return writeBuffer(graphicsContext.graphicsContext, mRenderableMetadataBuffer,
                       renderables.data(),
                       renderables.size() * sizeof(graphics::GpuRenderableMetadata));
}

bool RenderSceneUploader::uploadRenderableQueueInfo(
    const std::vector<graphics::GpuRenderableQueueInfo> &queueInfo)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext graphicsContext{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsContext) ||
        graphicsContext.renderDevice == nullptr || graphicsContext.graphicsContext == nullptr)
    {
        return false;
    }

    if (!ensureRenderableCapacity(
            graphicsContext.renderDevice,
            std::max(mRenderableCount, static_cast<std::uint32_t>(queueInfo.size()))))
    {
        return false;
    }

    if (queueInfo.empty())
    {
        return true;
    }

    return writeBuffer(graphicsContext.graphicsContext, mRenderableQueueInfoBuffer,
                       queueInfo.data(),
                       queueInfo.size() * sizeof(graphics::GpuRenderableQueueInfo));
}

bool RenderSceneUploader::uploadSoftBodyVertexBindings(
    const std::vector<graphics::GpuSoftBodyVertexBinding> &bindings)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext graphicsContext{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsContext) ||
        graphicsContext.renderDevice == nullptr || graphicsContext.graphicsContext == nullptr)
    {
        return false;
    }

    mSoftBodyVertexBindingCount                       = static_cast<std::uint32_t>(bindings.size());
    Diligent::IBuffer *oldSoftBodyVertexBindingBuffer = mSoftBodyVertexBindingBuffer;
    if (!ensureSoftBodyBufferCapacity(graphicsContext.renderDevice, mGraphicsContextMask,
                                      "CRESSimNeo.Gpu.SoftBodyVertexBindings",
                                      sizeof(graphics::GpuSoftBodyVertexBinding),
                                      mSoftBodyVertexBindingCount, mSoftBodyVertexBindingBuffer,
                                      mSoftBodyVertexBindingCapacity))
    {
        return false;
    }
    if (oldSoftBodyVertexBindingBuffer != mSoftBodyVertexBindingBuffer)
    {
        bumpGeneration(mSceneBindingGeneration);
    }

    if (bindings.empty())
    {
        return true;
    }

    return writeBuffer(graphicsContext.graphicsContext, mSoftBodyVertexBindingBuffer,
                       bindings.data(),
                       bindings.size() * sizeof(graphics::GpuSoftBodyVertexBinding));
}

bool RenderSceneUploader::uploadCameraInputs(const std::vector<graphics::GpuCameraInput> &cameras)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext graphicsContext{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsContext) ||
        graphicsContext.renderDevice == nullptr || graphicsContext.graphicsContext == nullptr)
    {
        return false;
    }

    mCameraCount = static_cast<std::uint32_t>(cameras.size());
    if (!ensureCameraCapacity(graphicsContext.renderDevice, mCameraCount))
    {
        return false;
    }

    if (mCameraCount == 0u)
    {
        return true;
    }

    return writeBuffer(graphicsContext.graphicsContext, mCameraInputsBuffer, cameras.data(),
                       cameras.size() * sizeof(graphics::GpuCameraInput));
}

bool RenderSceneUploader::uploadLightInputs(const std::vector<graphics::GpuLightInput> &lights)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext graphicsContext{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsContext) ||
        graphicsContext.renderDevice == nullptr || graphicsContext.graphicsContext == nullptr)
    {
        return false;
    }

    mLightCount = static_cast<std::uint32_t>(lights.size());
    if (!ensureLightCapacity(graphicsContext.renderDevice, mLightCount))
    {
        return false;
    }

    if (mLightCount == 0u)
    {
        return true;
    }

    return writeBuffer(graphicsContext.graphicsContext, mLightInputsBuffer, lights.data(),
                       lights.size() * sizeof(graphics::GpuLightInput));
}

bool RenderSceneUploader::uploadLocalLightSelections(
    const std::vector<graphics::GpuLocalLightSelection> &selections)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext graphicsContext{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsContext) ||
        graphicsContext.renderDevice == nullptr || graphicsContext.graphicsContext == nullptr)
    {
        return false;
    }

    const std::uint32_t selectionCount = static_cast<std::uint32_t>(selections.size());
    if (!ensureLocalLightSelectionCapacity(graphicsContext.renderDevice, selectionCount))
    {
        return false;
    }

    if (selectionCount == 0u)
    {
        return true;
    }

    return writeBuffer(graphicsContext.graphicsContext, mLocalLightSelectionBuffer,
                       selections.data(),
                       selections.size() * sizeof(graphics::GpuLocalLightSelection));
}

bool RenderSceneUploader::writeBuffer(Diligent::IDeviceContext *computeContext,
                                      Diligent::IBuffer *buffer, const void *data,
                                      std::size_t sizeBytes)
{
    if (computeContext == nullptr || buffer == nullptr || data == nullptr || sizeBytes == 0u)
    {
        return false;
    }

    const Diligent::BufferDesc &desc = buffer->GetDesc();
    if (desc.Usage != Diligent::USAGE_DYNAMIC)
    {
        computeContext->UpdateBuffer(buffer, 0u, static_cast<Diligent::Uint32>(sizeBytes), data,
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        return true;
    }

    void *mapped = nullptr;
    computeContext->MapBuffer(buffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }

    std::memcpy(mapped, data, sizeBytes);
    computeContext->UnmapBuffer(buffer, Diligent::MAP_WRITE);
    return true;
}

bool RenderSceneUploader::uploadEntityPoseData(const std::vector<Diligent::float4> &positions,
                                               const std::vector<Diligent::float4> &orientations,
                                               const std::vector<Diligent::float4> &scales)
{
    if (!mInitialized)
    {
        return false;
    }
    if (positions.size() != orientations.size() || positions.size() != scales.size())
    {
        return false;
    }

    gpu::GpuComputeBackendContext physicsContext{};
    if (!mDevice.tryGetPhysicsBackendContext(physicsContext) ||
        physicsContext.renderDevice == nullptr || physicsContext.computeContext == nullptr)
    {
        return false;
    }
    if (!mDevice.waitForGraphicsOnPhysics())
    {
        return false;
    }

    mEntityCount = static_cast<std::uint32_t>(positions.size());
    if (mEntityCount == 0u)
    {
        return ensureSharedPoseCapacity(physicsContext.renderDevice, 1u);
    }
    if (!ensureSharedPoseCapacity(physicsContext.renderDevice, mEntityCount))
    {
        return false;
    }

    return writeBuffer(physicsContext.computeContext, mEntityPositionsBuffer, positions.data(),
                       positions.size() * sizeof(Diligent::float4)) &&
           writeBuffer(physicsContext.computeContext, mEntityOrientationsBuffer,
                       orientations.data(), orientations.size() * sizeof(Diligent::float4)) &&
           writeBuffer(physicsContext.computeContext, mEntityScalesBuffer, scales.data(),
                       scales.size() * sizeof(Diligent::float4));
}

bool RenderSceneUploader::applyMappedEntityPoses(
    const common::PoseBufferView &sourcePoses, const std::vector<EntityPoseMappingEntry> &mappings)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr || computeContext.computeContext == nullptr)
    {
        return false;
    }

    const std::uint32_t mappingCount = static_cast<std::uint32_t>(mappings.size());
    if (mappingCount == 0u)
    {
        return true;
    }
    if (sourcePoses.positionsBuffer == nullptr || sourcePoses.orientationsBuffer == nullptr ||
        sourcePoses.scalesBuffer == nullptr || sourcePoses.count == 0u)
    {
        return false;
    }
    if (!ensureSharedPoseCapacity(computeContext.renderDevice,
                                  std::max(mEntityCount, mappingCount)) ||
        !ensurePhysicsSyncCapacity(computeContext.renderDevice, mappingCount))
    {
        return false;
    }

    if (!writeBuffer(computeContext.computeContext, mMappingBuffer, mappings.data(),
                     mappings.size() * sizeof(EntityPoseMappingEntry)))
    {
        return false;
    }

    const GpuEntityPoseSyncConstants constants{mappingCount, 0u, 0u, 0u};
    if (!writeBuffer(computeContext.computeContext, mConstantsBuffer, &constants,
                     sizeof(constants)))
    {
        return false;
    }

    const std::array bindings{
        gpu::GpuBufferBinding{"GpuEntityPoseSyncConstantsBuffer", mConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SourcePositions", sourcePoses.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SourceOrientations", sourcePoses.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SourceScales", sourcePoses.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_Mappings", mMappingBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityPositions", mEntityPositionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_EntityOrientations", mEntityOrientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_EntityScales", mEntityScalesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const bool syncBindingsChanged =
        mLastMappedSourcePoseBindingGeneration != sourcePoses.bindingGeneration ||
        mLastMappedOutputPoseBindingGeneration != mPoseBindingGeneration ||
        mLastMappedPhysicsSyncBindingGeneration != mPhysicsSyncBindingGeneration;
    if (syncBindingsChanged && !mEntityPoseSyncPass.forceRecreateAllVariants())
    {
        return false;
    }

    mLastMappedSourcePoseBindingGeneration  = sourcePoses.bindingGeneration;
    mLastMappedOutputPoseBindingGeneration  = mPoseBindingGeneration;
    mLastMappedPhysicsSyncBindingGeneration = mPhysicsSyncBindingGeneration;

    return mEntityPoseSyncPass.dispatch(computeContext.computeContext, 0u, bindings,
                                        dispatchGroupCount(mappingCount));
}

graphics::GpuEntitySceneView RenderSceneUploader::sceneView() const noexcept
{
    common::PoseBufferView poses{};
    poses.positionsBuffer    = mEntityPositionsBuffer;
    poses.orientationsBuffer = mEntityOrientationsBuffer;
    poses.scalesBuffer       = mEntityScalesBuffer;
    poses.count              = mEntityCount;
    poses.bindingGeneration  = mPoseBindingGeneration;
    return sceneView(poses, mEntityCount);
}

graphics::GpuEntitySceneView RenderSceneUploader::sceneView(
    const common::PoseBufferView &poses, std::uint32_t entityCount) const noexcept
{
    graphics::GpuEntitySceneView view{};
    view.layout                             = mLayout;
    view.poses                              = poses;
    view.renderableMetadataBuffer           = mRenderableMetadataBuffer;
    view.renderableQueueInfoBuffer          = mRenderableQueueInfoBuffer;
    view.renderableVisibilityFlagsBuffer    = mRenderableVisibilityFlagsBuffer;
    view.renderableShadowCascadeMasksBuffer = mRenderableShadowCascadeMasksBuffer;
    view.cameraInputsBuffer                 = mCameraInputsBuffer;
    view.preparedCamerasBuffer              = mPreparedCamerasBuffer;
    view.lightInputsBuffer                  = mLightInputsBuffer;
    view.localLightSelectionBuffer          = mLocalLightSelectionBuffer;
    view.softBodyVertexBindingBuffer        = mSoftBodyVertexBindingBuffer;
    view.entityCount                        = entityCount;
    view.renderableCount                    = mRenderableCount;
    view.cameraCount                        = mCameraCount;
    view.lightCount                         = mLightCount;
    view.bindingGeneration = combineGenerations(mSceneBindingGeneration, poses.bindingGeneration);
    return view;
}

} // namespace cressim::neo::engine
