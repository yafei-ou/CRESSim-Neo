#include "gpu/gpu_scene_sync.h"

#include "gpu/gpu_buffer_utils.h"
#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"

#include <array>
#include <cstring>

namespace cressim::neo::gpu
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

bool ensureStructuredBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                            std::uint32_t elementStride, std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage,
                            Diligent::CPU_ACCESS_FLAGS cpuAccess,
                            Diligent::Uint64 immediateContextMask,
                            Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer,
                            std::uint32_t &inOutCapacity, std::uint32_t minimumCapacity)
{
    return detail::ensureStructuredBufferCapacity(renderDevice, name, elementStride, elementCount,
                                                  minimumCapacity, bindFlags, usage, cpuAccess,
                                                  immediateContextMask, outBuffer, inOutCapacity);
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

const GpuComputePassDefinition kEntityPoseSyncPassDefinition = {
    "gpu/gpu_entity_pose_sync.cs.hlsl",  "CRESSimNeo.Gpu.EntityPoseSync",
    "CRESSimNeo.Gpu.EntityPoseSync.PSO", kEntityPoseSyncVars,
    std::size(kEntityPoseSyncVars),
};

GpuComputePass &entityPoseSyncPass()
{
    static GpuComputePass pass;
    return pass;
}

bool initializeEntityPoseSyncPass(Diligent::IRenderDevice *renderDevice,
                                  Diligent::IShaderSourceInputStreamFactory *streamFactory,
                                  Diligent::Uint64 contextMask)
{
    return entityPoseSyncPass().initialize(renderDevice, streamFactory, contextMask,
                                           kEntityPoseSyncPassDefinition);
}

} // namespace

GpuSceneSync::GpuSceneSync(GpuDevice &device) : mDevice(device) {}

bool GpuSceneSync::initialize(const GpuSceneLayoutDesc &layout)
{
    shutdown();
    mLayout = layout;

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr)
    {
        return false;
    }

    mContextMask = static_cast<Diligent::Uint64>(1ull) << computeContext.contextId;

    ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        return false;
    }

    if (!initializeEntityPoseSyncPass(computeContext.renderDevice, streamFactory, mContextMask))
    {
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Name                 = "CRESSimNeo.Gpu.EntityPoseSyncConstants";
    constantsDesc.Size                 = sizeof(GpuEntityPoseSyncConstants);
    constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    constantsDesc.ImmediateContextMask = mContextMask;
    computeContext.renderDevice->CreateBuffer(constantsDesc, nullptr, &mConstantsBuffer);
    if (mConstantsBuffer == nullptr)
    {
        return false;
    }

    mInitialized = true;
    return true;
}

void GpuSceneSync::shutdown()
{
    mLayout                             = {};
    mInitialized                        = false;
    mCapacity                           = 0;
    mEntityCount                        = 0;
    mRenderableCapacity                 = 0;
    mRenderableCount                    = 0;
    mCameraCapacity                     = 0;
    mCameraCount                        = 0;
    mLightCapacity                      = 0;
    mLightCount                         = 0;
    mContextMask                        = 0;
    mMappingBuffer                      = nullptr;
    mEntityPositionsBuffer              = nullptr;
    mEntityOrientationsBuffer           = nullptr;
    mEntityScalesBuffer                 = nullptr;
    mRenderableMetadataBuffer           = nullptr;
    mRenderableQueueInfoBuffer          = nullptr;
    mRenderableVisibilityFlagsBuffer    = nullptr;
    mRenderableShadowCascadeMasksBuffer = nullptr;
    mCameraInputsBuffer                 = nullptr;
    mPreparedCamerasBuffer              = nullptr;
    mLightInputsBuffer                  = nullptr;
    mLocalLightSelectionBuffer          = nullptr;
    mConstantsBuffer                    = nullptr;
}

bool GpuSceneSync::ensureCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t entityCount,
                                  std::uint32_t contextId)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(entityCount, 1u);
    if (mCapacity >= requiredCapacity && mMappingBuffer != nullptr &&
        mEntityPositionsBuffer != nullptr && mEntityOrientationsBuffer != nullptr &&
        mEntityScalesBuffer != nullptr)
    {
        return true;
    }

    const std::uint32_t newCapacity    = std::max<std::uint32_t>(requiredCapacity, 64u);
    const Diligent::Uint64 contextMask = static_cast<Diligent::Uint64>(1ull) << contextId;

    return ensureStructuredBuffer(
               renderDevice, "CRESSimNeo.Gpu.EntityPoseMappings", sizeof(GpuEntityPoseMappingEntry),
               newCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
               Diligent::CPU_ACCESS_NONE, contextMask, mMappingBuffer, mCapacity, 64u) &&
           ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.EntityPositions",
                                  sizeof(Diligent::float4), newCapacity,
                                  Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                  Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                  mEntityPositionsBuffer, mCapacity, 64u) &&
           ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.EntityOrientations",
                                  sizeof(Diligent::float4), newCapacity,
                                  Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                  Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                  mEntityOrientationsBuffer, mCapacity, 64u) &&
           ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.EntityScales",
                                  sizeof(Diligent::float4), newCapacity,
                                  Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                  Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                  mEntityScalesBuffer, mCapacity, 64u);
}

bool GpuSceneSync::syncRenderableMetadata(const std::vector<GpuRenderableMetadata> &renderables)
{
    if (!mInitialized)
    {
        return false;
    }

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr || computeContext.computeContext == nullptr)
    {
        return false;
    }

    mRenderableCount                       = static_cast<std::uint32_t>(renderables.size());
    const std::uint32_t requiredCapacity   = std::max<std::uint32_t>(mRenderableCount, 1u);
    const std::uint32_t visibilityCapacity = std::max<std::uint32_t>(
        mLayout.maxObjectsPerEnv * std::max(mLayout.totalCameraCapacity(), 1u), 1u);
    if (mRenderableCapacity < requiredCapacity || mRenderableMetadataBuffer == nullptr ||
        mRenderableQueueInfoBuffer == nullptr || mRenderableVisibilityFlagsBuffer == nullptr ||
        mRenderableShadowCascadeMasksBuffer == nullptr)
    {
        const std::uint32_t newCapacity        = std::max<std::uint32_t>(requiredCapacity, 64u);
        const Diligent::Uint64 contextMask     = static_cast<Diligent::Uint64>(1ull)
                                                 << computeContext.contextId;
        std::uint32_t visibilityBufferCapacity = visibilityCapacity;
        if (!ensureStructuredBuffer(
                computeContext.renderDevice, "CRESSimNeo.Gpu.RenderableMetadata",
                sizeof(GpuRenderableMetadata), newCapacity, Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mRenderableMetadataBuffer, mRenderableCapacity, 64u) ||
            !ensureStructuredBuffer(
                computeContext.renderDevice, "CRESSimNeo.Gpu.RenderableQueueInfo",
                sizeof(GpuRenderableQueueInfo), newCapacity, Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mRenderableQueueInfoBuffer, mRenderableCapacity, 64u) ||
            !ensureStructuredBuffer(
                computeContext.renderDevice, "CRESSimNeo.Gpu.RenderableVisibilityFlags",
                sizeof(std::uint32_t), visibilityCapacity,
                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mRenderableVisibilityFlagsBuffer, visibilityBufferCapacity, visibilityCapacity) ||
            !ensureStructuredBuffer(
                computeContext.renderDevice, "CRESSimNeo.Gpu.RenderableShadowCascadeMasks",
                sizeof(std::uint32_t), visibilityCapacity,
                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mRenderableShadowCascadeMasksBuffer, visibilityBufferCapacity, visibilityCapacity))
        {
            return false;
        }
    }

    if (mRenderableCount == 0u)
    {
        return true;
    }

    return writeBuffer(computeContext.computeContext, mRenderableMetadataBuffer, renderables.data(),
                       renderables.size() * sizeof(GpuRenderableMetadata));
}

bool GpuSceneSync::syncRenderableQueueInfo(const std::vector<GpuRenderableQueueInfo> &queueInfo)
{
    if (!mInitialized)
    {
        return false;
    }

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.computeContext == nullptr || mRenderableQueueInfoBuffer == nullptr)
    {
        return false;
    }

    if (queueInfo.empty())
    {
        return true;
    }

    return writeBuffer(computeContext.computeContext, mRenderableQueueInfoBuffer, queueInfo.data(),
                       queueInfo.size() * sizeof(GpuRenderableQueueInfo));
}

bool GpuSceneSync::syncCameraInputs(const std::vector<GpuCameraInput> &cameras)
{
    if (!mInitialized)
    {
        return false;
    }

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr || computeContext.computeContext == nullptr)
    {
        return false;
    }

    mCameraCount                         = static_cast<std::uint32_t>(cameras.size());
    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(mCameraCount, 1u);
    if (mCameraCapacity < requiredCapacity || mCameraInputsBuffer == nullptr ||
        mPreparedCamerasBuffer == nullptr)
    {
        const std::uint32_t newCapacity    = std::max<std::uint32_t>(requiredCapacity, 1u);
        const Diligent::Uint64 contextMask = static_cast<Diligent::Uint64>(1ull)
                                             << computeContext.contextId;
        if (!ensureStructuredBuffer(
                computeContext.renderDevice, "CRESSimNeo.Gpu.CameraInputs", sizeof(GpuCameraInput),
                newCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                Diligent::CPU_ACCESS_NONE, contextMask, mCameraInputsBuffer, mCameraCapacity, 1u) ||
            !ensureStructuredBuffer(computeContext.renderDevice, "CRESSimNeo.Gpu.PreparedCameras",
                                    sizeof(GpuPreparedCamera), newCapacity,
                                    Diligent::BIND_SHADER_RESOURCE |
                                        Diligent::BIND_UNORDERED_ACCESS,
                                    Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                    mPreparedCamerasBuffer, mCameraCapacity, 1u))
        {
            return false;
        }
    }

    if (mCameraCount == 0u)
    {
        return true;
    }

    return writeBuffer(computeContext.computeContext, mCameraInputsBuffer, cameras.data(),
                       cameras.size() * sizeof(GpuCameraInput));
}

bool GpuSceneSync::syncLightInputs(const std::vector<GpuLightInput> &lights)
{
    if (!mInitialized)
    {
        return false;
    }

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr || computeContext.computeContext == nullptr)
    {
        return false;
    }

    mLightCount                          = static_cast<std::uint32_t>(lights.size());
    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(mLightCount, 1u);
    if (mLightCapacity < requiredCapacity || mLightInputsBuffer == nullptr)
    {
        const std::uint32_t newCapacity    = std::max<std::uint32_t>(requiredCapacity, 1u);
        const Diligent::Uint64 contextMask = static_cast<Diligent::Uint64>(1ull)
                                             << computeContext.contextId;
        if (!ensureStructuredBuffer(
                computeContext.renderDevice, "CRESSimNeo.Gpu.LightInputs", sizeof(GpuLightInput),
                newCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                Diligent::CPU_ACCESS_NONE, contextMask, mLightInputsBuffer, mLightCapacity, 1u))
        {
            return false;
        }
    }

    if (mLightCount == 0u)
    {
        return true;
    }

    return writeBuffer(computeContext.computeContext, mLightInputsBuffer, lights.data(),
                       lights.size() * sizeof(GpuLightInput));
}

bool GpuSceneSync::syncLocalLightSelections(const std::vector<GpuLocalLightSelection> &selections)
{
    if (!mInitialized)
    {
        return false;
    }

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr || computeContext.computeContext == nullptr)
    {
        return false;
    }

    const std::uint32_t selectionCount   = static_cast<std::uint32_t>(selections.size());
    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(selectionCount, 1u);
    if (mLocalLightSelectionBuffer == nullptr ||
        mLocalLightSelectionBuffer->GetDesc().ElementByteStride != sizeof(GpuLocalLightSelection) ||
        mLocalLightSelectionBuffer->GetDesc().Size <
            static_cast<Diligent::Uint64>(requiredCapacity) * sizeof(GpuLocalLightSelection))
    {
        const Diligent::Uint64 contextMask = static_cast<Diligent::Uint64>(1ull)
                                             << computeContext.contextId;
        std::uint32_t ignoredCapacity      = 0u;
        if (!ensureStructuredBuffer(
                computeContext.renderDevice, "CRESSimNeo.Gpu.LocalLightSelections",
                sizeof(GpuLocalLightSelection), requiredCapacity, Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mLocalLightSelectionBuffer, ignoredCapacity, 1u))
        {
            return false;
        }
    }

    if (selectionCount == 0u)
    {
        return true;
    }

    return writeBuffer(computeContext.computeContext, mLocalLightSelectionBuffer, selections.data(),
                       selections.size() * sizeof(GpuLocalLightSelection));
}

bool GpuSceneSync::writeBuffer(Diligent::IDeviceContext *computeContext, Diligent::IBuffer *buffer,
                               const void *data, std::size_t sizeBytes)
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

bool GpuSceneSync::syncEntityPoseData(const std::vector<Diligent::float4> &positions,
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

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr || computeContext.computeContext == nullptr)
    {
        return false;
    }

    mEntityCount = static_cast<std::uint32_t>(positions.size());
    if (mEntityCount == 0u)
    {
        return ensureCapacity(computeContext.renderDevice, 1u, computeContext.contextId);
    }
    if (!ensureCapacity(computeContext.renderDevice, mEntityCount, computeContext.contextId))
    {
        return false;
    }

    return writeBuffer(computeContext.computeContext, mEntityPositionsBuffer, positions.data(),
                       positions.size() * sizeof(Diligent::float4)) &&
           writeBuffer(computeContext.computeContext, mEntityOrientationsBuffer,
                       orientations.data(), orientations.size() * sizeof(Diligent::float4)) &&
           writeBuffer(computeContext.computeContext, mEntityScalesBuffer, scales.data(),
                       scales.size() * sizeof(Diligent::float4));
}

bool GpuSceneSync::syncEntityPoses(const GpuPoseBufferView &sourcePoses,
                                   const std::vector<GpuEntityPoseMappingEntry> &mappings)
{
    if (!mInitialized)
    {
        return false;
    }

    GpuComputeBackendContext computeContext{};
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
    if (!ensureCapacity(computeContext.renderDevice, std::max(mEntityCount, mappingCount),
                        computeContext.contextId))
    {
        return false;
    }

    if (!writeBuffer(computeContext.computeContext, mMappingBuffer, mappings.data(),
                     mappings.size() * sizeof(GpuEntityPoseMappingEntry)))
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
        GpuBufferBinding{"GpuEntityPoseSyncConstantsBuffer", mConstantsBuffer,
                         Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        GpuBufferBinding{"g_SourcePositions", sourcePoses.positionsBuffer,
                         Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        GpuBufferBinding{"g_SourceOrientations", sourcePoses.orientationsBuffer,
                         Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        GpuBufferBinding{"g_SourceScales", sourcePoses.scalesBuffer,
                         Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        GpuBufferBinding{"g_Mappings", mMappingBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        GpuBufferBinding{"g_EntityPositions", mEntityPositionsBuffer,
                         Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        GpuBufferBinding{"g_EntityOrientations", mEntityOrientationsBuffer,
                         Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        GpuBufferBinding{"g_EntityScales", mEntityScalesBuffer,
                         Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return entityPoseSyncPass().dispatch(computeContext.computeContext, 0u, bindings,
                                         dispatchGroupCount(mappingCount));
}

GpuEntitySceneView GpuSceneSync::sceneView() const noexcept
{
    GpuEntitySceneView view{};
    view.layout                             = mLayout;
    view.poses.positionsBuffer              = mEntityPositionsBuffer;
    view.poses.orientationsBuffer           = mEntityOrientationsBuffer;
    view.poses.scalesBuffer                 = mEntityScalesBuffer;
    view.poses.count                        = mEntityCount;
    view.renderableMetadataBuffer           = mRenderableMetadataBuffer;
    view.renderableQueueInfoBuffer          = mRenderableQueueInfoBuffer;
    view.renderableVisibilityFlagsBuffer    = mRenderableVisibilityFlagsBuffer;
    view.renderableShadowCascadeMasksBuffer = mRenderableShadowCascadeMasksBuffer;
    view.cameraInputsBuffer                 = mCameraInputsBuffer;
    view.preparedCamerasBuffer              = mPreparedCamerasBuffer;
    view.lightInputsBuffer                  = mLightInputsBuffer;
    view.localLightSelectionBuffer          = mLocalLightSelectionBuffer;
    view.entityCount                        = mEntityCount;
    view.renderableCount                    = mRenderableCount;
    view.cameraCount                        = mCameraCount;
    view.lightCount                         = mLightCount;
    return view;
}

} // namespace cressim::neo::gpu
