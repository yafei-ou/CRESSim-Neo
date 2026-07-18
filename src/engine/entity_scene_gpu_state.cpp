#include "entity_scene_gpu_state.h"

#include "common/logger.h"
#include "gpu/gpu_buffer_utils.h"
#include "gpu/shader_source_provider.h"

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

EntitySceneGpuState::EntitySceneGpuState(gpu::GpuDevice &device) : mDevice(device) {}

bool EntitySceneGpuState::initialize()
{
    shutdown();

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

    gpu::ShaderSourceProvider shaderSourceProvider(mDevice.shaderSourceConfig());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderSourceProvider.streamFactory();
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

void EntitySceneGpuState::shutdown()
{
    mInitialized                            = false;
    mPoseCapacity                           = 0;
    mPhysicsSyncCapacity                    = 0;
    mEntityCount                            = 0;
    mGraphicsContextMask                    = 0;
    mPhysicsContextMask                     = 0;
    mSharedPoseContextMask                  = 0;
    mMappingBuffer                          = nullptr;
    mConstantsBuffer                        = nullptr;
    mEntityPositionsBuffer                  = nullptr;
    mEntityOrientationsBuffer               = nullptr;
    mEntityScalesBuffer                     = nullptr;
    mEntityPoseSyncPass                     = {};
    mPoseBindingGeneration                  = 1u;
    mPhysicsSyncBindingGeneration           = 1u;
    mLastMappedSourcePoseBindingGeneration  = 0u;
    mLastMappedOutputPoseBindingGeneration  = 0u;
    mLastMappedPhysicsSyncBindingGeneration = 0u;
}

bool EntitySceneGpuState::ensureSharedPoseCapacity(Diligent::IRenderDevice *renderDevice,
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
    }
    return true;
}

bool EntitySceneGpuState::ensurePhysicsSyncCapacity(Diligent::IRenderDevice *renderDevice,
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

    Diligent::IBuffer *oldMapping = mMappingBuffer;
    const bool success            = ensureStructuredBuffer(
        renderDevice, "CRESSimNeo.Gpu.EntityPoseMappings", sizeof(EntityPoseMappingEntry),
        requiredCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DYNAMIC,
        Diligent::CPU_ACCESS_WRITE, mPhysicsContextMask, mMappingBuffer, mPhysicsSyncCapacity, 64u);
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

bool EntitySceneGpuState::writeBuffer(Diligent::IDeviceContext *computeContext,
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

bool EntitySceneGpuState::uploadAuthoredEntityPoses(
    const std::vector<Diligent::float4> &positions,
    const std::vector<Diligent::float4> &orientations, const std::vector<Diligent::float4> &scales)
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

bool EntitySceneGpuState::applyMappedEntityPoses(
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

    std::uint32_t invalidMappingCount = 0u;
    for (std::uint32_t i = 0u; i < mappingCount; ++i)
    {
        const EntityPoseMappingEntry &mapping = mappings[i];
        const bool invalidSource              = mapping.sourcePoseIndex >= sourcePoses.count;
        const bool invalidDest                = mapping.entityPoseIndex >= mEntityCount;
        if (!invalidSource && !invalidDest)
        {
            continue;
        }

        if (invalidMappingCount < 8u)
        {
            CRESSIM_LOG_ERROR(
                "EntitySceneGpuState::applyMappedEntityPoses invalid mapping[", i,
                "]: sourcePoseIndex=", mapping.sourcePoseIndex, " sourceCount=", sourcePoses.count,
                " entityPoseIndex=", mapping.entityPoseIndex, " entityCount=", mEntityCount, ".");
        }
        ++invalidMappingCount;
    }
    if (invalidMappingCount > 0u)
    {
        CRESSIM_LOG_ERROR("EntitySceneGpuState::applyMappedEntityPoses rejected ",
                          invalidMappingCount, " invalid mappings.");
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

common::PoseBufferView EntitySceneGpuState::poseView() const noexcept
{
    common::PoseBufferView view{};
    view.positionsBuffer    = mEntityPositionsBuffer;
    view.orientationsBuffer = mEntityOrientationsBuffer;
    view.scalesBuffer       = mEntityScalesBuffer;
    view.count              = mEntityCount;
    view.bindingGeneration  = mPoseBindingGeneration;
    return view;
}

std::uint32_t EntitySceneGpuState::entityCount() const noexcept
{
    return mEntityCount;
}

} // namespace cressim::neo::engine
