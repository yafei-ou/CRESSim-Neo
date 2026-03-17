#include "gpu/gpu_scene_sync.h"

#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Primitives/interface/Errors.hpp"

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
    std::uint32_t _padding0 = 0;
    std::uint32_t _padding1 = 0;
    std::uint32_t _padding2 = 0;
};

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + kComputeThreadGroupSize - 1u) / kComputeThreadGroupSize;
}

bool ensureStructuredBuffer(Diligent::IRenderDevice* renderDevice, const char* name,
                            std::uint32_t elementStride, std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage,
                            Diligent::CPU_ACCESS_FLAGS cpuAccess,
                            Diligent::Uint64 immediateContextMask,
                            Diligent::RefCntAutoPtr<Diligent::IBuffer>& outBuffer)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    Diligent::BufferDesc desc{};
    desc.Name                 = name;
    desc.Size                 = static_cast<Diligent::Uint64>(elementStride) * elementCount;
    desc.BindFlags            = bindFlags;
    desc.Usage                = usage;
    desc.CPUAccessFlags       = cpuAccess;
    desc.ImmediateContextMask = immediateContextMask;
    if (usage != Diligent::USAGE_STAGING)
    {
        desc.Mode              = Diligent::BUFFER_MODE_STRUCTURED;
        desc.ElementByteStride = elementStride;
    }

    renderDevice->CreateBuffer(desc, nullptr, &outBuffer);
    return outBuffer != nullptr;
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
    "gpu/gpu_entity_pose_sync.cs.hlsl",
    "CRESSimNeo.Gpu.EntityPoseSync",
    "CRESSimNeo.Gpu.EntityPoseSync.PSO",
    kEntityPoseSyncVars,
    std::size(kEntityPoseSyncVars),
};

GpuComputePass& entityPoseSyncPass()
{
    static GpuComputePass pass;
    return pass;
}

bool initializeEntityPoseSyncPass(Diligent::IRenderDevice* renderDevice,
                                  Diligent::IShaderSourceInputStreamFactory* streamFactory,
                                  Diligent::Uint64 contextMask)
{
    return entityPoseSyncPass().initialize(renderDevice, streamFactory, contextMask,
                                           kEntityPoseSyncPassDefinition);
}

} // namespace

GpuSceneSync::GpuSceneSync(GpuDevice& device) : mDevice(device) {}

bool GpuSceneSync::initialize()
{
    shutdown();

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) || computeContext.renderDevice == nullptr)
    {
        return false;
    }

    mContextMask = static_cast<Diligent::Uint64>(1ull) << computeContext.contextId;

    ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory* streamFactory = shaderLibrary.streamFactory();
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
    mInitialized = false;
    mCapacity = 0;
    mEntityCount = 0;
    mRenderableCapacity = 0;
    mRenderableCount = 0;
    mContextMask = 0;
    mMappingBuffer = nullptr;
    mEntityPositionsBuffer = nullptr;
    mEntityOrientationsBuffer = nullptr;
    mEntityScalesBuffer = nullptr;
    mRenderableMetadataBuffer = nullptr;
    mRenderableModelMatricesBuffer = nullptr;
    mRenderableNormalMatricesBuffer = nullptr;
    mRenderableVisibilityFlagsBuffer = nullptr;
    mRenderableShadowCascadeMasksBuffer = nullptr;
    mConstantsBuffer = nullptr;
}

bool GpuSceneSync::ensureCapacity(Diligent::IRenderDevice* renderDevice, std::uint32_t entityCount,
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

    const std::uint32_t newCapacity = std::max<std::uint32_t>(requiredCapacity, 64u);
    const Diligent::Uint64 contextMask = static_cast<Diligent::Uint64>(1ull) << contextId;

    return ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.EntityPoseMappings",
                                  sizeof(GpuEntityPoseMappingEntry), newCapacity,
                                  Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                  Diligent::CPU_ACCESS_NONE, contextMask, mMappingBuffer) &&
           ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.EntityPositions",
                                  sizeof(Diligent::float4), newCapacity,
                                  Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                  Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                  mEntityPositionsBuffer) &&
           ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.EntityOrientations",
                                  sizeof(Diligent::float4), newCapacity,
                                  Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                  Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                  mEntityOrientationsBuffer) &&
           ensureStructuredBuffer(renderDevice, "CRESSimNeo.Gpu.EntityScales",
                                  sizeof(Diligent::float4), newCapacity,
                                  Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                  Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                  mEntityScalesBuffer) &&
           ((mCapacity = newCapacity), true);
}

bool GpuSceneSync::syncRenderableMetadata(const std::vector<GpuRenderableMetadata>& renderables)
{
    if (!mInitialized)
    {
        return false;
    }

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) || computeContext.renderDevice == nullptr ||
        computeContext.computeContext == nullptr)
    {
        return false;
    }

    mRenderableCount = static_cast<std::uint32_t>(renderables.size());
    const std::uint32_t requiredCapacity = std::max<std::uint32_t>(mRenderableCount, 1u);
    if (mRenderableCapacity < requiredCapacity || mRenderableMetadataBuffer == nullptr ||
        mRenderableModelMatricesBuffer == nullptr || mRenderableNormalMatricesBuffer == nullptr ||
        mRenderableVisibilityFlagsBuffer == nullptr || mRenderableShadowCascadeMasksBuffer == nullptr)
    {
        const std::uint32_t newCapacity = std::max<std::uint32_t>(requiredCapacity, 64u);
        const Diligent::Uint64 contextMask =
            static_cast<Diligent::Uint64>(1ull) << computeContext.contextId;
        if (!ensureStructuredBuffer(computeContext.renderDevice, "CRESSimNeo.Gpu.RenderableMetadata",
                                    sizeof(GpuRenderableMetadata), newCapacity,
                                    Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                    Diligent::CPU_ACCESS_NONE, contextMask, mRenderableMetadataBuffer) ||
            !ensureStructuredBuffer(computeContext.renderDevice,
                                    "CRESSimNeo.Gpu.RenderableModelMatrices",
                                    sizeof(Diligent::float4x4), newCapacity,
                                    Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                    Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                    mRenderableModelMatricesBuffer) ||
            !ensureStructuredBuffer(computeContext.renderDevice,
                                    "CRESSimNeo.Gpu.RenderableNormalMatrices",
                                    sizeof(Diligent::float4x4), newCapacity,
                                    Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                    Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                    mRenderableNormalMatricesBuffer) ||
            !ensureStructuredBuffer(computeContext.renderDevice,
                                    "CRESSimNeo.Gpu.RenderableVisibilityFlags",
                                    sizeof(std::uint32_t), newCapacity,
                                    Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                    Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                    mRenderableVisibilityFlagsBuffer) ||
            !ensureStructuredBuffer(computeContext.renderDevice,
                                    "CRESSimNeo.Gpu.RenderableShadowCascadeMasks",
                                    sizeof(std::uint32_t), newCapacity,
                                    Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                    Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                    mRenderableShadowCascadeMasksBuffer))
        {
            return false;
        }
        mRenderableCapacity = newCapacity;
    }

    if (mRenderableCount == 0u)
    {
        return true;
    }

    return writeBuffer(computeContext.computeContext, mRenderableMetadataBuffer, renderables.data(),
                       renderables.size() * sizeof(GpuRenderableMetadata));
}

bool GpuSceneSync::writeBuffer(Diligent::IDeviceContext* computeContext, Diligent::IBuffer* buffer,
                               const void* data, std::size_t sizeBytes)
{
    if (computeContext == nullptr || buffer == nullptr || data == nullptr || sizeBytes == 0u)
    {
        return false;
    }

    const Diligent::BufferDesc& desc = buffer->GetDesc();
    if (desc.Usage != Diligent::USAGE_DYNAMIC)
    {
        computeContext->UpdateBuffer(buffer, 0u, static_cast<Diligent::Uint32>(sizeBytes), data,
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        return true;
    }

    void* mapped = nullptr;
    computeContext->MapBuffer(buffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }

    std::memcpy(mapped, data, sizeBytes);
    computeContext->UnmapBuffer(buffer, Diligent::MAP_WRITE);
    return true;
}

bool GpuSceneSync::syncEntityPoses(const GpuPoseBufferView& sourcePoses,
                                   const std::vector<GpuEntityPoseMappingEntry>& mappings)
{
    if (!mInitialized)
    {
        return false;
    }

    GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) || computeContext.renderDevice == nullptr ||
        computeContext.computeContext == nullptr)
    {
        return false;
    }

    mEntityCount = static_cast<std::uint32_t>(mappings.size());
    if (mEntityCount == 0u)
    {
        return true;
    }
    if (sourcePoses.positionsBuffer == nullptr || sourcePoses.orientationsBuffer == nullptr ||
        sourcePoses.scalesBuffer == nullptr || sourcePoses.count == 0u)
    {
        return false;
    }
    if (!ensureCapacity(computeContext.renderDevice, mEntityCount, computeContext.contextId))
    {
        return false;
    }

    if (!writeBuffer(computeContext.computeContext, mMappingBuffer, mappings.data(),
                     mappings.size() * sizeof(GpuEntityPoseMappingEntry)))
    {
        return false;
    }

    const GpuEntityPoseSyncConstants constants{mEntityCount, 0u, 0u, 0u};
    if (!writeBuffer(computeContext.computeContext, mConstantsBuffer, &constants, sizeof(constants)))
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
                                         dispatchGroupCount(mEntityCount));
}

GpuEntitySceneView GpuSceneSync::sceneView() const noexcept
{
    GpuEntitySceneView view{};
    view.poses.positionsBuffer = mEntityPositionsBuffer;
    view.poses.orientationsBuffer = mEntityOrientationsBuffer;
    view.poses.scalesBuffer = mEntityScalesBuffer;
    view.poses.count = mEntityCount;
    view.renderableMetadataBuffer = mRenderableMetadataBuffer;
    view.renderableModelMatricesBuffer = mRenderableModelMatricesBuffer;
    view.renderableNormalMatricesBuffer = mRenderableNormalMatricesBuffer;
    view.renderableVisibilityFlagsBuffer = mRenderableVisibilityFlagsBuffer;
    view.renderableShadowCascadeMasksBuffer = mRenderableShadowCascadeMasksBuffer;
    view.entityCount = mEntityCount;
    view.renderableCount = mRenderableCount;
    return view;
}

} // namespace cressim::neo::gpu
