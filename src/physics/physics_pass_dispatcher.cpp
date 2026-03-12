#include "physics/physics_pass_dispatcher.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/ShaderResourceVariable.h"
#include "DiligentEngine/DiligentCore/Primitives/interface/Errors.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uint32_t kComputeThreadGroupSize = 64u;
constexpr std::uint32_t kNarrowPhaseChunkSize   = 128u;

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + kComputeThreadGroupSize - 1u) / kComputeThreadGroupSize;
}

} // namespace

template <std::size_t N>
bool PhysicsPassDispatcher::bindBufferVariables(Diligent::IShaderResourceBinding* srb,
                                                const std::array<BufferBinding, N>& bindings)
{
    for (const BufferBinding& binding : bindings)
    {
        if (!bindBufferVariable(srb, binding.variableName, binding.buffer, binding.viewType))
        {
            return false;
        }
    }
    return true;
}

template <std::size_t N>
bool PhysicsPassDispatcher::bindBufferVariables(const ComputePass& pass,
                                                const std::array<BufferBinding, N>& bindings)
{
    return bindBufferVariables(pass.srb, bindings);
}

template <std::size_t N>
bool PhysicsPassDispatcher::dispatchComputePass(
    Diligent::IDeviceContext* computeContext, const ComputePass& pass,
    const std::array<BufferBinding, N>& bindings, const GpuRigidDispatchConstants* constants,
    std::uint32_t groupCountX, std::uint32_t groupCountY, std::uint32_t groupCountZ)
{
    if (computeContext == nullptr || pass.pso == nullptr || pass.srb == nullptr)
    {
        return false;
    }

    if (!bindBufferVariables(pass, bindings))
    {
        return false;
    }

    if (constants != nullptr && !writeDispatchConstants(computeContext, *constants))
    {
        return false;
    }

    computeContext->SetPipelineState(pass.pso);
    computeContext->CommitShaderResources(pass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{groupCountX, groupCountY, groupCountZ});
    return true;
}

bool PhysicsPassDispatcher::writeDispatchConstants(Diligent::IDeviceContext* computeContext,
                                                   const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr || mDispatchConstantsBuffer == nullptr)
    {
        return false;
    }

    void* mapped = nullptr;
    computeContext->MapBuffer(mDispatchConstantsBuffer, Diligent::MAP_WRITE,
                              Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }

    std::memcpy(mapped, &constants, sizeof(constants));
    computeContext->UnmapBuffer(mDispatchConstantsBuffer, Diligent::MAP_WRITE);
    return true;
}

bool PhysicsPassDispatcher::bindBufferVariable(Diligent::IShaderResourceBinding* srb,
                                               const char* variableName, Diligent::IBuffer* buffer,
                                               Diligent::BUFFER_VIEW_TYPE viewType)
{
    if (srb == nullptr || buffer == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: bindBufferVariable invalid input for '",
                          variableName, "'.");
        return false;
    }

    Diligent::IShaderResourceVariable* variable =
        srb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE, variableName);
    if (variable == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: shader variable not found: '", variableName,
                          "'.");
        return false;
    }

    Diligent::IBufferView* view = buffer->GetDefaultView(viewType);
    if (view != nullptr)
    {
        variable->Set(view);
        return true;
    }

    variable->Set(buffer);
    return true;
}

bool PhysicsPassDispatcher::createShaderResourceBinding(
    Diligent::IPipelineState* pso,
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>& outSrb)
{
    if (pso == nullptr)
    {
        return false;
    }

    pso->CreateShaderResourceBinding(&outSrb, true);
    return outSrb != nullptr;
}

bool PhysicsPassDispatcher::createComputePass(
    Diligent::IRenderDevice* renderDevice, Diligent::IShaderSourceInputStreamFactory* streamFactory,
    const char* shaderPath, const char* shaderName, const char* psoName,
    const Diligent::ShaderResourceVariableDesc* variables, std::size_t variableCount,
    ComputePass& outPass)
{
    outPass = {};

    if (renderDevice == nullptr || streamFactory == nullptr)
    {
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Desc.ShaderType                 = Diligent::SHADER_TYPE_COMPUTE;
    shaderCreateInfo.Desc.Name                       = shaderName;
    shaderCreateInfo.FilePath                        = shaderPath;
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::RefCntAutoPtr<Diligent::IShader> computeShader;
    renderDevice->CreateShader(shaderCreateInfo, &computeShader);
    if (computeShader == nullptr)
    {
        return false;
    }

    Diligent::ComputePipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                 = psoName;
    psoCreateInfo.PSODesc.PipelineType         = Diligent::PIPELINE_TYPE_COMPUTE;
    psoCreateInfo.PSODesc.ImmediateContextMask = mPhysicsContextMask;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    psoCreateInfo.PSODesc.ResourceLayout.Variables = variables;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(variableCount);
    psoCreateInfo.pCS = computeShader;

    renderDevice->CreateComputePipelineState(psoCreateInfo, &outPass.pso);
    if (outPass.pso == nullptr)
    {
        return false;
    }

    return createShaderResourceBinding(outPass.pso, outPass.srb);
}

bool PhysicsPassDispatcher::initialize(Diligent::IRenderDevice* renderDevice,
                                       std::uint32_t physicsContextId,
                                       const char* shaderSourceDirectory)
{
    if (renderDevice == nullptr || shaderSourceDirectory == nullptr)
    {
        return false;
    }

    mShaderLibrary      = gpu::ShaderLibrary(shaderSourceDirectory);
    mPhysicsContextMask = static_cast<Diligent::Uint64>(1ull) << physicsContextId;

    Diligent::IShaderSourceInputStreamFactory* streamFactory = mShaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: shader stream factory is null.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kPredictVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyKinematicTargetPositions",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyKinematicTargetOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyKinematicTargetFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory, "physics/physics_rigid_predict.cs.hlsl",
                           "CRESSimNeo.Physics.RigidPredict.CS",
                           "CRESSimNeo.Physics.RigidPredict.PSO", kPredictVars,
                           std::size(kPredictVars), mPredictPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create rigid predict pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kUpdateWorldAabbsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderShapeTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderParams",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBodyFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory,
                           "physics/physics_rigid_update_world_aabbs.cs.hlsl",
                           "CRESSimNeo.Physics.RigidUpdateWorldAabbs.CS",
                           "CRESSimNeo.Physics.RigidUpdateWorldAabbs.PSO", kUpdateWorldAabbsVars,
                           std::size(kUpdateWorldAabbsVars), mUpdateWorldAabbsPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create world AABB pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kScanBlockVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ScanInput",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ScanOutput",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BlockSums",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePass(renderDevice, streamFactory, "physics/physics_scan_block.cs.hlsl",
                           "CRESSimNeo.Physics.ScanBlock.CS", "CRESSimNeo.Physics.ScanBlock.PSO",
                           kScanBlockVars, std::size(kScanBlockVars), mScanBlockPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create scan block pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kScanAddOffsetsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ScannedBlockOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ScanOutput",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePass(renderDevice, streamFactory, "physics/physics_scan_add_offsets.cs.hlsl",
                           "CRESSimNeo.Physics.ScanAddOffsets.CS",
                           "CRESSimNeo.Physics.ScanAddOffsets.PSO", kScanAddOffsetsVars,
                           std::size(kScanAddOffsetsVars), mScanAddOffsetsPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create scan add-offsets pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kCompactActiveBodiesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_compact_active_bodies.cs.hlsl",
            "CRESSimNeo.Physics.RigidCompactActiveBodies.CS",
            "CRESSimNeo.Physics.RigidCompactActiveBodies.PSO", kCompactActiveBodiesVars,
            std::size(kCompactActiveBodiesVars), mCompactActiveBodiesPass))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsPassDispatcher: failed to create active body compaction pipeline.");
        return false;
    }
    if (!createShaderResourceBinding(mCompactActiveBodiesPass.pso, mCompactStaticBodiesSrb))
    {
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kFinalizeActiveBodiesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBodyFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBodyOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_finalize_active_bodies.cs.hlsl",
            "CRESSimNeo.Physics.RigidFinalizeActiveBodies.CS",
            "CRESSimNeo.Physics.RigidFinalizeActiveBodies.PSO", kFinalizeActiveBodiesVars,
            std::size(kFinalizeActiveBodiesVars), mFinalizeActiveBodiesPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create active body finalize pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kBuildBroadPhaseElementsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_build_broad_phase_elements.cs.hlsl",
            "CRESSimNeo.Physics.RigidBuildBroadPhaseElements.CS",
            "CRESSimNeo.Physics.RigidBuildBroadPhaseElements.PSO", kBuildBroadPhaseElementsVars,
            std::size(kBuildBroadPhaseElementsVars), mBuildBroadPhaseElementsPass))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsPassDispatcher: failed to create broad-phase element build pipeline.");
        return false;
    }
    if (!createShaderResourceBinding(mBuildBroadPhaseElementsPass.pso,
                                     mBuildStaticBroadPhaseElementsSrb))
    {
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kReduceExtentElementsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_GroupExtents",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_reduce_extent_elements.cs.hlsl",
            "CRESSimNeo.Physics.RigidReduceExtentElements.CS",
            "CRESSimNeo.Physics.RigidReduceExtentElements.PSO", kReduceExtentElementsVars,
            std::size(kReduceExtentElementsVars), mReduceExtentElementsPass))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsPassDispatcher: failed to create element extent reduction pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kReduceExtentExtentsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_InputExtents",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_OutputExtents",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_reduce_extent_extents.cs.hlsl",
            "CRESSimNeo.Physics.RigidReduceExtentExtents.CS",
            "CRESSimNeo.Physics.RigidReduceExtentExtents.PSO", kReduceExtentExtentsVars,
            std::size(kReduceExtentExtentsVars), mReduceExtentExtentsPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create extent reduction pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kMortonCodesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_GlobalExtent",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_morton_codes.cs.hlsl",
            "CRESSimNeo.Physics.RigidMortonCodes.CS", "CRESSimNeo.Physics.RigidMortonCodes.PSO",
            kMortonCodesVars, std::size(kMortonCodesVars), mMortonCodesPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create morton code pipeline.");
        return false;
    }
    if (!createShaderResourceBinding(mMortonCodesPass.pso, mStaticMortonCodesSrb))
    {
        return false;
    }

    // TODO: maybe it's better to create separate SRBs for these as well.

    constexpr Diligent::ShaderResourceVariableDesc kRadixClassifyVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesIn",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_radix_classify.cs.hlsl",
            "CRESSimNeo.Physics.RigidRadixClassify.CS", "CRESSimNeo.Physics.RigidRadixClassify.PSO",
            kRadixClassifyVars, std::size(kRadixClassifyVars), mRadixClassifyPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create radix classify pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kRadixFinalizeVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_radix_finalize.cs.hlsl",
            "CRESSimNeo.Physics.RigidRadixFinalize.CS", "CRESSimNeo.Physics.RigidRadixFinalize.PSO",
            kRadixFinalizeVars, std::size(kRadixFinalizeVars), mRadixFinalizePass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create radix finalize pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kRadixScatterVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesIn",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesOut",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_radix_scatter.cs.hlsl",
            "CRESSimNeo.Physics.RigidRadixScatter.CS", "CRESSimNeo.Physics.RigidRadixScatter.PSO",
            kRadixScatterVars, std::size(kRadixScatterVars), mRadixScatterPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create radix scatter pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kBvhHierarchyVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_SortedMortonCodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhConstructionInfos",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_bvh_hierarchy.cs.hlsl",
            "CRESSimNeo.Physics.RigidBvhHierarchy.CS", "CRESSimNeo.Physics.RigidBvhHierarchy.PSO",
            kBvhHierarchyVars, std::size(kBvhHierarchyVars), mBvhHierarchyPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create BVH hierarchy pipeline.");
        return false;
    }
    if (!createShaderResourceBinding(mBvhHierarchyPass.pso, mStaticBvhHierarchySrb))
    {
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kBvhBoundingBoxesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhConstructionInfos",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory,
                           "physics/physics_rigid_bvh_bounding_boxes.cs.hlsl",
                           "CRESSimNeo.Physics.RigidBvhBoundingBoxes.CS",
                           "CRESSimNeo.Physics.RigidBvhBoundingBoxes.PSO", kBvhBoundingBoxesVars,
                           std::size(kBvhBoundingBoxesVars), mBvhBoundingBoxesPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create BVH bounding box pipeline.");
        return false;
    }
    if (!createShaderResourceBinding(mBvhBoundingBoxesPass.pso, mStaticBvhBoundingBoxesSrb))
    {
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kCountPairsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderShapeTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereSphere",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereBox",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsBoxBox",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsBoxCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsCapsuleCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory, "physics/physics_rigid_count_pairs.cs.hlsl",
                           "CRESSimNeo.Physics.RigidCountPairs.CS",
                           "CRESSimNeo.Physics.RigidCountPairs.PSO", kCountPairsVars,
                           std::size(kCountPairsVars), mCountPairsPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create pair count pipeline.");
        return false;
    }
    if (!createShaderResourceBinding(mCountPairsPass.pso, mCountPairsMovingSrb))
    {
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kFinalizePairsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereSphere",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereBox",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsBoxBox",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsBoxCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsCapsuleCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereSphere",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereBox",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsBoxBox",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsBoxCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsCapsuleCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidPairRanges",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_finalize_pairs.cs.hlsl",
            "CRESSimNeo.Physics.RigidFinalizePairs.CS", "CRESSimNeo.Physics.RigidFinalizePairs.PSO",
            kFinalizePairsVars, std::size(kFinalizePairsVars), mFinalizePairsPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create pair finalize pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kEmitPairsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderShapeTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereSphere",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereBox",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsBoxBox",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsBoxCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsCapsuleCapsule",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidPairRanges",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_CandidatePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory, "physics/physics_rigid_emit_pairs.cs.hlsl",
                           "CRESSimNeo.Physics.RigidEmitPairs.CS",
                           "CRESSimNeo.Physics.RigidEmitPairs.PSO", kEmitPairsVars,
                           std::size(kEmitPairsVars), mEmitPairsPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create pair emit pipeline.");
        return false;
    }
    if (!createShaderResourceBinding(mEmitPairsPass.pso, mEmitPairsMovingSrb))
    {
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kBuildNarrowPhaseChunksVars[] = {
        // {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
        //  Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidPairRanges",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseChunks",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseChunkCounter",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_build_narrow_phase_chunks.cs.hlsl",
            "CRESSimNeo.Physics.RigidBuildNarrowPhaseChunks.CS",
            "CRESSimNeo.Physics.RigidBuildNarrowPhaseChunks.PSO", kBuildNarrowPhaseChunksVars,
            std::size(kBuildNarrowPhaseChunksVars), mBuildNarrowPhaseChunksPass))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsPassDispatcher: failed to create narrow-phase chunk build pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kGenerateContactsVars[] = {
        // {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
        //  Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderParams",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_CandidatePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseChunks",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseChunkCounter",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory,
                           "physics/physics_rigid_generate_contacts.cs.hlsl",
                           "CRESSimNeo.Physics.RigidGenerateContacts.CS",
                           "CRESSimNeo.Physics.RigidGenerateContacts.PSO", kGenerateContactsVars,
                           std::size(kGenerateContactsVars), mGenerateContactsPass))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsPassDispatcher: failed to create rigid contact generation pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kSolveGatherVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(
            renderDevice, streamFactory, "physics/physics_rigid_solve_gather.cs.hlsl",
            "CRESSimNeo.Physics.RigidSolveGather.CS", "CRESSimNeo.Physics.RigidSolveGather.PSO",
            kSolveGatherVars, std::size(kSolveGatherVars), mSolveGatherPass))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to create rigid solve gather pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kClearCorrectionsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory,
                           "physics/physics_rigid_clear_corrections.cs.hlsl",
                           "CRESSimNeo.Physics.RigidClearCorrections.CS",
                           "CRESSimNeo.Physics.RigidClearCorrections.PSO", kClearCorrectionsVars,
                           std::size(kClearCorrectionsVars), mClearCorrectionsPass))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsPassDispatcher: failed to create rigid correction clear pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kApplyCorrectionsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory,
                           "physics/physics_rigid_apply_corrections.cs.hlsl",
                           "CRESSimNeo.Physics.RigidApplyCorrections.CS",
                           "CRESSimNeo.Physics.RigidApplyCorrections.PSO", kApplyCorrectionsVars,
                           std::size(kApplyCorrectionsVars), mApplyCorrectionsPass))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsPassDispatcher: failed to create rigid correction apply pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kUpdateVelocitiesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePass(renderDevice, streamFactory,
                           "physics/physics_rigid_update_velocities.cs.hlsl",
                           "CRESSimNeo.Physics.RigidUpdateVelocities.CS",
                           "CRESSimNeo.Physics.RigidUpdateVelocities.PSO", kUpdateVelocitiesVars,
                           std::size(kUpdateVelocitiesVars), mUpdateVelocitiesPass))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsPassDispatcher: failed to create rigid velocity update pipeline.");
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Name                 = "CRESSimNeo.Physics.DispatchConstants";
    constantsDesc.Size                 = sizeof(GpuRigidDispatchConstants);
    constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    constantsDesc.ImmediateContextMask = mPhysicsContextMask;
    renderDevice->CreateBuffer(constantsDesc, nullptr, &mDispatchConstantsBuffer);
    return mDispatchConstantsBuffer != nullptr;
}

bool PhysicsPassDispatcher::dispatchScanBlockPass(Diligent::IDeviceContext* computeContext,
                                                  const PhysicsSceneGpuState&,
                                                  Diligent::IBuffer* input,
                                                  Diligent::IBuffer* output,
                                                  Diligent::IBuffer* blockSums, std::uint32_t count,
                                                  const GpuRigidDispatchConstants& constants)
{
    if (count == 0u)
    {
        return false;
    }

    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ScanInput", input, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ScanOutput", output, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BlockSums", blockSums, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    GpuRigidDispatchConstants scanConstants = constants;
    scanConstants.candidatePairCapacity     = count;

    return dispatchComputePass(computeContext, mScanBlockPass, bindings, &scanConstants,
                               dispatchGroupCount(count));
}

bool PhysicsPassDispatcher::dispatchScanAddOffsetsPass(Diligent::IDeviceContext* computeContext,
                                                       Diligent::IBuffer* output,
                                                       Diligent::IBuffer* scannedBlockOffsets,
                                                       std::uint32_t count,
                                                       const GpuRigidDispatchConstants& constants)
{
    if (count == 0u)
    {
        return false;
    }

    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ScannedBlockOffsets", scannedBlockOffsets,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ScanOutput", output, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    GpuRigidDispatchConstants scanConstants = constants;
    scanConstants.candidatePairCapacity     = count;

    return dispatchComputePass(computeContext, mScanAddOffsetsPass, bindings, &scanConstants,
                               dispatchGroupCount(count));
}

bool PhysicsPassDispatcher::dispatchExclusiveScanPass(
    Diligent::IDeviceContext* computeContext, const PhysicsSceneGpuState& sceneState,
    Diligent::IBuffer* input, Diligent::IBuffer* output, std::uint32_t count,
    const GpuRigidDispatchConstants& constants, std::uint32_t recursionLevel)
{
    if (count == 0u)
    {
        return true;
    }

    const auto& transientState = sceneState.transientBuffers();
    if (recursionLevel >= transientState.scanBlockSumsBuffers.size() ||
        recursionLevel >= transientState.scanScannedBlockSumsBuffers.size())
    {
        return false;
    }

    Diligent::IBuffer* blockSums = transientState.scanBlockSumsBuffers[recursionLevel];
    Diligent::IBuffer* scannedBlockSums =
        transientState.scanScannedBlockSumsBuffers[recursionLevel];
    if (!dispatchScanBlockPass(computeContext, sceneState, input, output, blockSums, count,
                               constants))
    {
        return false;
    }

    const std::uint32_t groupCount = dispatchGroupCount(count);
    if (groupCount <= 1u)
    {
        return true;
    }

    if (!dispatchExclusiveScanPass(computeContext, sceneState, blockSums, scannedBlockSums,
                                   groupCount, constants, recursionLevel + 1u))
    {
        return false;
    }

    return dispatchScanAddOffsetsPass(computeContext, output, scannedBlockSums, count, constants);
}

bool PhysicsPassDispatcher::clearCorrections(Diligent::IDeviceContext* computeContext,
                                             PhysicsSceneGpuState& sceneState,
                                             std::uint32_t bodyCount,
                                             const GpuRigidDispatchConstants& constants)
{
    if (bodyCount == 0u)
    {
        sceneState.setCorrectionBuffersNeedClear(false);
        return true;
    }

    const auto& transientState = sceneState.transientBuffers();
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyTranslationCorrections",
                      transientState.translationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_RigidBodyRotationCorrections", transientState.rotationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!dispatchComputePass(computeContext, mClearCorrectionsPass, bindings, &constants,
                             dispatchGroupCount(bodyCount)))
    {
        return false;
    }

    sceneState.setCorrectionBuffersNeedClear(false);
    return true;
}

bool PhysicsPassDispatcher::predict(Diligent::IDeviceContext* computeContext,
                                    const PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                                    const GpuRigidDispatchConstants& constants)
{
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyPositionsInvMass", persistent.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyOrientations", persistent.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyLinearVelocities", persistent.linearVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyAngularVelocities", persistent.angularVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyKinematicTargetPositions",
                      persistent.kinematicTargetPositionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyKinematicTargetOrientations",
                      persistent.kinematicTargetOrientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyKinematicTargetFlags", persistent.kinematicTargetFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                      transient.previousRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PreviousRigidBodyOrientations",
                      transient.previousRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transient.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transient.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyLinearVelocities",
                      transient.predictedRigidBodies.linearVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyAngularVelocities",
                      transient.predictedRigidBodies.angularVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return dispatchComputePass(computeContext, mPredictPass, bindings, &constants,
                               dispatchGroupCount(bodyCount));
}

bool PhysicsPassDispatcher::updateWorldAabbs(Diligent::IDeviceContext* computeContext,
                                             const PhysicsSceneGpuState& sceneState,
                                             std::uint32_t bodyCount,
                                             const GpuRigidDispatchConstants& constants)
{
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transient.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transient.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyScales", persistent.scalesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderShapeTypes", persistent.colliderShapeTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderParams", persistent.colliderParamsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_ActiveBodyFlags", transient.activeBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_StaticBodyFlags", transient.staticBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return dispatchComputePass(computeContext, mUpdateWorldAabbsPass, bindings, &constants,
                               dispatchGroupCount(bodyCount));
}

bool PhysicsPassDispatcher::compactActiveBodies(Diligent::IDeviceContext* computeContext,
                                                const PhysicsSceneGpuState& sceneState,
                                                std::uint32_t bodyCount,
                                                const GpuRigidDispatchConstants& constants)
{
    // Conceptual example:
    //
    // full body domain:
    // - bodyIndex:          0 1 2 3 4 5
    // - active flags:       0 1 0 1 1 0
    // - exclusive offsets:  0 0 1 1 2 3
    //
    // compact domain:
    // - activeIndex:        0 1 2
    // - activeBodyIndices:  1 3 4

    if (!dispatchExclusiveScanPass(
            computeContext, sceneState, sceneState.transientBuffers().activeBodyFlagsBuffer,
            sceneState.transientBuffers().activeBodyOffsetsBuffer, bodyCount, constants))
    {
        return false;
    }
    if (!dispatchExclusiveScanPass(
            computeContext, sceneState, sceneState.transientBuffers().staticBodyFlagsBuffer,
            sceneState.transientBuffers().staticBodyOffsetsBuffer, bodyCount, constants))
    {
        return false;
    }

    const auto& transient = sceneState.transientBuffers();
    const std::array compactBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyFlags", transient.activeBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyOffsets", transient.activeBodyOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transient.activeBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array finalizeBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyFlags", transient.activeBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyOffsets", transient.activeBodyOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_StaticBodyFlags", transient.staticBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_StaticBodyOffsets", transient.staticBodyOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mCompactActiveBodiesPass, compactBindings) ||
        !bindBufferVariables(mFinalizeActiveBodiesPass, finalizeBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }

    computeContext->SetPipelineState(mCompactActiveBodiesPass.pso);
    computeContext->CommitShaderResources(mCompactActiveBodiesPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(bodyCount), 1u, 1u});

    const std::array staticCompactBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyFlags", transient.staticBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyOffsets", transient.staticBodyOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transient.staticBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mCompactStaticBodiesSrb, staticCompactBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mCompactActiveBodiesPass.pso);
    computeContext->CommitShaderResources(mCompactStaticBodiesSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(bodyCount), 1u, 1u});

    computeContext->SetPipelineState(mFinalizeActiveBodiesPass.pso);
    computeContext->CommitShaderResources(mFinalizeActiveBodiesPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{1u, 1u, 1u});
    return true;
}

bool PhysicsPassDispatcher::dispatchReduceBroadPhaseExtentPass(
    Diligent::IDeviceContext* computeContext, const PhysicsSceneGpuState& sceneState,
    std::uint32_t activeMovingCount, bool useStaticSet)
{
    const auto& transient = sceneState.transientBuffers();
    if (computeContext == nullptr || activeMovingCount == 0u ||
        (useStaticSet ? transient.staticBroadPhaseExtentScratchBuffers.empty()
                      : transient.broadPhaseExtentScratchBuffers.empty()) ||
        (useStaticSet ? transient.staticGlobalBroadPhaseExtentBuffer == nullptr
                      : transient.globalBroadPhaseExtentBuffer == nullptr))
    {
        return false;
    }

    const std::uint32_t initialGroupCount = dispatchGroupCount(activeMovingCount);
    Diligent::IBuffer* currentOutput =
        (initialGroupCount <= 1u)
            ? (useStaticSet ? transient.staticGlobalBroadPhaseExtentBuffer
                            : transient.globalBroadPhaseExtentBuffer)
            : (useStaticSet ? transient.staticBroadPhaseExtentScratchBuffers.front()
                            : transient.broadPhaseExtentScratchBuffers.front());
    GpuRigidDispatchConstants reductionConstants{};
    reductionConstants.activeMovingCount     = activeMovingCount;
    reductionConstants.staticBodyCount       = activeMovingCount;
    reductionConstants.candidatePairCapacity = activeMovingCount;

    const std::array firstBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements",
                      useStaticSet ? transient.staticBroadPhaseElementsBuffer
                                   : transient.broadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_GroupExtents", currentOutput, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mReduceExtentElementsPass, firstBindings) ||
        !writeDispatchConstants(computeContext, reductionConstants))
    {
        return false;
    }

    computeContext->SetPipelineState(mReduceExtentElementsPass.pso);
    computeContext->CommitShaderResources(mReduceExtentElementsPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    std::uint32_t currentCount = activeMovingCount;
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(currentCount), 1u, 1u});

    currentCount                    = dispatchGroupCount(currentCount);
    std::uint32_t level             = 1u;
    Diligent::IBuffer* currentInput = currentOutput;
    while (currentCount > 1u)
    {
        const std::uint32_t nextGroupCount = dispatchGroupCount(currentCount);
        const auto& scratchBuffers = useStaticSet ? transient.staticBroadPhaseExtentScratchBuffers
                                                  : transient.broadPhaseExtentScratchBuffers;
        if (nextGroupCount > 1u && level >= scratchBuffers.size())
        {
            return false;
        }

        currentOutput                            = (nextGroupCount <= 1u)
                                                       ? (useStaticSet ? transient.staticGlobalBroadPhaseExtentBuffer
                                                                       : transient.globalBroadPhaseExtentBuffer)
                                                       : scratchBuffers[level];
        reductionConstants.candidatePairCapacity = currentCount;
        const std::array reduceBindings{
            BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_InputExtents", currentInput, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_OutputExtents", currentOutput, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!bindBufferVariables(mReduceExtentExtentsPass.srb, reduceBindings) ||
            !writeDispatchConstants(computeContext, reductionConstants))
        {
            return false;
        }

        computeContext->SetPipelineState(mReduceExtentExtentsPass.pso);
        computeContext->CommitShaderResources(mReduceExtentExtentsPass.srb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(
            Diligent::DispatchComputeAttribs{dispatchGroupCount(currentCount), 1u, 1u});

        currentInput = currentOutput;
        currentCount = nextGroupCount;
        ++level;
    }

    return true;
}

bool PhysicsPassDispatcher::dispatchRadixSortPass(Diligent::IDeviceContext* computeContext,
                                                  const PhysicsSceneGpuState& sceneState,
                                                  std::uint32_t activeMovingCount,
                                                  bool useStaticSet,
                                                  const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr || activeMovingCount == 0u)
    {
        return false;
    }

    const auto& transient = sceneState.transientBuffers();
    Diligent::IBuffer* currentInput =
        useStaticSet ? transient.staticMortonCodesBuffer : transient.mortonCodesBuffer;
    Diligent::IBuffer* currentOutput = useStaticSet ? transient.staticMortonCodesScratchBuffer
                                                    : transient.mortonCodesScratchBuffer;
    for (std::uint32_t bit = 0u; bit < 32u; ++bit)
    {
        GpuRigidDispatchConstants radixConstants = constants;
        radixConstants.activeMovingCount         = activeMovingCount;
        radixConstants.staticBodyCount           = activeMovingCount;
        radixConstants.candidatePairCapacity     = activeMovingCount;
        radixConstants.iterationIndex            = bit;
        if (!writeDispatchConstants(computeContext, radixConstants))
        {
            return false;
        }

        const std::array classifyBindings{
            BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_MortonCodesIn", currentInput, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_RadixBitFlags",
                          useStaticSet ? transient.staticRadixBitFlagsBuffer
                                       : transient.radixBitFlagsBuffer,
                          Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!bindBufferVariables(mRadixClassifyPass.srb, classifyBindings))
        {
            return false;
        }
        computeContext->SetPipelineState(mRadixClassifyPass.pso);
        computeContext->CommitShaderResources(mRadixClassifyPass.srb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(
            Diligent::DispatchComputeAttribs{dispatchGroupCount(activeMovingCount), 1u, 1u});

        if (!dispatchExclusiveScanPass(computeContext, sceneState,
                                       useStaticSet ? transient.staticRadixBitFlagsBuffer
                                                    : transient.radixBitFlagsBuffer,
                                       useStaticSet ? transient.staticRadixBitOffsetsBuffer
                                                    : transient.radixBitOffsetsBuffer,
                                       activeMovingCount, radixConstants))
        {
            return false;
        }

        const std::array finalizeBindings{
            BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_RadixBitFlags",
                          useStaticSet ? transient.staticRadixBitFlagsBuffer
                                       : transient.radixBitFlagsBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_RadixBitOffsets",
                          useStaticSet ? transient.staticRadixBitOffsetsBuffer
                                       : transient.radixBitOffsetsBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_RadixMeta",
                          useStaticSet ? transient.staticRadixMetaBuffer
                                       : transient.radixMetaBuffer,
                          Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!bindBufferVariables(mRadixFinalizePass.srb, finalizeBindings))
        {
            return false;
        }
        computeContext->SetPipelineState(mRadixFinalizePass.pso);
        computeContext->CommitShaderResources(mRadixFinalizePass.srb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{1u, 1u, 1u});

        const std::array scatterBindings{
            BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_MortonCodesIn", currentInput, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_RadixBitFlags",
                          useStaticSet ? transient.staticRadixBitFlagsBuffer
                                       : transient.radixBitFlagsBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_RadixBitOffsets",
                          useStaticSet ? transient.staticRadixBitOffsetsBuffer
                                       : transient.radixBitOffsetsBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_RadixMeta",
                          useStaticSet ? transient.staticRadixMetaBuffer
                                       : transient.radixMetaBuffer,
                          Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            BufferBinding{"g_MortonCodesOut", currentOutput,
                          Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!bindBufferVariables(mRadixScatterPass.srb, scatterBindings))
        {
            return false;
        }
        computeContext->SetPipelineState(mRadixScatterPass.pso);
        computeContext->CommitShaderResources(mRadixScatterPass.srb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(
            Diligent::DispatchComputeAttribs{dispatchGroupCount(activeMovingCount), 1u, 1u});

        std::swap(currentInput, currentOutput);
    }

    Diligent::IBuffer* finalMortonBuffer =
        useStaticSet ? transient.staticMortonCodesBuffer : transient.mortonCodesBuffer;
    if (currentInput != finalMortonBuffer)
    {
        const Diligent::Uint64 bytes =
            static_cast<Diligent::Uint64>(activeMovingCount) * sizeof(GpuMortonCodeElement);
        computeContext->CopyBuffer(
            currentInput, 0u, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            finalMortonBuffer, 0u, bytes, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    return true;
}

bool PhysicsPassDispatcher::buildBroadPhase(Diligent::IDeviceContext* computeContext,
                                            const PhysicsSceneGpuState& sceneState,
                                            std::uint32_t activeMovingCount,
                                            const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto& transient = sceneState.transientBuffers();
    const std::array buildElementsBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transient.activeBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mBuildBroadPhaseElementsPass.srb, buildElementsBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mBuildBroadPhaseElementsPass.pso);
    computeContext->CommitShaderResources(mBuildBroadPhaseElementsPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeMovingCount), 1u, 1u});

    if (!dispatchReduceBroadPhaseExtentPass(computeContext, sceneState, activeMovingCount, false))
    {
        return false;
    }

    const std::array mortonBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_GlobalExtent", transient.globalBroadPhaseExtentBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_MortonCodes", transient.mortonCodesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mMortonCodesPass.srb, mortonBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mMortonCodesPass.pso);
    computeContext->CommitShaderResources(mMortonCodesPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeMovingCount), 1u, 1u});

    if (!dispatchRadixSortPass(computeContext, sceneState, activeMovingCount, false, constants))
    {
        return false;
    }

    const std::array bvhHierarchyBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_SortedMortonCodes", transient.mortonCodesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transient.bvhBuffer, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BvhConstructionInfos", transient.bvhConstructionInfoBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mBvhHierarchyPass.srb, bvhHierarchyBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mBvhHierarchyPass.pso);
    computeContext->CommitShaderResources(mBvhHierarchyPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeMovingCount), 1u, 1u});

    const std::array bvhBoundsBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transient.bvhBuffer, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BvhConstructionInfos", transient.bvhConstructionInfoBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mBvhBoundingBoxesPass.srb, bvhBoundsBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mBvhBoundingBoxesPass.pso);
    computeContext->CommitShaderResources(mBvhBoundingBoxesPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeMovingCount), 1u, 1u});

    if (constants.staticBodyCount == 0u)
    {
        return true;
    }
    if (!sceneState.staticBroadPhaseDirty())
    {
        return true;
    }
    GpuRigidDispatchConstants staticConstants = constants;
    staticConstants.activeMovingCount         = constants.staticBodyCount;

    const std::array staticBuildElementsBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transient.staticBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mBuildStaticBroadPhaseElementsSrb, staticBuildElementsBindings) ||
        !writeDispatchConstants(computeContext, staticConstants))
    {
        return false;
    }
    computeContext->SetPipelineState(mBuildBroadPhaseElementsPass.pso);
    computeContext->CommitShaderResources(mBuildStaticBroadPhaseElementsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(constants.staticBodyCount), 1u, 1u});

    if (!dispatchReduceBroadPhaseExtentPass(computeContext, sceneState, constants.staticBodyCount,
                                            true))
    {
        return false;
    }

    const std::array staticMortonBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_GlobalExtent", transient.staticGlobalBroadPhaseExtentBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_MortonCodes", transient.staticMortonCodesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mStaticMortonCodesSrb, staticMortonBindings) ||
        !writeDispatchConstants(computeContext, staticConstants))
    {
        return false;
    }
    computeContext->SetPipelineState(mMortonCodesPass.pso);
    computeContext->CommitShaderResources(mStaticMortonCodesSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(constants.staticBodyCount), 1u, 1u});

    if (!dispatchRadixSortPass(computeContext, sceneState, constants.staticBodyCount, true,
                               staticConstants))
    {
        return false;
    }

    const std::array staticBvhHierarchyBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_SortedMortonCodes", transient.staticMortonCodesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transient.staticBvhBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BvhConstructionInfos", transient.staticBvhConstructionInfoBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mStaticBvhHierarchySrb, staticBvhHierarchyBindings) ||
        !writeDispatchConstants(computeContext, staticConstants))
    {
        return false;
    }
    computeContext->SetPipelineState(mBvhHierarchyPass.pso);
    computeContext->CommitShaderResources(mStaticBvhHierarchySrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(constants.staticBodyCount), 1u, 1u});

    const std::array staticBvhBoundsBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transient.staticBvhBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BvhConstructionInfos", transient.staticBvhConstructionInfoBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mStaticBvhBoundingBoxesSrb, staticBvhBoundsBindings) ||
        !writeDispatchConstants(computeContext, staticConstants))
    {
        return false;
    }
    computeContext->SetPipelineState(mBvhBoundingBoxesPass.pso);
    computeContext->CommitShaderResources(mStaticBvhBoundingBoxesSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(constants.staticBodyCount), 1u, 1u});
    return true;
}

bool PhysicsPassDispatcher::finalizeBroadPhasePairs(Diligent::IDeviceContext* computeContext,
                                                    const PhysicsSceneGpuState& sceneState,
                                                    std::uint32_t activeMovingCount,
                                                    const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();
    const std::array countBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transient.activeBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transient.bvhBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderShapeTypes", persistent.colliderShapeTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCountsSphereSphere", transient.pairCountBuffers[0],
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PairCountsSphereBox", transient.pairCountBuffers[1],
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PairCountsSphereCapsule", transient.pairCountBuffers[2],
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PairCountsBoxBox", transient.pairCountBuffers[3],
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PairCountsBoxCapsule", transient.pairCountBuffers[4],
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PairCountsCapsuleCapsule", transient.pairCountBuffers[5],
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mCountPairsMovingSrb, countBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mCountPairsPass.pso);
    computeContext->CommitShaderResources(mCountPairsMovingSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeMovingCount), 1u, 1u});

    for (std::uint32_t pairType = 0u; pairType < kRigidPairTypeCount; ++pairType)
    {
        if (!dispatchExclusiveScanPass(
                computeContext, sceneState, transient.pairCountBuffers[pairType],
                transient.pairOffsetBuffers[pairType], activeMovingCount, constants))
        {
            return false;
        }
    }

    const std::array finalizeBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCountsSphereSphere", transient.pairCountBuffers[0],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCountsSphereBox", transient.pairCountBuffers[1],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCountsSphereCapsule", transient.pairCountBuffers[2],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCountsBoxBox", transient.pairCountBuffers[3],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCountsBoxCapsule", transient.pairCountBuffers[4],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCountsCapsuleCapsule", transient.pairCountBuffers[5],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsSphereSphere", transient.pairOffsetBuffers[0],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsSphereBox", transient.pairOffsetBuffers[1],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsSphereCapsule", transient.pairOffsetBuffers[2],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsBoxBox", transient.pairOffsetBuffers[3],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsBoxCapsule", transient.pairOffsetBuffers[4],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsCapsuleCapsule", transient.pairOffsetBuffers[5],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mFinalizePairsPass.srb, finalizeBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mFinalizePairsPass.pso);
    computeContext->CommitShaderResources(mFinalizePairsPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{1u, 1u, 1u});
    return true;
}

bool PhysicsPassDispatcher::emitBroadPhasePairs(Diligent::IDeviceContext* computeContext,
                                                const PhysicsSceneGpuState& sceneState,
                                                std::uint32_t activeMovingCount,
                                                const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();
    const std::array emitBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transient.activeBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transient.bvhBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderShapeTypes", persistent.colliderShapeTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsSphereSphere", transient.pairOffsetBuffers[0],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsSphereBox", transient.pairOffsetBuffers[1],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsSphereCapsule", transient.pairOffsetBuffers[2],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsBoxBox", transient.pairOffsetBuffers[3],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsBoxCapsule", transient.pairOffsetBuffers[4],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsetsCapsuleCapsule", transient.pairOffsetBuffers[5],
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_CandidatePairs", transient.candidatePairsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mEmitPairsMovingSrb, emitBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mEmitPairsPass.pso);
    computeContext->CommitShaderResources(mEmitPairsMovingSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeMovingCount), 1u, 1u});

    const std::array chunkBindings{
        // BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
        //   Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_NarrowPhaseChunks", transient.narrowPhaseChunksBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_NarrowPhaseChunkCounter", transient.narrowPhaseChunkCounterBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mBuildNarrowPhaseChunksPass.srb, chunkBindings) ||
        !writeDispatchConstants(computeContext, constants))
    {
        return false;
    }
    computeContext->SetPipelineState(mBuildNarrowPhaseChunksPass.pso);
    computeContext->CommitShaderResources(mBuildNarrowPhaseChunksPass.srb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{1u, 1u, 1u});
    return true;
}

bool PhysicsPassDispatcher::dispatchGenerateContactsPass(Diligent::IDeviceContext* computeContext,
                                                         const PhysicsSceneGpuState& sceneState,
                                                         std::uint32_t pairCount)
{
    if (pairCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();
    const std::array bindings{
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transient.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transient.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyScales", persistent.scalesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderParams", persistent.colliderParamsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_CandidatePairs", transient.candidatePairsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_NarrowPhaseChunks", transient.narrowPhaseChunksBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_NarrowPhaseChunkCounter", transient.narrowPhaseChunkCounterBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_RigidContacts", transient.contactsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::uint32_t dispatchGroupUpperBound =
        ((pairCount + kNarrowPhaseChunkSize - 1u) / kNarrowPhaseChunkSize) +
        (kRigidPairTypeCount - 1u);

    return dispatchComputePass(computeContext, mGenerateContactsPass, bindings, nullptr,
                               dispatchGroupUpperBound);
}

bool PhysicsPassDispatcher::generateContacts(Diligent::IDeviceContext* computeContext,
                                             const PhysicsSceneGpuState& sceneState,
                                             std::uint32_t pairCount,
                                             const GpuRigidDispatchConstants& constants)
{
    return writeDispatchConstants(computeContext, constants) &&
           dispatchGenerateContactsPass(computeContext, sceneState, pairCount);
}

bool PhysicsPassDispatcher::dispatchSolveGatherPass(Diligent::IDeviceContext* computeContext,
                                                    const PhysicsSceneGpuState& sceneState,
                                                    std::uint32_t pairCount)
{
    if (pairCount == 0u)
    {
        return false;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transient.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transient.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyInverseInertiaLocal", persistent.inverseInertiaLocalBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidContacts", transient.contactsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyTranslationCorrections", transient.translationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::uint32_t contactSlotCount = pairCount * kRigidContactsPerPair;
    return dispatchComputePass(computeContext, mSolveGatherPass, bindings, nullptr,
                               dispatchGroupCount(contactSlotCount));
}

bool PhysicsPassDispatcher::solveConstraints(Diligent::IDeviceContext* computeContext,
                                             const PhysicsSceneGpuState& sceneState,
                                             std::uint32_t rigidBodyCount, std::uint32_t pairCount,
                                             std::uint32_t iterations,
                                             const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (pairCount == 0u || iterations == 0u)
    {
        return true;
    }

    const auto& transient = sceneState.transientBuffers();
    const std::array applyBindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transient.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transient.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_RigidBodyTypes", sceneState.persistentRigidBodies().bodyTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyTranslationCorrections", transient.translationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!bindBufferVariables(mApplyCorrectionsPass.srb, applyBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        GpuRigidDispatchConstants iterationConstants = constants;
        iterationConstants.iterationIndex            = iteration;
        if (!writeDispatchConstants(computeContext, iterationConstants) ||
            !dispatchSolveGatherPass(computeContext, sceneState, pairCount))
        {
            return false;
        }

        computeContext->SetPipelineState(mApplyCorrectionsPass.pso);
        computeContext->CommitShaderResources(mApplyCorrectionsPass.srb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(
            Diligent::DispatchComputeAttribs{dispatchGroupCount(rigidBodyCount), 1u, 1u});
    }
    return true;
}

bool PhysicsPassDispatcher::updateVelocities(Diligent::IDeviceContext* computeContext,
                                             const PhysicsSceneGpuState& sceneState,
                                             std::uint32_t bodyCount,
                                             const GpuRigidDispatchConstants& constants)
{
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto& transient  = sceneState.transientBuffers();
    const auto& persistent = sceneState.persistentRigidBodies();
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", mDispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                      transient.previousRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PreviousRigidBodyOrientations",
                      transient.previousRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transient.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transient.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyLinearVelocities",
                      transient.predictedRigidBodies.linearVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyAngularVelocities",
                      transient.predictedRigidBodies.angularVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return dispatchComputePass(computeContext, mUpdateVelocitiesPass, bindings, &constants,
                               dispatchGroupCount(bodyCount));
}

} // namespace cressim::neo::physics
