#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H

#include "physics/physics_scene_gpu_state.h"
#include "physics/rigid_body_common.h"

#include "gpu/shader_library.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace cressim::neo::physics
{

class PhysicsPassDispatcher
{
public:
    bool initialize(Diligent::IRenderDevice* renderDevice, std::uint32_t physicsContextId,
                    const char* shaderSourceDirectory);

    bool clearCorrections(Diligent::IDeviceContext* computeContext,
                          PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants& constants);
    bool predict(Diligent::IDeviceContext* computeContext, const PhysicsSceneGpuState& sceneState,
                 std::uint32_t bodyCount, const GpuRigidDispatchConstants& constants);
    bool updateWorldAabbs(Diligent::IDeviceContext* computeContext,
                          const PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants& constants);
    bool compactActiveBodies(Diligent::IDeviceContext* computeContext,
                             const PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                             const GpuRigidDispatchConstants& constants);
    bool buildBroadPhase(Diligent::IDeviceContext* computeContext,
                         const PhysicsSceneGpuState& sceneState, std::uint32_t activeMovingCount,
                         const GpuRigidDispatchConstants& constants);
    bool finalizeBroadPhasePairs(Diligent::IDeviceContext* computeContext,
                                 const PhysicsSceneGpuState& sceneState,
                                 std::uint32_t activeMovingCount,
                                 const GpuRigidDispatchConstants& constants);
    bool emitBroadPhasePairs(Diligent::IDeviceContext* computeContext,
                             const PhysicsSceneGpuState& sceneState,
                             std::uint32_t activeMovingCount,
                             const GpuRigidDispatchConstants& constants);
    bool generateContacts(Diligent::IDeviceContext* computeContext,
                          const PhysicsSceneGpuState& sceneState, std::uint32_t pairCount,
                          const GpuRigidDispatchConstants& constants);
    bool solveConstraints(Diligent::IDeviceContext* computeContext,
                          const PhysicsSceneGpuState& sceneState, std::uint32_t rigidBodyCount,
                          std::uint32_t pairCount, std::uint32_t iterations,
                          const GpuRigidDispatchConstants& constants);
    bool updateVelocities(Diligent::IDeviceContext* computeContext,
                          const PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants& constants);

private:
    struct BufferBinding
    {
        const char* variableName;
        Diligent::IBuffer* buffer;
        Diligent::BUFFER_VIEW_TYPE viewType;
    };

    struct ComputePass
    {
        Diligent::RefCntAutoPtr<Diligent::IPipelineState> pso;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    };

    template <std::size_t N>
    bool bindBufferVariables(Diligent::IShaderResourceBinding* srb,
                             const std::array<BufferBinding, N>& bindings);

    template <std::size_t N>
    bool bindBufferVariables(const ComputePass& pass, const std::array<BufferBinding, N>& bindings);

    bool writeDispatchConstants(Diligent::IDeviceContext* computeContext,
                                const GpuRigidDispatchConstants& constants);
    bool bindBufferVariable(Diligent::IShaderResourceBinding* srb, const char* variableName,
                            Diligent::IBuffer* buffer, Diligent::BUFFER_VIEW_TYPE viewType);

    bool createComputePass(Diligent::IRenderDevice* renderDevice,
                           Diligent::IShaderSourceInputStreamFactory* streamFactory,
                           const char* shaderPath, const char* shaderName, const char* psoName,
                           const Diligent::ShaderResourceVariableDesc* variables,
                           std::size_t variableCount, ComputePass& outPass);

    bool createShaderResourceBinding(
        Diligent::IPipelineState* pso,
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>& outSrb);

    template <std::size_t N>
    bool dispatchComputePass(Diligent::IDeviceContext* computeContext, const ComputePass& pass,
                             const std::array<BufferBinding, N>& bindings,
                             const GpuRigidDispatchConstants* constants, std::uint32_t groupCountX,
                             std::uint32_t groupCountY = 1u, std::uint32_t groupCountZ = 1u);

    bool dispatchScanBlockPass(Diligent::IDeviceContext* computeContext,
                               const PhysicsSceneGpuState& sceneState, Diligent::IBuffer* input,
                               Diligent::IBuffer* output, Diligent::IBuffer* blockSums,
                               std::uint32_t count, const GpuRigidDispatchConstants& constants);
    bool dispatchScanAddOffsetsPass(Diligent::IDeviceContext* computeContext,
                                    Diligent::IBuffer* output,
                                    Diligent::IBuffer* scannedBlockOffsets, std::uint32_t count,
                                    const GpuRigidDispatchConstants& constants);
    bool dispatchExclusiveScanPass(Diligent::IDeviceContext* computeContext,
                                   const PhysicsSceneGpuState& sceneState, Diligent::IBuffer* input,
                                   Diligent::IBuffer* output, std::uint32_t count,
                                   const GpuRigidDispatchConstants& constants,
                                   std::uint32_t recursionLevel = 0u);
    bool dispatchReduceBroadPhaseExtentPass(Diligent::IDeviceContext* computeContext,
                                            const PhysicsSceneGpuState& sceneState,
                                            std::uint32_t activeMovingCount, bool useStaticSet);
    bool dispatchRadixSortPass(Diligent::IDeviceContext* computeContext,
                               const PhysicsSceneGpuState& sceneState,
                               std::uint32_t activeMovingCount, bool useStaticSet,
                               const GpuRigidDispatchConstants& constants);
    bool dispatchGenerateContactsPass(Diligent::IDeviceContext* computeContext,
                                      const PhysicsSceneGpuState& sceneState,
                                      std::uint32_t pairCount);
    bool dispatchSolveGatherPass(Diligent::IDeviceContext* computeContext,
                                 const PhysicsSceneGpuState& sceneState, std::uint32_t pairCount);

    gpu::ShaderLibrary mShaderLibrary{""};
    Diligent::Uint64 mPhysicsContextMask = 0;

    ComputePass mPredictPass;
    ComputePass mUpdateWorldAabbsPass;
    ComputePass mScanBlockPass;
    ComputePass mScanAddOffsetsPass;

    ComputePass mCompactActiveBodiesPass;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mCompactStaticBodiesSrb;

    ComputePass mFinalizeActiveBodiesPass;

    ComputePass mBuildBroadPhaseElementsPass;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mBuildStaticBroadPhaseElementsSrb;

    ComputePass mReduceExtentElementsPass;
    ComputePass mReduceExtentExtentsPass;

    ComputePass mMortonCodesPass;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mStaticMortonCodesSrb;

    ComputePass mRadixClassifyPass;
    ComputePass mRadixFinalizePass;
    ComputePass mRadixScatterPass;

    ComputePass mBvhHierarchyPass;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mStaticBvhHierarchySrb;

    ComputePass mBvhBoundingBoxesPass;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mStaticBvhBoundingBoxesSrb;

    ComputePass mCountPairsPass;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mCountPairsMovingSrb;

    ComputePass mFinalizePairsPass;

    ComputePass mEmitPairsPass;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mEmitPairsMovingSrb;

    ComputePass mBuildNarrowPhaseChunksPass;
    ComputePass mGenerateContactsPass;
    ComputePass mClearCorrectionsPass;
    ComputePass mSolveGatherPass;
    ComputePass mApplyCorrectionsPass;
    ComputePass mUpdateVelocitiesPass;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> mDispatchConstantsBuffer;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
