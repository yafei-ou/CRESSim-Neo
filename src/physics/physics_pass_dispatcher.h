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

    template <std::size_t N>
    bool bindBufferVariables(Diligent::IShaderResourceBinding* srb,
                             const std::array<BufferBinding, N>& bindings);

    bool writeDispatchConstants(Diligent::IDeviceContext* computeContext,
                                const GpuRigidDispatchConstants& constants);
    bool bindBufferVariable(Diligent::IShaderResourceBinding* srb, const char* variableName,
                            Diligent::IBuffer* buffer, Diligent::BUFFER_VIEW_TYPE viewType);
    bool createComputePipeline(Diligent::IRenderDevice* renderDevice,
                               Diligent::IShaderSourceInputStreamFactory* streamFactory,
                               const char* shaderPath, const char* shaderName, const char* psoName,
                               const Diligent::ShaderResourceVariableDesc* variables,
                               std::size_t variableCount,
                               Diligent::RefCntAutoPtr<Diligent::IPipelineState>& outPso,
                               Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>& outSrb);
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

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mPredictPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mPredictSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mUpdateWorldAabbsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mUpdateWorldAabbsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mScanBlockPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mScanBlockSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mScanAddOffsetsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mScanAddOffsetsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mCompactActiveBodiesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mCompactActiveBodiesSrb;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mCompactStaticBodiesSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mFinalizeActiveBodiesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mFinalizeActiveBodiesSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mBuildBroadPhaseElementsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mBuildBroadPhaseElementsSrb;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mBuildStaticBroadPhaseElementsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mReduceExtentElementsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mReduceExtentElementsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mReduceExtentExtentsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mReduceExtentExtentsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mMortonCodesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mMortonCodesSrb;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mStaticMortonCodesSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mRadixClassifyPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mRadixClassifySrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mRadixFinalizePso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mRadixFinalizeSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mRadixScatterPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mRadixScatterSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mBvhHierarchyPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mBvhHierarchySrb;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mStaticBvhHierarchySrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mBvhBoundingBoxesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mBvhBoundingBoxesSrb;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mStaticBvhBoundingBoxesSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mCountPairsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mCountPairsSrb;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mCountPairsMovingSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mFinalizePairsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mFinalizePairsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mEmitPairsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mEmitPairsSrb;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mEmitPairsMovingSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mBuildNarrowPhaseChunksPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mBuildNarrowPhaseChunksSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mGenerateContactsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mGenerateContactsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mClearCorrectionsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mClearCorrectionsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mSolveGatherPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mSolveGatherSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mApplyCorrectionsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mApplyCorrectionsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mUpdateVelocitiesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mUpdateVelocitiesSrb;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mDispatchConstantsBuffer;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
