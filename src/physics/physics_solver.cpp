#include "physics/physics_solver.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/ShaderResourceVariable.h"
#include "gpu/shader_library.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"
#include "DiligentEngine/DiligentCore/Primitives/interface/Errors.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <limits>

namespace cressim::neo::physics
{

namespace
{

struct PhysicsDispatchConstants
{
    float dt                     = 0.0f;
    std::uint32_t rigidBodyCount = 0;
    std::uint32_t particleCount  = 0;
    std::uint32_t substepIndex   = 0;
    std::uint32_t iterationIndex = 0;
    std::uint32_t reserved0      = 0;
    std::uint32_t reserved1      = 0;
    std::uint32_t reserved2      = 0;
};

struct GpuPair
{
    std::uint32_t a = 0;
    std::uint32_t b = 0;
};

constexpr std::uint32_t kComputeThreadGroupSize = 64;

constexpr std::size_t stageIndex(PhysicsSolverStage stage)
{
    return static_cast<std::size_t>(stage);
}

void markStage(PhysicsSolverStageStats& stats, PhysicsSolverStage stage, bool executed)
{
    stats.executed[stageIndex(stage)] = executed;
    if (executed)
    {
        ++stats.dispatchedStages;
    }
    else
    {
        ++stats.skippedStages;
    }
}

Diligent::Uint64 contextMaskForId(std::uint32_t contextId)
{
    return static_cast<Diligent::Uint64>(1ull) << contextId;
}

} // namespace

struct PhysicsSolver::Impl
{
    struct PersistentRigidBodyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderShapeTypesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderParamsBuffer;
    };

    struct PredictedRigidBodyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
    };

    struct SolverTransientBuffers
    {
        PredictedRigidBodyBuffers predictedRigidBodies;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> spatialKeysBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> sortedIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> candidatePairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> contactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> constraintScratchBuffer;
    };

    struct RigidBodyReadbackBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
    };

    gpu::ShaderLibrary shaderLibrary{""};
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> predictPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> predictSrb;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> dispatchConstantsBuffer;

    PersistentRigidBodyBuffers persistentRigidBodies;
    SolverTransientBuffers transientState;
    RigidBodyReadbackBuffers readbackRigidBodies;

    std::uint32_t bufferCapacity  = 0;
    std::uint32_t scratchCapacity = 0;
    PhysicsSolverStageStats stageStats{};

    bool bindPredictBuffers();
    bool ensureCapacity(Diligent::IRenderDevice* renderDevice, std::uint32_t bodyCount,
                        std::uint32_t physicsContextId);
    bool uploadPersistentRigidBodyState(Diligent::IDeviceContext* computeContext, PhysicsWorld& world,
                                        std::uint32_t bodyCount);
    bool copyPersistentRigidBodiesToPredictedState(Diligent::IDeviceContext* computeContext,
                                                   std::uint32_t bodyCount);
    bool dispatchPredictPass(Diligent::IDeviceContext* computeContext, std::uint32_t bodyCount);
    bool readbackPredictedRigidStateBlocking(Diligent::IDeviceContext* computeContext,
                                             PhysicsWorld& world, std::uint32_t bodyCount);
};

PhysicsSolver::PhysicsSolver(gpu::GpuDevice& device, const PhysicsSolverDesc& desc)
    : mDevice(device), mDesc(desc), mImpl(std::make_unique<Impl>())
{
}

PhysicsSolver::~PhysicsSolver() = default;

namespace
{

bool ensureStructuredBuffer(Diligent::IRenderDevice* renderDevice,
                            const char* name,
                            std::uint32_t elementStride,
                            std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags,
                            Diligent::USAGE usage,
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

bool writeDispatchConstants(Diligent::IDeviceContext* computeContext,
                            Diligent::IBuffer* constantsBuffer,
                            const PhysicsDispatchConstants& constants)
{
    if (computeContext == nullptr || constantsBuffer == nullptr)
    {
        return false;
    }

    void* mapped = nullptr;
    computeContext->MapBuffer(constantsBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                              mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    computeContext->UnmapBuffer(constantsBuffer, Diligent::MAP_WRITE);
    return true;
}

} // namespace

bool PhysicsSolver::Impl::bindPredictBuffers()
{
    auto bindView = [&](const char* varName, Diligent::IBuffer* buffer,
                        Diligent::BUFFER_VIEW_TYPE viewType)
    {
        Diligent::IShaderResourceVariable* variable =
            predictSrb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE, varName);
        if (variable == nullptr || buffer == nullptr)
        {
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
    };

    return bindView("g_PredictedRigidBodyPositionsInvMass",
                    transientState.predictedRigidBodies.positionsBuffer,
                    Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindView("g_PredictedRigidBodyOrientations",
                    transientState.predictedRigidBodies.orientationsBuffer,
                    Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindView("g_PredictedRigidBodyLinearVelocities",
                    transientState.predictedRigidBodies.linearVelocitiesBuffer,
                    Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindView("g_PredictedRigidBodyAngularVelocities",
                    transientState.predictedRigidBodies.angularVelocitiesBuffer,
                    Diligent::BUFFER_VIEW_UNORDERED_ACCESS);
}

bool PhysicsSolver::Impl::ensureCapacity(Diligent::IRenderDevice* renderDevice,
                                         std::uint32_t bodyCount,
                                         std::uint32_t physicsContextId)
{
    const bool hasAllCoreBuffers = persistentRigidBodies.positionsBuffer != nullptr &&
                                   persistentRigidBodies.orientationsBuffer != nullptr &&
                                   persistentRigidBodies.linearVelocitiesBuffer != nullptr &&
                                   persistentRigidBodies.angularVelocitiesBuffer != nullptr &&
                                   persistentRigidBodies.colliderShapeTypesBuffer != nullptr &&
                                   persistentRigidBodies.colliderParamsBuffer != nullptr &&
                                   transientState.predictedRigidBodies.positionsBuffer != nullptr &&
                                   transientState.predictedRigidBodies.orientationsBuffer != nullptr &&
                                   transientState.predictedRigidBodies.linearVelocitiesBuffer !=
                                       nullptr &&
                                   transientState.predictedRigidBodies.angularVelocitiesBuffer !=
                                       nullptr &&
                                   readbackRigidBodies.positionsBuffer != nullptr &&
                                   readbackRigidBodies.orientationsBuffer != nullptr &&
                                   readbackRigidBodies.linearVelocitiesBuffer != nullptr &&
                                   readbackRigidBodies.angularVelocitiesBuffer != nullptr;
    if (hasAllCoreBuffers && bufferCapacity >= bodyCount)
    {
        return true;
    }

    // TODO: when capacity is full, allocate 1.5x needed to reduce re-allocation
    const std::uint32_t newCapacity = std::max<std::uint32_t>(bodyCount, 64u);
    const Diligent::Uint64 contextMask = contextMaskForId(physicsContextId);
    if (!ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PositionsInvMass",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Orientations",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.LinearVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.AngularVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderShapeTypes",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.colliderShapeTypesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderParams",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.colliderParamsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedPositionsInvMass",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.predictedRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedOrientations",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.predictedRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedLinearVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.predictedRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedAngularVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.predictedRigidBodies.angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedPositionsInvMass.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedOrientations.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedLinearVelocities.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedAngularVelocities.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.angularVelocitiesBuffer))
    {
        return false;
    }

    bufferCapacity = newCapacity;

    const std::uint32_t newScratchCapacity = std::max<std::uint32_t>(newCapacity * 8u, 64u);
    if (!ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SpatialKeys",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.spatialKeysBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SortedIndices",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.sortedIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CandidatePairs", sizeof(GpuPair),
                                newScratchCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.candidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Contacts",
                                sizeof(Diligent::float4), newScratchCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.contactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ConstraintScratch",
                                sizeof(Diligent::float4), newScratchCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.constraintScratchBuffer))
    {
        return false;
    }
    scratchCapacity = newScratchCapacity;

    return bindPredictBuffers();
}

bool PhysicsSolver::Impl::uploadPersistentRigidBodyState(Diligent::IDeviceContext* computeContext,
                                                         PhysicsWorld& world,
                                                         std::uint32_t bodyCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const RigidBodySoAHost& rigidBodies = world.rigidBodySoA();
    if (static_cast<std::uint32_t>(rigidBodies.size()) != bodyCount)
    {
        return false;
    }

    if (bodyCount == 0u)
    {
        return true;
    }

    const Diligent::Uint64 float4Bytes = static_cast<Diligent::Uint64>(bodyCount) *
                                         sizeof(Diligent::float4);
    const Diligent::Uint64 shapeTypeBytes = static_cast<Diligent::Uint64>(bodyCount) *
                                            sizeof(std::uint32_t);

    // TODO(PBD-GPU): use world.rigidBodyDirtyRange() to upload only changed slices.
    computeContext->UpdateBuffer(persistentRigidBodies.positionsBuffer, 0u, float4Bytes,
                                 rigidBodies.positionsInvMass.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.orientationsBuffer, 0u, float4Bytes,
                                 rigidBodies.orientations.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.linearVelocitiesBuffer, 0u, float4Bytes,
                                 rigidBodies.linearVelocities.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.angularVelocitiesBuffer, 0u, float4Bytes,
                                 rigidBodies.angularVelocities.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.colliderShapeTypesBuffer, 0u, shapeTypeBytes,
                                 rigidBodies.colliderShapeTypes.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.colliderParamsBuffer, 0u, float4Bytes,
                                 rigidBodies.colliderParams.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    world.clearRigidBodyDirtyRange();
    return true;
}

bool PhysicsSolver::Impl::copyPersistentRigidBodiesToPredictedState(
    Diligent::IDeviceContext* computeContext, std::uint32_t bodyCount)
{
    if (computeContext == nullptr || bodyCount == 0u)
    {
        return false;
    }

    const Diligent::Uint64 bytes = static_cast<Diligent::Uint64>(bodyCount) * sizeof(Diligent::float4);
    computeContext->CopyBuffer(persistentRigidBodies.positionsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               transientState.predictedRigidBodies.positionsBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(persistentRigidBodies.orientationsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               transientState.predictedRigidBodies.orientationsBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(persistentRigidBodies.linearVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               transientState.predictedRigidBodies.linearVelocitiesBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(persistentRigidBodies.angularVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               transientState.predictedRigidBodies.angularVelocitiesBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
}

bool PhysicsSolver::Impl::dispatchPredictPass(Diligent::IDeviceContext* computeContext,
                                              std::uint32_t bodyCount)
{
    if (computeContext == nullptr || predictPso == nullptr || predictSrb == nullptr ||
        bodyCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(predictPso);
    computeContext->CommitShaderResources(predictSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    const std::uint32_t groupCountX = (bodyCount + kComputeThreadGroupSize - 1u) /
                                      kComputeThreadGroupSize;
    computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{groupCountX, 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::readbackPredictedRigidStateBlocking(Diligent::IDeviceContext* computeContext,
                                                              PhysicsWorld& world,
                                                              std::uint32_t bodyCount)
{
    if (computeContext == nullptr || bodyCount == 0u)
    {
        return false;
    }

    const Diligent::Uint64 bytes = static_cast<Diligent::Uint64>(bodyCount) * sizeof(Diligent::float4);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.positionsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.positionsBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.orientationsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.orientationsBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.linearVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.linearVelocitiesBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.angularVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.angularVelocitiesBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    computeContext->Flush();
    computeContext->WaitForIdle();

    void* mappedPositions    = nullptr;
    void* mappedOrientations = nullptr;
    void* mappedLinear       = nullptr;
    void* mappedAngular      = nullptr;

    computeContext->MapBuffer(readbackRigidBodies.positionsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_NONE, mappedPositions);
    computeContext->MapBuffer(readbackRigidBodies.orientationsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_NONE, mappedOrientations);
    computeContext->MapBuffer(readbackRigidBodies.linearVelocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_NONE, mappedLinear);
    computeContext->MapBuffer(readbackRigidBodies.angularVelocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_NONE, mappedAngular);
    if (mappedPositions == nullptr || mappedOrientations == nullptr || mappedLinear == nullptr ||
        mappedAngular == nullptr)
    {
        if (mappedPositions != nullptr)
        {
            computeContext->UnmapBuffer(readbackRigidBodies.positionsBuffer, Diligent::MAP_READ);
        }
        if (mappedOrientations != nullptr)
        {
            computeContext->UnmapBuffer(readbackRigidBodies.orientationsBuffer,
                                        Diligent::MAP_READ);
        }
        if (mappedLinear != nullptr)
        {
            computeContext->UnmapBuffer(readbackRigidBodies.linearVelocitiesBuffer,
                                        Diligent::MAP_READ);
        }
        if (mappedAngular != nullptr)
        {
            computeContext->UnmapBuffer(readbackRigidBodies.angularVelocitiesBuffer,
                                        Diligent::MAP_READ);
        }
        return false;
    }

    const auto* positions         = static_cast<const Diligent::float4*>(mappedPositions);
    const auto* orientations      = static_cast<const Diligent::float4*>(mappedOrientations);
    const auto* linearVelocities  = static_cast<const Diligent::float4*>(mappedLinear);
    const auto* angularVelocities = static_cast<const Diligent::float4*>(mappedAngular);
    for (std::uint32_t i = 0; i < bodyCount; ++i)
    {
        (void)world.writeBackRigidBodyState(i, positions[i], orientations[i], linearVelocities[i],
                                            angularVelocities[i]);
    }
    world.finalizeRigidBodyWriteback();

    computeContext->UnmapBuffer(readbackRigidBodies.positionsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(readbackRigidBodies.orientationsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(readbackRigidBodies.linearVelocitiesBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(readbackRigidBodies.angularVelocitiesBuffer, Diligent::MAP_READ);
    return true;
}

bool PhysicsSolver::initialize()
{
    shutdown();

    if (!mDesc.enableGpuCompute)
    {
        mInitialized = true;
        return true;
    }

    gpu::GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to get physics GPU context.");
        return false;
    }

    mImpl                = std::make_unique<Impl>();
    mImpl->shaderLibrary = gpu::ShaderLibrary(mDevice.shaderSourceDirectory());

    std::string shaderPath;
    constexpr const char* kShaderRelativePath = "physics/physics_placeholder_integrate.cs.hlsl";
    if (!mImpl->shaderLibrary.resolveShaderPath(kShaderRelativePath, shaderPath))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to resolve compute shader path.");
        return false;
    }

    Diligent::IShaderSourceInputStreamFactory* streamFactory = mImpl->shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: shader stream factory is null.");
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Desc.ShaderType                 = Diligent::SHADER_TYPE_COMPUTE;
    shaderCreateInfo.Desc.Name                       = "CRESSimNeo.Physics.PlaceholderPredict.CS";
    shaderCreateInfo.FilePath                        = kShaderRelativePath;
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::RefCntAutoPtr<Diligent::IShader> computeShader;
    computeContext.renderDevice->CreateShader(shaderCreateInfo, &computeShader);
    if (computeShader == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to compile compute shader.");
        return false;
    }

    Diligent::ComputePipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name         = "CRESSimNeo.Physics.PlaceholderPredict.PSO";
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_COMPUTE;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    Diligent::ShaderResourceVariableDesc vars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
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
    psoCreateInfo.PSODesc.ResourceLayout.Variables    = vars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(std::size(vars));
    psoCreateInfo.pCS = computeShader;

    computeContext.renderDevice->CreateComputePipelineState(psoCreateInfo, &mImpl->predictPso);
    if (mImpl->predictPso == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create compute PSO.");
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Name                 = "CRESSimNeo.Physics.DispatchConstants";
    constantsDesc.Size                 = sizeof(PhysicsDispatchConstants);
    constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    constantsDesc.ImmediateContextMask = contextMaskForId(computeContext.contextId);
    computeContext.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                              &mImpl->dispatchConstantsBuffer);
    if (mImpl->dispatchConstantsBuffer == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create dispatch constants buffer.");
        return false;
    }

    mImpl->predictPso->CreateShaderResourceBinding(&mImpl->predictSrb, true);
    if (mImpl->predictSrb == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create SRB.");
        return false;
    }

    Diligent::IShaderResourceVariable* constantsVar =
        mImpl->predictSrb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE,
                                             "PhysicsDispatchConstantsBuffer");
    if (constantsVar == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: variable PhysicsDispatchConstantsBuffer not found.");
        return false;
    }
    constantsVar->Set(mImpl->dispatchConstantsBuffer);

    mInitialized = true;
    return true;
}

void PhysicsSolver::shutdown()
{
    mImpl        = std::make_unique<Impl>();
    mInitialized = false;
}

bool PhysicsSolver::step(const common::FrameContext& frameContext, PhysicsWorld& world)
{
    if (!mInitialized)
    {
        return false;
    }

    mImpl->stageStats = PhysicsSolverStageStats{};

    if (!mDesc.enableGpuCompute)
    {
        world.integrateRigidBodiesCpu(frameContext.deltaSeconds);
        markStage(mImpl->stageStats, PhysicsSolverStage::PredictState, true);
        markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, true);
        return true;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr)
    {
        return false;
    }

    const std::uint32_t rigidBodyCount = world.rigidBodyCount();
    if (rigidBodyCount == 0u)
    {
        return true;
    }

    if (!mImpl->ensureCapacity(computeBackend.renderDevice, rigidBodyCount,
                               computeBackend.contextId))
    {
        return false;
    }
    if (!mImpl->uploadPersistentRigidBodyState(computeBackend.computeContext, world,
                                               rigidBodyCount))
    {
        return false;
    }
    if (!mImpl->copyPersistentRigidBodiesToPredictedState(computeBackend.computeContext,
                                                          rigidBodyCount))
    {
        return false;
    }
    if (!mImpl->bindPredictBuffers())
    {
        return false;
    }

    const std::uint32_t substeps   = std::max<std::uint32_t>(mDesc.substeps, 1u);
    const std::uint32_t iterations = std::max<std::uint32_t>(mDesc.solverIterations, 1u);
    const float substepDt          = frameContext.deltaSeconds / static_cast<float>(substeps);

    for (std::uint32_t substep = 0; substep < substeps; ++substep)
    {
        const PhysicsDispatchConstants constants{
            substepDt,
            rigidBodyCount,
            0u,
            substep,
            0u,
        };
        if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                    constants) ||
            !mImpl->dispatchPredictPass(computeBackend.computeContext, rigidBodyCount))
        {
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::PredictState, true);

        // TODO(PBD-GPU): implement spatial key generation dispatch and key layout.
        // TODO(PBD-GPU): implement spatial sort/bucket stage (radix sort or uniform grid bucket build).
        // TODO(PBD-GPU): implement contact generation stage and contact pair schema.
        // TODO(PBD-GPU): implement constraint construction stage.
        if (mDesc.enableCollisionPipelineScaffold)
        {
            markStage(mImpl->stageStats, PhysicsSolverStage::BuildSpatialIndices, false);
            markStage(mImpl->stageStats, PhysicsSolverStage::SortSpatialIndices, false);
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts, false);
            markStage(mImpl->stageStats, PhysicsSolverStage::BuildConstraintData, false);
        }

        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            (void)iteration;
            // TODO(PBD-GPU): implement iterative constraint solve kernels and lambda buffers.
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::SolveConstraints, false);

        // TODO(PBD-GPU): implement velocity update pass from predicted state deltas.
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateVelocities, false);
    }

    if (!mDesc.enableBlockingReadback)
    {
        // TODO(PBD-GPU): replace blocking readback with async readback ring and fence polling.
        // TODO(PBD-GPU): add direct physics->render interop path to bypass CPU readback.
        markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, false);
        return true;
    }

    if (!mImpl->readbackPredictedRigidStateBlocking(computeBackend.computeContext, world,
                                                    rigidBodyCount))
    {
        return false;
    }
    markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, true);
    return true;
}

const PhysicsSolverStageStats& PhysicsSolver::lastStageStats() const noexcept
{
    return mImpl->stageStats;
}

} // namespace cressim::neo::physics
