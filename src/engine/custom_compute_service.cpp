#include "engine/custom_compute_service.h"

#include "common/logger.h"
#include "engine/shared_buffer_service.h"
#include "gpu/gpu_device.h"
#include "gpu/gpu_types.h"
#include "gpu/shader_library.h"
#include "physics/physics_gpu_scene_view.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace cressim::neo::engine
{

namespace
{

struct BufferBindingEntry
{
    std::string variableName;
    Diligent::IBuffer *buffer           = nullptr;
    Diligent::BUFFER_VIEW_TYPE viewType = Diligent::BUFFER_VIEW_UNDEFINED;
};

bool bindBufferVariable(Diligent::IShaderResourceBinding *srb, const BufferBindingEntry &binding)
{
    if (srb == nullptr || binding.buffer == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceVariable *variable =
        srb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE, binding.variableName.c_str());
    if (variable == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: shader variable not found: '",
                          binding.variableName, "'.");
        return false;
    }

    if (binding.viewType == Diligent::BUFFER_VIEW_UNDEFINED)
    {
        variable->Set(binding.buffer);
    }
    else
    {
        Diligent::IBufferView *view = binding.buffer->GetDefaultView(binding.viewType);
        if (view == nullptr)
        {
            CRESSIM_LOG_ERROR("CustomComputeService: buffer view is unavailable for variable '",
                              binding.variableName, "'.");
            return false;
        }
        variable->Set(view);
    }
    return true;
}

} // namespace

struct CustomComputeService::ResourceEntry
{
    CustomComputeResourceDesc desc{};
    Diligent::IBuffer *buffer = nullptr;
};

struct CustomComputeService::PassState
{
    CustomComputePassDesc desc{};
    gpu::GpuComputePass pass{};
    std::vector<std::string> variableNames;
    std::vector<Diligent::ShaderResourceVariableDesc> variables;
    std::vector<CustomComputeResourceBindingDesc> resourceBindings;
    std::unordered_map<std::string, std::uint64_t> expectedBindingGenerations;
    std::unordered_map<std::uint64_t, std::uint64_t> expectedSharedBufferGenerations;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> constantBuffer;
    std::uint32_t constantBufferSizeBytes = 0u;
};

CustomComputeService::CustomComputeService(gpu::GpuDevice &device) : mDevice(device) {}

CustomComputeService::~CustomComputeService() = default;

std::vector<CustomComputeResourceDesc> CustomComputeService::listResources(
    physics::PhysicsSolver &solver, physics::PhysicsWorld &world)
{
    std::vector<CustomComputeResourceDesc> resources;
    ResourceMap resourceMap;
    if (!buildResourceRegistry(solver, world, resourceMap, &resources))
    {
        return {};
    }
    return resources;
}

CustomComputePassHandle CustomComputeService::createPass(physics::PhysicsSolver &solver,
                                                         physics::PhysicsWorld &world,
                                                         const SharedBufferService *sharedBuffers,
                                                         const CustomComputePassDesc &desc)
{
    CustomComputePassHandle handle{};
    const bool hasShaderPath   = !desc.shaderPath.empty();
    const bool hasShaderSource = !desc.shaderSource.empty();
    if (hasShaderPath == hasShaderSource)
    {
        CRESSIM_LOG_ERROR(
            "CustomComputeService: createPass requires exactly one of shaderPath or shaderSource.");
        return handle;
    }
    if (desc.resourceBindings.empty())
    {
        CRESSIM_LOG_ERROR(
            "CustomComputeService: createPass requires at least one resource binding.");
        return handle;
    }
    if (desc.threadGroupSizeX == 0u || desc.threadGroupSizeY == 0u || desc.threadGroupSizeZ == 0u)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: threadgroup sizes must be non-zero.");
        return handle;
    }
    if (desc.dispatch.mode == CustomComputeDispatchMode::ExplicitGroupCount &&
        (desc.dispatch.groupCountX == 0u || desc.dispatch.groupCountY == 0u ||
         desc.dispatch.groupCountZ == 0u))
    {
        CRESSIM_LOG_ERROR("CustomComputeService: explicit dispatch group counts must be non-zero.");
        return handle;
    }
    if (desc.dispatch.mode == CustomComputeDispatchMode::ResourceElementCount &&
        desc.dispatch.countResourceKey.empty())
    {
        CRESSIM_LOG_ERROR(
            "CustomComputeService: resource-count dispatch requires countResourceKey.");
        return handle;
    }
    if (desc.constantBufferVariableName.empty() &&
        (desc.constantBufferSizeBytes != 0u || !desc.constantData.empty()))
    {
        CRESSIM_LOG_ERROR(
            "CustomComputeService: constants require constantBufferVariableName to be set.");
        return handle;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: physics compute backend is unavailable.");
        return handle;
    }

    ResourceMap resources;
    if (!buildResourceRegistry(solver, world, resources, nullptr))
    {
        return handle;
    }

    auto passState              = std::make_unique<PassState>();
    passState->desc             = desc;
    passState->resourceBindings = desc.resourceBindings;

    passState->variableNames.reserve(desc.resourceBindings.size() +
                                     (desc.constantBufferVariableName.empty() ? 0u : 1u));
    for (const CustomComputeResourceBindingDesc &binding : desc.resourceBindings)
    {
        const bool hasResourceKey  = !binding.resourceKey.empty();
        const bool hasSharedBuffer = binding.sharedBufferHandle.isValid();
        if (binding.shaderVariableName.empty() || hasResourceKey == hasSharedBuffer)
        {
            CRESSIM_LOG_ERROR("CustomComputeService: each binding requires shaderVariableName and "
                              "exactly one of resourceKey or sharedBufferHandle.");
            return handle;
        }
        if (hasResourceKey)
        {
            const auto resourceIt = resources.find(binding.resourceKey);
            if (resourceIt == resources.end())
            {
                CRESSIM_LOG_ERROR("CustomComputeService: unknown resource key '",
                                  binding.resourceKey, "'.");
                return handle;
            }
            if (!isAccessCompatible(binding.access, resourceIt->second.desc.access))
            {
                CRESSIM_LOG_ERROR(
                    "CustomComputeService: binding access is not allowed for resource '",
                    binding.resourceKey, "'.");
                return handle;
            }

            passState->expectedBindingGenerations[binding.resourceKey] =
                resourceIt->second.desc.bindingGeneration;
        }
        else
        {
            if (sharedBuffers == nullptr)
            {
                CRESSIM_LOG_ERROR("CustomComputeService: shared buffer bindings require a shared "
                                  "buffer service.");
                return handle;
            }
            if (!sharedBuffers->tryGetBuffer(binding.sharedBufferHandle))
            {
                CRESSIM_LOG_ERROR("CustomComputeService: invalid shared buffer handle ",
                                  binding.sharedBufferHandle.id, ".");
                return handle;
            }
            const SharedBufferAccess sharedAccess =
                binding.access == CustomComputeResourceAccess::ReadOnly
                    ? SharedBufferAccess::ReadOnly
                    : (binding.access == CustomComputeResourceAccess::WriteOnly
                           ? SharedBufferAccess::WriteOnly
                           : SharedBufferAccess::ReadWrite);
            if (!sharedBuffers->isAccessCompatible(binding.sharedBufferHandle, sharedAccess))
            {
                CRESSIM_LOG_ERROR("CustomComputeService: binding access is not allowed for shared "
                                  "buffer handle ",
                                  binding.sharedBufferHandle.id, ".");
                return handle;
            }
            passState->expectedSharedBufferGenerations[binding.sharedBufferHandle.id] = 1u;
        }
        passState->variableNames.push_back(binding.shaderVariableName);
    }

    if (desc.dispatch.mode == CustomComputeDispatchMode::ResourceElementCount)
    {
        const auto countIt = resources.find(desc.dispatch.countResourceKey);
        if (countIt == resources.end())
        {
            CRESSIM_LOG_ERROR("CustomComputeService: unknown dispatch count resource key '",
                              desc.dispatch.countResourceKey, "'.");
            return handle;
        }
        passState->expectedBindingGenerations[desc.dispatch.countResourceKey] =
            countIt->second.desc.bindingGeneration;
    }

    if (!desc.constantBufferVariableName.empty())
    {
        passState->variableNames.push_back(desc.constantBufferVariableName);
        passState->constantBufferSizeBytes = roundUpConstantBufferSize(std::max(
            desc.constantBufferSizeBytes, static_cast<std::uint32_t>(desc.constantData.size())));
        if (passState->constantBufferSizeBytes == 0u)
        {
            CRESSIM_LOG_ERROR("CustomComputeService: constant buffer size must be non-zero when "
                              "constants are enabled.");
            return handle;
        }

        Diligent::BufferDesc constantsDesc{};
        constantsDesc.Name =
            desc.debugName.empty() ? "CRESSimNeo.CustomCompute.Constants" : desc.debugName.c_str();
        constantsDesc.Size                 = passState->constantBufferSizeBytes;
        constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
        constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
        constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
        constantsDesc.ImmediateContextMask = gpu::contextMaskForId(computeBackend.contextId);

        computeBackend.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                                  &passState->constantBuffer);
        if (passState->constantBuffer == nullptr)
        {
            CRESSIM_LOG_ERROR("CustomComputeService: failed to create constant buffer.");
            return handle;
        }
        if (!uploadConstantData(*passState, desc.constantData))
        {
            return handle;
        }
    }

    passState->variables.reserve(passState->variableNames.size());
    for (const std::string &variableName : passState->variableNames)
    {
        passState->variables.push_back(Diligent::ShaderResourceVariableDesc{
            Diligent::SHADER_TYPE_COMPUTE, variableName.c_str(),
            Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE});
    }

    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: shader stream factory is unavailable.");
        return handle;
    }

    const std::string shaderName =
        desc.debugName.empty() ? desc.shaderPath : desc.debugName + ".Shader";
    const std::string psoName = desc.debugName.empty() ? desc.shaderPath : desc.debugName + ".PSO";
    gpu::GpuComputePassDefinition definition{};
    definition.shaderPath    = desc.shaderPath.c_str();
    definition.shaderSource  = desc.shaderSource.c_str();
    definition.shaderName    = shaderName.c_str();
    definition.entryPoint    = desc.entryPoint.empty() ? "main" : desc.entryPoint.c_str();
    definition.psoName       = psoName.c_str();
    definition.variables     = passState->variables.data();
    definition.variableCount = passState->variables.size();
    if (!passState->pass.initialize(mDevice, streamFactory,
                                    gpu::contextMaskForId(computeBackend.contextId), definition))
    {
        CRESSIM_LOG_ERROR("CustomComputeService: failed to initialize compute pass for shader '",
                          desc.shaderPath, "'.");
        return handle;
    }

    if (!bindPassResources(*passState, resources, sharedBuffers))
    {
        return handle;
    }

    handle.id          = mNextPassId++;
    mPasses[handle.id] = std::move(passState);
    return handle;
}

bool CustomComputeService::updatePassConstants(CustomComputePassHandle handle,
                                               const std::vector<std::uint8_t> &data)
{
    const auto it = mPasses.find(handle.id);
    if (it == mPasses.end())
    {
        CRESSIM_LOG_ERROR("CustomComputeService: invalid pass handle ", handle.id, ".");
        return false;
    }
    if (it->second->constantBuffer == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: pass ", handle.id,
                          " does not expose a constant buffer.");
        return false;
    }
    return uploadConstantData(*it->second, data);
}

bool CustomComputeService::executePass(physics::PhysicsSolver &solver, physics::PhysicsWorld &world,
                                       const SharedBufferService *sharedBuffers,
                                       CustomComputePassHandle handle)
{
    const auto it = mPasses.find(handle.id);
    if (it == mPasses.end())
    {
        CRESSIM_LOG_ERROR("CustomComputeService: invalid pass handle ", handle.id, ".");
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: physics compute backend is unavailable.");
        return false;
    }

    ResourceMap resources;
    if (!buildResourceRegistry(solver, world, resources, nullptr))
    {
        return false;
    }

    PassState &pass = *it->second;
    for (const auto &entry : pass.expectedBindingGenerations)
    {
        const auto resourceIt = resources.find(entry.first);
        if (resourceIt == resources.end())
        {
            CRESSIM_LOG_ERROR("CustomComputeService: required resource '", entry.first,
                              "' is no longer available.");
            return false;
        }
        if (resourceIt->second.desc.bindingGeneration != entry.second)
        {
            CRESSIM_LOG_ERROR("CustomComputeService: resource layout changed for '", entry.first,
                              "'. Recreate the custom compute pass.");
            return false;
        }
    }

    for (const auto &entry : pass.expectedSharedBufferGenerations)
    {
        if (sharedBuffers == nullptr ||
            !sharedBuffers->tryGetBuffer(SharedBufferHandle{entry.first}))
        {
            CRESSIM_LOG_ERROR("CustomComputeService: shared buffer handle ", entry.first,
                              " is no longer available.");
            return false;
        }
    }

    if (!bindPassResources(pass, resources, sharedBuffers))
    {
        return false;
    }

    std::uint32_t groupCountX = pass.desc.dispatch.groupCountX;
    std::uint32_t groupCountY = pass.desc.dispatch.groupCountY;
    std::uint32_t groupCountZ = pass.desc.dispatch.groupCountZ;
    if (pass.desc.dispatch.mode == CustomComputeDispatchMode::ResourceElementCount)
    {
        const auto countIt = resources.find(pass.desc.dispatch.countResourceKey);
        if (countIt == resources.end())
        {
            CRESSIM_LOG_ERROR("CustomComputeService: dispatch count resource '",
                              pass.desc.dispatch.countResourceKey, "' is unavailable.");
            return false;
        }
        if (countIt->second.desc.elementCount == 0u)
        {
            return true;
        }
        groupCountX = (countIt->second.desc.elementCount + pass.desc.threadGroupSizeX - 1u) /
                      pass.desc.threadGroupSizeX;
        groupCountY = 1u;
        groupCountZ = 1u;
    }

    Diligent::IShaderResourceBinding *srb   = pass.pass.defaultSrb();
    Diligent::IPipelineState *pipelineState = pass.pass.pipelineState();
    if (srb == nullptr || pipelineState == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: pass handle ", handle.id,
                          " is missing pipeline state.");
        return false;
    }

    computeBackend.computeContext->SetPipelineState(pipelineState);
    computeBackend.computeContext->CommitShaderResources(
        srb, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeBackend.computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{groupCountX, groupCountY, groupCountZ});
    return true;
}

bool CustomComputeService::destroyPass(CustomComputePassHandle handle)
{
    return mPasses.erase(handle.id) != 0u;
}

void CustomComputeService::clear()
{
    mPasses.clear();
    mNextPassId = 1u;
}

bool CustomComputeService::buildResourceRegistry(physics::PhysicsSolver &solver,
                                                 physics::PhysicsWorld &world,
                                                 ResourceMap &outResources,
                                                 std::vector<CustomComputeResourceDesc> *outDescs)
{
    outResources.clear();
    if (outDescs != nullptr)
    {
        outDescs->clear();
    }

    const physics::PhysicsGpuSceneView sceneView = solver.gpuSceneView();
    const auto addBuffer = [&](const std::string &key, Diligent::IBuffer *buffer,
                               CustomComputeResourceAccess access, std::uint32_t elementCount,
                               std::uint64_t bindingGeneration)
    {
        if (buffer == nullptr)
        {
            return;
        }

        const Diligent::BufferDesc &bufferDesc = buffer->GetDesc();
        CustomComputeResourceDesc desc{};
        desc.key                = key;
        desc.kind               = CustomComputeResourceKind::Buffer;
        desc.access             = access;
        desc.elementCount       = elementCount;
        desc.elementStrideBytes = static_cast<std::uint32_t>(bufferDesc.ElementByteStride);
        desc.bindingGeneration  = bindingGeneration;

        outResources.emplace(desc.key, ResourceEntry{desc, buffer});
        if (outDescs != nullptr)
        {
            outDescs->push_back(desc);
        }
    };

    addBuffer("rigid.positions", sceneView.rigid.statePositionsBuffer,
              CustomComputeResourceAccess::ReadOnly, sceneView.rigid.bodyCount,
              sceneView.rigid.bindingGeneration);
    addBuffer("rigid.orientations", sceneView.rigid.stateOrientationsBuffer,
              CustomComputeResourceAccess::ReadOnly, sceneView.rigid.bodyCount,
              sceneView.rigid.bindingGeneration);
    addBuffer("rigid.kinematic_target_positions", sceneView.rigid.kinematicTargetPositionsBuffer,
              CustomComputeResourceAccess::ReadWrite, sceneView.rigid.bodyCount,
              sceneView.rigid.bindingGeneration);
    addBuffer("rigid.kinematic_target_orientations",
              sceneView.rigid.kinematicTargetOrientationsBuffer,
              CustomComputeResourceAccess::ReadWrite, sceneView.rigid.bodyCount,
              sceneView.rigid.bindingGeneration);
    addBuffer("rigid.kinematic_target_flags", sceneView.rigid.kinematicTargetFlagsBuffer,
              CustomComputeResourceAccess::ReadWrite, sceneView.rigid.bodyCount,
              sceneView.rigid.bindingGeneration);

    return true;
}

bool CustomComputeService::isAccessCompatible(CustomComputeResourceAccess requested,
                                              CustomComputeResourceAccess allowed) noexcept
{
    switch (allowed)
    {
    case CustomComputeResourceAccess::ReadOnly:
        return requested == CustomComputeResourceAccess::ReadOnly;
    case CustomComputeResourceAccess::WriteOnly:
        return requested != CustomComputeResourceAccess::ReadOnly;
    case CustomComputeResourceAccess::ReadWrite:
        return true;
    }
    return false;
}

Diligent::BUFFER_VIEW_TYPE CustomComputeService::bufferViewTypeForAccess(
    CustomComputeResourceAccess access) noexcept
{
    return access == CustomComputeResourceAccess::ReadOnly ? Diligent::BUFFER_VIEW_SHADER_RESOURCE
                                                           : Diligent::BUFFER_VIEW_UNORDERED_ACCESS;
}

std::uint32_t CustomComputeService::roundUpConstantBufferSize(std::uint32_t sizeBytes) noexcept
{
    if (sizeBytes == 0u)
    {
        return 0u;
    }
    return (sizeBytes + 15u) & ~15u;
}

bool CustomComputeService::bindPassResources(PassState &pass, const ResourceMap &resources,
                                             const SharedBufferService *sharedBuffers)
{
    Diligent::IShaderResourceBinding *srb = pass.pass.defaultSrb();
    if (srb == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: pass is missing a shader resource binding.");
        return false;
    }

    std::vector<BufferBindingEntry> bindings;
    bindings.reserve(pass.resourceBindings.size() + (pass.constantBuffer == nullptr ? 0u : 1u));
    for (const CustomComputeResourceBindingDesc &bindingDesc : pass.resourceBindings)
    {
        Diligent::IBuffer *buffer = nullptr;
        if (!bindingDesc.resourceKey.empty())
        {
            const auto resourceIt = resources.find(bindingDesc.resourceKey);
            if (resourceIt == resources.end())
            {
                CRESSIM_LOG_ERROR("CustomComputeService: unknown resource key '",
                                  bindingDesc.resourceKey, "'.");
                return false;
            }
            buffer = resourceIt->second.buffer;
        }
        else
        {
            if (sharedBuffers == nullptr)
            {
                CRESSIM_LOG_ERROR("CustomComputeService: shared buffer service is unavailable.");
                return false;
            }
            buffer = sharedBuffers->tryGetBuffer(bindingDesc.sharedBufferHandle);
            if (buffer == nullptr)
            {
                CRESSIM_LOG_ERROR("CustomComputeService: shared buffer handle ",
                                  bindingDesc.sharedBufferHandle.id, " is unavailable.");
                return false;
            }
        }
        bindings.push_back(BufferBindingEntry{bindingDesc.shaderVariableName, buffer,
                                              bufferViewTypeForAccess(bindingDesc.access)});
    }
    if (pass.constantBuffer != nullptr)
    {
        bindings.push_back(BufferBindingEntry{pass.desc.constantBufferVariableName,
                                              pass.constantBuffer,
                                              Diligent::BUFFER_VIEW_UNDEFINED});
    }

    for (const BufferBindingEntry &binding : bindings)
    {
        if (!bindBufferVariable(srb, binding))
        {
            return false;
        }
    }
    return true;
}

bool CustomComputeService::uploadConstantData(PassState &pass,
                                              const std::vector<std::uint8_t> &data)
{
    if (pass.constantBuffer == nullptr)
    {
        return data.empty();
    }
    if (data.size() > pass.constantBufferSizeBytes)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: constant payload exceeds constant buffer size.");
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: physics compute backend is unavailable.");
        return false;
    }

    void *mapped = nullptr;
    computeBackend.computeContext->MapBuffer(pass.constantBuffer, Diligent::MAP_WRITE,
                                             Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        CRESSIM_LOG_ERROR("CustomComputeService: failed to map constant buffer.");
        return false;
    }

    std::memset(mapped, 0, pass.constantBufferSizeBytes);
    if (!data.empty())
    {
        std::memcpy(mapped, data.data(), data.size());
    }
    computeBackend.computeContext->UnmapBuffer(pass.constantBuffer, Diligent::MAP_WRITE);
    return true;
}

} // namespace cressim::neo::engine
