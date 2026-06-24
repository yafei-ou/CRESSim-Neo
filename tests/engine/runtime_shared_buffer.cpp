#include "common/logger.h"
#include "engine/custom_compute.h"
#include "engine/runtime.h"
#include "gpu/cuda_interop.h"

namespace
{

constexpr const char *kSharedBufferWriteShader = R"(
#include "include/structured_buffer_compat.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Output);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    uint count = 0u;
    uint stride = 0u;
    g_Output.GetDimensions(count, stride);
    if (idx >= count)
    {
        return;
    }

    CRESSIM_SB_STORE(g_Output, idx, idx);
}
)";

} // namespace

int main()
{
    using namespace cressim::neo;

    if (!gpu::CudaSharedBuffer::supportsCudaInteropBuild())
    {
        CRESSIM_LOG_WARNING("Skipping runtime shared buffer test because CUDA interop is not "
                            "enabled in this build.");
        return 0;
    }

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping runtime shared buffer test because runtime initialization "
                            "failed.");
        return 0;
    }

    engine::SharedBufferDesc sharedDesc{};
    sharedDesc.debugName          = "RuntimeSharedBufferSmoke";
    sharedDesc.elementStrideBytes = sizeof(std::uint32_t);
    sharedDesc.elementCount       = 64u;
    sharedDesc.access             = engine::SharedBufferAccess::ReadWrite;
    sharedDesc.bindFlags = engine::SharedBufferBindFlags::ShaderResource |
                           engine::SharedBufferBindFlags::UnorderedAccess;
    const engine::SharedBufferHandle sharedBuffer = runtime.createSharedBuffer(sharedDesc);
    if (!sharedBuffer.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create shared buffer.");
        runtime.shutdown();
        return 1;
    }

    engine::SharedBufferInfo sharedInfo{};
    if (!runtime.tryGetSharedBufferInfo(sharedBuffer, sharedInfo) || !sharedInfo.importedIntoCuda)
    {
        CRESSIM_LOG_ERROR("Shared buffer metadata is unavailable or CUDA import failed.");
        runtime.shutdown();
        return 1;
    }

    engine::SharedBufferCudaView cudaView{};
    if (!runtime.tryGetSharedBufferCudaView(sharedBuffer, cudaView) || !cudaView.isValid())
    {
        CRESSIM_LOG_ERROR("Shared buffer CUDA view is unavailable.");
        runtime.shutdown();
        return 1;
    }

    runtime.prepare();
    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload world before shared-buffer custom compute.");
        runtime.shutdown();
        return 1;
    }

    engine::CustomComputePassDesc passDesc{};
    passDesc.debugName        = "RuntimeSharedBufferWrite";
    passDesc.shaderSource     = kSharedBufferWriteShader;
    passDesc.threadGroupSizeX = 64u;
    passDesc.resourceBindings.resize(1u);
    passDesc.resourceBindings[0].shaderVariableName = "g_Output";
    passDesc.resourceBindings[0].sharedBufferHandle = sharedBuffer;
    passDesc.resourceBindings[0].access = engine::CustomComputeResourceAccess::ReadWrite;
    passDesc.dispatch.mode        = engine::CustomComputeDispatchMode::ExplicitGroupCount;
    passDesc.dispatch.groupCountX = 1u;

    const engine::CustomComputePassHandle pass = runtime.createCustomComputePass(passDesc);
    if (!pass.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create shared-buffer custom compute pass.");
        runtime.shutdown();
        return 1;
    }
    if (!runtime.executeCustomComputePass(pass))
    {
        CRESSIM_LOG_ERROR("Failed to execute shared-buffer custom compute pass.");
        runtime.shutdown();
        return 1;
    }

    engine::SharedBufferDesc readOnlyDesc{};
    readOnlyDesc.debugName          = "RuntimeSharedBufferReadOnly";
    readOnlyDesc.elementStrideBytes = sizeof(std::uint32_t);
    readOnlyDesc.elementCount       = 8u;
    readOnlyDesc.access             = engine::SharedBufferAccess::ReadOnly;
    readOnlyDesc.bindFlags          = engine::SharedBufferBindFlags::ShaderResource;
    const engine::SharedBufferHandle readOnlyBuffer = runtime.createSharedBuffer(readOnlyDesc);
    if (!readOnlyBuffer.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create read-only shared buffer.");
        runtime.shutdown();
        return 1;
    }

    engine::CustomComputePassDesc invalidPassDesc = passDesc;
    invalidPassDesc.resourceBindings.resize(1u);
    invalidPassDesc.resourceBindings[0].shaderVariableName = "g_Output";
    invalidPassDesc.resourceBindings[0].sharedBufferHandle = readOnlyBuffer;
    invalidPassDesc.resourceBindings[0].access =
        engine::CustomComputeResourceAccess::ReadWrite;
    if (runtime.createCustomComputePass(invalidPassDesc).isValid())
    {
        CRESSIM_LOG_ERROR("Expected read-only shared buffer binding to reject read-write access.");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    return 0;
}
