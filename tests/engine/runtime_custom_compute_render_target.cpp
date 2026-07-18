#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/custom_compute.h"
#include "engine/runtime.h"
#include "gpu/cuda_interop.h"

#include <cstdint>
#include <vector>

namespace
{

constexpr const char *kRenderTargetSampleShader = R"(
#include "structured_buffer_compat.hlsli"

Texture2DArray<uint> g_SegmentationTarget;
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_Output);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadID.x;
    uint envCount = 0u;
    uint stride = 0u;
    g_Output.GetDimensions(envCount, stride);
    if (envIndex >= envCount)
    {
        return;
    }

    uint width = 0u;
    uint height = 0u;
    uint layers = 0u;
    g_SegmentationTarget.GetDimensions(width, height, layers);
    const uint sampled = g_SegmentationTarget.Load(int4(int(width / 2u), int(height / 2u),
                                                        int(envIndex), 0));
    CRESSIM_SB_STORE(g_Output, envIndex, sampled);
}
)";

using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "RuntimeCustomComputeRenderTarget.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3 &normal, const Diligent::float3 &v0,
                             const Diligent::float3 &v1, const Diligent::float3 &v2,
                             const Diligent::float3 &v3)
    {
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f});
        mesh.vertices.push_back({v1, normal, 1.0f, 0.0f});
        mesh.vertices.push_back({v2, normal, 1.0f, 1.0f});
        mesh.vertices.push_back({v3, normal, 0.0f, 1.0f});

        mesh.indices.push_back(baseIndex + 0u);
        mesh.indices.push_back(baseIndex + 2u);
        mesh.indices.push_back(baseIndex + 1u);
        mesh.indices.push_back(baseIndex + 0u);
        mesh.indices.push_back(baseIndex + 3u);
        mesh.indices.push_back(baseIndex + 2u);
    };

    const float h = halfExtent;
    addFace({0.0f, 0.0f, 1.0f}, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h});
    addFace({0.0f, 0.0f, -1.0f}, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h});
    addFace({-1.0f, 0.0f, 0.0f}, {-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h});
    addFace({1.0f, 0.0f, 0.0f}, {h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h});
    addFace({0.0f, 1.0f, 0.0f}, {-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h});
    addFace({0.0f, -1.0f, 0.0f}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h});

    return mesh;
}

} // namespace

int main()
{
    using namespace cressim::neo;

    if (!gpu::CudaSharedBuffer::supportsCudaInteropBuild())
    {
        CRESSIM_LOG_WARNING("Skipping render-target custom compute test because CUDA interop is "
                            "not enabled in this build.");
        return 0;
    }

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.sceneLayout.envCount           = 2u;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping render-target custom compute test because runtime "
                            "initialization failed.");
        return 0;
    }

    auto &world = runtime.getWorld();
    auto &resources = runtime.getResources();
    gpu::GpuRenderTargetDesc targetDesc{};
    targetDesc.width            = 64u;
    targetDesc.height           = 64u;
    targetDesc.arraySize        = 2u;
    targetDesc.color            = true;
    targetDesc.depth            = true;
    targetDesc.layeredRendering = true;
    targetDesc.shaderReadable   = true;
    targetDesc.colorFormat      = Diligent::TEX_FORMAT_R32_UINT;
    targetDesc.debugName        = "RuntimeCustomComputeRenderTarget.Target";
    const gpu::GpuRenderTargetHandle target =
        runtime.getGpuDevice()->renderTargetSystem().createRenderTarget(targetDesc);
    if (!runtime.getGpuDevice()->renderTargetSystem().isValidRenderTarget(target))
    {
        CRESSIM_LOG_ERROR("Failed to create explicit render target.");
        runtime.shutdown();
        return 1;
    }

    const graphics::MeshHandle cubeMesh = resources.registerMesh(makeCubeMesh(0.65f));
    MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "RuntimeCustomComputeRenderTarget.Material";
    const graphics::MaterialHandle material = resources.registerMaterial(materialDesc);

    for (std::uint32_t envIndex = 0u; envIndex < 2u; ++envIndex)
    {
        const common::EntityId cubeEntity = world.createEntity(envIndex);
        engine::TransformComponent cubeTransform{};
        cubeTransform.worldTransform.position = {0.0f, 0.0f, 0.5f};
        world.setTransform(cubeEntity, cubeTransform);

        engine::MeshRendererComponent cubeRenderer{};
        cubeRenderer.mesh           = cubeMesh;
        cubeRenderer.material       = material;
        cubeRenderer.segmentationId = 17u + envIndex;
        cubeRenderer.visible        = true;
        world.setMeshRenderer(cubeEntity, cubeRenderer);

        const common::EntityId cameraEntity = world.createEntity(envIndex);
        engine::TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, 0.0f, -4.0f};
        world.setTransform(cameraEntity, cameraTransform);

        engine::CameraComponent camera{};
        camera.product             = engine::CameraComponent::Product::SegmentationDepth;
        camera.output.mode         = gpu::RenderOutputMode::ExplicitSurface;
        camera.output.binding      = gpu::GpuRenderTargetBinding{target, envIndex, 1u};
        camera.outputWidth         = targetDesc.width;
        camera.outputHeight        = targetDesc.height;
        world.setCamera(cameraEntity, camera);

        const common::EntityId lightEntity = world.createEntity(envIndex);
        engine::DirectionalLightComponent light{};
        light.direction = {-0.55f, -1.0f, 0.25f};
        light.intensity = 6.0f;
        world.setDirectionalLight(lightEntity, light);
    }

    runtime.prepare();
    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload world before render-target compute test.");
        runtime.shutdown();
        return 1;
    }

    engine::SharedBufferDesc sharedDesc{};
    sharedDesc.debugName          = "RuntimeCustomComputeRenderTarget.Output";
    sharedDesc.elementStrideBytes = sizeof(std::uint32_t);
    sharedDesc.elementCount       = 2u;
    sharedDesc.access             = engine::SharedBufferAccess::ReadWrite;
    sharedDesc.bindFlags = engine::SharedBufferBindFlags::ShaderResource |
                           engine::SharedBufferBindFlags::UnorderedAccess;
    const engine::SharedBufferHandle sharedBuffer = runtime.createSharedBuffer(sharedDesc);
    if (!sharedBuffer.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create shared output buffer.");
        runtime.shutdown();
        return 1;
    }

    engine::CustomComputePassDesc passDesc{};
    passDesc.debugName        = "RuntimeCustomComputeRenderTarget";
    passDesc.shaderSource     = kRenderTargetSampleShader;
    passDesc.threadGroupSizeX = 64u;
    passDesc.resourceBindings.resize(2u);
    passDesc.resourceBindings[0].shaderVariableName     = "g_SegmentationTarget";
    passDesc.resourceBindings[0].renderTargetBinding    = gpu::GpuRenderTargetBinding{target, 0u, 2u};
    passDesc.resourceBindings[0].renderTargetTexturePlane =
        gpu::GpuRenderTargetTexturePlane::Color;
    passDesc.resourceBindings[0].access = engine::CustomComputeResourceAccess::ReadOnly;
    passDesc.resourceBindings[1].shaderVariableName = "g_Output";
    passDesc.resourceBindings[1].sharedBufferHandle = sharedBuffer;
    passDesc.resourceBindings[1].access = engine::CustomComputeResourceAccess::ReadWrite;
    passDesc.dispatch.mode        = engine::CustomComputeDispatchMode::ExplicitGroupCount;
    passDesc.dispatch.groupCountX = 1u;

    const engine::CustomComputePassHandle pass = runtime.createCustomComputePass(passDesc);
    if (!pass.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create render-target custom compute pass.");
        runtime.shutdown();
        return 1;
    }

    gpu::GpuRenderTargetDesc colorOnlyTargetDesc{};
    colorOnlyTargetDesc.width          = 32u;
    colorOnlyTargetDesc.height         = 32u;
    colorOnlyTargetDesc.arraySize      = 1u;
    colorOnlyTargetDesc.color          = true;
    colorOnlyTargetDesc.depth          = false;
    colorOnlyTargetDesc.shaderReadable = true;
    colorOnlyTargetDesc.colorFormat    = Diligent::TEX_FORMAT_RGBA8_UNORM;
    colorOnlyTargetDesc.debugName      = "RuntimeCustomComputeRenderTarget.ColorOnly";
    const gpu::GpuRenderTargetHandle colorOnlyTarget =
        runtime.getGpuDevice()->renderTargetSystem().createRenderTarget(colorOnlyTargetDesc);
    engine::CustomComputePassDesc invalidDepthPass = passDesc;
    invalidDepthPass.resourceBindings.resize(2u);
    invalidDepthPass.resourceBindings[0].shaderVariableName = "g_SegmentationTarget";
    invalidDepthPass.resourceBindings[0].renderTargetBinding =
        gpu::GpuRenderTargetBinding{colorOnlyTarget, 0u, 1u};
    invalidDepthPass.resourceBindings[0].renderTargetTexturePlane =
        gpu::GpuRenderTargetTexturePlane::Depth;
    invalidDepthPass.resourceBindings[0].access = engine::CustomComputeResourceAccess::ReadOnly;
    invalidDepthPass.resourceBindings[1].shaderVariableName = "g_Output";
    invalidDepthPass.resourceBindings[1].sharedBufferHandle = sharedBuffer;
    invalidDepthPass.resourceBindings[1].access = engine::CustomComputeResourceAccess::ReadWrite;
    if (runtime.createCustomComputePass(invalidDepthPass).isValid())
    {
        CRESSIM_LOG_ERROR("Expected invalid depth-plane binding on a color-only target to fail.");
        runtime.shutdown();
        return 1;
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex   = 0u;
    frame.timeSeconds  = 0.0;
    runtime.stepVisualSensors(frame);
    if (!runtime.executeCustomComputePass(pass))
    {
        CRESSIM_LOG_ERROR("Failed to execute render-target custom compute pass.");
        runtime.shutdown();
        return 1;
    }
    if (!runtime.syncSharedBufferToCuda(sharedBuffer))
    {
        CRESSIM_LOG_ERROR("Failed to synchronize render-target custom compute output to CUDA.");
        runtime.shutdown();
        return 1;
    }

    engine::SharedBufferCudaView cudaView{};
    if (!runtime.tryGetSharedBufferCudaView(sharedBuffer, cudaView) || !cudaView.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to retrieve CUDA view for render-target custom compute output.");
        runtime.shutdown();
        return 1;
    }

    gpu::CudaStream cudaStream;
    if (!cudaStream.initialize())
    {
        CRESSIM_LOG_ERROR("Failed to create CUDA stream for render-target custom compute test.");
        runtime.shutdown();
        return 1;
    }

    std::vector<std::uint32_t> sampledIds(2u, 0u);
    if (!cudaStream.copyDeviceToHostAsync(sampledIds.data(), cudaView.devicePointer,
                                          sampledIds.size() * sizeof(std::uint32_t)) ||
        !cudaStream.synchronize())
    {
        CRESSIM_LOG_ERROR("Failed to copy render-target custom compute output from CUDA.");
        runtime.shutdown();
        return 1;
    }
    if (sampledIds[0] != 17u || sampledIds[1] != 18u)
    {
        CRESSIM_LOG_ERROR("Unexpected sampled segmentation ids. got=[", sampledIds[0], ", ",
                          sampledIds[1], "] expected=[17, 18].");
        runtime.shutdown();
        return 1;
    }
    runtime.endFrame(frame);

    runtime.shutdown();
    return 0;
}
