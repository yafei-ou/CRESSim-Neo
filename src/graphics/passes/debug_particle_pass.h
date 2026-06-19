#ifndef CRESSIM_NEO_GRAPHICS_DEBUG_PARTICLE_PASS_H
#define CRESSIM_NEO_GRAPHICS_DEBUG_PARTICLE_PASS_H

#include "gpu/gpu_device.h"
#include "graphics/renderer.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"

#include <cstdint>
#include <unordered_map>

namespace cressim::neo::physics
{
struct PhysicsGpuSceneView;
}

namespace cressim::neo::graphics
{
struct ResolvedCameraView;
}

namespace cressim::neo::graphics::detail
{

class DebugParticlePass
{
public:
    explicit DebugParticlePass(gpu::GpuDevice &device);

    bool initialize();
    bool draw(const gpu::GpuRenderTargetBinding &targetBinding,
              const gpu::GpuRenderTargetDesc &targetDesc, const GpuEntitySceneView &gpuScene,
              const cressim::neo::physics::PhysicsGpuSceneView &physicsScene,
              const ResolvedCameraView &camera, std::uint32_t targetLayer,
              const RenderFrameOptions::DebugParticleOptions &options);

private:
    struct DrawConstants
    {
        Diligent::float4 color{0.2f, 0.8f, 1.0f, 1.0f};
        Diligent::float4 staticColor{1.0f, 0.18f, 0.08f, 1.0f};
        Diligent::float4 edgeColor{1.0f, 1.0f, 1.0f, 1.0f};
        Diligent::float4 edgeHighStrainColor{1.0f, 0.08f, 0.04f, 1.0f};
        Diligent::float4 edgeDamagedColor{1.0f, 0.48f, 0.04f, 1.0f};
        Diligent::float4 edgeDisabledColor{0.0f, 0.0f, 0.0f, 1.0f};
        std::uint32_t cameraIndex = 0u;
        std::uint32_t targetLayer = 0u;
        std::uint32_t envIndex    = 0u;
        std::uint32_t flags       = 0u;
        std::uint32_t shapeModes         = 0u;
        std::uint32_t shapePrimitiveMode = 0u;
        std::uint32_t maxMembershipCount = 1u;
        std::uint32_t padding0           = 0u;
        float fallbackRadius             = 0.15f;
        float highStrainThreshold        = 0.35f;
        float damageDisplayThreshold     = 0.01f;
        float shapeCorrectionScale       = 40.0f;
        float shapeCenterRadius          = 0.055f;
        float shapeAxisLength            = 0.09f;
        float padding1                   = 0.0f;
        float padding2                   = 0.0f;
    };

    struct PipelineKey
    {
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;

        bool operator==(const PipelineKey &rhs) const noexcept
        {
            return colorFormat == rhs.colorFormat && depthFormat == rhs.depthFormat;
        }
    };

    struct PipelineKeyHasher
    {
        std::size_t operator()(const PipelineKey &key) const noexcept;
    };

    bool ensureConstants(Diligent::IRenderDevice *renderDevice);
    Diligent::IPipelineState *getOrCreatePipeline(Diligent::IRenderDevice *renderDevice,
                                                  const PipelineKey &key);
    Diligent::IPipelineState *getOrCreateEdgePipeline(Diligent::IRenderDevice *renderDevice,
                                                      const PipelineKey &key);
    Diligent::IShaderResourceBinding *getOrCreateBinding(Diligent::IPipelineState *pipeline);
    bool drawConstraintEdges(const gpu::GpuRenderTargetDesc &targetDesc,
                             const GpuEntitySceneView &gpuScene,
                             const cressim::neo::physics::PhysicsGpuSceneView &physicsScene,
                             Diligent::IRenderDevice *renderDevice,
                             Diligent::IDeviceContext *graphicsContext,
                             const DrawConstants &constants);

    gpu::GpuDevice &mDevice;
    bool mInitialized = false;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mFallbackRadiusBuffer;
    std::unordered_map<PipelineKey, Diligent::RefCntAutoPtr<Diligent::IPipelineState>,
                       PipelineKeyHasher>
        mPipelines;
    std::unordered_map<PipelineKey, Diligent::RefCntAutoPtr<Diligent::IPipelineState>,
                       PipelineKeyHasher>
        mEdgePipelines;
    std::unordered_map<Diligent::IPipelineState *,
                       Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>>
        mBindings;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_DEBUG_PARTICLE_PASS_H
