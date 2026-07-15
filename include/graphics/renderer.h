#ifndef CRESSIM_NEO_GRAPHICS_H
#define CRESSIM_NEO_GRAPHICS_H

#include "common/frame_context.h"
#include "common/id.h"
#include "gpu/gpu_device.h"
#include "graphics/export.h"
#include "graphics/host_scene.h"
#include "graphics/render_resource_manager.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace cressim::neo::physics
{
struct PhysicsGpuSceneView;
}

namespace cressim::neo::graphics
{

enum class ToneMapper : std::uint32_t
{
    Disabled = 0u,
    Reinhard = 1u,
    Filmic   = 2u,
};

struct RendererDesc
{
    IblQualityTier iblQualityTier = IblQualityTier::Off;
};

struct RenderFrameOptions
{
    struct PresentedExplicitOutput
    {
        enum class SourceKind : std::uint32_t
        {
            Color        = 0u,
            Depth        = 1u,
            Segmentation = 2u,
        };

        gpu::GpuRenderTargetBinding binding{};
        gpu::GpuRenderTargetDesc sourceTargetDesc{};
        SourceKind sourceKind       = SourceKind::Color;
        bool sourceIsDisplayEncoded = false;
        float nearClip              = 0.01f;
        float farClip               = 1000.0f;

        bool isValid() const noexcept
        {
            return binding.isValid();
        }
    };

    struct DebugParticleOptions
    {
        bool enabled                  = false;
        Diligent::float4 color        = {0.2f, 0.8f, 1.0f, 1.0f};
        Diligent::float4 staticColor  = {1.0f, 0.18f, 0.08f, 1.0f};
        Diligent::float4 edgeColor    = {1.0f, 0.86f, 0.18f, 1.0f};
        bool useParticleRadii         = true;
        bool highlightStaticParticles = true;
        bool drawConstraintEdges      = false;
        float fallbackRadius          = 0.15f;
    };

    struct DebugRoutedCableOptions
    {
        bool enabled   = false;
        float radius   = 0.03f;
        float opacity  = 1.0f;
        bool depthTest = true;
    };

    struct DebugStrandFrameOptions
    {
        bool enabled     = false;
        float axisLength = 0.18f;
        float thickness  = 0.02f;
        float opacity    = 1.0f;
    };

    common::EntityId presentedCameraEntity = common::kInvalidEntityId;
    std::optional<PresentedExplicitOutput> presentedExplicitOutput{};
    std::optional<gpu::GpuPresentationTargetDesc> presentationTarget{};
    ToneMapper toneMapper = ToneMapper::Reinhard;
    float exposure        = 1.0f;
    DebugParticleOptions debugParticles{};
    DebugRoutedCableOptions debugRoutedCables{};
    DebugStrandFrameOptions debugStrandFrames{};

    RenderFrameOptions() = default;

    RenderFrameOptions(common::EntityId presentedCameraEntityIn,
                       std::optional<gpu::GpuPresentationTargetDesc> presentationTargetIn,
                       ToneMapper toneMapperIn = ToneMapper::Reinhard, float exposureIn = 1.0f)
        : presentedCameraEntity(presentedCameraEntityIn),
          presentationTarget(std::move(presentationTargetIn)), toneMapper(toneMapperIn),
          exposure(exposureIn)
    {
    }

    RenderFrameOptions(
        common::EntityId presentedCameraEntityIn, PresentedExplicitOutput presentedExplicitOutputIn,
        std::optional<gpu::GpuPresentationTargetDesc> presentationTargetIn = std::nullopt,
        ToneMapper toneMapperIn = ToneMapper::Reinhard, float exposureIn = 1.0f)
        : presentedCameraEntity(presentedCameraEntityIn),
          presentedExplicitOutput(std::move(presentedExplicitOutputIn)),
          presentationTarget(std::move(presentationTargetIn)), toneMapper(toneMapperIn),
          exposure(exposureIn)
    {
    }
};

struct RenderStats
{
    // Current counters are framework-level instrumentation, not GPU timestamps.
    std::uint32_t drawCalls                   = 0;
    std::uint32_t opaqueDrawCalls             = 0;
    std::uint32_t transparentDrawCalls        = 0;
    std::uint32_t shadowDrawCalls             = 0;
    std::uint32_t renderableCount             = 0;
    std::uint32_t lightCount                  = 0;
    std::uint32_t renderedCameraCount         = 0;
    std::uint32_t renderTargetResizeRequests  = 0;
    std::uint32_t renderTargetResizeNoOps     = 0;
    std::uint32_t renderTargetRecreateCount   = 0;
    std::uint32_t renderTargetResizeConflicts = 0;
};

class CRESSIM_NEO_GRAPHICS_API Renderer
{
public:
    Renderer(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
             const RendererDesc &desc = RendererDesc{});
    ~Renderer();

    bool initialize();
    RenderStats render(const common::FrameContext &frameContext, const HostSceneView &sceneView,
                       const physics::PhysicsGpuSceneView *physicsScene,
                       const RenderFrameOptions &options = RenderFrameOptions{});

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_H
