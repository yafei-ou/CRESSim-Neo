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

/// @file renderer.h
/// @brief Multi-pass clustered forward HDR renderer, post-processing tonemapping, and visual debug
/// overlays.

namespace cressim::neo::physics
{
struct PhysicsGpuSceneView;
}

namespace cressim::neo::graphics
{

/// @brief HDR tonemapping algorithm applied during display presentation.
enum class ToneMapper : std::uint32_t
{
    Disabled = 0u, ///< No tonemapping applied; raw HDR clamped to [0, 1].
    Reinhard = 1u, ///< Extended Reinhard rational curve operator.
    Filmic   = 2u, ///< ACES filmic S-curve tonemapping.
};

/// @brief Global configuration descriptor for creating the forward renderer.
struct RendererDesc
{
    IblQualityTier iblQualityTier = IblQualityTier::Off; ///< Image-based lighting quality tier.
};

/// @brief Per-frame rendering options, camera presentation routing, and debug visualization
/// switches.
struct RenderFrameOptions
{
    /// @brief Destination display routing configuration for an explicit offscreen target.
    struct PresentedExplicitOutput
    {
        /// @brief Source plane classification for display blitting.
        enum class SourceKind : std::uint32_t
        {
            Color        = 0u, ///< RGBA color buffer.
            Depth        = 1u, ///< Linearized depth buffer.
            Segmentation = 2u, ///< Categorical segmentation ID mask.
        };

        gpu::GpuRenderTargetBinding binding{};           ///< Source target binding.
        gpu::GpuRenderTargetDesc sourceTargetDesc{};     ///< Source target descriptor.
        SourceKind sourceKind       = SourceKind::Color; ///< Output channel category.
        bool sourceIsDisplayEncoded = false;   ///< True if source buffer is already sRGB encoded.
        float nearClip              = 0.01f;   ///< Near plane for depth visualization.
        float farClip               = 1000.0f; ///< Far plane for depth visualization.

        /// @brief Checks if the output binding refers to a valid target.
        /// @return True if valid.
        bool isValid() const noexcept
        {
            return binding.isValid();
        }
    };

    /// @brief Debug visualization options for particle physics point clouds.
    struct DebugParticleOptions
    {
        bool enabled                 = false; ///< Enable particle sphere/billboard debug rendering.
        Diligent::float4 color       = {0.2f, 0.8f, 1.0f, 1.0f}; ///< Default particle color (RGBA).
        Diligent::float4 staticColor = {1.0f, 0.18f, 0.08f,
                                        1.0f}; ///< Color for pinned/static particles.
        Diligent::float4 edgeColor   = {1.0f, 0.86f, 0.18f,
                                        1.0f}; ///< Color for distance constraint debug lines.
        bool useParticleRadii = true; ///< Use actual particle radii instead of fallbackRadius.
        bool highlightStaticParticles = true; ///< Visually tint static particles differently.
        bool drawConstraintEdges =
            false;                    ///< Draw wireframe edges between constrained particle pairs.
        float fallbackRadius = 0.15f; ///< Fallback radius in world units if unassigned.
    };

    /// @brief Debug visualization options for routed cable paths.
    struct DebugRoutedCableOptions
    {
        bool enabled   = false; ///< Enable cable curve wireframe debug rendering.
        float radius   = 0.03f; ///< Visual tube radius.
        float opacity  = 1.0f;  ///< Visual opacity.
        bool depthTest = true;  ///< Whether debug cable geometry tests against scene depth buffer.
    };

    /// @brief Debug visualization options for 1D strand coordinate frames.
    struct DebugStrandFrameOptions
    {
        bool enabled     = false; ///< Enable strand coordinate axis triad rendering.
        float axisLength = 0.18f; ///< Length of RGB triad axis lines.
        float thickness  = 0.02f; ///< Line thickness.
        float opacity    = 1.0f;  ///< Visual opacity.
    };

    common::EntityId presentedCameraEntity =
        common::kInvalidEntityId; ///< Primary camera entity blitted to the window presentation
                                  ///< swapchain.
    std::optional<PresentedExplicitOutput>
        presentedExplicitOutput{}; ///< Optional explicit offscreen output blitted to presentation.
    std::optional<gpu::GpuPresentationTargetDesc>
        presentationTarget{};                     ///< Presentation swapchain description.
    ToneMapper toneMapper = ToneMapper::Reinhard; ///< Tonemapping operator.
    float exposure        = 1.0f;                 ///< Exposure compensation scale factor.
    DebugParticleOptions debugParticles{};        ///< Particle debug rendering options.
    DebugRoutedCableOptions debugRoutedCables{};  ///< Routed cable debug options.
    DebugStrandFrameOptions debugStrandFrames{};  ///< Strand frame debug options.

    /// @brief Default constructor.
    RenderFrameOptions() = default;

    /// @brief Convenience constructor specifying presented camera entity and presentation target.
    /// @param presentedCameraEntityIn Entity ID of camera to present.
    /// @param presentationTargetIn Presentation target description.
    /// @param toneMapperIn Selected tonemapper.
    /// @param exposureIn Exposure multiplier.
    RenderFrameOptions(common::EntityId presentedCameraEntityIn,
                       std::optional<gpu::GpuPresentationTargetDesc> presentationTargetIn,
                       ToneMapper toneMapperIn = ToneMapper::Reinhard, float exposureIn = 1.0f)
        : presentedCameraEntity(presentedCameraEntityIn),
          presentationTarget(std::move(presentationTargetIn)), toneMapper(toneMapperIn),
          exposure(exposureIn)
    {
    }

    /// @brief Convenience constructor specifying presented camera, explicit output, and
    /// presentation target.
    /// @param presentedCameraEntityIn Camera entity ID.
    /// @param presentedExplicitOutputIn Explicit output binding to present.
    /// @param presentationTargetIn Presentation target description.
    /// @param toneMapperIn Selected tonemapper.
    /// @param exposureIn Exposure multiplier.
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

/// @brief Frame rendering performance and draw call instrumentation metrics.
struct RenderStats
{
    std::uint32_t drawCalls            = 0; ///< Total graphics draw calls executed.
    std::uint32_t opaqueDrawCalls      = 0; ///< Opaque forward pass draw calls.
    std::uint32_t transparentDrawCalls = 0; ///< Transparent pass draw calls.
    std::uint32_t shadowDrawCalls      = 0; ///< Shadow map cascade and local light draw calls.
    std::uint32_t renderableCount      = 0; ///< Total candidate renderable objects evaluated.
    std::uint32_t lightCount           = 0; ///< Total scene light sources.
    std::uint32_t renderedCameraCount  = 0; ///< Total cameras rendered during the frame.
    std::uint32_t renderTargetResizeRequests =
        0; ///< Number of offscreen render target resize requests.
    std::uint32_t renderTargetResizeNoOps =
        0; ///< Resize requests where target dimensions were unchanged.
    std::uint32_t renderTargetRecreateCount = 0; ///< Number of render target texture recreations.
    std::uint32_t renderTargetResizeConflicts =
        0; ///< Conflicts where shared targets had mismatched requests.
};

/// @brief Primary clustered forward HDR renderer executing scene culling, shadow passes, surface
/// shading, and post-processing.
class CRESSIM_NEO_GRAPHICS_API Renderer
{
public:
    /// @brief Constructs a renderer instance bound to a GPU device and asset resource manager.
    /// @param device GPU device interface.
    /// @param resourceManager Mesh/material/texture resource manager.
    /// @param desc Renderer configuration descriptor.
    Renderer(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
             const RendererDesc &desc = RendererDesc{});
    /// @brief Destructor releasing renderer pipelines and GPU passes.
    ~Renderer();

    /// @brief Initializes graphics pipelines, shadow passes, compute culling, and post-processing
    /// filters.
    /// @return True on success.
    bool initialize();
    /// @brief Executes complete scene rendering for all active cameras and outputs.
    /// @param frameContext Temporal frame context.
    /// @param sceneView Scene graph and camera/light data view.
    /// @param physicsScene Optional physics GPU buffer views for deformable skinning and debug
    /// overlays.
    /// @param options Per-frame presentation and tonemapping options.
    /// @return RenderStats struct containing draw call and execution metrics.
    RenderStats render(const common::FrameContext &frameContext, const HostSceneView &sceneView,
                       const physics::PhysicsGpuSceneView *physicsScene,
                       const RenderFrameOptions &options = RenderFrameOptions{});

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_H
