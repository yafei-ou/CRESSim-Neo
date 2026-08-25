#ifndef CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_H
#define CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_H

#include "common/frame_context.h"
#include "gpu/export.h"
#include "gpu/gpu_types.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>

/// @file gpu_render_target_system.h
/// @brief Offscreen render target allocation, resizing, binding, viewport manipulation, and
/// readback system interface.

namespace cressim::neo::gpu
{

/// @brief Subsystem managing offscreen render targets, multi-layer attachments, viewports, and
/// pixel readbacks.
class CRESSIM_NEO_GPU_API GpuRenderTargetSystem
{
public:
    /// @brief Virtual destructor.
    virtual ~GpuRenderTargetSystem() = default;

    /// @brief Allocates an offscreen render target texture set based on the provided descriptor.
    /// @param desc Target geometry and texture format specification.
    /// @return Allocated render target handle.
    virtual GpuRenderTargetHandle createRenderTarget(const GpuRenderTargetDesc &desc) = 0;
    /// @brief Resizes an existing render target's pixel dimensions.
    /// @param target Handle of the target to resize.
    /// @param width New width in pixels.
    /// @param height New height in pixels.
    /// @return GpuRenderTargetUpdateResult status code.
    virtual GpuRenderTargetUpdateResult resizeRenderTarget(GpuRenderTargetHandle target,
                                                           std::uint32_t width,
                                                           std::uint32_t height)      = 0;
    /// @brief Reconfigures all properties (formats, flags, layers, dimensions) of a render target.
    /// @param target Handle of the target to reconfigure.
    /// @param desc Updated target descriptor.
    /// @return GpuRenderTargetUpdateResult status code.
    virtual GpuRenderTargetUpdateResult reconfigureRenderTarget(
        GpuRenderTargetHandle target, const GpuRenderTargetDesc &desc)      = 0;
    /// @brief Destroys and deallocates an offscreen render target.
    /// @param target Handle of target to destroy.
    virtual void destroyRenderTarget(GpuRenderTargetHandle target)          = 0;
    /// @brief Checks whether the given target handle refers to an allocated, active render target.
    /// @param target Target handle to check.
    /// @return True if valid.
    virtual bool isValidRenderTarget(GpuRenderTargetHandle target) const    = 0;
    /// @brief Queries the configuration descriptor of an active render target.
    /// @param target Handle to inspect.
    /// @param outDesc Output descriptor to populate.
    /// @return True on success.
    virtual bool tryGetRenderTargetDesc(GpuRenderTargetHandle target,
                                        GpuRenderTargetDesc &outDesc) const = 0;

    /// @brief Sets the normalized viewport for subsequent rendering operations targeting this
    /// binding.
    /// @param binding Target binding and layer range.
    /// @param viewport Normalized viewport rectangle.
    virtual void setRenderTargetViewport(const GpuRenderTargetBinding &binding,
                                         const GpuRenderViewport &viewport) = 0;
    /// @brief Binds the render target attachments to the active graphics context and applies clear
    /// values.
    /// @param binding Target binding and layer range.
    /// @param frameContext Temporal frame context.
    /// @param beginDesc Clear flags and values.
    virtual void beginRenderTarget(const GpuRenderTargetBinding &binding,
                                   const common::FrameContext &frameContext,
                                   const GpuRenderPassBeginDesc &beginDesc) = 0;
    /// @brief Unbinds the render target attachments and transitions textures for subsequent
    /// sampling/compute.
    /// @param binding Target binding.
    /// @param frameContext Temporal frame context.
    virtual void endRenderTarget(const GpuRenderTargetBinding &binding,
                                 const common::FrameContext &frameContext)  = 0;

    /// @brief Queues an asynchronous readback copy of the specified render target's pixels to CPU
    /// host memory.
    /// @param binding Target binding to read back.
    /// @return Monotonic tracking readback request handle.
    virtual GpuRenderTargetReadbackRequest requestRenderTargetReadback(
        const GpuRenderTargetBinding &binding)                                          = 0;
    /// @brief Attempts to fetch downloaded pixel data for an enqueued readback request.
    /// @param request Tracking handle.
    /// @param outEvent Output readback event payload to populate.
    /// @return True if data was downloaded and ready, false if still pending.
    virtual bool tryGetRenderTargetReadback(GpuRenderTargetReadbackRequest request,
                                            GpuRenderTargetReadbackEvent &outEvent)     = 0;
    /// @brief Retrieves the raw Diligent texture pointer for the target's color attachment.
    /// @param target Render target handle.
    /// @param outTexture Output texture reference.
    /// @return True if target has a color attachment.
    virtual bool tryGetRenderTargetColorTexture(GpuRenderTargetHandle target,
                                                Diligent::ITexture *&outTexture)        = 0;
    /// @brief Retrieves the raw Diligent texture pointer for the target's depth attachment.
    /// @param target Render target handle.
    /// @param outTexture Output texture reference.
    /// @return True if target has a depth attachment.
    virtual bool tryGetRenderTargetDepthTexture(GpuRenderTargetHandle target,
                                                Diligent::ITexture *&outTexture)        = 0;
    /// @brief Retrieves the Diligent shader resource view (SRV) for sampling this target in later
    /// passes.
    /// @param binding Target binding and layer slice.
    /// @param plane Texture plane (Color or Depth).
    /// @param outView Output texture view reference.
    /// @return True if SRV is available.
    virtual bool tryGetRenderTargetShaderResourceView(const GpuRenderTargetBinding &binding,
                                                      GpuRenderTargetTexturePlane plane,
                                                      Diligent::ITextureView *&outView) = 0;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_H
