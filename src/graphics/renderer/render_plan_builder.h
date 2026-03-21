#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_RENDER_PLAN_BUILDER_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_RENDER_PLAN_BUILDER_H

#include "graphics/renderer/passes/render_pass_types.h"

namespace cressim::neo::graphics::detail
{

FrameRenderPlan buildFrameRenderPlan(std::vector<ResolvedCameraView> cameras,
                                     const ForwardDirectionalLightData& light,
                                     std::optional<DisplayResolveRequest> displayResolve);

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_RENDER_PLAN_BUILDER_H
