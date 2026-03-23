#ifndef CRESSIM_NEO_GRAPHICS_RENDER_PLAN_BUILDER_H
#define CRESSIM_NEO_GRAPHICS_RENDER_PLAN_BUILDER_H

#include "graphics/passes/render_pass_types.h"

namespace cressim::neo::graphics::detail
{

FrameRenderPlan buildFrameRenderPlan(std::vector<ResolvedCameraView> cameras,
                                     std::optional<DisplayResolveRequest> displayResolve);

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDER_PLAN_BUILDER_H
