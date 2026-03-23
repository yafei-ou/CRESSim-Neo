#include "graphics/renderer/render_plan_builder.h"
#include "common/logger.h"

#include <optional>
#include <vector>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::kInvalidEntityId;
using cressim::neo::gpu::GpuRenderTargetBinding;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::graphics::CameraBatchView;
using cressim::neo::graphics::ForwardDirectionalLightData;
using cressim::neo::graphics::FrameRenderPlan;
using cressim::neo::graphics::ResolvedCameraView;
using cressim::neo::graphics::detail::buildFrameRenderPlan;

ResolvedCameraView makeCamera(EntityId entityId, GpuRenderTargetHandle target,
                              std::uint32_t firstLayer, bool layeredRendering,
                              bool useOutputViewport)
{
    ResolvedCameraView camera{};
    camera.entityId                  = entityId;
    camera.outputBinding            = GpuRenderTargetBinding{target, firstLayer, 1u};
    camera.outputTargetDesc.width   = 320u;
    camera.outputTargetDesc.height  = 180u;
    camera.outputTargetDesc.arraySize = layeredRendering ? 2u : 1u;
    camera.outputTargetDesc.color   = true;
    camera.outputTargetDesc.depth   = true;
    camera.outputTargetDesc.layeredRendering = layeredRendering;
    camera.viewport                 = {0.0f, 0.0f, 0.5f, 1.0f};
    camera.useOutputViewport        = useOutputViewport;
    return camera;
}

bool isSingleCameraBatches(const FrameRenderPlan& plan)
{
    for (const CameraBatchView& batch : plan.cameraBatches)
    {
        if (batch.cameras.size() != 1u)
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    const ForwardDirectionalLightData light{};
    const GpuRenderTargetHandle target{42u};

    std::vector<ResolvedCameraView> explicitNonLayeredCameras;
    explicitNonLayeredCameras.push_back(makeCamera(1u, target, 0u, false, true));
    explicitNonLayeredCameras.push_back(makeCamera(2u, target, 0u, false, true));

    const FrameRenderPlan explicitPlan =
        buildFrameRenderPlan(std::move(explicitNonLayeredCameras), light, std::nullopt);
    if (explicitPlan.cameraBatches.size() != 2u || !isSingleCameraBatches(explicitPlan))
    {
        CRESSIM_LOG_ERROR( "Expected explicit non-layered viewport cameras to remain single-camera batches.\n");
        return 1;
    }

    std::vector<ResolvedCameraView> layeredCameras;
    layeredCameras.push_back(makeCamera(3u, target, 0u, true, false));
    layeredCameras.push_back(makeCamera(4u, target, 1u, true, false));

    const FrameRenderPlan layeredPlan =
        buildFrameRenderPlan(std::move(layeredCameras), light, std::nullopt);
    if (layeredPlan.cameraBatches.size() != 1u || layeredPlan.cameraBatches.front().cameras.size() != 2u)
    {
        CRESSIM_LOG_ERROR( "Expected layered cameras to preserve existing batching behavior.\n");
        return 1;
    }

    CRESSIM_LOG_INFO( "Render plan viewport policy checks passed.\n");
    return 0;
}
