#include "graphics/renderer/render_plan_builder.h"

#include <algorithm>

namespace cressim::neo::graphics::detail
{

namespace
{

bool sameBatchCompatibility(const ResolvedCameraView &lhs, const ResolvedCameraView &rhs)
{
    const gpu::GpuRenderTargetDesc &lhsDesc = lhs.outputTargetDesc;
    const gpu::GpuRenderTargetDesc &rhsDesc = rhs.outputTargetDesc;
    return lhs.outputBinding.target.id == rhs.outputBinding.target.id &&
           lhsDesc.width == rhsDesc.width && lhsDesc.height == rhsDesc.height &&
           lhsDesc.color == rhsDesc.color && lhsDesc.depth == rhsDesc.depth &&
           lhsDesc.shaderReadable == rhsDesc.shaderReadable &&
           lhsDesc.colorFormat == rhsDesc.colorFormat &&
           lhsDesc.depthFormat == rhsDesc.depthFormat && lhs.clearColor == rhs.clearColor &&
           lhs.clearDepth == rhs.clearDepth && lhs.clearColorValue.x == rhs.clearColorValue.x &&
           lhs.clearColorValue.y == rhs.clearColorValue.y &&
           lhs.clearColorValue.z == rhs.clearColorValue.z &&
           lhs.clearColorValue.w == rhs.clearColorValue.w &&
           lhs.clearDepthValue == rhs.clearDepthValue;
}

bool requiresDedicatedBatch(const ResolvedCameraView &camera)
{
    return camera.useOutputViewport;
}

} // namespace

FrameRenderPlan buildFrameRenderPlan(std::vector<ResolvedCameraView> cameras,
                                     const ForwardDirectionalLightData &light,
                                     std::optional<DisplayResolveRequest> displayResolve)
{
    FrameRenderPlan plan{};
    plan.displayResolve = std::move(displayResolve);

    CameraBatchView currentBatch{};
    bool hasOpenBatch = false;
    std::vector<std::uint32_t> usedLayers;

    const auto flushBatch = [&]()
    {
        if (!hasOpenBatch)
        {
            return;
        }

        std::uint32_t minLayer = currentBatch.cameras.front().outputBinding.firstLayer;
        std::uint32_t maxLayer = minLayer;
        for (const ResolvedCameraView &camera : currentBatch.cameras)
        {
            minLayer = std::min(minLayer, camera.outputBinding.firstLayer);
            maxLayer = std::max(maxLayer, camera.outputBinding.firstLayer);
        }

        currentBatch.renderBinding = gpu::GpuRenderTargetBinding{
            currentBatch.cameras.front().outputBinding.target, minLayer, maxLayer - minLayer + 1u};
        plan.cameraBatches.push_back(std::move(currentBatch));
        currentBatch = {};
        hasOpenBatch = false;
        usedLayers.clear();
    };

    for (ResolvedCameraView &camera : cameras)
    {
        const bool duplicateLayer = std::find(usedLayers.begin(), usedLayers.end(),
                                              camera.outputBinding.firstLayer) != usedLayers.end();
        const bool dedicatedBatch =
            requiresDedicatedBatch(camera) ||
            (hasOpenBatch && requiresDedicatedBatch(currentBatch.cameras.front()));
        const bool canJoin = hasOpenBatch && !dedicatedBatch &&
                             sameBatchCompatibility(currentBatch.cameras.front(), camera) &&
                             !duplicateLayer;
        if (!canJoin)
        {
            flushBatch();
            currentBatch.light            = light;
            currentBatch.renderTargetDesc = camera.outputTargetDesc;
            hasOpenBatch                  = true;
        }

        usedLayers.push_back(camera.outputBinding.firstLayer);
        currentBatch.cameras.push_back(std::move(camera));
    }

    flushBatch();
    return plan;
}

} // namespace cressim::neo::graphics::detail
