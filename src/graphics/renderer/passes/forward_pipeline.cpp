#include "graphics/renderer/passes/forward_pipeline.h"

#include "graphics/renderer/passes/pbr_pass.h"

namespace cressim::neo::graphics::detail
{

ForwardPipeline::ForwardPipeline(GraphicsDeviceImpl& device) :
    mDevice(device)
{
}

ForwardPipeline::~ForwardPipeline() = default;

bool ForwardPipeline::initialize()
{
    mPbrPass = std::make_unique<PbrPass>(mDevice);
    if (!mPbrPass->initialize())
    {
        mPbrPass.reset();
        return false;
    }

    mInitialized = true;
    return true;
}

bool ForwardPipeline::draw(RenderTargetHandle target, const ForwardDrawCommand& drawCommand)
{
    if (!mInitialized || mPbrPass == nullptr)
    {
        return false;
    }

    switch (drawCommand.shadingModel)
    {
    case ForwardShadingModel::Pbr:
        return mPbrPass->draw(target, drawCommand);
    case ForwardShadingModel::Phong:
    case ForwardShadingModel::BlinnPhong:
        // Reserved for upcoming forward shading pass variants.
        return false;
    default:
        return false;
    }
}

} // namespace cressim::neo::graphics::detail
