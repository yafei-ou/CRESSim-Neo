#include "graphics/renderer/renderer_internal.h"

namespace cressim::neo::graphics::detail
{

bool isVisibleByFrustum(const PreparedRenderable& renderable, const Diligent::ViewFrustum& frustum)
{
    if (renderable.instance == nullptr)
    {
        return false;
    }

    if (!renderable.hasWorldBounds)
    {
        return true;
    }

    const Diligent::BoxVisibility visibility = Diligent::GetBoxVisibility(frustum, renderable.worldBounds);
    return visibility != Diligent::BoxVisibility::Invisible;
}

} // namespace cressim::neo::graphics::detail
