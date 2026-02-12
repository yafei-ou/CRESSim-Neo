#include "graphics/graphics_device.h"

#include "graphics/device/diligent_graphics_device.h"

#include <memory>

namespace cressim::neo::graphics
{

std::unique_ptr<IGraphicsDevice> createGraphicsDevice()
{
    return std::make_unique<DiligentGraphicsDevice>();
}

} // namespace cressim::neo::graphics
