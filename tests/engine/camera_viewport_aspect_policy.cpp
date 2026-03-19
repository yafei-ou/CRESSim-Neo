#include "common/math_utils_runtime.h"

#include <cmath>
#include <iostream>

int main()
{
    using cressim::neo::common::runtime_math::effectiveViewportAspect;
    using cressim::neo::gpu::GpuRenderViewport;

    const float full = effectiveViewportAspect(1600.0f, 900.0f, GpuRenderViewport{0.0f, 0.0f, 1.0f, 1.0f});
    if (std::fabs(full - (1600.0f / 900.0f)) > 1e-5f)
    {
        std::cerr << "Fullscreen viewport aspect mismatch.\n";
        return 1;
    }

    const float halfWidth =
        effectiveViewportAspect(1600.0f, 900.0f, GpuRenderViewport{0.0f, 0.0f, 0.5f, 1.0f});
    if (std::fabs(halfWidth - (800.0f / 900.0f)) > 1e-5f)
    {
        std::cerr << "Half-width viewport aspect mismatch.\n";
        return 1;
    }

    const float quadTile =
        effectiveViewportAspect(1600.0f, 900.0f, GpuRenderViewport{0.0f, 0.0f, 0.5f, 0.5f});
    if (std::fabs(quadTile - (800.0f / 450.0f)) > 1e-5f)
    {
        std::cerr << "2x2 tiled viewport aspect mismatch.\n";
        return 1;
    }

    const float tallTile =
        effectiveViewportAspect(1920.0f, 1080.0f, GpuRenderViewport{0.25f, 0.0f, 0.25f, 1.0f});
    if (std::fabs(tallTile - (480.0f / 1080.0f)) > 1e-5f)
    {
        std::cerr << "Non-square tiled viewport aspect mismatch.\n";
        return 1;
    }

    std::cout << "Camera viewport aspect policy checks passed.\n";
    return 0;
}
