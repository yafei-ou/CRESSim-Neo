#pragma once

#include <cstdlib>
#include <filesystem>

#ifndef CRESSIM_NEO_EXAMPLE_ASSET_DIR
#error "CRESSIM_NEO_EXAMPLE_ASSET_DIR must be defined by the example build."
#endif

namespace cressim::neo::examples::helpers
{

inline std::filesystem::path assetRoot()
{
    if (const char *overridePath = std::getenv("CRESSIM_NEO_ASSET_DIR"); overridePath != nullptr &&
        overridePath[0] != '\0')
    {
        return overridePath;
    }
    return CRESSIM_NEO_EXAMPLE_ASSET_DIR;
}

inline std::filesystem::path assetPath(const std::filesystem::path &relativePath)
{
    return assetRoot() / relativePath;
}

} // namespace cressim::neo::examples::helpers
