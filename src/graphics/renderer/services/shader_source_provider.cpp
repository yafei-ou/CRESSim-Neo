#include "graphics/renderer/services/shader_source_provider.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

namespace cressim::neo::graphics::detail
{

ShaderSourceProvider::ShaderSourceProvider(std::string shaderDirectory, bool allowFallback) :
    mShaderDirectory(std::move(shaderDirectory)),
    mAllowFallback(allowFallback)
{
}

bool ShaderSourceProvider::resolveShaderDirectory()
{
    if (mShaderDirectoryResolved)
    {
        return !mResolvedShaderDirectory.empty();
    }

    mShaderDirectoryResolved = true;

    std::vector<std::filesystem::path> candidates;
    if (!mShaderDirectory.empty())
    {
        candidates.emplace_back(mShaderDirectory);
    }

#ifdef CRESSIM_NEO_SHADER_SOURCE_DIR
    candidates.emplace_back(CRESSIM_NEO_SHADER_SOURCE_DIR);
#endif

    candidates.emplace_back("shaders");

    for (const auto& candidate : candidates)
    {
        std::error_code error;
        if (!std::filesystem::exists(candidate, error))
        {
            continue;
        }
        if (!std::filesystem::is_directory(candidate, error))
        {
            continue;
        }

        mResolvedShaderDirectory = candidate.lexically_normal().string();
        return true;
    }

    return false;
}

bool ShaderSourceProvider::loadSource(const char* relativePath, const char* fallbackSource, std::string& outSource)
{
    if (relativePath == nullptr || relativePath[0] == '\0')
    {
        return false;
    }

    const std::string shaderKey = relativePath;
    const auto cachedIt = mShaderSourceCache.find(shaderKey);
    if (cachedIt != mShaderSourceCache.end())
    {
        outSource = cachedIt->second;
        return true;
    }

    if (resolveShaderDirectory())
    {
        const std::filesystem::path shaderPath = std::filesystem::path(mResolvedShaderDirectory) / relativePath;
        std::ifstream shaderFile(shaderPath, std::ios::binary);
        if (shaderFile.is_open())
        {
            std::string fileSource{
                std::istreambuf_iterator<char>{shaderFile},
                std::istreambuf_iterator<char>{}};
            if (!fileSource.empty())
            {
                mShaderSourceCache.emplace(shaderKey, fileSource);
                outSource = std::move(fileSource);
                return true;
            }
        }
    }

    if (mAllowFallback && fallbackSource != nullptr)
    {
        outSource = fallbackSource;
        return true;
    }

    return false;
}

} // namespace cressim::neo::graphics::detail
