#include "gpu/shader_library.h"

#include "common/logger.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/include/DefaultShaderSourceStreamFactory.h"

#include <filesystem>
#include <utility>
#include <vector>

namespace cressim::neo::gpu
{

ShaderLibrary::ShaderLibrary(std::string shaderDirectory)
    : mShaderDirectory(std::move(shaderDirectory))
{
}

bool ShaderLibrary::resolveShaderDirectory()
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

bool ShaderLibrary::resolveShaderPath(const char* relativePath, std::string& outPath)
{
    if (relativePath == nullptr || relativePath[0] == '\0')
    {
        CRESSIM_LOG_ERROR("Shader path resolution failed: empty shader relative path.");
        return false;
    }
    if (!resolveShaderDirectory())
    {
        CRESSIM_LOG_ERROR("Shader path resolution failed for '", relativePath,
                          "': shader directory could not be resolved.");
        return false;
    }

    const std::filesystem::path shaderPath =
        std::filesystem::path(mResolvedShaderDirectory) / relativePath;
    std::error_code error;
    if (!std::filesystem::exists(shaderPath, error) ||
        !std::filesystem::is_regular_file(shaderPath, error))
    {
        CRESSIM_LOG_ERROR("Shader file not found: relative='", relativePath, "' resolved='",
                          shaderPath.lexically_normal().string(), "'");
        return false;
    }

    outPath = shaderPath.lexically_normal().string();
    return true;
}

Diligent::IShaderSourceInputStreamFactory* ShaderLibrary::streamFactory()
{
    if (!ensureStreamFactory())
    {
        return nullptr;
    }
    return mStreamFactory;
}

bool ShaderLibrary::ensureStreamFactory()
{
    if (mStreamFactory != nullptr)
    {
        return true;
    }
    if (!resolveShaderDirectory())
    {
        CRESSIM_LOG_ERROR("Failed to create shader source stream factory: shader directory could "
                          "not be resolved.");
        return false;
    }

    Diligent::CreateDefaultShaderSourceStreamFactory(mResolvedShaderDirectory.c_str(),
                                                     &mStreamFactory);
    if (mStreamFactory == nullptr)
    {
        CRESSIM_LOG_ERROR("Failed to create shader source stream factory for directory '",
                          mResolvedShaderDirectory, "'.");
        return false;
    }
    return true;
}

} // namespace cressim::neo::gpu
