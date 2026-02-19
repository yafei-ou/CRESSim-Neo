#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_SHADER_SOURCE_PROVIDER_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_SHADER_SOURCE_PROVIDER_H

#include <string>
#include <unordered_map>

namespace cressim::neo::graphics::detail
{

class ShaderSourceProvider
{
public:
    ShaderSourceProvider(std::string shaderDirectory, bool allowFallback);

    bool loadSource(const char* relativePath, const char* fallbackSource, std::string& outSource);

private:
    bool resolveShaderDirectory();

private:
    std::string mShaderDirectory;
    bool mAllowFallback = true;
    bool mShaderDirectoryResolved = false;
    std::string mResolvedShaderDirectory;
    std::unordered_map<std::string, std::string> mShaderSourceCache;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_SHADER_SOURCE_PROVIDER_H
