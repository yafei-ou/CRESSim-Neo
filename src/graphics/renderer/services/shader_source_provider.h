#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_SHADER_SOURCE_PROVIDER_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_SHADER_SOURCE_PROVIDER_H

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <string>

namespace cressim::neo::graphics::detail
{

class ShaderSourceProvider
{
public:
    explicit ShaderSourceProvider(std::string shaderDirectory);

    bool resolveShaderPath(const char* relativePath, std::string& outPath);
    Diligent::IShaderSourceInputStreamFactory* streamFactory();

private:
    bool resolveShaderDirectory();
    bool ensureStreamFactory();

private:
    std::string mShaderDirectory;
    bool mShaderDirectoryResolved = false;
    std::string mResolvedShaderDirectory;
    Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> mStreamFactory;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_SHADER_SOURCE_PROVIDER_H
