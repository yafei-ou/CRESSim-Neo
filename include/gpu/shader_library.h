#ifndef CRESSIM_NEO_GPU_SHADER_LIBRARY_H
#define CRESSIM_NEO_GPU_SHADER_LIBRARY_H

#include "gpu/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <string>

namespace cressim::neo::gpu
{

class CRESSIM_NEO_GPU_API ShaderLibrary
{
public:
    explicit ShaderLibrary(std::string shaderDirectory);

    bool resolveShaderPath(const char *relativePath, std::string &outPath);
    Diligent::IShaderSourceInputStreamFactory *streamFactory();

private:
    bool resolveShaderDirectory();
    bool ensureStreamFactory();

private:
    std::string mShaderDirectory;
    bool mShaderDirectoryResolved = false;
    std::string mResolvedShaderDirectory;
    Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> mStreamFactory;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_SHADER_LIBRARY_H
