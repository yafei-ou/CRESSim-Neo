#ifndef CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H
#define CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H

#include "gpu/export.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <memory>
#include <string>

namespace cressim::neo::gpu
{

class CRESSIM_NEO_GPU_API ShaderSourceProvider
{
public:
    explicit ShaderSourceProvider(std::string shaderDirectory);
    ~ShaderSourceProvider();

    ShaderSourceProvider(ShaderSourceProvider &&) noexcept;
    ShaderSourceProvider &operator=(ShaderSourceProvider &&) noexcept;

    ShaderSourceProvider(const ShaderSourceProvider &)            = delete;
    ShaderSourceProvider &operator=(const ShaderSourceProvider &) = delete;

    bool resolveShaderPath(const char *relativePath, std::string &outPath);
    Diligent::IShaderSourceInputStreamFactory *streamFactory();

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H
