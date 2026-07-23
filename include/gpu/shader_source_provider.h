#ifndef CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H
#define CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H

#include "gpu/export.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cressim::neo::gpu
{

struct ShaderSourceConfig
{
    // Entry-point shader paths are resolved relative to this directory.
    std::filesystem::path sourceDirectory;
    // Adds <sourceDirectory>/include and its child directories before includeDirectories when true.
    bool includeSourceDirectory = true;
    // Additional ordered include roots searched after the engine shader headers.
    std::vector<std::filesystem::path> includeDirectories;
};

class CRESSIM_NEO_GPU_API ShaderSourceProvider
{
public:
    // With no explicit source directory, the provider resolves shaders from
    // CRESSIM_NEO_ASSET_DIR/shaders or the asset directory beside its module.
    explicit ShaderSourceProvider(ShaderSourceConfig config);
    // Compatibility shorthand: uses <shaderDirectory>/include as the sole include root.
    explicit ShaderSourceProvider(std::string shaderDirectory);
    ~ShaderSourceProvider();

    ShaderSourceProvider(ShaderSourceProvider &&) noexcept;
    ShaderSourceProvider &operator=(ShaderSourceProvider &&) noexcept;

    ShaderSourceProvider(const ShaderSourceProvider &)            = delete;
    ShaderSourceProvider &operator=(const ShaderSourceProvider &) = delete;

    Diligent::IShaderSourceInputStreamFactory *streamFactory();
    std::filesystem::path sourceDirectory();

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H
