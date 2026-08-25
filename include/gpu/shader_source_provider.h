#ifndef CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H
#define CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H

#include "gpu/export.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

/// @file shader_source_provider.h
/// @brief Shader file resolution, include directory path management, and stream factory
/// provisioning for HLSL compilation.

namespace cressim::neo::gpu
{

/// @brief Configuration settings controlling shader source file paths and `#include` search roots.
struct ShaderSourceConfig
{
    std::filesystem::path sourceDirectory; ///< Root directory containing entry-point shader files.
    bool includeSourceDirectory = true; ///< Automatically adds `<sourceDirectory>/include` and its
                                        ///< subdirectories to include paths.
    std::vector<std::filesystem::path>
        includeDirectories; ///< Additional search directories checked in order when resolving
                            ///< shader header includes.
};

/// @brief Service providing filesystem-backed input stream factories for Diligent shader
/// compilation.
class CRESSIM_NEO_GPU_API ShaderSourceProvider
{
public:
    /// @brief Constructs a shader source provider using full path configuration.
    /// @param config Shader source and include directory configuration.
    explicit ShaderSourceProvider(ShaderSourceConfig config);
    /// @brief Compatibility shorthand constructor configuring `<shaderDirectory>/include` as the
    /// include search root.
    /// @param shaderDirectory Path string to base shader directory.
    explicit ShaderSourceProvider(std::string shaderDirectory);
    /// @brief Destructor.
    ~ShaderSourceProvider();

    /// @brief Move constructor.
    /// @param other Source provider to move from.
    ShaderSourceProvider(ShaderSourceProvider &&other) noexcept;
    /// @brief Move assignment operator.
    /// @param other Source provider to move from.
    /// @return Reference to this.
    ShaderSourceProvider &operator=(ShaderSourceProvider &&other) noexcept;

    ShaderSourceProvider(const ShaderSourceProvider &)            = delete;
    ShaderSourceProvider &operator=(const ShaderSourceProvider &) = delete;

    /// @brief Retrieves the Diligent input stream factory for compiling shaders with nested
    /// includes.
    /// @return Pointer to Diligent::IShaderSourceInputStreamFactory.
    Diligent::IShaderSourceInputStreamFactory *streamFactory();
    /// @brief Gets the root directory from which entry-point shaders are loaded.
    /// @return Filesystem path to shader source directory.
    std::filesystem::path sourceDirectory();

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_SHADER_SOURCE_PROVIDER_H
