#include "gpu/shader_source_provider.h"

#include "common/logger.h"

#include "DiligentEngine/DiligentCore/Common/interface/ObjectBase.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCountedObjectImpl.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/include/DefaultShaderSourceStreamFactory.h"

#include <filesystem>
#include <utility>
#include <vector>

namespace cressim::neo::gpu
{

namespace
{

class NormalizingShaderSourceFactory final
    : public Diligent::ObjectBase<Diligent::IShaderSourceInputStreamFactory>
{
public:
    using TBase = Diligent::ObjectBase<Diligent::IShaderSourceInputStreamFactory>;

    static Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> Create(
        Diligent::IShaderSourceInputStreamFactory *sourceFactory,
        std::vector<Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory>>
            includeFactories)
    {
        return Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory>{
            Diligent::MakeNewRCObj<NormalizingShaderSourceFactory>()(sourceFactory,
                                                                     std::move(includeFactories))};
    }

    NormalizingShaderSourceFactory(
        Diligent::IReferenceCounters *refCounters,
        Diligent::IShaderSourceInputStreamFactory *sourceFactory,
        std::vector<Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory>>
            includeFactories)
        : TBase{refCounters}, mSourceFactory{sourceFactory},
          mIncludeFactories{std::move(includeFactories)}
    {
    }

    void DILIGENT_CALL_TYPE QueryInterface(const Diligent::INTERFACE_ID &IID,
                                           Diligent::IObject **ppInterface) override
    {
        if (ppInterface == nullptr)
        {
            return;
        }
        if (IID == Diligent::IID_IShaderSourceInputStreamFactory)
        {
            *ppInterface = this;
            (*ppInterface)->AddRef();
            return;
        }
        TBase::QueryInterface(IID, ppInterface);
    }

    using Diligent::IObject::QueryInterface;

    void DILIGENT_CALL_TYPE CreateInputStream(const Diligent::Char *name,
                                              Diligent::IFileStream **ppStream) override final
    {
        CreateInputStream2(name, Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_NONE, ppStream);
    }

    void DILIGENT_CALL_TYPE CreateInputStream2(
        const Diligent::Char *name, Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAGS flags,
        Diligent::IFileStream **ppStream) override final
    {
        if (ppStream == nullptr)
        {
            return;
        }

        *ppStream = nullptr;
        if (name == nullptr || name[0] == '\0' || mSourceFactory == nullptr)
        {
            return;
        }

        const std::string requestedName = normalizeSeparators(name);
        if (!isSafeRelativePath(requestedName))
        {
            if ((flags & Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT) == 0)
            {
                LOG_ERROR_MESSAGE("Rejected shader source path outside configured roots: ", name);
            }
            return;
        }

        mSourceFactory->CreateInputStream2(requestedName.c_str(),
                                           Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT,
                                           ppStream);
        if (*ppStream == nullptr)
        {
            for (const auto &includeFactory : mIncludeFactories)
            {
                includeFactory->CreateInputStream2(
                    requestedName.c_str(), Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT,
                    ppStream);
                if (*ppStream != nullptr)
                {
                    break;
                }
            }
        }

        if (*ppStream == nullptr &&
            (flags & Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT) == 0)
        {
            LOG_ERROR_MESSAGE("Failed to create input stream for source file ", name);
        }
    }

private:
    static std::string normalizeSeparators(std::string value)
    {
        for (char &ch : value)
        {
            if (ch == '\\')
            {
                ch = '/';
            }
        }
        return value;
    }

    static bool isSafeRelativePath(const std::string &value)
    {
        const std::filesystem::path path{value};
        if (path.empty() || path.is_absolute())
        {
            return false;
        }
        for (const auto &part : path)
        {
            if (part == "..")
            {
                return false;
            }
        }
        return true;
    }

private:
    Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> mSourceFactory;
    std::vector<Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory>>
        mIncludeFactories;
};

} // namespace

struct ShaderSourceProvider::Impl
{
    explicit Impl(ShaderSourceConfig configIn) : config{std::move(configIn)} {}

    bool resolveConfig();
    bool ensureStreamFactory();
    Diligent::IShaderSourceInputStreamFactory *streamFactory();

    ShaderSourceConfig config;
    bool configResolved = false;
    std::filesystem::path resolvedSourceDirectory;
    std::vector<std::filesystem::path> resolvedIncludeDirectories;
    Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> streamFactoryInstance;
};

ShaderSourceProvider::ShaderSourceProvider(ShaderSourceConfig config)
    : mImpl{std::make_unique<Impl>(std::move(config))}
{
}

ShaderSourceProvider::ShaderSourceProvider(std::string shaderDirectory)
    : ShaderSourceProvider{
          ShaderSourceConfig{std::filesystem::path{std::move(shaderDirectory)}, true, {}}}
{
}

ShaderSourceProvider::~ShaderSourceProvider() = default;

ShaderSourceProvider::ShaderSourceProvider(ShaderSourceProvider &&) noexcept = default;

ShaderSourceProvider &ShaderSourceProvider::operator=(ShaderSourceProvider &&) noexcept = default;

bool ShaderSourceProvider::Impl::resolveConfig()
{
    if (configResolved)
    {
        return !resolvedSourceDirectory.empty();
    }

    configResolved = true;

    if (!config.sourceDirectory.empty())
    {
        std::error_code error;
        if (!std::filesystem::is_directory(config.sourceDirectory, error))
        {
            CRESSIM_LOG_ERROR("Shader source directory does not exist: '",
                              config.sourceDirectory.string(), "'.");
            return false;
        }
        resolvedSourceDirectory = config.sourceDirectory.lexically_normal();
    }
    else
    {
        std::vector<std::filesystem::path> candidates;
        candidates.emplace_back("shaders");

#ifdef CRESSIM_NEO_SHADER_INSTALL_DIR
        candidates.emplace_back(CRESSIM_NEO_SHADER_INSTALL_DIR);
#endif

        for (const auto &candidate : candidates)
        {
            std::error_code error;
            if (!std::filesystem::is_directory(candidate, error))
            {
                continue;
            }

            resolvedSourceDirectory = candidate.lexically_normal();
            break;
        }
    }
    if (resolvedSourceDirectory.empty())
    {
        return false;
    }

    if (config.includeSourceDirectory)
    {
        resolvedIncludeDirectories.push_back(resolvedSourceDirectory / "include");
    }
    resolvedIncludeDirectories.insert(resolvedIncludeDirectories.end(),
                                      config.includeDirectories.begin(),
                                      config.includeDirectories.end());
    for (const auto &directory : resolvedIncludeDirectories)
    {
        std::error_code error;
        if (!std::filesystem::is_directory(directory, error))
        {
            CRESSIM_LOG_ERROR("Shader include directory does not exist: '", directory.string(),
                              "'.");
            resolvedSourceDirectory.clear();
            return false;
        }
    }
    return true;
}

Diligent::IShaderSourceInputStreamFactory *ShaderSourceProvider::Impl::streamFactory()
{
    if (!ensureStreamFactory())
    {
        return nullptr;
    }
    return streamFactoryInstance;
}

bool ShaderSourceProvider::Impl::ensureStreamFactory()
{
    if (streamFactoryInstance != nullptr)
    {
        return true;
    }
    if (!resolveConfig())
    {
        CRESSIM_LOG_ERROR("Failed to create shader source stream factory: shader directory could "
                          "not be resolved.");
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> sourceFactory;
    Diligent::CreateDefaultShaderSourceStreamFactory(resolvedSourceDirectory.string().c_str(),
                                                     &sourceFactory);
    if (sourceFactory == nullptr)
    {
        CRESSIM_LOG_ERROR("Failed to create shader source stream factory for directory '",
                          resolvedSourceDirectory.string(), "'.");
        return false;
    }
    std::vector<Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory>>
        includeFactories;
    includeFactories.reserve(resolvedIncludeDirectories.size());
    for (const auto &directory : resolvedIncludeDirectories)
    {
        Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> includeFactory;
        Diligent::CreateDefaultShaderSourceStreamFactory(directory.string().c_str(),
                                                         &includeFactory);
        if (includeFactory == nullptr)
        {
            CRESSIM_LOG_ERROR("Failed to create shader include stream factory for directory '",
                              directory.string(), "'.");
            return false;
        }
        includeFactories.push_back(std::move(includeFactory));
    }
    streamFactoryInstance =
        NormalizingShaderSourceFactory::Create(sourceFactory, std::move(includeFactories));
    return true;
}

Diligent::IShaderSourceInputStreamFactory *ShaderSourceProvider::streamFactory()
{
    return mImpl->streamFactory();
}

std::filesystem::path ShaderSourceProvider::sourceDirectory()
{
    if (!mImpl->resolveConfig())
    {
        return {};
    }
    return mImpl->resolvedSourceDirectory;
}

} // namespace cressim::neo::gpu
