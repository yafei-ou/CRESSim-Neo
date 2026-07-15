#include "gpu/shader_source_provider.h"

#include "common/logger.h"

#include "DiligentEngine/DiligentCore/Common/interface/ObjectBase.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCountedObjectImpl.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/include/DefaultShaderSourceStreamFactory.h"

#include <algorithm>
#include <filesystem>
#include <set>
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
        Diligent::IShaderSourceInputStreamFactory *baseFactory, std::string shaderRoot)
    {
        return Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory>{
            Diligent::MakeNewRCObj<NormalizingShaderSourceFactory>()(baseFactory,
                                                                     std::move(shaderRoot))};
    }

    NormalizingShaderSourceFactory(Diligent::IReferenceCounters *refCounters,
                                   Diligent::IShaderSourceInputStreamFactory *baseFactory,
                                   std::string shaderRoot)
        : TBase{refCounters}, mBaseFactory{baseFactory}, mShaderRoot{std::move(shaderRoot)}
    {
        buildSearchBases();
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
        if (name == nullptr || name[0] == '\0' || mBaseFactory == nullptr)
        {
            return;
        }

        mBaseFactory->CreateInputStream2(
            name, Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT, ppStream);
        if (*ppStream != nullptr)
        {
            return;
        }

        const std::string requestedName = normalizeSeparators(name);
        const bool isRelativeNestedInclude =
            requestedName.rfind("./", 0) == 0 || requestedName.rfind("../", 0) == 0;
        const bool isLocalHeaderInclude = requestedName.find('/') == std::string::npos &&
                                          requestedName.find('\\') == std::string::npos;
        if (!isRelativeNestedInclude && !isLocalHeaderInclude)
        {
            if ((flags & Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT) == 0)
            {
                LOG_ERROR_MESSAGE("Failed to create input stream for source file ", name);
            }
            return;
        }

        const std::filesystem::path requestedPath{requestedName};
        std::set<std::string> triedCandidates;

        for (const auto &base : mSearchBases)
        {
            const std::filesystem::path candidatePath = (base / requestedPath).lexically_normal();
            if (candidatePath.empty() || candidatePath.is_absolute())
            {
                continue;
            }

            bool escapesRoot = false;
            for (const auto &part : candidatePath)
            {
                if (part == "..")
                {
                    escapesRoot = true;
                    break;
                }
            }
            if (escapesRoot)
            {
                continue;
            }

            const std::string candidate = normalizeSeparators(candidatePath.string());
            if (!triedCandidates.insert(candidate).second)
            {
                continue;
            }

            mBaseFactory->CreateInputStream2(
                candidate.c_str(), Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT,
                ppStream);
            if (*ppStream != nullptr)
            {
                return;
            }
        }

        if ((flags & Diligent::CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT) == 0)
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

    void buildSearchBases()
    {
        mSearchBases.clear();
        mSearchBases.emplace_back();

        const std::filesystem::path includeRoot = std::filesystem::path(mShaderRoot) / "include";
        std::error_code error;
        if (!std::filesystem::exists(includeRoot, error) ||
            !std::filesystem::is_directory(includeRoot, error))
        {
            return;
        }

        std::vector<std::filesystem::path> discoveredBases;
        discoveredBases.emplace_back("include");
        for (std::filesystem::recursive_directory_iterator it{includeRoot, error}, end;
             !error && it != end; it.increment(error))
        {
            if (!it->is_directory())
            {
                continue;
            }

            const std::filesystem::path relative =
                std::filesystem::relative(it->path(), mShaderRoot, error);
            if (error || relative.empty())
            {
                error.clear();
                continue;
            }
            discoveredBases.push_back(relative);
        }

        std::sort(discoveredBases.begin(), discoveredBases.end(),
                  [](const std::filesystem::path &lhs, const std::filesystem::path &rhs)
                  { return lhs.native().size() > rhs.native().size(); });
        discoveredBases.erase(std::unique(discoveredBases.begin(), discoveredBases.end()),
                              discoveredBases.end());
        mSearchBases.insert(mSearchBases.end(), discoveredBases.begin(), discoveredBases.end());
    }

private:
    Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> mBaseFactory;
    std::string mShaderRoot;
    std::vector<std::filesystem::path> mSearchBases;
};

} // namespace

struct ShaderSourceProvider::Impl
{
    explicit Impl(std::string shaderDirectoryIn) : shaderDirectory{std::move(shaderDirectoryIn)} {}

    bool resolveShaderDirectory();
    bool ensureStreamFactory();
    bool resolveShaderPath(const char *relativePath, std::string &outPath);
    Diligent::IShaderSourceInputStreamFactory *streamFactory();

    std::string shaderDirectory;
    bool shaderDirectoryResolved = false;
    std::string resolvedShaderDirectory;
    Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> streamFactoryInstance;
};

ShaderSourceProvider::ShaderSourceProvider(std::string shaderDirectory)
    : mImpl{std::make_unique<Impl>(std::move(shaderDirectory))}
{
}

ShaderSourceProvider::~ShaderSourceProvider() = default;

ShaderSourceProvider::ShaderSourceProvider(ShaderSourceProvider &&) noexcept = default;

ShaderSourceProvider &ShaderSourceProvider::operator=(ShaderSourceProvider &&) noexcept = default;

bool ShaderSourceProvider::Impl::resolveShaderDirectory()
{
    if (shaderDirectoryResolved)
    {
        return !resolvedShaderDirectory.empty();
    }

    shaderDirectoryResolved = true;

    std::vector<std::filesystem::path> candidates;
    if (!shaderDirectory.empty())
    {
        candidates.emplace_back(shaderDirectory);
    }

#ifdef CRESSIM_NEO_SHADER_SOURCE_DIR
    candidates.emplace_back(CRESSIM_NEO_SHADER_SOURCE_DIR);
#endif

    candidates.emplace_back("shaders");

    for (const auto &candidate : candidates)
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

        resolvedShaderDirectory = candidate.lexically_normal().string();
        return true;
    }

    return false;
}

bool ShaderSourceProvider::Impl::resolveShaderPath(const char *relativePath, std::string &outPath)
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
        std::filesystem::path(resolvedShaderDirectory) / relativePath;
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
    if (!resolveShaderDirectory())
    {
        CRESSIM_LOG_ERROR("Failed to create shader source stream factory: shader directory could "
                          "not be resolved.");
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderSourceInputStreamFactory> baseFactory;
    Diligent::CreateDefaultShaderSourceStreamFactory(resolvedShaderDirectory.c_str(), &baseFactory);
    if (baseFactory == nullptr)
    {
        CRESSIM_LOG_ERROR("Failed to create shader source stream factory for directory '",
                          resolvedShaderDirectory, "'.");
        return false;
    }
    streamFactoryInstance =
        NormalizingShaderSourceFactory::Create(baseFactory, resolvedShaderDirectory);
    return true;
}

bool ShaderSourceProvider::resolveShaderPath(const char *relativePath, std::string &outPath)
{
    return mImpl->resolveShaderPath(relativePath, outPath);
}

Diligent::IShaderSourceInputStreamFactory *ShaderSourceProvider::streamFactory()
{
    return mImpl->streamFactory();
}

} // namespace cressim::neo::gpu
