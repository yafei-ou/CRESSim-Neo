#include "gpu/shader_cache.h"

#include "common/logger.h"

#include "DiligentEngine/DiligentCore/Common/interface/DataBlobImpl.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/FileWrapper.hpp"
#include "DiligentEngine/DiligentCore/Graphics/Archiver/interface/ArchiverFactoryLoader.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#include <system_error>

namespace cressim::neo::gpu
{

namespace
{

constexpr Diligent::Uint32 kShaderCacheContentVersion = 1u;

} // namespace

bool ShaderCache::initialize(Diligent::IRenderDevice *renderDevice)
{
    shutdown();
    if (renderDevice == nullptr)
    {
        return false;
    }

    Diligent::IArchiverFactory *archiverFactory = Diligent::LoadAndGetArchiverFactory();
    if (archiverFactory == nullptr)
    {
        CRESSIM_LOG_ERROR("failed to acquire Diligent archiver factory.");
        return true;
    }

    Diligent::RenderStateCacheCreateInfo createInfo{};
    createInfo.pDevice          = renderDevice;
    createInfo.pArchiverFactory = archiverFactory;
    createInfo.FileHashMode     = Diligent::RENDER_STATE_CACHE_FILE_HASH_MODE_BY_CONTENT;
    createInfo.EnableHotReload  = false;

    Diligent::CreateRenderStateCache(createInfo, &mStateCache);
    if (mStateCache == nullptr)
    {
        CRESSIM_LOG_ERROR("failed to create render state cache.");
        return true;
    }

    mCacheFilePath = makeCacheFilePath(renderDevice);
    CRESSIM_LOG_INFO("using cache file '", mCacheFilePath.string(), "'.");
    loadCache();
    return true;
}

void ShaderCache::shutdown()
{
    saveCache();
    mStateCache      = nullptr;
    mLoadedCacheBlob = nullptr;
    mCacheFilePath.clear();
}

bool ShaderCache::createShader(const Diligent::ShaderCreateInfo &createInfo,
                               Diligent::IShader **shader)
{
    if (shader == nullptr)
    {
        return false;
    }
    *shader = nullptr;

    if (mStateCache != nullptr)
    {
        mStateCache->CreateShader(createInfo, shader);
        return *shader != nullptr;
    }

    return false;
}

bool ShaderCache::createGraphicsPipelineState(
    const Diligent::GraphicsPipelineStateCreateInfo &createInfo,
    Diligent::IPipelineState **pipelineState)
{
    if (pipelineState == nullptr)
    {
        return false;
    }
    *pipelineState = nullptr;

    if (mStateCache != nullptr)
    {
        mStateCache->CreateGraphicsPipelineState(createInfo, pipelineState);
        return *pipelineState != nullptr;
    }

    return false;
}

bool ShaderCache::createComputePipelineState(
    const Diligent::ComputePipelineStateCreateInfo &createInfo,
    Diligent::IPipelineState **pipelineState)
{
    if (pipelineState == nullptr)
    {
        return false;
    }
    *pipelineState = nullptr;

    if (mStateCache != nullptr)
    {
        mStateCache->CreateComputePipelineState(createInfo, pipelineState);
        return *pipelineState != nullptr;
    }

    return false;
}

void ShaderCache::loadCache()
{
    if (mStateCache == nullptr || mCacheFilePath.empty())
    {
        return;
    }

    mLoadedCacheBlob = nullptr;

    std::error_code error;
    if (!std::filesystem::exists(mCacheFilePath, error))
    {
        CRESSIM_LOG_INFO("cache file does not exist yet: '", mCacheFilePath.string(), "'.");
        return;
    }

    Diligent::FileWrapper cacheFile{mCacheFilePath.string().c_str()};
    if (!cacheFile)
    {
        CRESSIM_LOG_ERROR("failed to open cache file '", mCacheFilePath.string(), "'.");
        return;
    }

    Diligent::RefCntAutoPtr<Diligent::DataBlobImpl> cacheBlob = Diligent::DataBlobImpl::Create();
    if (!cacheFile->Read(cacheBlob))
    {
        CRESSIM_LOG_ERROR("failed to read cache file '", mCacheFilePath.string(), "'.");
        return;
    }

    // Work around Diligent archive deserialization relying on the original blob memory even when
    // MakeCopy=true by keeping the source blob alive and asking the cache to retain it directly.
    mLoadedCacheBlob = cacheBlob;
    if (!mStateCache->Load(mLoadedCacheBlob, kShaderCacheContentVersion, false))
    {
        CRESSIM_LOG_ERROR("failed to load cache file '", mCacheFilePath.string(), "'.");
        mLoadedCacheBlob = nullptr;
        return;
    }

    CRESSIM_LOG_INFO("loaded cache file '", mCacheFilePath.string(), "' (",
                     static_cast<std::uint64_t>(cacheBlob->GetSize()), " bytes).");
}

void ShaderCache::saveCache()
{
    if (mStateCache == nullptr || mCacheFilePath.empty())
    {
        return;
    }

    std::error_code error;
    std::filesystem::create_directories(mCacheFilePath.parent_path(), error);
    if (error)
    {
        CRESSIM_LOG_ERROR("failed to create cache directory '",
                          mCacheFilePath.parent_path().string(), "'.");
        return;
    }

    Diligent::RefCntAutoPtr<Diligent::IDataBlob> cacheBlob;
    const bool writeOk = mStateCache->WriteToBlob(kShaderCacheContentVersion, &cacheBlob);
    if (!writeOk)
    {
        CRESSIM_LOG_ERROR("WriteToBlob failed for cache file '", mCacheFilePath.string(), "'.");
        return;
    }
    if (cacheBlob == nullptr)
    {
        CRESSIM_LOG_ERROR("WriteToBlob succeeded but produced a null blob for cache "
                          "file '",
                          mCacheFilePath.string(), "'.");
        return;
    }

    if (!Diligent::FileWrapper::WriteFile(mCacheFilePath.string().c_str(),
                                          cacheBlob->GetConstDataPtr(), cacheBlob->GetSize(), true))
    {
        CRESSIM_LOG_ERROR("failed to write cache file '", mCacheFilePath.string(), "'.");
        return;
    }

    CRESSIM_LOG_INFO("saved cache file '", mCacheFilePath.string(), "' (",
                     static_cast<std::uint64_t>(cacheBlob->GetSize()), " bytes).");
}

std::filesystem::path ShaderCache::makeCacheFilePath(Diligent::IRenderDevice *renderDevice) const
{
    if (renderDevice == nullptr)
    {
        return {};
    }

    const Diligent::RENDER_DEVICE_TYPE deviceType = renderDevice->GetDeviceInfo().Type;
    std::string fileName                          = "state_cache_";
    fileName += Diligent::GetRenderDeviceTypeShortString(deviceType);
#ifdef DILIGENT_DEBUG
    fileName += "_d";
#else
    fileName += "_r";
#endif
    fileName += ".bin";

    // TODO: we shouldn't make the cache path dependent on the current launch path
    return std::filesystem::current_path() / ".cache" / "cressim_neo" / fileName;
}

} // namespace cressim::neo::gpu
