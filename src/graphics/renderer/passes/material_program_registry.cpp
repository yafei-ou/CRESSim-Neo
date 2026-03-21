#include "graphics/renderer/passes/material_program_registry.h"

#include "DiligentEngine/DiligentCore/Primitives/interface/Errors.hpp"

#include <array>
#include <functional>
#include <string>

namespace cressim::neo::graphics::detail
{

namespace
{

template <typename T>
void hashCombine(std::size_t& seed, T value)
{
    const auto hashed = std::hash<T>{}(value);
    seed ^= hashed + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

const char* passClassName(MainPassClass passClass)
{
    switch (passClass)
    {
    case MainPassClass::ForwardOpaque:
        return "ForwardOpaque";
    default:
        return "Unknown";
    }
}

Diligent::ShaderMacroArray buildFeatureMacros(std::array<Diligent::ShaderMacro, 5>& macros,
                                              MaterialFeatureFlags featureFlags)
{
    Diligent::Uint32 count = 0;
    macros[count++] = Diligent::ShaderMacro{"MANUAL_LAYER_EXPORT", "1"};
    if (hasFlag(featureFlags, MaterialFeatureFlags::AlphaTest))
    {
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_FEATURE_ALPHA_TEST", "1"};
    }
    if (hasFlag(featureFlags, MaterialFeatureFlags::NormalMap))
    {
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_FEATURE_NORMAL_MAP", "1"};
    }
    if (hasFlag(featureFlags, MaterialFeatureFlags::ClearCoat))
    {
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_FEATURE_CLEAR_COAT", "1"};
    }
    if (hasFlag(featureFlags, MaterialFeatureFlags::DoubleSided))
    {
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_FEATURE_DOUBLE_SIDED", "1"};
    }
    return Diligent::ShaderMacroArray{count > 0 ? macros.data() : nullptr, count};
}

} // namespace

std::size_t MaterialProgramRegistry::ProgramKeyHasher::operator()(
    const ProgramKey& key) const noexcept
{
    std::size_t seed = 0;
    hashCombine(seed, static_cast<std::uint32_t>(key.passClass));
    hashCombine(seed, static_cast<std::uint32_t>(key.programFamily));
    hashCombine(seed, static_cast<std::uint32_t>(key.featureFlags));
    hashCombine(seed, static_cast<std::uint32_t>(key.colorFormat));
    hashCombine(seed, static_cast<std::uint32_t>(key.depthFormat));
    hashCombine(seed, key.depthEnable);
    hashCombine(seed, key.depthWrite);
    hashCombine(seed, key.blendingEnabled);
    return seed;
}

MaterialProgramRegistry::MaterialProgramRegistry(gpu::ShaderLibrary& shaderSourceProvider)
    : mShaderLibrary(shaderSourceProvider)
{
}

MaterialProgramRegistry::ProgramResources* MaterialProgramRegistry::getOrCreateProgram(
    Diligent::IRenderDevice* renderDevice, const ProgramKey& key)
{
    if (renderDevice == nullptr || key.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return nullptr;
    }

    auto it = mPrograms.find(key);
    if (it != mPrograms.end())
    {
        if (it->second.pipelineState != nullptr)
        {
            return &it->second;
        }
    }

    ProgramResources resources{};
    if (!createProgram(renderDevice, key, resources))
    {
        return nullptr;
    }

    auto insertResult = mPrograms.emplace(key, std::move(resources));
    return &insertResult.first->second;
}

MaterialProgramRegistry::ProgramKey MaterialProgramRegistry::buildProgramKey(
    MainPassClass passClass, MaterialProgramFamily programFamily, MaterialFeatureFlags featureFlags,
    Diligent::TEXTURE_FORMAT colorFormat, Diligent::TEXTURE_FORMAT depthFormat, bool depthEnable,
    bool depthWrite, bool blendingEnabled) noexcept
{
    ProgramKey key{};
    key.passClass       = passClass;
    key.programFamily   = programFamily;
    key.featureFlags    = featureFlags;
    key.colorFormat     = colorFormat;
    key.depthFormat     = depthFormat;
    key.depthEnable     = depthEnable;
    key.depthWrite      = depthWrite;
    key.blendingEnabled = blendingEnabled;
    return key;
}

std::size_t MaterialProgramRegistry::cachedProgramCount() const noexcept
{
    return mPrograms.size();
}

bool MaterialProgramRegistry::createProgram(Diligent::IRenderDevice* renderDevice,
                                            const ProgramKey& key, ProgramResources& outResources)
{
    if (renderDevice == nullptr || key.passClass != MainPassClass::ForwardOpaque ||
        key.programFamily != MaterialProgramFamily::StandardLit)
    {
        return false;
    }

    constexpr const char* kPbrVsRelativePath = "graphics/pbr.vs.hlsl";
    constexpr const char* kPbrPsRelativePath = "graphics/pbr.ps.hlsl";

    std::string pbrVsPath;
    if (!mShaderLibrary.resolveShaderPath(kPbrVsRelativePath, pbrVsPath))
    {
        LOG_ERROR_MESSAGE("MaterialProgramRegistry failed to resolve vertex shader path for pass=",
                          passClassName(key.passClass), " relative='", kPbrVsRelativePath, "'.");
        return false;
    }

    std::string pbrPsPath;
    if (!mShaderLibrary.resolveShaderPath(kPbrPsRelativePath, pbrPsPath))
    {
        LOG_ERROR_MESSAGE("MaterialProgramRegistry failed to resolve pixel shader path for pass=",
                          passClassName(key.passClass), " relative='", kPbrPsRelativePath, "'.");
        return false;
    }

    Diligent::IShaderSourceInputStreamFactory* streamFactory = mShaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        LOG_ERROR_MESSAGE("MaterialProgramRegistry failed to get stream factory for pass=",
                          passClassName(key.passClass), ".");
        return false;
    }

    std::array<Diligent::ShaderMacro, 5> macros{};
    const Diligent::ShaderMacroArray macroArray = buildFeatureMacros(macros, key.featureFlags);

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Macros                          = macroArray;
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.ForwardOpaque.StandardLit.VS";
    shaderCreateInfo.FilePath        = kPbrVsRelativePath;
    shaderCreateInfo.Source          = nullptr;
    renderDevice->CreateShader(shaderCreateInfo, &vertexShader);
    if (vertexShader == nullptr)
    {
        LOG_ERROR_MESSAGE(
            "MaterialProgramRegistry failed to compile VS. pass=", passClassName(key.passClass),
            " programFamily=", static_cast<std::uint32_t>(key.programFamily),
            " featureFlags=", static_cast<std::uint32_t>(key.featureFlags), " shader='", pbrVsPath,
            "'.");
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.ForwardOpaque.StandardLit.PS";
    shaderCreateInfo.FilePath        = kPbrPsRelativePath;
    shaderCreateInfo.Source          = nullptr;
    renderDevice->CreateShader(shaderCreateInfo, &pixelShader);
    if (pixelShader == nullptr)
    {
        LOG_ERROR_MESSAGE(
            "MaterialProgramRegistry failed to compile PS. pass=", passClassName(key.passClass),
            " programFamily=", static_cast<std::uint32_t>(key.programFamily),
            " featureFlags=", static_cast<std::uint32_t>(key.featureFlags), " shader='", pbrPsPath,
            "'.");
        return false;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                      = "CRESSimNeo.ForwardOpaque.StandardLit.PSO";
    psoCreateInfo.PSODesc.PipelineType              = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]    = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat =
        key.depthEnable ? key.depthFormat : Diligent::TEX_FORMAT_UNKNOWN;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode =
        hasFlag(key.featureFlags, MaterialFeatureFlags::DoubleSided) ? Diligent::CULL_MODE_NONE
                                                                     : Diligent::CULL_MODE_BACK;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable =
        key.depthEnable ? Diligent::True : Diligent::False;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable =
        key.depthWrite ? Diligent::True : Diligent::False;

    auto& blendDesc       = psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0];
    blendDesc.BlendEnable = key.blendingEnabled ? Diligent::True : Diligent::False;
    if (key.blendingEnabled)
    {
        blendDesc.SrcBlend       = Diligent::BLEND_FACTOR_SRC_ALPHA;
        blendDesc.DestBlend      = Diligent::BLEND_FACTOR_INV_SRC_ALPHA;
        blendDesc.BlendOp        = Diligent::BLEND_OPERATION_ADD;
        blendDesc.SrcBlendAlpha  = Diligent::BLEND_FACTOR_ONE;
        blendDesc.DestBlendAlpha = Diligent::BLEND_FACTOR_INV_SRC_ALPHA;
        blendDesc.BlendOpAlpha   = Diligent::BLEND_OPERATION_ADD;
    }

    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_VERTEX, "g_EntityPositions",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_EntityOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_EntityScales",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_RenderableMetadata",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_RenderableVisibilityFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_VisiblePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap0",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap1",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap2",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap3",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
    psoCreateInfo.PSODesc.ResourceLayout.Variables = kVars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(std::size(kVars));

    // Fixed vertex layout for this milestone: position, normal, uv.
    constexpr Diligent::LayoutElement kLayoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False}};
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = kLayoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements    = 3;
    psoCreateInfo.pVS                                         = vertexShader;
    psoCreateInfo.pPS                                         = pixelShader;

    renderDevice->CreateGraphicsPipelineState(psoCreateInfo, &outResources.pipelineState);
    if (outResources.pipelineState == nullptr)
    {
        LOG_ERROR_MESSAGE(
            "MaterialProgramRegistry failed to create PSO. pass=", passClassName(key.passClass),
            " programFamily=", static_cast<std::uint32_t>(key.programFamily),
            " featureFlags=", static_cast<std::uint32_t>(key.featureFlags), ".");
        return false;
    }

    return true;
}

} // namespace cressim::neo::graphics::detail
