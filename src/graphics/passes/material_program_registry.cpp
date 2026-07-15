#include "graphics/passes/material_program_registry.h"

#include "common/logger.h"

#include <array>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace cressim::neo::graphics::detail
{

namespace
{

template <typename T>
void hashCombine(std::size_t &seed, T value)
{
    const auto hashed = std::hash<T>{}(value);
    seed ^= hashed + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

const char *passClassName(MainPassClass passClass)
{
    switch (passClass)
    {
    case MainPassClass::ForwardOpaque:
        return "ForwardOpaque";
    case MainPassClass::ForwardTransparent:
        return "ForwardTransparent";
    default:
        return "Unknown";
    }
}

void appendVariable(std::vector<Diligent::ShaderResourceVariableDesc> &vars,
                    Diligent::SHADER_TYPE shaderType, const char *name)
{
    vars.push_back(Diligent::ShaderResourceVariableDesc{
        shaderType, name, Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
}

Diligent::ShaderMacroArray buildFeatureMacros(std::array<Diligent::ShaderMacro, 12> &macros,
                                              MaterialProgramFamily programFamily,
                                              MaterialFeatureFlags featureFlags,
                                              IblQualityTier iblQualityTier)
{
    Diligent::Uint32 count = 0;
    macros[count++]        = Diligent::ShaderMacro{"MANUAL_LAYER_EXPORT", "1"};
    if (programFamily == MaterialProgramFamily::SoftBodyLit)
    {
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_PROGRAM_FAMILY_SOFT_BODY", "1"};
    }
    else if (programFamily == MaterialProgramFamily::CurveLit)
    {
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_PROGRAM_FAMILY_CURVE", "1"};
    }
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
    switch (iblQualityTier)
    {
    case IblQualityTier::Off:
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_IBL_OFF", "1"};
        break;
    case IblQualityTier::DiffuseOnly:
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_IBL_DIFFUSE_ONLY", "1"};
        break;
    case IblQualityTier::Full:
        macros[count++] = Diligent::ShaderMacro{"CRESSIM_IBL_FULL", "1"};
        break;
    default:
        break;
    }
    return Diligent::ShaderMacroArray{count > 0 ? macros.data() : nullptr, count};
}

std::vector<Diligent::ShaderResourceVariableDesc> buildResourceLayoutVariables(
    MaterialProgramFamily programFamily, IblQualityTier iblQualityTier)
{
    std::vector<Diligent::ShaderResourceVariableDesc> vars;
    vars.reserve(26u);
    appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_EntityPositions");
    appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_EntityOrientations");
    appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_EntityScales");
    appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_RenderableMetadata");
    appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_RenderableVisibilityFlags");
    appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_VisiblePairs");
    appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras");
    appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_PreparedCameras");
    if (programFamily == MaterialProgramFamily::SoftBodyLit)
    {
        appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_SoftBodyRenderPositions");
        appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_SoftBodyVertexNormals");
    }
    else if (programFamily == MaterialProgramFamily::CurveLit)
    {
        appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_CurveRenderPositions");
        appendVariable(vars, Diligent::SHADER_TYPE_VERTEX, "g_CurveRenderNormals");
    }
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_LightInputs");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_LocalLightSelections");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_LightShadowAssignments");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_LocalShadowViews");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_NormalTexture");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_BaseColorTexture");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_MetallicRoughnessTexture");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_EmissiveTexture");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_AoTexture");

    if (iblQualityTier == IblQualityTier::DiffuseOnly || iblQualityTier == IblQualityTier::Full)
    {
        appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_IrradianceMap");
        appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_EnvironmentIblLookup");
    }
    if (iblQualityTier == IblQualityTier::Full)
    {
        appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_PrefilteredSpecularMap");
        appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_BrdfLut");
    }

    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap0");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap1");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap2");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap3");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_LocalShadowMap");
    appendVariable(vars, Diligent::SHADER_TYPE_PIXEL, "g_PointShadowMap");
    return vars;
}

std::string buildVariantSuffix(const MaterialProgramRegistry::ProgramKey &key)
{
    std::ostringstream stream;
    stream << "pass_" << static_cast<std::uint32_t>(key.passClass) << "_family_"
           << static_cast<std::uint32_t>(key.programFamily) << "_features_"
           << static_cast<std::uint32_t>(key.featureFlags) << "_ibl_"
           << static_cast<std::uint32_t>(key.iblQualityTier) << "_rt_"
           << static_cast<std::uint32_t>(key.colorFormat) << "_ds_"
           << static_cast<std::uint32_t>(key.depthFormat) << "_de_"
           << static_cast<std::uint32_t>(key.depthEnable ? 1u : 0u) << "_dw_"
           << static_cast<std::uint32_t>(key.depthWrite ? 1u : 0u) << "_blend_"
           << static_cast<std::uint32_t>(key.blendingEnabled ? 1u : 0u);
    return stream.str();
}

} // namespace

std::size_t MaterialProgramRegistry::ProgramKeyHasher::operator()(
    const ProgramKey &key) const noexcept
{
    std::size_t seed = 0;
    hashCombine(seed, static_cast<std::uint32_t>(key.passClass));
    hashCombine(seed, static_cast<std::uint32_t>(key.programFamily));
    hashCombine(seed, static_cast<std::uint32_t>(key.featureFlags));
    hashCombine(seed, static_cast<std::uint32_t>(key.iblQualityTier));
    hashCombine(seed, static_cast<std::uint32_t>(key.colorFormat));
    hashCombine(seed, static_cast<std::uint32_t>(key.depthFormat));
    hashCombine(seed, key.depthEnable);
    hashCombine(seed, key.depthWrite);
    hashCombine(seed, key.blendingEnabled);
    return seed;
}

MaterialProgramRegistry::MaterialProgramRegistry(gpu::GpuDevice &device,
                                                 gpu::ShaderSourceProvider &shaderSourceProvider)
    : mDevice(device), mShaderSourceProvider(shaderSourceProvider)
{
}

MaterialProgramRegistry::ProgramResources *MaterialProgramRegistry::getOrCreateProgram(
    const ProgramKey &key)
{
    if (key.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return nullptr;
    }

    auto [it, inserted] = mPrograms.try_emplace(key);
    if (!inserted)
    {
        if (it->second.pipelineState != nullptr)
        {
            return &it->second;
        }
    }

    if (!createProgram(key, it->second))
    {
        mPrograms.erase(it);
        return nullptr;
    }

    return &it->second;
}

MaterialProgramRegistry::ProgramKey MaterialProgramRegistry::buildProgramKey(
    MainPassClass passClass, MaterialProgramFamily programFamily, MaterialFeatureFlags featureFlags,
    IblQualityTier iblQualityTier, Diligent::TEXTURE_FORMAT colorFormat,
    Diligent::TEXTURE_FORMAT depthFormat, bool depthEnable, bool depthWrite,
    bool blendingEnabled) noexcept
{
    ProgramKey key{};
    key.passClass       = passClass;
    key.programFamily   = programFamily;
    key.featureFlags    = featureFlags;
    key.iblQualityTier  = iblQualityTier;
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

bool MaterialProgramRegistry::createProgram(const ProgramKey &key, ProgramResources &outResources)
{
    if ((key.passClass != MainPassClass::ForwardOpaque &&
         key.passClass != MainPassClass::ForwardTransparent) ||
        (key.programFamily != MaterialProgramFamily::StandardLit &&
         key.programFamily != MaterialProgramFamily::SoftBodyLit &&
         key.programFamily != MaterialProgramFamily::CurveLit))
    {
        return false;
    }

    constexpr const char *kPbrVsRelativePath = "graphics/pbr.vs.hlsl";
    constexpr const char *kPbrPsRelativePath = "graphics/pbr.ps.hlsl";

    std::string pbrVsPath;
    if (!mShaderSourceProvider.resolveShaderPath(kPbrVsRelativePath, pbrVsPath))
    {
        CRESSIM_LOG_ERROR("MaterialProgramRegistry failed to resolve vertex shader path for pass=",
                          passClassName(key.passClass), " relative='", kPbrVsRelativePath, "'.");
        return false;
    }

    std::string pbrPsPath;
    if (!mShaderSourceProvider.resolveShaderPath(kPbrPsRelativePath, pbrPsPath))
    {
        CRESSIM_LOG_ERROR("MaterialProgramRegistry failed to resolve pixel shader path for pass=",
                          passClassName(key.passClass), " relative='", kPbrPsRelativePath, "'.");
        return false;
    }

    Diligent::IShaderSourceInputStreamFactory *streamFactory = mShaderSourceProvider.streamFactory();
    if (streamFactory == nullptr)
    {
        CRESSIM_LOG_ERROR("MaterialProgramRegistry failed to get stream factory for pass=",
                          passClassName(key.passClass), ".");
        return false;
    }

    std::array<Diligent::ShaderMacro, 12> macros{};
    const Diligent::ShaderMacroArray macroArray =
        buildFeatureMacros(macros, key.programFamily, key.featureFlags, key.iblQualityTier);
    const std::string variantSuffix = buildVariantSuffix(key);
    const std::string passName      = passClassName(key.passClass);
    outResources.vertexShaderName   = "CRESSimNeo." + passName + ".StandardLit.VS." + variantSuffix;
    outResources.pixelShaderName    = "CRESSimNeo." + passName + ".StandardLit.PS." + variantSuffix;
    outResources.pipelineStateName = "CRESSimNeo." + passName + ".StandardLit.PSO." + variantSuffix;

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Macros                          = macroArray;
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name       = outResources.vertexShaderName.c_str();
    shaderCreateInfo.FilePath        = kPbrVsRelativePath;
    shaderCreateInfo.Source          = nullptr;
    if (!mDevice.createShader(shaderCreateInfo, &vertexShader))
    {
        vertexShader = nullptr;
    }
    if (vertexShader == nullptr)
    {
        CRESSIM_LOG_ERROR(
            "MaterialProgramRegistry failed to compile VS. pass=", passClassName(key.passClass),
            " programFamily=", static_cast<std::uint32_t>(key.programFamily),
            " featureFlags=", static_cast<std::uint32_t>(key.featureFlags), " shader='", pbrVsPath,
            "'.");
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name       = outResources.pixelShaderName.c_str();
    shaderCreateInfo.FilePath        = kPbrPsRelativePath;
    shaderCreateInfo.Source          = nullptr;
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        pixelShader = nullptr;
    }
    if (pixelShader == nullptr)
    {
        CRESSIM_LOG_ERROR(
            "MaterialProgramRegistry failed to compile PS. pass=", passClassName(key.passClass),
            " programFamily=", static_cast<std::uint32_t>(key.programFamily),
            " featureFlags=", static_cast<std::uint32_t>(key.featureFlags), " shader='", pbrPsPath,
            "'.");
        return false;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                      = outResources.pipelineStateName.c_str();
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

    auto &blendDesc       = psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0];
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
    const std::vector<Diligent::ShaderResourceVariableDesc> vars =
        buildResourceLayoutVariables(key.programFamily, key.iblQualityTier);
    psoCreateInfo.PSODesc.ResourceLayout.Variables    = vars.data();
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables = static_cast<Diligent::Uint32>(vars.size());

    // Fixed vertex layout: position, normal, uv, tangent.
    constexpr Diligent::LayoutElement kLayoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{3, 0, 4, Diligent::VT_FLOAT32, Diligent::False}};
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = kLayoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements    = 4;
    psoCreateInfo.pVS                                         = vertexShader;
    psoCreateInfo.pPS                                         = pixelShader;

    if (!mDevice.createGraphicsPipelineState(psoCreateInfo, &outResources.pipelineState))
    {
        outResources.pipelineState = nullptr;
    }
    if (outResources.pipelineState == nullptr)
    {
        CRESSIM_LOG_ERROR(
            "MaterialProgramRegistry failed to create PSO. pass=", passClassName(key.passClass),
            " programFamily=", static_cast<std::uint32_t>(key.programFamily),
            " featureFlags=", static_cast<std::uint32_t>(key.featureFlags), ".");
        return false;
    }

    return true;
}

} // namespace cressim::neo::graphics::detail
