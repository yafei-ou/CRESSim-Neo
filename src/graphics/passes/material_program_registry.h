#ifndef CRESSIM_NEO_GRAPHICS_PASSES_MATERIAL_PROGRAM_REGISTRY_H
#define CRESSIM_NEO_GRAPHICS_PASSES_MATERIAL_PROGRAM_REGISTRY_H

#include "gpu/shader_library.h"
#include "graphics/passes/render_pass_types.h"
#include "graphics/render_resource_manager.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#include <unordered_map>

namespace cressim::neo::graphics::detail
{

class MaterialProgramRegistry
{
public:
    struct ProgramKey
    {
        MainPassClass passClass              = MainPassClass::ForwardOpaque;
        MaterialProgramFamily programFamily  = MaterialProgramFamily::StandardLit;
        MaterialFeatureFlags featureFlags    = MaterialFeatureFlags::None;
        IblQualityTier iblQualityTier        = IblQualityTier::Off;
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;
        bool depthEnable                     = true;
        bool depthWrite                      = true;
        bool blendingEnabled                 = false;

        bool operator==(const ProgramKey &rhs) const noexcept
        {
            return passClass == rhs.passClass && programFamily == rhs.programFamily &&
                   featureFlags == rhs.featureFlags && iblQualityTier == rhs.iblQualityTier &&
                   colorFormat == rhs.colorFormat && depthFormat == rhs.depthFormat &&
                   depthEnable == rhs.depthEnable && depthWrite == rhs.depthWrite &&
                   blendingEnabled == rhs.blendingEnabled;
        }
    };

    struct ProgramKeyHasher
    {
        std::size_t operator()(const ProgramKey &key) const noexcept;
    };

    struct ProgramResources
    {
        Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipelineState;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> shaderResourceBinding;
    };

public:
    explicit MaterialProgramRegistry(gpu::ShaderLibrary &shaderSourceProvider);

    ProgramResources *getOrCreateProgram(Diligent::IRenderDevice *renderDevice,
                                         const ProgramKey &key);

    static ProgramKey buildProgramKey(MainPassClass passClass, MaterialProgramFamily programFamily,
                                      MaterialFeatureFlags featureFlags,
                                      IblQualityTier iblQualityTier,
                                      Diligent::TEXTURE_FORMAT colorFormat,
                                      Diligent::TEXTURE_FORMAT depthFormat, bool depthEnable,
                                      bool depthWrite, bool blendingEnabled) noexcept;

    std::size_t cachedProgramCount() const noexcept;

private:
    bool createProgram(Diligent::IRenderDevice *renderDevice, const ProgramKey &key,
                       ProgramResources &outResources);

private:
    gpu::ShaderLibrary &mShaderLibrary;
    std::unordered_map<ProgramKey, ProgramResources, ProgramKeyHasher> mPrograms;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_PASSES_MATERIAL_PROGRAM_REGISTRY_H
