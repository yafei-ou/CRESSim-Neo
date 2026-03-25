#include "common/logger.h"
#include "engine/world.h"

namespace
{

using cressim::neo::engine::World;
using cressim::neo::gpu::GpuSceneLayoutDesc;
using cressim::neo::graphics::EnvironmentIblDesc;
using cressim::neo::graphics::IblQualityTier;
using cressim::neo::graphics::TextureHandle;

} // namespace

int main()
{
    World world;
    GpuSceneLayoutDesc layout{};
    layout.envCount = 2u;
    world.setSceneLayout(layout);

    EnvironmentIblDesc envA{};
    envA.irradianceCubemap = TextureHandle{11u};
    envA.prefilteredSpecularCubemap = TextureHandle{12u};
    envA.intensity = 1.5f;

    EnvironmentIblDesc envB{};
    envB.irradianceCubemap = TextureHandle{21u};
    envB.prefilteredSpecularCubemap = TextureHandle{22u};
    envB.intensity = 0.75f;

    if (!world.setEnvironmentIbl(0u, envA) || !world.setEnvironmentIbl(1u, envB))
    {
        CRESSIM_LOG_ERROR("Failed to assign environment IBL state.\n");
        return 1;
    }

    const EnvironmentIblDesc *storedA = world.tryGetEnvironmentIbl(0u);
    const EnvironmentIblDesc *storedB = world.tryGetEnvironmentIbl(1u);
    if (storedA == nullptr || storedB == nullptr || !storedA->enabled(IblQualityTier::Full) ||
        !storedB->enabled(IblQualityTier::Full))
    {
        CRESSIM_LOG_ERROR("Environment IBL state could not be retrieved.\n");
        return 1;
    }

    const auto hostScene = world.hostSceneView();
    if (hostScene.environmentIbls == nullptr || hostScene.environmentIbls->size() != 2u)
    {
        CRESSIM_LOG_ERROR("Host scene did not expose per-environment IBL state.\n");
        return 1;
    }

    if ((*hostScene.environmentIbls)[0].irradianceCubemap.id != envA.irradianceCubemap.id ||
        (*hostScene.environmentIbls)[1].prefilteredSpecularCubemap.id != envB.prefilteredSpecularCubemap.id ||
        (*hostScene.environmentIbls)[1].intensity != envB.intensity)
    {
        CRESSIM_LOG_ERROR("Host scene environment IBL contents did not match assignments.\n");
        return 1;
    }

    EnvironmentIblDesc irradianceOnly{};
    irradianceOnly.irradianceCubemap = TextureHandle{31u};
    if (!irradianceOnly.enabled(IblQualityTier::DiffuseOnly) ||
        irradianceOnly.enabled(IblQualityTier::Off) ||
        irradianceOnly.enabled(IblQualityTier::Full))
    {
        CRESSIM_LOG_ERROR("Environment IBL tier-dependent enabled() semantics were incorrect.\n");
        return 1;
    }

    if (world.setEnvironmentIbl(2u, envA) || world.tryGetEnvironmentIbl(2u) != nullptr)
    {
        CRESSIM_LOG_ERROR("Out-of-range environment IBL access did not fail safely.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Environment IBL state checks passed.\n");
    return 0;
}
