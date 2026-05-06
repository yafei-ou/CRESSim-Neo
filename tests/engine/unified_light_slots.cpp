#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/world.h"
#include "graphics/render_resource_manager.h"

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::PointLightComponent;
using cressim::neo::engine::World;
using cressim::neo::engine::SpotLightComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::graphics::GpuLightType;
using cressim::neo::common::SceneLayoutDesc;
using cressim::neo::graphics::kForwardLocalLightCap;
using cressim::neo::graphics::kMainDirectionalLightSlot;
using cressim::neo::graphics::RenderResourceManager;

} // namespace

int main()
{
    constexpr std::uint32_t kEnv0InitialPointLightCount = kForwardLocalLightCap + 1u;
    constexpr std::uint32_t kEnv0InitialLocalLightCount = kEnv0InitialPointLightCount + 3u;

    World world;
    RenderResourceManager resources;
    SceneLayoutDesc layout{};
    // This fixture intentionally authors more local lights than the forward shading cap to verify
    // deterministic truncation, mixed-type rejection, and environment moves independently of
    // storage capacity.
    layout.maxLightsPerEnv = 1u + kEnv0InitialLocalLightCount;
    layout.envCount = 3u;
    world.setSceneLayout(layout);

    const auto mainLightEntity = world.createEntity();
    DirectionalLightComponent mainLight{};
    mainLight.direction    = {0.1f, -1.0f, 0.2f};
    mainLight.castsShadows = true;
    world.setDirectionalLight(mainLightEntity, mainLight);

    for (std::uint32_t i = 0u; i < kEnv0InitialPointLightCount; ++i)
    {
        const auto entity = world.createEntity();
        TransformComponent transform{};
        transform.worldTransform.position = {static_cast<float>(i + 1u), 0.5f, 2.0f};
        world.setTransform(entity, transform);

        PointLightComponent point{};
        point.range        = 5.0f + static_cast<float>(i);
        point.intensity    = 2.0f + static_cast<float>(i);
        point.castsShadows = (i == 0u);
        world.setPointLight(entity, point);
    }

    const auto spotEntity = world.createEntity();
    TransformComponent spotTransform{};
    spotTransform.worldTransform.position = {10.0f, 4.0f, 1.0f};
    world.setTransform(spotEntity, spotTransform);
    SpotLightComponent spot{};
    spot.direction      = {0.0f, -1.0f, 0.0f};
    spot.range          = 12.0f;
    spot.innerConeAngle = 20.0f;
    spot.outerConeAngle = 30.0f;
    spot.castsShadows   = true;
    world.setSpotLight(spotEntity, spot);
    spot.range = 16.0f;
    world.setSpotLight(spotEntity, spot);

    const auto conflictingLightEntity = world.createEntity();
    TransformComponent conflictingTransform{};
    conflictingTransform.worldTransform.position = {-3.0f, 1.5f, 8.0f};
    world.setTransform(conflictingLightEntity, conflictingTransform);
    PointLightComponent conflictingPoint{};
    conflictingPoint.range        = 7.0f;
    conflictingPoint.intensity    = 4.0f;
    conflictingPoint.castsShadows = true;
    world.setPointLight(conflictingLightEntity, conflictingPoint);

    SpotLightComponent conflictingSpot{};
    conflictingSpot.direction      = {1.0f, -1.0f, 0.0f};
    conflictingSpot.range          = 20.0f;
    conflictingSpot.innerConeAngle = 10.0f;
    conflictingSpot.outerConeAngle = 15.0f;
    conflictingSpot.castsShadows   = false;
    world.setSpotLight(conflictingLightEntity, conflictingSpot);

    const auto movablePointEntity = world.createEntity();
    TransformComponent movablePointTransform{};
    movablePointTransform.worldTransform.position = {13.0f, 2.0f, -4.0f};
    world.setTransform(movablePointEntity, movablePointTransform);
    PointLightComponent movablePoint{};
    movablePoint.range        = 11.0f;
    movablePoint.intensity    = 2.5f;
    movablePoint.castsShadows = false;
    world.setPointLight(movablePointEntity, movablePoint);
    world.setEntityEnvironment(movablePointEntity, 1u);

    const auto env1MainLightEntity = world.createEntity(1u);
    DirectionalLightComponent env1MainLight{};
    env1MainLight.direction = {-0.3f, -1.0f, 0.1f};
    env1MainLight.castsShadows = true;
    world.setDirectionalLight(env1MainLightEntity, env1MainLight);

    const auto env1PointEntity = world.createEntity(1u);
    TransformComponent env1PointTransform{};
    env1PointTransform.worldTransform.position = {42.0f, 3.0f, -7.0f};
    world.setTransform(env1PointEntity, env1PointTransform);
    PointLightComponent env1Point{};
    env1Point.range = 9.0f;
    env1Point.intensity = 3.5f;
    env1Point.castsShadows = true;
    world.setPointLight(env1PointEntity, env1Point);

    const auto env2PointFirstEntity = world.createEntity(2u);
    TransformComponent env2PointFirstTransform{};
    env2PointFirstTransform.worldTransform.position = {55.0f, 2.0f, 4.0f};
    world.setTransform(env2PointFirstEntity, env2PointFirstTransform);
    PointLightComponent env2PointFirst{};
    env2PointFirst.range = 6.0f;
    env2PointFirst.intensity = 1.5f;
    env2PointFirst.castsShadows = false;
    world.setPointLight(env2PointFirstEntity, env2PointFirst);

    const auto env2MainEntity = world.createEntity(2u);
    DirectionalLightComponent env2Main{};
    env2Main.direction = {0.25f, -1.0f, -0.15f};
    env2Main.castsShadows = false;
    world.setDirectionalLight(env2MainEntity, env2Main);

    world.ensureRenderStateUpToDate(resources);

    const auto &lights = world.lights();
    if (lights.empty() || lights[0].lightSlot != kMainDirectionalLightSlot ||
        lights[0].type != GpuLightType::Directional)
    {
        CRESSIM_LOG_ERROR("Expected slot 0 to remain reserved for the main directional light.");
        return 1;
    }

    const auto &selections = world.localLightSelections();
    if (selections.empty())
    {
        CRESSIM_LOG_ERROR("Expected local light selections to be populated.");
        return 1;
    }

    const auto &selection = selections[0];
    if (selection.localLightCount != kForwardLocalLightCap)
    {
        CRESSIM_LOG_ERROR("Expected deterministic truncation to the forward local light cap.");
        return 1;
    }
    if (selection.shadowedPointLightCount != 1u || selection.shadowedLocalLightCount != 1u)
    {
        CRESSIM_LOG_ERROR("Expected deterministic bounded shadow-light bookkeeping per environment.");
        return 1;
    }

    for (std::uint32_t i = 0u; i < selection.localLightCount; ++i)
    {
        if (lights[selection.lightIndices[i]].lightSlot != i + 1u)
        {
            CRESSIM_LOG_ERROR("Expected local lights to preserve ascending slot order.");
            return 1;
        }
    }

    if (selections.size() < 2u || selections[1].localLightCount != 2u)
    {
        CRESSIM_LOG_ERROR("Expected environment-local light selection state for env 1.");
        return 1;
    }

    const auto &lightInputs = world.lightInputs();
    const std::uint32_t env1MainLightIndex = layout.maxLightsPerEnv * 1u + kMainDirectionalLightSlot;
    std::uint32_t env1PointLightIndex       = layout.totalLightCapacity();
    for (std::uint32_t i = 0u; i < selections[1].localLightCount; ++i)
    {
        const std::uint32_t candidateIndex = selections[1].lightIndices[i];
        if (candidateIndex >= lightInputs.size())
        {
            continue;
        }
        if (lightInputs[candidateIndex].type != static_cast<std::uint32_t>(GpuLightType::Point))
        {
            continue;
        }
        if (std::abs(lightInputs[candidateIndex].positionRange.x - 42.0f) > 1.0e-4f ||
            std::abs(lightInputs[candidateIndex].positionRange.y - 3.0f) > 1.0e-4f ||
            std::abs(lightInputs[candidateIndex].positionRange.z + 7.0f) > 1.0e-4f)
        {
            continue;
        }
        env1PointLightIndex = candidateIndex;
        break;
    }
    if (env1MainLightIndex >= lightInputs.size() || env1PointLightIndex >= lightInputs.size())
    {
        CRESSIM_LOG_ERROR("Expected dense env-scoped light indices to stay within bounds.");
        return 1;
    }
    if (lightInputs[env1MainLightIndex].active == 0u ||
        lightInputs[env1MainLightIndex].lightSlot != kMainDirectionalLightSlot ||
        lightInputs[env1MainLightIndex].envIndex != 1u)
    {
        CRESSIM_LOG_ERROR("Expected env 1 main directional light at slot 0.");
        return 1;
    }
    if (lightInputs[env1PointLightIndex].type != static_cast<std::uint32_t>(GpuLightType::Point) ||
        std::abs(lightInputs[env1PointLightIndex].positionRange.x - 42.0f) > 1.0e-4f ||
        std::abs(lightInputs[env1PointLightIndex].positionRange.y - 3.0f) > 1.0e-4f ||
        std::abs(lightInputs[env1PointLightIndex].positionRange.z + 7.0f) > 1.0e-4f)
    {
        CRESSIM_LOG_ERROR("Expected env 1 point light position to refresh from its transform.");
        return 1;
    }

    const auto &lightsAfter = world.lights();
    const std::uint32_t env2BaseIndex = layout.maxLightsPerEnv * 2u;
    if (env2BaseIndex + 1u >= lightsAfter.size())
    {
        CRESSIM_LOG_ERROR("Expected env 2 light storage to be within bounds.");
        return 1;
    }
    if (lightsAfter[env2BaseIndex].type != GpuLightType::Directional ||
        lightsAfter[env2BaseIndex].lightSlot != kMainDirectionalLightSlot ||
        lightsAfter[env2BaseIndex + 1u].type != GpuLightType::Point ||
        lightsAfter[env2BaseIndex + 1u].lightSlot != 1u)
    {
        CRESSIM_LOG_ERROR("Expected slot 0 to remain reserved for a later-created main directional light.");
        return 1;
    }

    if (!world.tryGetSpotLight(spotEntity).has_value() || world.tryGetPointLight(spotEntity).has_value())
    {
        CRESSIM_LOG_ERROR("Expected typed light getters to respect the unified light type.");
        return 1;
    }
    if (std::abs(world.tryGetSpotLight(spotEntity)->range - 16.0f) > 1.0e-4f)
    {
        CRESSIM_LOG_ERROR("Expected setting the same light type twice to update in place.");
        return 1;
    }
    if (!world.tryGetPointLight(conflictingLightEntity).has_value() ||
        world.tryGetSpotLight(conflictingLightEntity).has_value())
    {
        CRESSIM_LOG_ERROR("Expected mixed light assignment to leave the original light type unchanged.");
        return 1;
    }
    if (std::abs(world.tryGetPointLight(conflictingLightEntity)->range - conflictingPoint.range) > 1.0e-4f)
    {
        CRESSIM_LOG_ERROR("Expected rejected mixed light assignment to preserve the original light data.");
        return 1;
    }
    if (!world.tryGetPointLight(movablePointEntity).has_value())
    {
        CRESSIM_LOG_ERROR("Expected point lights to survive environment moves.");
        return 1;
    }
    if (!world.removePointLight(movablePointEntity) || world.tryGetPointLight(movablePointEntity).has_value())
    {
        CRESSIM_LOG_ERROR("Expected point lights to remain removable after environment moves.");
        return 1;
    }

    return 0;
}
