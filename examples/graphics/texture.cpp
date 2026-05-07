#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialFeatureFlags;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::TextureColorSpace;
using cressim::neo::graphics::TextureHandle;
using cressim::neo::graphics::TextureResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
}

void setPixel(std::vector<std::uint8_t> &pixels, std::uint32_t width, std::uint32_t x,
              std::uint32_t y, const std::array<std::uint8_t, 4> &rgba)
{
    const std::size_t offset =
        (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + x) * 4u;
    pixels[offset + 0u] = rgba[0];
    pixels[offset + 1u] = rgba[1];
    pixels[offset + 2u] = rgba[2];
    pixels[offset + 3u] = rgba[3];
}

TextureResourceDesc makeTextureDesc(const std::string &debugName, std::uint32_t width,
                                    std::uint32_t height, TextureColorSpace colorSpace,
                                    std::vector<std::uint8_t> pixels)
{
    TextureResourceDesc desc{};
    desc.debugName = debugName;
    desc.width = width;
    desc.height = height;
    desc.colorSpace = colorSpace;
    desc.pixelData = std::move(pixels);
    return desc;
}

TextureResourceDesc makeBaseColorCheckerTexture()
{
    constexpr std::uint32_t kSize = 4u;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4u, 255u);
    const std::array<std::array<std::uint8_t, 4>, 4> colors = {{
        {220u, 48u, 48u, 255u},
        {240u, 208u, 48u, 255u},
        {44u, 164u, 80u, 255u},
        {56u, 92u, 220u, 255u},
    }};

    for (std::uint32_t y = 0; y < kSize; ++y)
    {
        for (std::uint32_t x = 0; x < kSize; ++x)
        {
            const std::uint32_t colorIndex = (x / 2u) + 2u * (y / 2u);
            setPixel(pixels, kSize, x, y, colors[colorIndex]);
        }
    }

    return makeTextureDesc("TextureViewer.BaseColorChecker", kSize, kSize,
                           TextureColorSpace::Srgb, std::move(pixels));
}

TextureResourceDesc makeMetallicRoughnessTexture()
{
    constexpr std::uint32_t kSize = 4u;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4u, 0u);

    for (std::uint32_t y = 0; y < kSize; ++y)
    {
        for (std::uint32_t x = 0; x < kSize; ++x)
        {
            const std::uint8_t roughness =
                static_cast<std::uint8_t>(32u + y * 64u);
            const std::uint8_t metallic =
                static_cast<std::uint8_t>(x * 85u);
            setPixel(pixels, kSize, x, y, {0u, roughness, metallic, 255u});
        }
    }

    return makeTextureDesc("TextureViewer.MetallicRoughness", kSize, kSize,
                           TextureColorSpace::Linear, std::move(pixels));
}

TextureResourceDesc makeEmissiveTexture()
{
    constexpr std::uint32_t kSize = 4u;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4u, 0u);

    for (std::uint32_t y = 0; y < kSize; ++y)
    {
        for (std::uint32_t x = 0; x < kSize; ++x)
        {
            const bool on = (x == y) || (x + y == kSize - 1u);
            setPixel(pixels, kSize, x, y, on ? std::array<std::uint8_t, 4>{255u, 168u, 48u, 255u}
                                             : std::array<std::uint8_t, 4>{0u, 0u, 0u, 255u});
        }
    }

    return makeTextureDesc("TextureViewer.Emissive", kSize, kSize, TextureColorSpace::Srgb,
                           std::move(pixels));
}

TextureResourceDesc makeAoTexture()
{
    constexpr std::uint32_t kSize = 4u;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4u, 255u);

    for (std::uint32_t y = 0; y < kSize; ++y)
    {
        for (std::uint32_t x = 0; x < kSize; ++x)
        {
            const std::uint8_t ao =
                (x == 0u || x == kSize - 1u || y == 0u || y == kSize - 1u) ? 48u : 255u;
            setPixel(pixels, kSize, x, y, {ao, ao, ao, 255u});
        }
    }

    return makeTextureDesc("TextureViewer.AO", kSize, kSize, TextureColorSpace::Linear,
                           std::move(pixels));
}

TextureResourceDesc makeFlatNormalTexture()
{
    constexpr std::uint32_t kSize = 4u;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4u, 255u);
    for (std::uint32_t y = 0; y < kSize; ++y)
    {
        for (std::uint32_t x = 0; x < kSize; ++x)
        {
            setPixel(pixels, kSize, x, y, {128u, 128u, 255u, 255u});
        }
    }

    return makeTextureDesc("TextureViewer.FlatNormal", kSize, kSize, TextureColorSpace::Linear,
                           std::move(pixels));
}

TextureResourceDesc makePerturbedNormalTexture()
{
    constexpr std::uint32_t kSize = 4u;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4u, 255u);

    for (std::uint32_t y = 0; y < kSize; ++y)
    {
        for (std::uint32_t x = 0; x < kSize; ++x)
        {
            const std::uint8_t nx = static_cast<std::uint8_t>(48u + x * 48u);
            const std::uint8_t ny = static_cast<std::uint8_t>(208u - y * 40u);
            setPixel(pixels, kSize, x, y, {nx, ny, 220u, 255u});
        }
    }

    return makeTextureDesc("TextureViewer.PerturbedNormal", kSize, kSize,
                           TextureColorSpace::Linear, std::move(pixels));
}

TextureResourceDesc makeAlphaCutoutTexture()
{
    constexpr std::uint32_t kSize = 4u;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * kSize * 4u, 0u);

    for (std::uint32_t y = 0; y < kSize; ++y)
    {
        for (std::uint32_t x = 0; x < kSize; ++x)
        {
            const bool solid = (x == 1u || x == 2u || y == 1u || y == 2u);
            setPixel(pixels, kSize, x, y,
                     solid ? std::array<std::uint8_t, 4>{116u, 196u, 96u, 255u}
                           : std::array<std::uint8_t, 4>{32u, 72u, 24u, 0u});
        }
    }

    return makeTextureDesc("TextureViewer.AlphaCutout", kSize, kSize, TextureColorSpace::Srgb,
                           std::move(pixels));
}

void spawnRenderable(cressim::neo::engine::World &world, cressim::neo::graphics::MeshHandle mesh,
                     cressim::neo::graphics::MaterialHandle material,
                     const Diligent::float3 &position, const Diligent::float3 &scale,
                     const Diligent::QuaternionF &rotation = Diligent::QuaternionF{})
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale = scale;
    transform.worldTransform.rotation = rotation;
    world.setTransform(entity, transform);

    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(entity, renderer);
}

} // namespace

int main(int argc, char **argv)
{
    CommonExampleOptions options{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, false))
            {
                continue;
            }

            printUsage(argv[0]);
            return 2;
        }
    }
    catch (const std::invalid_argument &error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        printUsage(argv[0]);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);

    DebugViewerApp viewer;
    const auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(
        options, ViewerExampleDefaults{
                     .windowTitle = "CRESSim Neo Texture Validation Viewer",
                     .width = 1440u,
                     .height = 900u,
                     .showStats = true,
                     .vSync = false,
                     .startFullscreen = false,
                     .startFullscreenWindowed = true,
                 });

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        return 1;
    }

    auto &world = runtime.getWorld();
    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.6f, -10.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.55f, -1.0f, 0.25f};
    light.color = {1.0f, 0.98f, 0.95f};
    light.intensity = 7.5f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();

    const auto cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(0.8f, "TextureViewer.CubeMesh"));
    const auto wallPanelMesh =
        resources.registerMesh(
            cressim::neo::examples::helpers::makePlaneMesh(1.0f, 1.0f, "TextureViewer.PanelMesh"));
    const auto tiledPanelMesh =
        resources.registerMesh(cressim::neo::examples::helpers::makePlaneMesh(
            1.0f, 4.0f, "TextureViewer.TiledPanelMesh"));
    const auto groundMesh =
        resources.registerMesh(cressim::neo::examples::helpers::makePlaneMesh(
            12.0f, 12.0f, "TextureViewer.GroundMesh"));

    const TextureHandle baseColorTexture =
        resources.registerTexture(makeBaseColorCheckerTexture());
    const TextureHandle metallicRoughnessTexture =
        resources.registerTexture(makeMetallicRoughnessTexture());
    const TextureHandle emissiveTexture = resources.registerTexture(makeEmissiveTexture());
    const TextureHandle aoTexture = resources.registerTexture(makeAoTexture());
    const TextureHandle flatNormalTexture = resources.registerTexture(makeFlatNormalTexture());
    const TextureHandle perturbedNormalTexture =
        resources.registerTexture(makePerturbedNormalTexture());
    const TextureHandle alphaTexture = resources.registerTexture(makeAlphaCutoutTexture());

    MaterialResourceDesc groundMaterial{};
    groundMaterial.debugName = "TextureViewer.Ground";
    groundMaterial.baseColor = {0.74f, 0.75f, 0.78f};
    groundMaterial.roughness = 0.92f;
    const auto groundMaterialHandle = resources.registerMaterial(groundMaterial);

    MaterialResourceDesc fallbackMaterial{};
    fallbackMaterial.debugName = "TextureViewer.FallbackScalar";
    fallbackMaterial.baseColor = {0.85f, 0.85f, 0.85f};
    fallbackMaterial.metallic = 0.0f;
    fallbackMaterial.roughness = 0.55f;
    const auto fallbackMaterialHandle = resources.registerMaterial(fallbackMaterial);

    MaterialResourceDesc baseColorMaterial{};
    baseColorMaterial.debugName = "TextureViewer.BaseColor";
    baseColorMaterial.baseColorTexture = baseColorTexture;
    baseColorMaterial.metallic = 0.0f;
    baseColorMaterial.roughness = 0.6f;
    const auto baseColorMaterialHandle = resources.registerMaterial(baseColorMaterial);

    MaterialResourceDesc mrMaterial{};
    mrMaterial.debugName = "TextureViewer.MetallicRoughness";
    mrMaterial.baseColor = {0.92f, 0.92f, 0.92f};
    mrMaterial.baseColorTexture = baseColorTexture;
    mrMaterial.metallic = 1.0f;
    mrMaterial.roughness = 1.0f;
    mrMaterial.metallicRoughnessTexture = metallicRoughnessTexture;
    const auto mrMaterialHandle = resources.registerMaterial(mrMaterial);

    MaterialResourceDesc emissiveMaterial{};
    emissiveMaterial.debugName = "TextureViewer.Emissive";
    emissiveMaterial.baseColor = {0.08f, 0.08f, 0.08f};
    emissiveMaterial.baseColorTexture = baseColorTexture;
    emissiveMaterial.emissiveTexture = emissiveTexture;
    emissiveMaterial.emissiveFactor = {1.8f, 1.6f, 1.3f};
    emissiveMaterial.roughness = 0.5f;
    const auto emissiveMaterialHandle = resources.registerMaterial(emissiveMaterial);

    MaterialResourceDesc aoMaterial{};
    aoMaterial.debugName = "TextureViewer.AO";
    aoMaterial.baseColor = {0.78f, 0.80f, 0.84f};
    aoMaterial.baseColorTexture = baseColorTexture;
    aoMaterial.aoTexture = aoTexture;
    aoMaterial.roughness = 0.95f;
    const auto aoMaterialHandle = resources.registerMaterial(aoMaterial);

    MaterialResourceDesc flatNormalMaterial{};
    flatNormalMaterial.debugName = "TextureViewer.FlatNormal";
    flatNormalMaterial.baseColor = {0.80f, 0.80f, 0.84f};
    flatNormalMaterial.normalTexture = flatNormalTexture;
    flatNormalMaterial.roughness = 0.5f;
    const auto flatNormalMaterialHandle = resources.registerMaterial(flatNormalMaterial);

    MaterialResourceDesc perturbedNormalMaterial{};
    perturbedNormalMaterial.debugName = "TextureViewer.PerturbedNormal";
    perturbedNormalMaterial.baseColor = {0.80f, 0.80f, 0.84f};
    perturbedNormalMaterial.normalTexture = perturbedNormalTexture;
    perturbedNormalMaterial.roughness = 0.5f;
    const auto perturbedNormalMaterialHandle = resources.registerMaterial(perturbedNormalMaterial);

    MaterialResourceDesc doubleSidedNormalMaterial = perturbedNormalMaterial;
    doubleSidedNormalMaterial.debugName = "TextureViewer.DoubleSidedNormal";
    doubleSidedNormalMaterial.pipeline.featureFlags |= MaterialFeatureFlags::DoubleSided;
    const auto doubleSidedNormalMaterialHandle =
        resources.registerMaterial(doubleSidedNormalMaterial);

    MaterialResourceDesc alphaCutoutMaterial{};
    alphaCutoutMaterial.debugName = "TextureViewer.AlphaCutout";
    alphaCutoutMaterial.baseColorTexture = alphaTexture;
    alphaCutoutMaterial.pipeline.featureFlags =
        MaterialFeatureFlags::AlphaTest | MaterialFeatureFlags::DoubleSided;
    alphaCutoutMaterial.pipeline.alphaCutoff = 0.5f;
    alphaCutoutMaterial.castsShadows = false;
    alphaCutoutMaterial.roughness = 0.9f;
    const auto alphaCutoutMaterialHandle = resources.registerMaterial(alphaCutoutMaterial);

    const Diligent::QuaternionF panelRotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, -Diligent::PI_F * 0.5f);
    const Diligent::QuaternionF crossPanelRotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, Diligent::PI_F * 0.5f) *
        panelRotation;

    spawnRenderable(world, groundMesh, groundMaterialHandle, {0.0f, -1.25f, 5.0f},
                    {1.0f, 1.0f, 1.0f});

    spawnRenderable(world, wallPanelMesh, fallbackMaterialHandle, {-5.8f, 0.2f, 5.6f},
                    {1.0f, 1.0f, 1.0f}, panelRotation);
    spawnRenderable(world, wallPanelMesh, baseColorMaterialHandle, {-3.5f, 0.2f, 5.6f},
                    {1.0f, 1.0f, 1.0f}, panelRotation);
    spawnRenderable(world, wallPanelMesh, flatNormalMaterialHandle, {-1.2f, 0.2f, 5.6f},
                    {1.0f, 1.0f, 1.0f}, panelRotation);
    spawnRenderable(world, wallPanelMesh, perturbedNormalMaterialHandle, {1.1f, 0.2f, 5.6f},
                    {1.0f, 1.0f, 1.0f}, panelRotation);
    spawnRenderable(world, wallPanelMesh, aoMaterialHandle, {3.4f, 0.2f, 5.6f},
                    {1.0f, 1.0f, 1.0f}, panelRotation);
    spawnRenderable(world, wallPanelMesh, emissiveMaterialHandle, {5.7f, 0.2f, 5.6f},
                    {1.0f, 1.0f, 1.0f}, panelRotation);
    spawnRenderable(world, wallPanelMesh, doubleSidedNormalMaterialHandle, {8.0f, 0.2f, 5.6f},
                    {1.0f, 1.0f, 1.0f}, panelRotation);
    spawnRenderable(world, wallPanelMesh, doubleSidedNormalMaterialHandle, {10.3f, 0.2f, 5.6f},
                    {1.0f, 1.0f, 1.0f}, panelRotation *
                                              Diligent::QuaternionF::RotationFromAxisAngle(
                                                  {0.0f, 1.0f, 0.0f}, Diligent::PI_F));

    spawnRenderable(world, cubeMesh, fallbackMaterialHandle, {-4.2f, -0.35f, 2.0f},
                    {1.0f, 1.0f, 1.0f});
    spawnRenderable(world, cubeMesh, mrMaterialHandle, {-1.2f, -0.35f, 2.4f},
                    {1.0f, 1.0f, 1.0f});
    spawnRenderable(world, cubeMesh, mrMaterialHandle, {1.8f, -0.35f, 2.8f},
                    {1.0f, 1.0f, 1.0f});

    spawnRenderable(world, tiledPanelMesh, alphaCutoutMaterialHandle, {4.8f, 0.0f, 3.4f},
                    {1.0f, 1.0f, 1.0f}, panelRotation);
    spawnRenderable(world, tiledPanelMesh, alphaCutoutMaterialHandle, {4.8f, 0.0f, 4.4f},
                    {1.0f, 1.0f, 1.0f}, crossPanelRotation);

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, {});

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Texture viewer run failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Texture viewer finished.\n");
    CRESSIM_LOG_INFO("Visual checks: rear wall panels = scalar, base color, flat normal, perturbed normal, AO, emissive.\n");
    CRESSIM_LOG_INFO("Left cross panels = double-sided perturbed normal check. Front cubes = fallback then metallic/roughness cases. Right panels = alpha cutout.\n");
    return 0;
}
