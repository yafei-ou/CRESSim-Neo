#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/passes/material_program_registry.h"
#include "common/logger.h"

#include <cstring>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MaterialFeatureFlags;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::MainPassClass;
using cressim::neo::graphics::RenderStats;
using cressim::neo::graphics::TextureColorSpace;
using cressim::neo::graphics::TextureDimension;
using cressim::neo::graphics::TexturePixelFormat;
using cressim::neo::graphics::TextureResourceDesc;
using cressim::neo::graphics::detail::MaterialProgramRegistry;

std::uint16_t encodeFloat16(float value)
{
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::int32_t exponent    = static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa   = bits & 0x007fffffu;

    if (exponent <= 0)
    {
        if (exponent < -10)
        {
            return static_cast<std::uint16_t>(sign);
        }

        mantissa = (mantissa | 0x00800000u) >> static_cast<std::uint32_t>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000u) >> 13u));
    }
    if (exponent >= 31)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10u) |
                                      ((mantissa + 0x00001000u) >> 13u));
}

bool sameStats(const RenderStats& lhs, const RenderStats& rhs)
{
    return lhs.drawCalls == rhs.drawCalls &&
        lhs.opaqueDrawCalls == rhs.opaqueDrawCalls &&
        lhs.shadowDrawCalls == rhs.shadowDrawCalls &&
        lhs.renderableCount == rhs.renderableCount &&
        lhs.lightCount == rhs.lightCount &&
        lhs.cameraCount == rhs.cameraCount &&
        lhs.renderTargetResizeRequests == rhs.renderTargetResizeRequests &&
        lhs.renderTargetResizeNoOps == rhs.renderTargetResizeNoOps &&
        lhs.renderTargetRecreateCount == rhs.renderTargetRecreateCount &&
        lhs.renderTargetResizeConflicts == rhs.renderTargetResizeConflicts;
}

TextureResourceDesc makeSolidTextureDesc(const char *debugName, TextureColorSpace colorSpace,
                                         std::uint8_t r, std::uint8_t g, std::uint8_t b,
                                         std::uint8_t a)
{
    TextureResourceDesc desc{};
    desc.debugName = debugName;
    desc.width = 1u;
    desc.height = 1u;
    desc.colorSpace = colorSpace;
    desc.pixelData = {r, g, b, a};
    return desc;
}

TextureResourceDesc makeHdrCubeDesc(const char *debugName, float r, float g, float b)
{
    TextureResourceDesc desc{};
    desc.debugName = debugName;
    desc.width = 1u;
    desc.height = 1u;
    desc.mipLevelCount = 1u;
    desc.dimension = TextureDimension::TextureCube;
    desc.pixelFormat = TexturePixelFormat::RGBA16F;
    desc.colorSpace = TextureColorSpace::Linear;
    desc.subresources.resize(6u);
    for (auto &subresource : desc.subresources)
    {
        subresource.pixelData.resize(sizeof(std::uint16_t) * 4u);
        auto *encoded = reinterpret_cast<std::uint16_t *>(subresource.pixelData.data());
        encoded[0] = encodeFloat16(r);
        encoded[1] = encodeFloat16(g);
        encoded[2] = encodeFloat16(b);
        encoded[3] = encodeFloat16(1.0f);
    }
    return desc;
}

TextureResourceDesc makeHdrBrdfLutDesc(const char *debugName)
{
    TextureResourceDesc desc{};
    desc.debugName = debugName;
    desc.width = 1u;
    desc.height = 1u;
    desc.mipLevelCount = 1u;
    desc.dimension = TextureDimension::Texture2D;
    desc.pixelFormat = TexturePixelFormat::RGBA16F;
    desc.colorSpace = TextureColorSpace::Linear;
    desc.subresources.resize(1u);
    desc.subresources.front().pixelData.resize(sizeof(std::uint16_t) * 4u);
    auto *encoded = reinterpret_cast<std::uint16_t *>(desc.subresources.front().pixelData.data());
    encoded[0] = encodeFloat16(0.5f);
    encoded[1] = encodeFloat16(0.5f);
    encoded[2] = encodeFloat16(0.0f);
    encoded[3] = encodeFloat16(1.0f);
    return desc;
}

} // namespace

int main()
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = cressim::neo::gpu::GpuBackend::Null;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
        return 1;
    }

    auto& world = runtime.getWorld();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, 0.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    world.setDirectionalLight(lightEntity, DirectionalLightComponent{});

    auto& resources = runtime.getResources();

    MeshResourceDesc meshDesc{};
    meshDesc.debugName = "ForwardPipeline.TestTriangle";
    meshDesc.vertices = {
        {{-0.4f, -0.4f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{0.4f, -0.4f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.0f, 0.4f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5f, 1.0f}};
    meshDesc.indices = {0u, 1u, 2u};
    const auto mesh = resources.registerMesh(meshDesc);

    const auto baseColorTexture =
        resources.registerTexture(makeSolidTextureDesc("ForwardPipeline.BaseColor",
                                                       TextureColorSpace::Srgb, 255u, 128u, 128u,
                                                       255u));
    const auto metallicRoughnessTexture =
        resources.registerTexture(makeSolidTextureDesc("ForwardPipeline.MetallicRoughness",
                                                       TextureColorSpace::Linear, 0u, 128u, 64u,
                                                       255u));
    const auto emissiveTexture =
        resources.registerTexture(makeSolidTextureDesc("ForwardPipeline.Emissive",
                                                       TextureColorSpace::Srgb, 32u, 16u, 8u,
                                                       255u));
    const auto normalTexture =
        resources.registerTexture(makeSolidTextureDesc("ForwardPipeline.Normal",
                                                       TextureColorSpace::Linear, 255u, 128u, 128u,
                                                       255u));
    const auto aoTexture = resources.registerTexture(
        makeSolidTextureDesc("ForwardPipeline.AO", TextureColorSpace::Linear, 192u, 0u, 0u, 255u));
    const auto irradianceTexture = resources.registerTexture(
        makeHdrCubeDesc("ForwardPipeline.Irradiance", 0.15f, 0.18f, 0.20f));
    const auto prefilteredTexture = resources.registerTexture(
        makeHdrCubeDesc("ForwardPipeline.PrefilteredSpecular", 0.30f, 0.28f, 0.25f));
    const auto brdfLut = resources.registerTexture(
        makeHdrBrdfLutDesc("ForwardPipeline.BrdfLut"));

    cressim::neo::graphics::EnvironmentIblDesc environmentIbl{};
    environmentIbl.irradianceCubemap = irradianceTexture;
    environmentIbl.prefilteredSpecularCubemap = prefilteredTexture;
    environmentIbl.brdfLut = brdfLut;
    if (!world.setEnvironmentIbl(0u, environmentIbl))
    {
        CRESSIM_LOG_ERROR("Failed to assign environment IBL.\n");
        runtime.shutdown();
        return 1;
    }

    MaterialResourceDesc opaqueMaterialDesc{};
    opaqueMaterialDesc.debugName = "ForwardPipeline.Opaque";
    const auto opaqueMaterial = resources.registerMaterial(opaqueMaterialDesc);

    MaterialResourceDesc texturedMaterialDesc{};
    texturedMaterialDesc.debugName = "ForwardPipeline.Textured";
    texturedMaterialDesc.baseColorTexture = baseColorTexture;
    texturedMaterialDesc.normalTexture = normalTexture;
    texturedMaterialDesc.metallicRoughnessTexture = metallicRoughnessTexture;
    texturedMaterialDesc.emissiveTexture = emissiveTexture;
    texturedMaterialDesc.aoTexture = aoTexture;
    texturedMaterialDesc.emissiveFactor = {1.0f, 1.0f, 1.0f};
    texturedMaterialDesc.castsShadows = false;
    const auto texturedMaterial = resources.registerMaterial(texturedMaterialDesc);

    MaterialResourceDesc transparentMaterialDesc{};
    transparentMaterialDesc.debugName = "ForwardPipeline.Transparent";
    transparentMaterialDesc.blendMode = cressim::neo::graphics::BlendMode::Transparent;
    transparentMaterialDesc.opacity = 0.5f;
    transparentMaterialDesc.castsShadows = false;
    const auto transparentMaterial = resources.registerMaterial(transparentMaterialDesc);

    const auto visibleOpaqueEntity = world.createEntity();
    TransformComponent visibleOpaqueTransform{};
    visibleOpaqueTransform.worldTransform.position = {0.0f, 0.0f, 2.0f};
    world.setTransform(visibleOpaqueEntity, visibleOpaqueTransform);
    MeshRendererComponent visibleOpaqueRenderer{};
    visibleOpaqueRenderer.mesh = mesh;
    visibleOpaqueRenderer.material = opaqueMaterial;
    visibleOpaqueRenderer.visible = true;
    world.setMeshRenderer(visibleOpaqueEntity, visibleOpaqueRenderer);

    const auto texturedEntity = world.createEntity();
    TransformComponent texturedTransform{};
    texturedTransform.worldTransform.position = {-0.8f, 0.0f, 2.5f};
    world.setTransform(texturedEntity, texturedTransform);
    MeshRendererComponent texturedRenderer{};
    texturedRenderer.mesh = mesh;
    texturedRenderer.material = texturedMaterial;
    texturedRenderer.visible = true;
    world.setMeshRenderer(texturedEntity, texturedRenderer);

    const auto visibleTransparentEntity = world.createEntity();
    TransformComponent visibleTransparentTransform{};
    visibleTransparentTransform.worldTransform.position = {0.8f, 0.0f, 3.0f};
    world.setTransform(visibleTransparentEntity, visibleTransparentTransform);
    MeshRendererComponent visibleTransparentRenderer{};
    visibleTransparentRenderer.mesh = mesh;
    visibleTransparentRenderer.material = transparentMaterial;
    visibleTransparentRenderer.visible = true;
    world.setMeshRenderer(visibleTransparentEntity, visibleTransparentRenderer);

    const auto culledEntity = world.createEntity();
    TransformComponent culledTransform{};
    culledTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};
    world.setTransform(culledEntity, culledTransform);
    MeshRendererComponent culledRenderer{};
    culledRenderer.mesh = mesh;
    culledRenderer.material = opaqueMaterial;
    culledRenderer.visible = true;
    world.setMeshRenderer(culledEntity, culledRenderer);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex = 0;
    frame.timeSeconds = 0.0;
    runtime.tick(frame);
    const RenderStats firstFrame = runtime.lastRenderStats();

    frame.frameIndex = 1;
    frame.timeSeconds = frame.deltaSeconds;
    runtime.tick(frame);
    const RenderStats secondFrame = runtime.lastRenderStats();

    frame.frameIndex = 2;
    frame.timeSeconds = static_cast<double>(frame.deltaSeconds) * 2.0;
    runtime.tick(frame);
    const RenderStats thirdFrame = runtime.lastRenderStats();

    runtime.shutdown();

    if (firstFrame.renderableCount != 4)
    {
        CRESSIM_LOG_ERROR( "Unexpected renderable counters.\n");
        return 1;
    }
    if (firstFrame.shadowDrawCalls != 4)
    {
        CRESSIM_LOG_ERROR( "Unexpected shadow draws.\n");
        return 1;
    }
    if (firstFrame.cameraCount != 1 || firstFrame.lightCount != 1)
    {
        CRESSIM_LOG_ERROR( "Unexpected camera/light counters.\n");
        return 1;
    }
    if (!sameStats(secondFrame, thirdFrame))
    {
        CRESSIM_LOG_ERROR( "Forward queue statistics were not stable across identical frames.\n");
        return 1;
    }

    MaterialResourceDesc runtimeVariantA{};
    runtimeVariantA.baseColor = {1.0f, 0.2f, 0.2f};
    runtimeVariantA.roughness = 0.15f;
    runtimeVariantA.pipeline.featureFlags = MaterialFeatureFlags::None;
    runtimeVariantA.baseColorTexture = baseColorTexture;

    MaterialResourceDesc runtimeVariantB = runtimeVariantA;
    runtimeVariantB.baseColor = {0.1f, 0.8f, 0.4f};
    runtimeVariantB.roughness = 0.9f;
    runtimeVariantB.metallic = 1.0f;
    runtimeVariantB.emissiveTexture = emissiveTexture;

    const auto keyA = MaterialProgramRegistry::buildProgramKey(
        MainPassClass::ForwardOpaque,
        runtimeVariantA.pipeline.programFamily,
        runtimeVariantA.pipeline.featureFlags,
        Diligent::TEX_FORMAT_RGBA8_UNORM,
        Diligent::TEX_FORMAT_D32_FLOAT,
        true,
        true,
        false);
    const auto keyB = MaterialProgramRegistry::buildProgramKey(
        MainPassClass::ForwardOpaque,
        runtimeVariantB.pipeline.programFamily,
        runtimeVariantB.pipeline.featureFlags,
        Diligent::TEX_FORMAT_RGBA8_UNORM,
        Diligent::TEX_FORMAT_D32_FLOAT,
        true,
        true,
        false);
    if (!(keyA == keyB))
    {
        CRESSIM_LOG_ERROR( "Program key unexpectedly changed with runtime-only material parameters.\n");
        return 1;
    }

    MaterialResourceDesc normalMappedVariant = runtimeVariantA;
    normalMappedVariant.normalTexture = normalTexture;
    const auto storedNormalMappedMaterial = resources.tryGetMaterial(texturedMaterial);
    if (storedNormalMappedMaterial == nullptr ||
        !cressim::neo::graphics::hasFlag(storedNormalMappedMaterial->pipeline.featureFlags,
                                         MaterialFeatureFlags::NormalMap))
    {
        CRESSIM_LOG_ERROR("Stored textured material did not retain NormalMap feature flags.\n");
        return 1;
    }
    const auto keyNormal = MaterialProgramRegistry::buildProgramKey(
        MainPassClass::ForwardOpaque,
        normalMappedVariant.pipeline.programFamily,
        normalMappedVariant.pipeline.featureFlags | MaterialFeatureFlags::NormalMap,
        Diligent::TEX_FORMAT_RGBA8_UNORM,
        Diligent::TEX_FORMAT_D32_FLOAT,
        true,
        true,
        false);
    if (keyA == keyNormal)
    {
        CRESSIM_LOG_ERROR("Program key should differ when normal mapping is enabled.\n");
        return 1;
    }

    MaterialResourceDesc featureVariant = runtimeVariantA;
    featureVariant.pipeline.featureFlags = MaterialFeatureFlags::AlphaTest;
    const auto keyC = MaterialProgramRegistry::buildProgramKey(
        MainPassClass::ForwardOpaque,
        featureVariant.pipeline.programFamily,
        featureVariant.pipeline.featureFlags,
        Diligent::TEX_FORMAT_RGBA8_UNORM,
        Diligent::TEX_FORMAT_D32_FLOAT,
        true,
        true,
        false);
    if (keyA == keyC)
    {
        CRESSIM_LOG_ERROR( "Program key should differ when compile-time feature flags differ.\n");
        return 1;
    }

    CRESSIM_LOG_INFO( "Forward pipeline vulkan-backend checks passed.\n");
    return 0;
}
