#include "graphics/renderer/output_planner.h"

#include "graphics/renderer/renderer_internal.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace cressim::neo::graphics::detail
{

namespace
{

struct RequestedExtent
{
    std::uint32_t width  = 0u;
    std::uint32_t height = 0u;
};

struct ManagedFamilyInfo
{
    gpu::GpuRenderTargetDesc desc{};
    gpu::GpuRenderTargetHandle target{};
    std::uint32_t nextLayer = 0u;
};

std::uint32_t buildGlobalCameraIndex(const CameraData& camera,
                                     const gpu::GpuEntitySceneView& gpuScene)
{
    return camera.envIndex * std::max(gpuScene.layout.maxCamerasPerEnv, 1u) + camera.cameraSlot;
}

gpu::GpuRenderTargetDesc buildManagedPrimaryDesc(const CameraData& camera,
                                                 const gpu::GpuRenderTargetDesc& defaultTargetDesc)
{
    gpu::GpuRenderTargetDesc desc = defaultTargetDesc;
    desc.width     = camera.outputWidth == 0u ? defaultTargetDesc.width : camera.outputWidth;
    desc.height    = camera.outputHeight == 0u ? defaultTargetDesc.height : camera.outputHeight;
    desc.arraySize = 1u;
    desc.layeredRendering = true;
    desc.shaderReadable   = true;
    desc.debugName        = "CRESSimNeo.ManagedPrimary";
    return desc;
}

void populateResolvedCameraView(const CameraData& camera, const gpu::GpuEntitySceneView& gpuScene,
                                ResolvedCameraView& outView)
{
    outView.entityId          = camera.entityId;
    outView.viewport          = normalizeViewport(camera.viewport);
    outView.clearColor        = camera.clearColor;
    outView.clearDepth        = camera.clearDepth;
    outView.clearColorValue   = camera.clearColorValue;
    outView.clearDepthValue   = camera.clearDepthValue;
    outView.envIndex          = camera.envIndex;
    outView.cameraSlot        = camera.cameraSlot;
    outView.globalCameraIndex = buildGlobalCameraIndex(camera, gpuScene);
}

void logInvalidExplicitTarget(const CameraData& camera)
{
    std::cerr << "Renderer: skipping ExplicitSurface camera entity " << camera.entityId
              << " because its render target binding is invalid.\n";
}

void logManagedPrimaryUnavailable(const CameraData& camera)
{
    std::cerr << "Renderer: skipping ManagedPrimary camera entity " << camera.entityId
              << " because no default target is available.\n";
}

} // namespace

CameraOutputPlanningResult planCameraOutputs(
    const std::vector<CameraData>& cameras, const gpu::GpuEntitySceneView& gpuScene,
    gpu::GpuRenderTargetSystem& renderTargetSystem,
    const gpu::GpuRenderTargetDesc& defaultTargetDesc,
    const gpu::GpuRenderTargetBinding& defaultTargetBinding, bool hasDefaultTarget,
    const RenderFrameOptions& options,
    std::unordered_map<RenderTargetFamilyKey, gpu::GpuRenderTargetHandle,
                       RenderTargetFamilyKeyHasher>& managedPrimaryTargets,
    RenderStats& inOutStats)
{
    CameraOutputPlanningResult result{};
    result.resolvedCameras.reserve(cameras.size());

    std::unordered_map<common::ResourceId, RequestedExtent> requestedExtents;
    std::unordered_map<RenderTargetFamilyKey, std::uint32_t, RenderTargetFamilyKeyHasher>
        managedFamilyCounts;
    std::vector<RenderTargetFamilyKey> managedFamilyOrder;

    if (hasDefaultTarget)
    {
        for (const CameraData& camera : cameras)
        {
            if (camera.output.mode != gpu::CameraOutputMode::ManagedPrimary)
            {
                continue;
            }

            const RenderTargetFamilyKey key =
                makeRenderTargetFamilyKey(buildManagedPrimaryDesc(camera, defaultTargetDesc));
            auto [it, inserted] = managedFamilyCounts.emplace(key, 0u);
            ++it->second;
            if (inserted)
            {
                managedFamilyOrder.push_back(key);
            }
        }
    }

    std::unordered_map<RenderTargetFamilyKey, ManagedFamilyInfo, RenderTargetFamilyKeyHasher>
        managedFamilies;
    for (const RenderTargetFamilyKey& key : managedFamilyOrder)
    {
        gpu::GpuRenderTargetDesc familyDesc{};
        familyDesc.width            = key.width;
        familyDesc.height           = key.height;
        familyDesc.arraySize        = managedFamilyCounts[key];
        familyDesc.color            = key.color;
        familyDesc.depth            = key.depth;
        familyDesc.shaderReadable   = key.shaderReadable;
        familyDesc.layeredRendering = key.layeredRendering;
        familyDesc.colorFormat      = key.colorFormat;
        familyDesc.depthFormat      = key.depthFormat;
        familyDesc.debugName        = key.debugName;

        gpu::GpuRenderTargetHandle target{};
        const auto targetIt = managedPrimaryTargets.find(key);
        if (targetIt != managedPrimaryTargets.end() &&
            renderTargetSystem.isValidRenderTarget(targetIt->second))
        {
            target = targetIt->second;
            gpu::GpuRenderTargetDesc existingDesc{};
            if (!renderTargetSystem.tryGetRenderTargetDesc(target, existingDesc) ||
                existingDesc.width != familyDesc.width ||
                existingDesc.height != familyDesc.height ||
                existingDesc.arraySize != familyDesc.arraySize ||
                existingDesc.color != familyDesc.color || existingDesc.depth != familyDesc.depth ||
                existingDesc.shaderReadable != familyDesc.shaderReadable ||
                existingDesc.layeredRendering != familyDesc.layeredRendering ||
                existingDesc.colorFormat != familyDesc.colorFormat ||
                existingDesc.depthFormat != familyDesc.depthFormat)
            {
                const gpu::GpuRenderTargetUpdateResult updateResult =
                    renderTargetSystem.reconfigureRenderTarget(target, familyDesc);
                if (updateResult == gpu::GpuRenderTargetUpdateResult::Recreated)
                {
                    ++inOutStats.renderTargetRecreateCount;
                }
                if (updateResult == gpu::GpuRenderTargetUpdateResult::Failed)
                {
                    target = {};
                }
            }
        }

        if (!renderTargetSystem.isValidRenderTarget(target))
        {
            target = renderTargetSystem.createRenderTarget(familyDesc);
            if (renderTargetSystem.isValidRenderTarget(target))
            {
                managedPrimaryTargets[key] = target;
            }
        }

        if (renderTargetSystem.isValidRenderTarget(target))
        {
            managedFamilies.emplace(key, ManagedFamilyInfo{familyDesc, target, 0u});
            result.usedManagedFamilies.insert(key);
        }
    }

    for (const CameraData& camera : cameras)
    {
        if (camera.output.mode == gpu::CameraOutputMode::ManagedPrimary)
        {
            if (!hasDefaultTarget)
            {
                logManagedPrimaryUnavailable(camera);
                continue;
            }

            const RenderTargetFamilyKey key =
                makeRenderTargetFamilyKey(buildManagedPrimaryDesc(camera, defaultTargetDesc));
            const auto familyIt = managedFamilies.find(key);
            if (familyIt == managedFamilies.end() ||
                !renderTargetSystem.isValidRenderTarget(familyIt->second.target))
            {
                std::cerr << "Renderer: skipping ManagedPrimary camera entity " << camera.entityId
                          << " because its managed render target could not be created.\n";
                continue;
            }

            ManagedFamilyInfo& family = familyIt->second;
            ResolvedCameraView resolved{};
            populateResolvedCameraView(camera, gpuScene, resolved);
            resolved.outputBinding =
                gpu::GpuRenderTargetBinding{family.target, family.nextLayer, 1u};
            resolved.outputTargetDesc = family.desc;
            ++family.nextLayer;

            if (camera.entityId == options.presentedCameraEntity)
            {
                result.displayResolve = DisplayResolveRequest{resolved.outputBinding,
                                                              family.desc,
                                                              defaultTargetBinding,
                                                              defaultTargetDesc,
                                                              false,
                                                              false,
                                                              resolved.clearColorValue,
                                                              resolved.clearDepthValue};
            }

            result.resolvedCameras.push_back(resolved);
            ++inOutStats.cameraCount;
            continue;
        }

        gpu::GpuRenderTargetHandle target = camera.output.binding.target;
        if (!renderTargetSystem.isValidRenderTarget(target))
        {
            logInvalidExplicitTarget(camera);
            continue;
        }

        gpu::GpuRenderTargetDesc targetDesc{};
        if (!renderTargetSystem.tryGetRenderTargetDesc(target, targetDesc))
        {
            logInvalidExplicitTarget(camera);
            continue;
        }

        if (camera.outputWidth > 0u || camera.outputHeight > 0u)
        {
            ++inOutStats.renderTargetResizeRequests;

            RequestedExtent desired{};
            desired.width  = camera.outputWidth == 0u ? targetDesc.width : camera.outputWidth;
            desired.height = camera.outputHeight == 0u ? targetDesc.height : camera.outputHeight;

            const auto requestedIt = requestedExtents.find(target.id);
            if (requestedIt == requestedExtents.end())
            {
                requestedExtents.emplace(target.id, desired);
            }
            else
            {
                const bool conflict = requestedIt->second.width != desired.width ||
                                      requestedIt->second.height != desired.height;
                if (conflict)
                {
                    ++inOutStats.renderTargetResizeConflicts;
                }
                desired = requestedIt->second;
            }

            if (targetDesc.width != desired.width || targetDesc.height != desired.height)
            {
                const gpu::GpuRenderTargetUpdateResult updateResult =
                    renderTargetSystem.resizeRenderTarget(target, desired.width, desired.height);
                if (updateResult == gpu::GpuRenderTargetUpdateResult::Unchanged)
                {
                    ++inOutStats.renderTargetResizeNoOps;
                }
                else if (updateResult == gpu::GpuRenderTargetUpdateResult::Recreated)
                {
                    ++inOutStats.renderTargetRecreateCount;
                }
                if (updateResult == gpu::GpuRenderTargetUpdateResult::Failed ||
                    !renderTargetSystem.tryGetRenderTargetDesc(target, targetDesc))
                {
                    std::cerr << "Renderer: skipping ExplicitSurface camera entity "
                              << camera.entityId << " because its render target resize failed.\n";
                    continue;
                }
            }
            else
            {
                ++inOutStats.renderTargetResizeNoOps;
            }
        }

        ResolvedCameraView resolved{};
        populateResolvedCameraView(camera, gpuScene, resolved);
        resolved.outputBinding            = camera.output.binding;
        resolved.outputBinding.target     = target;
        resolved.outputBinding.layerCount = 1u;
        resolved.outputBinding.firstLayer =
            std::min(resolved.outputBinding.firstLayer, targetDesc.arraySize - 1u);
        resolved.outputTargetDesc = targetDesc;
        result.resolvedCameras.push_back(resolved);
        ++inOutStats.cameraCount;
    }

    return result;
}

} // namespace cressim::neo::graphics::detail
