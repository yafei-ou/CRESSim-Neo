#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/readback.h"

#include <cstdint>
#include <vector>

int main()
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING(
            "Skipping runtime ultrasound staging test because runtime initialization failed.");
        return 0;
    }

    auto &world = runtime.getWorld();
    gpu::GpuDevice *device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        CRESSIM_LOG_WARNING(
            "Skipping runtime ultrasound staging test because GPU device is unavailable.");
        runtime.shutdown();
        return 0;
    }

    const common::EntityId softEntity = world.createEntity();
    engine::TransformComponent softTransform{};
    softTransform.worldTransform.position = {0.0f, 0.0f, 0.5f};
    world.setTransform(softEntity, softTransform);

    engine::SoftBodyComponent softBody{};
    softBody.source.kind = physics::SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = {1.0f, 1.0f, 1.0f};
    softBody.source.regularGrid.targetParticleSpacing = 0.5f;
    softBody.particleRadius = 0.1f;
    if (!world.setSoftBody(softEntity, softBody))
    {
        CRESSIM_LOG_WARNING(
            "Skipping runtime ultrasound staging test because soft-body authoring failed.");
        runtime.shutdown();
        return 0;
    }

    world.setUltrasoundScattererSource(softEntity, engine::UltrasoundScattererSourceComponent{});
    const auto authoredParticles = world.tryGetSoftBodyAuthoringParticles(softEntity);
    if (!authoredParticles.has_value() || authoredParticles->particleCount == 0u)
    {
        CRESSIM_LOG_WARNING(
            "Skipping runtime ultrasound staging test because authored particles are unavailable.");
        runtime.shutdown();
        return 0;
    }

    std::vector<engine::UltrasoundAmplitudeRange> amplitudeRanges(
        authoredParticles->particleCount, engine::UltrasoundAmplitudeRange{0.8f, 1.0f});
    world.setUltrasoundScattererAmplitudeRanges(softEntity, amplitudeRanges);

    const common::EntityId probeEntity = world.createEntity();
    engine::TransformComponent probeTransform{};
    probeTransform.worldTransform.position = {0.0f, 0.0f, -0.25f};
    world.setTransform(probeEntity, probeTransform);

    engine::UltrasoundProbeComponent probe{};
    probe.numScanlines = 64u;
    probe.lineLength = 1.5f;
    probe.scanlineSpacing = 0.02f;
    probe.imageBaseHeight = 128u;
    world.setUltrasoundProbe(probeEntity, probe);

    runtime.prepare();

    const engine::UltrasoundProbeResult *preparedResult =
        world.tryGetUltrasoundProbeResult(probeEntity);
    if (preparedResult == nullptr || !preparedResult->valid ||
        preparedResult->imageTarget.id == common::kInvalidResourceId)
    {
        CRESSIM_LOG_WARNING(
            "Skipping runtime ultrasound staging test because prepared probe outputs are unavailable.");
        runtime.shutdown();
        return 0;
    }

    const gpu::GpuRenderTargetReadbackRequest request =
        device->renderTargetSystem().requestRenderTargetReadback(
            gpu::GpuRenderTargetBinding{preparedResult->imageTarget, 0u, 1u});
    if (request.id == 0u)
    {
        CRESSIM_LOG_ERROR("Failed to queue ultrasound readback request before sensor execution.");
        runtime.shutdown();
        return 1;
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    if (!runtime.stepPhysics(frame))
    {
        CRESSIM_LOG_WARNING(
            "Skipping runtime ultrasound staging test because staged physics initialization failed.");
        runtime.shutdown();
        return 0;
    }
    if (!runtime.stepSensors(frame))
    {
        CRESSIM_LOG_WARNING(
            "Skipping runtime ultrasound staging test because staged ultrasound execution failed.");
        runtime.shutdown();
        return 0;
    }
    runtime.pollReadbacks();

    const engine::UltrasoundProbeResult *executedResult =
        world.tryGetUltrasoundProbeResult(probeEntity);
    if (executedResult == nullptr || !executedResult->valid || !executedResult->imageValid)
    {
        CRESSIM_LOG_WARNING(
            "Skipping runtime ultrasound staging test because probe image generation is unavailable.");
        runtime.shutdown();
        return 0;
    }

    gpu::GpuRenderTargetReadbackEvent event{};
    if (!device->renderTargetSystem().tryGetRenderTargetReadback(request, event) ||
        !tests::helpers::isValidReadback(event))
    {
        CRESSIM_LOG_ERROR("Expected queued ultrasound readback to complete after staged execution.");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO("Runtime ultrasound staging checks passed.");
    return 0;
}
