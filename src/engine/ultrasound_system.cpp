#include "engine/ultrasound_system.h"

#include "common/logger.h"
#include "engine/components.h"
#include "engine/world.h"
#include "gpu/cuda_interop.h"
#include "gpu/gpu_buffer_utils.h"
#include "gpu/gpu_compute_pass.h"
#include "gpu/gpu_device.h"
#include "gpu/shader_library.h"
#include "gpu/shared_export_buffer.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if CRESSIM_NEO_HAS_ULTRASOUND
extern "C"
{
#include "cru_c_extension.h"
#include "cru_compute.h"
}
#endif

namespace cressim::neo::engine
{

namespace
{

float resolvePointDistance(const physics::SoftBodyState &body,
                           const UltrasoundScattererSourceComponent &source)
{
    if (source.pointDistanceOverride > 0.0f)
    {
        return std::max(source.pointDistanceOverride, 1.0e-4f);
    }

    switch (body.source.kind)
    {
    case physics::SoftBodySourceKind::RegularGrid:
        return std::max(body.source.regularGrid.targetParticleSpacing, 1.0e-4f);
    case physics::SoftBodySourceKind::TetMesh:
    case physics::SoftBodySourceKind::TetGenFiles:
    default:
        return std::max(body.particleRadius * 2.0f, 1.0e-4f);
    }
}

bool equalsProbeComponent(const UltrasoundProbeComponent &lhs,
                          const UltrasoundProbeComponent &rhs) noexcept
{
    return lhs.enabled == rhs.enabled && lhs.geometry == rhs.geometry &&
           lhs.numScanlines == rhs.numScanlines && lhs.lineLength == rhs.lineLength &&
           lhs.scanlineSpacing == rhs.scanlineSpacing &&
           lhs.sectorAngleDegrees == rhs.sectorAngleDegrees && lhs.probeRadius == rhs.probeRadius &&
           lhs.soundSpeed == rhs.soundSpeed && lhs.worldUnitsPerMeter == rhs.worldUnitsPerMeter &&
           lhs.noiseAmplitude == rhs.noiseAmplitude &&
           lhs.samplingFrequency == rhs.samplingFrequency &&
           lhs.demodulationFrequency == rhs.demodulationFrequency &&
           lhs.centerFrequency == rhs.centerFrequency &&
           lhs.fractionalBandwidth == rhs.fractionalBandwidth &&
           lhs.beamSigmaLateral == rhs.beamSigmaLateral &&
           lhs.beamSigmaElevational == rhs.beamSigmaElevational &&
           lhs.radialDecimation == rhs.radialDecimation &&
           lhs.threadsPerBlock == rhs.threadsPerBlock && lhs.cudaNumStreams == rhs.cudaNumStreams &&
           lhs.numTimeSamples == rhs.numTimeSamples &&
           lhs.useArcProjection == rhs.useArcProjection &&
           lhs.enablePhaseDelay == rhs.enablePhaseDelay;
}

bool equalsRendererComponent(const UltrasoundRendererComponent &lhs,
                             const UltrasoundRendererComponent &rhs) noexcept
{
    return lhs.enabled == rhs.enabled && lhs.output.mode == rhs.output.mode &&
           lhs.output.binding == rhs.output.binding && lhs.outputWidth == rhs.outputWidth &&
           lhs.outputHeight == rhs.outputHeight &&
           lhs.useFixedMaxNormalization == rhs.useFixedMaxNormalization &&
           lhs.fixedMaxSignal == rhs.fixedMaxSignal;
}

bool equalsSourceComponent(const UltrasoundScattererSourceComponent &lhs,
                           const UltrasoundScattererSourceComponent &rhs) noexcept
{
    return lhs.enabled == rhs.enabled && lhs.density == rhs.density &&
           lhs.pointDistanceOverride == rhs.pointDistanceOverride;
}

UltrasoundRendererComponent disabledRendererComponent() noexcept
{
    UltrasoundRendererComponent renderer{};
    renderer.enabled = false;
    return renderer;
}

constexpr std::uint32_t kInvalidScenePoseSlot = 0xffffffffu;

std::uint32_t nextPowerOfTwo(std::uint32_t value)
{
    if (value <= 1u)
    {
        return 1u;
    }

    --value;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

struct UltrasoundImageSize
{
    std::uint32_t width      = 1u;
    std::uint32_t height     = 1u;
    float probeRadiusPixels  = 0.0f;
    float sectorAngleRadians = 0.0f;
};

std::uint32_t dispatchGroupCount(std::uint32_t threadCount, std::uint32_t groupSize) noexcept
{
    return groupSize == 0u ? 0u : (threadCount + groupSize - 1u) / groupSize;
}

float resolveCurvilinearSectorAngleRadians(float sectorAngleDegrees) noexcept
{
    return std::max(std::abs(sectorAngleDegrees) * (3.14159265359f / 180.0f), 1.0e-4f);
}

#if CRESSIM_NEO_HAS_ULTRASOUND
CruBounds3 computeBounds(const physics::SoftBodyState &body, float pointDistance)
{
    CruBounds3 bounds{};
    if (body.restPositions.empty())
    {
        return bounds;
    }

    Diligent::float3 minimum = body.restPositions.front();
    Diligent::float3 maximum = body.restPositions.front();
    for (const Diligent::float3 &position : body.restPositions)
    {
        minimum.x = std::min(minimum.x, position.x);
        minimum.y = std::min(minimum.y, position.y);
        minimum.z = std::min(minimum.z, position.z);
        maximum.x = std::max(maximum.x, position.x);
        maximum.y = std::max(maximum.y, position.y);
        maximum.z = std::max(maximum.z, position.z);
    }

    const float padding = pointDistance * 0.5f;
    bounds.minimum[0]   = minimum.x - padding;
    bounds.minimum[1]   = minimum.y - padding;
    bounds.minimum[2]   = minimum.z - padding;
    bounds.maximum[0]   = maximum.x + padding;
    bounds.maximum[1]   = maximum.y + padding;
    bounds.maximum[2]   = maximum.z + padding;
    return bounds;
}

bool equalsRfLayout(const CruRfLayout &lhs, const CruRfLayout &rhs) noexcept
{
    return lhs.numScanlines == rhs.numScanlines &&
           lhs.requiredTimeSamples == rhs.requiredTimeSamples &&
           lhs.timeSamples == rhs.timeSamples &&
           lhs.delayCompensationSamples == rhs.delayCompensationSamples &&
           lhs.samplesPerLine == rhs.samplesPerLine && lhs.totalSamples == rhs.totalSamples;
}

UltrasoundImageSize computeImageSize(const UltrasoundProbeComponent &probe,
                                     const UltrasoundRendererComponent &renderer,
                                     const CruRfLayout &layout) noexcept
{
    if (probe.geometry == UltrasoundProbeComponent::Geometry::Curvilinear)
    {
        const float nativeDepthSamples = static_cast<float>(std::max(layout.samplesPerLine, 1u));
        const float sectorAngleRadians =
            resolveCurvilinearSectorAngleRadians(probe.sectorAngleDegrees);
        const float nativeProbeRadiusPixels = std::max(probe.probeRadius, 0.0f) *
                                              nativeDepthSamples /
                                              std::max(probe.lineLength, 1.0e-4f);
        const float nativeWidth =
            std::max(1.0f, 2.0f * (nativeProbeRadiusPixels + nativeDepthSamples) *
                               std::sin(0.5f * sectorAngleRadians));
        const float nativeHeight =
            std::max(1.0f, nativeProbeRadiusPixels + nativeDepthSamples -
                               nativeProbeRadiusPixels * std::cos(0.5f * sectorAngleRadians));
        if (renderer.outputHeight > 0u)
        {
            const std::uint32_t height = std::max(renderer.outputHeight, 1u);
            const float scale          = static_cast<float>(height) / nativeHeight;
            const std::uint32_t width =
                renderer.outputWidth > 0u
                    ? std::max(renderer.outputWidth, 1u)
                    : std::max<std::uint32_t>(
                          1u, static_cast<std::uint32_t>(std::floor(nativeWidth * scale + 0.5f)));
            return UltrasoundImageSize{width, height, nativeProbeRadiusPixels * scale,
                                       sectorAngleRadians};
        }

        if (renderer.outputWidth > 0u)
        {
            const std::uint32_t width  = std::max(renderer.outputWidth, 1u);
            const float scale          = static_cast<float>(width) / nativeWidth;
            const std::uint32_t height = std::max<std::uint32_t>(
                1u, static_cast<std::uint32_t>(std::floor(nativeHeight * scale + 0.5f)));
            return UltrasoundImageSize{width, height, nativeProbeRadiusPixels * scale,
                                       sectorAngleRadians};
        }

        return UltrasoundImageSize{
            std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(std::floor(nativeWidth + 0.5f))),
            std::max<std::uint32_t>(1u,
                                    static_cast<std::uint32_t>(std::floor(nativeHeight + 0.5f))),
            nativeProbeRadiusPixels, sectorAngleRadians};
    }

    const std::uint32_t lateralIntervals =
        probe.numScanlines > 0u ? std::max(probe.numScanlines - 1u, 1u) : 1u;
    const float lateralSpan =
        static_cast<float>(lateralIntervals) * std::max(probe.scanlineSpacing, 1.0e-4f);
    const float aspect = lateralSpan / std::max(probe.lineLength, 1.0e-4f);
    if (renderer.outputHeight > 0u)
    {
        const std::uint32_t height = std::max(renderer.outputHeight, 1u);
        const std::uint32_t width =
            renderer.outputWidth > 0u
                ? std::max(renderer.outputWidth, 1u)
                : std::max<std::uint32_t>(
                      1u, static_cast<std::uint32_t>(std::floor(height * aspect + 0.5f)));
        return UltrasoundImageSize{width, height, 0.0f, 0.0f};
    }

    if (renderer.outputWidth > 0u)
    {
        const std::uint32_t width  = std::max(renderer.outputWidth, 1u);
        const std::uint32_t height = std::max<std::uint32_t>(
            1u, static_cast<std::uint32_t>(
                    std::floor(static_cast<float>(width) / std::max(aspect, 1.0e-6f) + 0.5f)));
        return UltrasoundImageSize{width, height, 0.0f, 0.0f};
    }

    const std::uint32_t height = std::max(layout.samplesPerLine, 1u);
    const std::uint32_t width =
        std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(std::floor(height * aspect + 0.5f)));
    return UltrasoundImageSize{width, height, 0.0f, 0.0f};
}

UltrasoundProbeLayout buildProbeLayout(const UltrasoundProbeComponent &probeComponent,
                                       const UltrasoundRendererComponent &rendererComponent,
                                       const CruRfLayout &rfLayout) noexcept
{
    UltrasoundProbeLayout layout{};
    const UltrasoundImageSize size = computeImageSize(probeComponent, rendererComponent, rfLayout);
    layout.numScanlines            = rfLayout.numScanlines;
    layout.samplesPerScanline      = rfLayout.samplesPerLine;
    layout.imageWidth              = size.width;
    layout.imageHeight             = size.height;
    layout.colorFormat             = Diligent::TEX_FORMAT_RGBA8_UNORM;
    layout.layeredOutputSupported  = true;
    return layout;
}

struct UltrasoundReductionConstants
{
    std::uint32_t dataLength = 0u;
    std::uint32_t padding0   = 0u;
    std::uint32_t padding1   = 0u;
    std::uint32_t padding2   = 0u;
};

struct UltrasoundImageConstants
{
    std::uint32_t numScanlines             = 0u;
    std::uint32_t samplesPerLine           = 0u;
    std::uint32_t imageWidth               = 1u;
    std::uint32_t imageHeight              = 1u;
    std::uint32_t outputLayer              = 0u;
    float fixedMaxSignal                   = 1.0f;
    std::uint32_t useFixedMaxNormalization = 0u;
    float sectorAngleRadians               = 0.0f;
    float probeRadiusPixels                = 0.0f;
    std::uint32_t padding0                 = 0u;
    std::uint32_t padding1                 = 0u;
    std::uint32_t padding2                 = 0u;
};

struct UltrasoundScanlineUpdateConstants
{
    std::uint32_t entityPoseSlot = kInvalidScenePoseSlot;
    std::uint32_t scanlineCount  = 0u;
    std::uint32_t padding0       = 0u;
    std::uint32_t padding1       = 0u;
};

constexpr std::uint32_t kUltrasoundReductionThreadGroupSize = 256u;
constexpr std::uint32_t kUltrasoundImageThreadGroupSizeX    = 8u;
constexpr std::uint32_t kUltrasoundImageThreadGroupSizeY    = 8u;
constexpr std::uint32_t kUltrasoundScanlineThreadGroupSize  = 64u;

constexpr Diligent::ShaderResourceVariableDesc kUltrasoundMaxReduceVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "UltrasoundReductionConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RfData", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_GroupMaxRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kUltrasoundMaxFinalizeVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "UltrasoundReductionConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_GroupMax", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FinalMaxRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kUltrasoundImageVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "UltrasoundImageConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RfData", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FinalMax", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_OutputImageRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderMacro kUltrasoundNonLayeredImageMacros[] = {
    {"CRESSIM_ULTRASOUND_LAYERED_OUTPUT", "0"},
};

constexpr Diligent::ShaderMacro kUltrasoundLayeredImageMacros[] = {
    {"CRESSIM_ULTRASOUND_LAYERED_OUTPUT", "1"},
};

constexpr Diligent::ShaderResourceVariableDesc kUltrasoundScanlineUpdateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "UltrasoundScanlineUpdateConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LocalScanlines",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_WorldScanlinesRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kUltrasoundMaxReducePassDefinition = {
    "graphics/ultrasound_rf_max_reduce.cs.hlsl", "CRESSimNeo.Ultrasound.MaxReduce.CS",
    "CRESSimNeo.Ultrasound.MaxReduce.PSO",       kUltrasoundMaxReduceVars,
    std::size(kUltrasoundMaxReduceVars),
};

constexpr gpu::GpuComputePassDefinition kUltrasoundMaxFinalizePassDefinition = {
    "graphics/ultrasound_rf_max_finalize.cs.hlsl", "CRESSimNeo.Ultrasound.MaxFinalize.CS",
    "CRESSimNeo.Ultrasound.MaxFinalize.PSO",       kUltrasoundMaxFinalizeVars,
    std::size(kUltrasoundMaxFinalizeVars),
};

constexpr gpu::GpuComputePassDefinition kUltrasoundImagePassDefinition = {
    "graphics/ultrasound_rf_image.cs.hlsl",
    "CRESSimNeo.Ultrasound.Image.CS",
    "CRESSimNeo.Ultrasound.Image.PSO",
    kUltrasoundImageVars,
    std::size(kUltrasoundImageVars),
    kUltrasoundNonLayeredImageMacros,
    std::size(kUltrasoundNonLayeredImageMacros),
};

constexpr gpu::GpuComputePassDefinition kUltrasoundLayeredImagePassDefinition = {
    "graphics/ultrasound_rf_image.cs.hlsl",   "CRESSimNeo.Ultrasound.LayeredImage.CS",
    "CRESSimNeo.Ultrasound.LayeredImage.PSO", kUltrasoundImageVars,
    std::size(kUltrasoundImageVars),          kUltrasoundLayeredImageMacros,
    std::size(kUltrasoundLayeredImageMacros),
};

constexpr gpu::GpuComputePassDefinition kUltrasoundCurvilinearImagePassDefinition = {
    "graphics/ultrasound_rf_curvilinear.cs.hlsl",
    "CRESSimNeo.Ultrasound.CurvilinearImage.CS",
    "CRESSimNeo.Ultrasound.CurvilinearImage.PSO",
    kUltrasoundImageVars,
    std::size(kUltrasoundImageVars),
    kUltrasoundNonLayeredImageMacros,
    std::size(kUltrasoundNonLayeredImageMacros),
};

constexpr gpu::GpuComputePassDefinition kUltrasoundLayeredCurvilinearImagePassDefinition = {
    "graphics/ultrasound_rf_curvilinear.cs.hlsl",
    "CRESSimNeo.Ultrasound.LayeredCurvilinearImage.CS",
    "CRESSimNeo.Ultrasound.LayeredCurvilinearImage.PSO",
    kUltrasoundImageVars,
    std::size(kUltrasoundImageVars),
    kUltrasoundLayeredImageMacros,
    std::size(kUltrasoundLayeredImageMacros),
};

constexpr gpu::GpuComputePassDefinition kUltrasoundScanlineUpdatePassDefinition = {
    "graphics/ultrasound_scanline_update.cs.hlsl", "CRESSimNeo.Ultrasound.ScanlineUpdate.CS",
    "CRESSimNeo.Ultrasound.ScanlineUpdate.PSO",    kUltrasoundScanlineUpdateVars,
    std::size(kUltrasoundScanlineUpdateVars),
};
#endif

} // namespace

bool computeUltrasoundProbeLayout(const UltrasoundProbeComponent &probeComponent,
                                  const UltrasoundRendererComponent &rendererComponent,
                                  UltrasoundProbeLayout &outLayout)
{
    outLayout = {};
#if !CRESSIM_NEO_HAS_ULTRASOUND
    (void)probeComponent;
    (void)rendererComponent;
    return false;
#else
    if (!probeComponent.enabled)
    {
        return false;
    }

    CruComputeHandle engine = CruComputeCreate();
    if (engine == nullptr)
    {
        return false;
    }

    const float worldUnitsPerMeter        = std::max(probeComponent.worldUnitsPerMeter, 1.0f);
    const float effectiveSoundSpeed       = probeComponent.soundSpeed * worldUnitsPerMeter;
    const float effectiveBeamSigmaLateral = probeComponent.beamSigmaLateral * worldUnitsPerMeter;
    const float effectiveBeamSigmaElevational =
        probeComponent.beamSigmaElevational * worldUnitsPerMeter;
    int radialDecimation = static_cast<int>(std::max(probeComponent.radialDecimation, 1u));
    int threadsPerBlock  = static_cast<int>(std::max(probeComponent.threadsPerBlock, 1u));
    int cudaNumStreams   = static_cast<int>(std::max(probeComponent.cudaNumStreams, 1u));
    int numTimeSamples   = static_cast<int>(probeComponent.numTimeSamples);
    if (numTimeSamples <= 0)
    {
        const float requiredMaxTime   = 2.0f * probeComponent.lineLength / effectiveSoundSpeed;
        const auto requiredNumSamples = static_cast<std::uint32_t>(
            std::floor(probeComponent.samplingFrequency * requiredMaxTime + 0.5f));
        numTimeSamples = static_cast<int>(
            std::max<std::uint32_t>(16384u, nextPowerOfTwo(requiredNumSamples + 2048u)));
    }

    CruComputeSetParameter(engine, CruSimulationParameterType_eSoundSpeed, &effectiveSoundSpeed);
    CruComputeSetParameter(engine, CruSimulationParameterType_eNoiseAmplitude,
                           &probeComponent.noiseAmplitude);
    CruComputeSetParameter(engine, CruSimulationParameterType_eUseArcProjection,
                           &probeComponent.useArcProjection);
    CruComputeSetParameter(engine, CruSimulationParameterType_eRadialDecimation, &radialDecimation);
    CruComputeSetParameter(engine, CruSimulationParameterType_eEnablePhaseDelay,
                           &probeComponent.enablePhaseDelay);
    CruComputeSetParameter(engine, CruSimulationParameterType_eNumTimeSamples, &numTimeSamples);
    CruComputeSetParameter(engine, CruSimulationParameterType_eThreadsPerBlock, &threadsPerBlock);
    CruComputeSetParameter(engine, CruSimulationParameterType_eCudaNumStreams, &cudaNumStreams);

    ExcitationSignalHandle excitation = CruCreateGaussianExcitationSignal(
        probeComponent.samplingFrequency, probeComponent.demodulationFrequency,
        probeComponent.centerFrequency, probeComponent.fractionalBandwidth);
    BeamProfileHandle beamProfile =
        CruCreateGaussianBeamProfile(effectiveBeamSigmaLateral, effectiveBeamSigmaElevational);
    if (excitation == nullptr || beamProfile == nullptr)
    {
        if (beamProfile != nullptr)
        {
            CruReleaseBeamProfile(beamProfile);
        }
        if (excitation != nullptr)
        {
            CruReleaseExcitationSignal(excitation);
        }
        CruComputeDestroy(engine);
        return false;
    }

    CruComputeSetExcitation(engine, excitation);
    CruComputeSetBeamProfile(engine, beamProfile);
    CruComputeBindScanlinesDevice(engine, nullptr, probeComponent.numScanlines,
                                  probeComponent.lineLength);

    const CruRfLayout rfLayout = CruComputeGetRfLayout(engine);
    outLayout                  = buildProbeLayout(probeComponent, rendererComponent, rfLayout);

    CruReleaseBeamProfile(beamProfile);
    CruReleaseExcitationSignal(excitation);
    CruComputeDestroy(engine);
    return true;
#endif
}

struct UltrasoundSystem::Impl
{
    struct SourceBinding
    {
#if CRESSIM_NEO_HAS_ULTRASOUND
        DeformationTrackerHandle tracker = nullptr;
        void *neighborIndices            = nullptr;
        void *neighborWeights            = nullptr;
        void *generatedScatterers        = nullptr;
#endif
        common::EntityId sourceEntityId = common::kInvalidEntityId;
        std::uint32_t particleOffset    = 0u;
        std::uint32_t particleCount     = 0u;
        std::uint32_t scattererOffset   = 0u;
        std::uint64_t scattererCount    = 0u;
        UltrasoundScattererSourceComponent component{};

        SourceBinding()                                 = default;
        SourceBinding(const SourceBinding &)            = delete;
        SourceBinding &operator=(const SourceBinding &) = delete;

#if CRESSIM_NEO_HAS_ULTRASOUND
        SourceBinding(SourceBinding &&other) noexcept
            : tracker(other.tracker), neighborIndices(other.neighborIndices),
              neighborWeights(other.neighborWeights),
              generatedScatterers(other.generatedScatterers), sourceEntityId(other.sourceEntityId),
              particleOffset(other.particleOffset), particleCount(other.particleCount),
              scattererOffset(other.scattererOffset), scattererCount(other.scattererCount),
              component(other.component)
        {
            other.tracker             = nullptr;
            other.neighborIndices     = nullptr;
            other.neighborWeights     = nullptr;
            other.generatedScatterers = nullptr;
            other.sourceEntityId      = common::kInvalidEntityId;
            other.particleOffset      = 0u;
            other.particleCount       = 0u;
            other.scattererOffset     = 0u;
            other.scattererCount      = 0u;
        }

        SourceBinding &operator=(SourceBinding &&other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }

            reset();
            tracker             = other.tracker;
            neighborIndices     = other.neighborIndices;
            neighborWeights     = other.neighborWeights;
            generatedScatterers = other.generatedScatterers;
            sourceEntityId      = other.sourceEntityId;
            particleOffset      = other.particleOffset;
            particleCount       = other.particleCount;
            scattererOffset     = other.scattererOffset;
            scattererCount      = other.scattererCount;
            component           = other.component;

            other.tracker             = nullptr;
            other.neighborIndices     = nullptr;
            other.neighborWeights     = nullptr;
            other.generatedScatterers = nullptr;
            other.sourceEntityId      = common::kInvalidEntityId;
            other.particleOffset      = 0u;
            other.particleCount       = 0u;
            other.scattererOffset     = 0u;
            other.scattererCount      = 0u;
            return *this;
        }
#else
        SourceBinding(SourceBinding &&other) noexcept
            : sourceEntityId(other.sourceEntityId), particleOffset(other.particleOffset),
              particleCount(other.particleCount), scattererOffset(other.scattererOffset),
              scattererCount(other.scattererCount), component(other.component)
        {
            other.sourceEntityId  = common::kInvalidEntityId;
            other.particleOffset  = 0u;
            other.particleCount   = 0u;
            other.scattererOffset = 0u;
            other.scattererCount  = 0u;
        }

        SourceBinding &operator=(SourceBinding &&other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }

            reset();
            sourceEntityId  = other.sourceEntityId;
            particleOffset  = other.particleOffset;
            particleCount   = other.particleCount;
            scattererOffset = other.scattererOffset;
            scattererCount  = other.scattererCount;
            component       = other.component;

            other.sourceEntityId  = common::kInvalidEntityId;
            other.particleOffset  = 0u;
            other.particleCount   = 0u;
            other.scattererOffset = 0u;
            other.scattererCount  = 0u;
            return *this;
        }
#endif

        void reset()
        {
#if CRESSIM_NEO_HAS_ULTRASOUND
            if (tracker != nullptr)
            {
                CruExtDestroyDeformationTracker(tracker);
                tracker = nullptr;
            }
            if (neighborIndices != nullptr)
            {
                CruFreeCudaBuffer(neighborIndices);
                neighborIndices = nullptr;
            }
            if (neighborWeights != nullptr)
            {
                CruFreeCudaBuffer(neighborWeights);
                neighborWeights = nullptr;
            }
            if (generatedScatterers != nullptr)
            {
                CruFreeCudaBuffer(generatedScatterers);
                generatedScatterers = nullptr;
            }
#endif
            scattererCount  = 0u;
            scattererOffset = 0u;
        }

        ~SourceBinding()
        {
            reset();
        }
    };

    struct EnvironmentRuntime
    {
        std::uint64_t softTopologyRevision       = 0u;
        std::uint64_t amplitudeAuthoringRevision = 0u;
        std::vector<common::EntityId> sourceOrder{};
        std::vector<SourceBinding> bindings{};
        std::uint64_t totalScattererCount = 0u;

#if CRESSIM_NEO_HAS_ULTRASOUND
        void *scatterersDevice = nullptr;
#endif

        EnvironmentRuntime()                                      = default;
        EnvironmentRuntime(const EnvironmentRuntime &)            = delete;
        EnvironmentRuntime &operator=(const EnvironmentRuntime &) = delete;
        EnvironmentRuntime(EnvironmentRuntime &&)                 = delete;
        EnvironmentRuntime &operator=(EnvironmentRuntime &&)      = delete;

        void reset()
        {
            for (SourceBinding &binding : bindings)
            {
                binding.reset();
            }
            bindings.clear();
            sourceOrder.clear();
            totalScattererCount        = 0u;
            amplitudeAuthoringRevision = 0u;
#if CRESSIM_NEO_HAS_ULTRASOUND
            if (scatterersDevice != nullptr)
            {
                CruFreeCudaBuffer(scatterersDevice);
                scatterersDevice = nullptr;
            }
#endif
        }

        ~EnvironmentRuntime()
        {
            reset();
        }
    };

    struct ProbeRuntime
    {
        common::EntityId entityId    = common::kInvalidEntityId;
        std::uint32_t entityPoseSlot = kInvalidScenePoseSlot;
        UltrasoundProbeComponent component{};
        UltrasoundRendererComponent renderer{};
        UltrasoundProbeLayout layout{};
        std::uint32_t envIndex            = 0u;
        std::uint64_t boundScattererCount = 0u;

#if CRESSIM_NEO_HAS_ULTRASOUND
        CruComputeHandle engine           = nullptr;
        ExcitationSignalHandle excitation = nullptr;
        BeamProfileHandle beamProfile     = nullptr;
        gpu::SharedExportBuffer rfSharedBuffer{};
        gpu::CudaSharedBuffer rfCudaBuffer{};
        gpu::SharedExportBuffer worldScanlineSharedBuffer{};
        gpu::CudaSharedBuffer worldScanlineCudaBuffer{};
        gpu::CudaSharedBufferBridge worldScanlineBridge{};
        gpu::CudaExternalTimelineSemaphore completionSemaphore{};
        std::uint64_t nextCompletionFenceValue = 1u;
        std::uint32_t rfCapacitySamples        = 0u;
        std::uint32_t scanlineCapacity         = 0u;
        CruRfLayout rfLayout{};
        gpu::GpuRenderTargetBinding imageBinding{};
        bool ownsImageTarget      = false;
        std::uint32_t imageWidth  = 0u;
        std::uint32_t imageHeight = 0u;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> localScanlineTemplateBuffer;
        std::uint32_t localScanlineTemplateCapacity = 0u;
#endif

        ProbeRuntime()                                = default;
        ProbeRuntime(const ProbeRuntime &)            = delete;
        ProbeRuntime &operator=(const ProbeRuntime &) = delete;
        ProbeRuntime(ProbeRuntime &&)                 = delete;
        ProbeRuntime &operator=(ProbeRuntime &&)      = delete;

        void reset()
        {
            entityId            = common::kInvalidEntityId;
            entityPoseSlot      = kInvalidScenePoseSlot;
            boundScattererCount = 0u;
            component           = UltrasoundProbeComponent{};
            renderer            = disabledRendererComponent();
            layout              = UltrasoundProbeLayout{};
#if CRESSIM_NEO_HAS_ULTRASOUND
            if (engine != nullptr)
            {
                CruComputeDestroy(engine);
                engine = nullptr;
            }
            localScanlineTemplateBuffer   = nullptr;
            localScanlineTemplateCapacity = 0u;
            worldScanlineCudaBuffer.reset();
            worldScanlineSharedBuffer.reset();
            worldScanlineBridge.reset();
            rfCudaBuffer.reset();
            rfSharedBuffer.reset();
            completionSemaphore.reset();
            nextCompletionFenceValue = 1u;
            rfCapacitySamples        = 0u;
            scanlineCapacity         = 0u;
            rfLayout                 = CruRfLayout{};
            imageBinding             = {};
            ownsImageTarget          = false;
            imageWidth               = 0u;
            imageHeight              = 0u;
            if (beamProfile != nullptr)
            {
                CruReleaseBeamProfile(beamProfile);
                beamProfile = nullptr;
            }
            if (excitation != nullptr)
            {
                CruReleaseExcitationSignal(excitation);
                excitation = nullptr;
            }
#endif
        }

        ~ProbeRuntime()
        {
            reset();
        }
    };

    struct ImagePassState
    {
        gpu::GpuComputePass maxReducePass;
        gpu::GpuComputePass maxFinalizePass;
        gpu::GpuComputePass linearImagePass;
        gpu::GpuComputePass layeredLinearImagePass;
        gpu::GpuComputePass curvilinearImagePass;
        gpu::GpuComputePass layeredCurvilinearImagePass;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> reductionConstantsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> imageConstantsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> groupMaxBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> finalMaxBuffer;
        std::uint32_t groupMaxCapacity = 0u;
        bool initialized               = false;
    };

    struct ScanlinePassState
    {
        gpu::GpuComputePass pass;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> constantsBuffer;
        bool initialized = false;
    };

    gpu::CudaSharedBufferBridge bridge;
    bool disabled = false;
    std::unordered_map<std::uint32_t, EnvironmentRuntime> environmentRuntimes{};
    std::unordered_map<common::EntityId, ProbeRuntime> probeRuntimes{};
    ImagePassState imagePassState{};
    ScanlinePassState scanlinePassState{};

#if CRESSIM_NEO_HAS_ULTRASOUND
    void destroyProbeImageTarget(gpu::GpuRenderTargetSystem &renderTargetSystem,
                                 ProbeRuntime &runtime)
    {
        if (runtime.ownsImageTarget &&
            renderTargetSystem.isValidRenderTarget(runtime.imageBinding.target))
        {
            renderTargetSystem.destroyRenderTarget(runtime.imageBinding.target);
        }
        runtime.imageBinding    = {};
        runtime.ownsImageTarget = false;
        runtime.imageWidth      = 0u;
        runtime.imageHeight     = 0u;
    }

    void resetProbeRuntime(gpu::GpuRenderTargetSystem &renderTargetSystem, ProbeRuntime &runtime)
    {
        destroyProbeImageTarget(renderTargetSystem, runtime);
        runtime.reset();
    }

    void resetProbeRuntimePreservingImageTarget(ProbeRuntime &runtime)
    {
        const gpu::GpuRenderTargetBinding preservedBinding = runtime.imageBinding;
        const bool preservedOwnership                      = runtime.ownsImageTarget;
        const std::uint32_t preservedWidth                 = runtime.imageWidth;
        const std::uint32_t preservedHeight                = runtime.imageHeight;
        runtime.imageBinding                               = {};
        runtime.ownsImageTarget                            = false;
        runtime.imageWidth                                 = 0u;
        runtime.imageHeight                                = 0u;
        runtime.reset();
        runtime.imageBinding    = preservedBinding;
        runtime.ownsImageTarget = preservedOwnership;
        runtime.imageWidth      = preservedWidth;
        runtime.imageHeight     = preservedHeight;
    }

    void clearPublishedProbeResult(World &world, const common::EntityId probeEntityId) const
    {
        world.clearUltrasoundProbeResult(probeEntityId);
    }

    bool rebuildProbeRuntime(gpu::GpuDevice &device,
                             const gpu::GpuGraphicsBackendContext &graphicsBackend, World &world,
                             const gpu::GpuComputeBackendContext &computeBackend,
                             const common::EntityId probeEntityId, ProbeRuntime &runtime,
                             const std::uint32_t envIndex,
                             const UltrasoundProbeComponent &probeComponent,
                             const UltrasoundRendererComponent &rendererComponent,
                             const std::uint64_t boundScattererCount,
                             const bool preserveImageTarget)
    {
        if (preserveImageTarget)
        {
            resetProbeRuntimePreservingImageTarget(runtime);
        }
        else
        {
            resetProbeRuntime(device.renderTargetSystem(), runtime);
        }

        runtime.component           = probeComponent;
        runtime.renderer            = rendererComponent;
        runtime.envIndex            = envIndex;
        runtime.boundScattererCount = boundScattererCount;
        runtime.entityId            = probeEntityId;
        runtime.entityPoseSlot      = world.entityPoseSlot(probeEntityId);
        if (runtime.entityPoseSlot == kInvalidScenePoseSlot)
        {
            clearPublishedProbeResult(world, probeEntityId);
            return false;
        }

        runtime.engine = CruComputeCreate();
        if (runtime.engine == nullptr)
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: failed to create simulation engine.");
            clearPublishedProbeResult(world, probeEntityId);
            return false;
        }

        // soundSpeed and beam sigmas are authored in physical units and converted to
        // scene units here. Probe geometry fields stay authored directly in scene units.
        const float worldUnitsPerMeter  = std::max(probeComponent.worldUnitsPerMeter, 1.0f);
        const float effectiveSoundSpeed = probeComponent.soundSpeed * worldUnitsPerMeter;
        const float effectiveBeamSigmaLateral =
            probeComponent.beamSigmaLateral * worldUnitsPerMeter;
        const float effectiveBeamSigmaElevational =
            probeComponent.beamSigmaElevational * worldUnitsPerMeter;
        int radialDecimation = static_cast<int>(std::max(probeComponent.radialDecimation, 1u));
        int threadsPerBlock  = static_cast<int>(std::max(probeComponent.threadsPerBlock, 1u));
        int cudaNumStreams   = static_cast<int>(std::max(probeComponent.cudaNumStreams, 1u));
        int numTimeSamples   = static_cast<int>(probeComponent.numTimeSamples);
        if (numTimeSamples <= 0)
        {
            const float requiredMaxTime   = 2.0f * probeComponent.lineLength / effectiveSoundSpeed;
            const auto requiredNumSamples = static_cast<std::uint32_t>(
                std::floor(probeComponent.samplingFrequency * requiredMaxTime + 0.5f));
            numTimeSamples = static_cast<int>(
                std::max<std::uint32_t>(16384u, nextPowerOfTwo(requiredNumSamples + 2048u)));
        }
        CruComputeSetParameter(runtime.engine, CruSimulationParameterType_eSoundSpeed,
                               &effectiveSoundSpeed);
        CruComputeSetParameter(runtime.engine, CruSimulationParameterType_eNoiseAmplitude,
                               &runtime.component.noiseAmplitude);
        CruComputeSetParameter(runtime.engine, CruSimulationParameterType_eUseArcProjection,
                               &runtime.component.useArcProjection);
        CruComputeSetParameter(runtime.engine, CruSimulationParameterType_eRadialDecimation,
                               &radialDecimation);
        CruComputeSetParameter(runtime.engine, CruSimulationParameterType_eEnablePhaseDelay,
                               &runtime.component.enablePhaseDelay);
        CruComputeSetParameter(runtime.engine, CruSimulationParameterType_eNumTimeSamples,
                               &numTimeSamples);
        CruComputeSetParameter(runtime.engine, CruSimulationParameterType_eThreadsPerBlock,
                               &threadsPerBlock);
        CruComputeSetParameter(runtime.engine, CruSimulationParameterType_eCudaNumStreams,
                               &cudaNumStreams);

        runtime.excitation = CruCreateGaussianExcitationSignal(
            runtime.component.samplingFrequency, runtime.component.demodulationFrequency,
            runtime.component.centerFrequency, runtime.component.fractionalBandwidth);
        runtime.beamProfile =
            CruCreateGaussianBeamProfile(effectiveBeamSigmaLateral, effectiveBeamSigmaElevational);
        if (runtime.excitation == nullptr || runtime.beamProfile == nullptr)
        {
            resetProbeRuntime(device.renderTargetSystem(), runtime);
            clearPublishedProbeResult(world, probeEntityId);
            return false;
        }
        CruComputeSetExcitation(runtime.engine, runtime.excitation);
        CruComputeSetBeamProfile(runtime.engine, runtime.beamProfile);
        if (!ensureProbeScanlineBuffers(computeBackend, runtime))
        {
            resetProbeRuntime(device.renderTargetSystem(), runtime);
            clearPublishedProbeResult(world, probeEntityId);
            return false;
        }
        runtime.rfLayout = CruComputeGetRfLayout(runtime.engine);
        if (!ensureProbeRfBuffer(runtime, graphicsBackend.renderDevice,
                                 gpu::contextMaskForId(graphicsBackend.contextId),
                                 runtime.rfLayout) ||
            !ensureProbeImageTarget(device.renderTargetSystem(), runtime))
        {
            resetProbeRuntime(device.renderTargetSystem(), runtime);
            clearPublishedProbeResult(world, probeEntityId);
            return false;
        }

        return true;
    }

    bool ensureProbeRfBuffer(ProbeRuntime &runtime, Diligent::IRenderDevice *renderDevice,
                             Diligent::Uint64 graphicsContextMask, const CruRfLayout &layout)
    {
        runtime.rfLayout = layout;

        if (layout.totalSamples == 0u)
        {
            runtime.rfCudaBuffer.reset();
            runtime.rfSharedBuffer.reset();
            runtime.rfCapacitySamples = 0u;
            CruComputeBindRfLinesDevice(runtime.engine, nullptr, 0u);
            return true;
        }

        if (!runtime.rfSharedBuffer.ensureStructuredBuffer(
                renderDevice, "CRESSimNeo.Engine.UltrasoundProbeRf", sizeof(Diligent::float2),
                layout.totalSamples, layout.totalSamples, Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, graphicsContextMask))
        {
            return false;
        }

        if (runtime.rfCapacitySamples != layout.totalSamples || !runtime.rfCudaBuffer.isImported())
        {
            runtime.rfCudaBuffer.reset();
            if (!runtime.rfCudaBuffer.importFromSharedExportBuffer(runtime.rfSharedBuffer))
            {
                runtime.rfCapacitySamples = 0u;
                return false;
            }
            runtime.rfCapacitySamples = layout.totalSamples;
        }

        CruComputeBindRfLinesDevice(runtime.engine, runtime.rfCudaBuffer.devicePointer(),
                                    runtime.rfCapacitySamples);
        return true;
    }

    bool ensureProbeImageTarget(gpu::GpuRenderTargetSystem &renderTargetSystem,
                                ProbeRuntime &runtime)
    {
        runtime.layout = buildProbeLayout(runtime.component, runtime.renderer, runtime.rfLayout);

        if (!runtime.renderer.enabled)
        {
            destroyProbeImageTarget(renderTargetSystem, runtime);
            return true;
        }

        const gpu::GpuRenderTargetBinding explicitBinding = runtime.renderer.output.binding;
        if (runtime.renderer.output.mode == gpu::RenderOutputMode::ExplicitSurface &&
            explicitBinding.isValid())
        {
            gpu::GpuRenderTargetDesc targetDesc{};
            if (!renderTargetSystem.tryGetRenderTargetDesc(explicitBinding.target, targetDesc) ||
                !targetDesc.color || targetDesc.width != runtime.layout.imageWidth ||
                targetDesc.height != runtime.layout.imageHeight ||
                targetDesc.colorFormat != runtime.layout.colorFormat ||
                explicitBinding.layerCount != 1u ||
                explicitBinding.firstLayer + explicitBinding.layerCount > targetDesc.arraySize)
            {
                return false;
            }

            if (runtime.ownsImageTarget)
            {
                destroyProbeImageTarget(renderTargetSystem, runtime);
            }
            runtime.imageBinding    = explicitBinding;
            runtime.ownsImageTarget = false;
            runtime.imageWidth      = runtime.layout.imageWidth;
            runtime.imageHeight     = runtime.layout.imageHeight;
            return true;
        }

        if (renderTargetSystem.isValidRenderTarget(runtime.imageBinding.target) &&
            runtime.ownsImageTarget && runtime.imageWidth == runtime.layout.imageWidth &&
            runtime.imageHeight == runtime.layout.imageHeight)
        {
            return true;
        }

        destroyProbeImageTarget(renderTargetSystem, runtime);

        gpu::GpuRenderTargetDesc desc{};
        desc.width            = runtime.layout.imageWidth;
        desc.height           = runtime.layout.imageHeight;
        desc.arraySize        = 1u;
        desc.color            = true;
        desc.depth            = false;
        desc.layeredRendering = false;
        desc.shaderReadable   = true;
        desc.unorderedAccess  = true;
        desc.colorFormat      = runtime.layout.colorFormat;
        desc.debugName        = "CRESSimNeo.Ultrasound.Image";

        const gpu::GpuRenderTargetHandle imageTarget = renderTargetSystem.createRenderTarget(desc);
        if (!renderTargetSystem.isValidRenderTarget(imageTarget))
        {
            return false;
        }

        runtime.imageBinding    = gpu::GpuRenderTargetBinding{imageTarget, 0u, 1u};
        runtime.ownsImageTarget = true;
        runtime.imageWidth      = runtime.layout.imageWidth;
        runtime.imageHeight     = runtime.layout.imageHeight;
        return true;
    }

    bool ensureProbeCompletionSemaphoreInitialized(ProbeRuntime &runtime,
                                                   Diligent::IRenderDevice *renderDevice,
                                                   const common::EntityId probeEntityId)
    {
        if (runtime.completionSemaphore.isInitialized())
        {
            return true;
        }
        const std::string semaphoreName =
            "CRESSimNeo.Ultrasound.Completion." + std::to_string(probeEntityId);
        if (renderDevice == nullptr ||
            !runtime.completionSemaphore.initialize(renderDevice, semaphoreName.c_str()) ||
            !runtime.completionSemaphore.importIntoCuda())
        {
            runtime.completionSemaphore.reset();
            runtime.nextCompletionFenceValue = 1u;
            return false;
        }
        return true;
    }

    bool writeBuffer(Diligent::IDeviceContext *context, Diligent::IBuffer *buffer, const void *data,
                     const std::size_t sizeBytes)
    {
        if (context == nullptr || buffer == nullptr || data == nullptr || sizeBytes == 0u)
        {
            return false;
        }

        const Diligent::BufferDesc &desc = buffer->GetDesc();
        if (desc.Usage != Diligent::USAGE_DYNAMIC)
        {
            context->UpdateBuffer(buffer, 0u, static_cast<Diligent::Uint32>(sizeBytes), data,
                                  Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
            return true;
        }

        void *mapped = nullptr;
        context->MapBuffer(buffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mapped);
        if (mapped == nullptr)
        {
            return false;
        }
        std::memcpy(mapped, data, sizeBytes);
        context->UnmapBuffer(buffer, Diligent::MAP_WRITE);
        return true;
    }

    std::vector<CruPackedScanline> buildLocalScanlineTemplate(
        const UltrasoundProbeComponent &probeComponent)
    {
        std::vector<CruPackedScanline> scanlines;
        scanlines.reserve(probeComponent.numScanlines);

        const Diligent::float3 forward{0.0f, 0.0f, 1.0f};
        const Diligent::float3 lateralBase{1.0f, 0.0f, 0.0f};
        for (std::uint32_t i = 0u; i < probeComponent.numScanlines; ++i)
        {
            Diligent::float3 localOrigin{0.0f, 0.0f, 0.0f};
            Diligent::float3 direction = forward;
            Diligent::float3 lateral   = lateralBase;
            if (probeComponent.geometry == UltrasoundProbeComponent::Geometry::Curvilinear)
            {
                const float angleSpanRadians =
                    resolveCurvilinearSectorAngleRadians(probeComponent.sectorAngleDegrees);
                const float angleStep =
                    probeComponent.numScanlines > 1u
                        ? angleSpanRadians / static_cast<float>(probeComponent.numScanlines - 1u)
                        : 0.0f;
                const float angle    = -0.5f * angleSpanRadians + static_cast<float>(i) * angleStep;
                const float sinAngle = std::sin(angle);
                const float cosAngle = std::cos(angle);
                const float radius   = std::max(probeComponent.probeRadius, 0.0f);

                localOrigin =
                    lateralBase * (sinAngle * radius) + forward * ((cosAngle - 1.0f) * radius);
                direction = Diligent::normalize(lateralBase * sinAngle + forward * cosAngle);
                lateral   = Diligent::normalize(lateralBase * cosAngle - forward * sinAngle);
            }
            else
            {
                const float centeredIndex =
                    static_cast<float>(i) -
                    0.5f * static_cast<float>(probeComponent.numScanlines - 1u);
                localOrigin = lateralBase * (centeredIndex * probeComponent.scanlineSpacing);
            }

            const Diligent::float3 elevational =
                Diligent::normalize(Diligent::cross(lateral, direction));
            CruPackedScanline packed{};
            packed.origin[0]               = localOrigin.x;
            packed.origin[1]               = localOrigin.y;
            packed.origin[2]               = localOrigin.z;
            packed.origin[3]               = 0.0f;
            packed.radialDirection[0]      = direction.x;
            packed.radialDirection[1]      = direction.y;
            packed.radialDirection[2]      = direction.z;
            packed.radialDirection[3]      = 0.0f;
            packed.lateralDirection[0]     = lateral.x;
            packed.lateralDirection[1]     = lateral.y;
            packed.lateralDirection[2]     = lateral.z;
            packed.lateralDirection[3]     = 0.0f;
            packed.elevationalDirection[0] = elevational.x;
            packed.elevationalDirection[1] = elevational.y;
            packed.elevationalDirection[2] = elevational.z;
            packed.elevationalDirection[3] = 0.0f;
            scanlines.push_back(packed);
        }

        return scanlines;
    }

    bool ensureScanlinePassInitialized(gpu::GpuDevice &device,
                                       const gpu::GpuComputeBackendContext &computeBackend)
    {
        if (scanlinePassState.initialized)
        {
            return true;
        }

        gpu::ShaderLibrary shaderLibrary(device.shaderSourceDirectory());
        Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
        if (streamFactory == nullptr)
        {
            return false;
        }

        const Diligent::Uint64 computeContextMask = gpu::contextMaskForId(computeBackend.contextId);
        if (!scanlinePassState.pass.initialize(device, streamFactory, computeContextMask,
                                               kUltrasoundScanlineUpdatePassDefinition))
        {
            return false;
        }

        Diligent::BufferDesc constantsDesc{};
        constantsDesc.Name                 = "CRESSimNeo.Ultrasound.ScanlineUpdateConstants";
        constantsDesc.Size                 = sizeof(UltrasoundScanlineUpdateConstants);
        constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
        constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
        constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
        constantsDesc.ImmediateContextMask = computeContextMask;
        computeBackend.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                                  &scanlinePassState.constantsBuffer);
        if (scanlinePassState.constantsBuffer == nullptr)
        {
            return false;
        }

        scanlinePassState.initialized = true;
        return true;
    }

    bool ensureProbeScanlineBuffers(const gpu::GpuComputeBackendContext &computeBackend,
                                    ProbeRuntime &runtime)
    {
        const auto localScanlines         = buildLocalScanlineTemplate(runtime.component);
        const std::uint32_t scanlineCount = static_cast<std::uint32_t>(localScanlines.size());

        if (!gpu::detail::ensureStructuredBufferCapacity(
                computeBackend.renderDevice, "CRESSimNeo.Ultrasound.LocalScanlines",
                sizeof(CruPackedScanline), scanlineCount,
                std::max<std::uint32_t>(scanlineCount, 1u), Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
                gpu::contextMaskForId(computeBackend.contextId),
                runtime.localScanlineTemplateBuffer, runtime.localScanlineTemplateCapacity))
        {
            return false;
        }

        if (scanlineCount > 0u &&
            !writeBuffer(computeBackend.computeContext, runtime.localScanlineTemplateBuffer,
                         localScanlines.data(), localScanlines.size() * sizeof(CruPackedScanline)))
        {
            return false;
        }

        if (!runtime.worldScanlineSharedBuffer.ensureStructuredBuffer(
                computeBackend.renderDevice, "CRESSimNeo.Engine.Ultrasound.WorldScanlines",
                sizeof(CruPackedScanline), scanlineCount,
                std::max<std::uint32_t>(scanlineCount, 1u),
                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
                gpu::contextMaskForId(computeBackend.contextId)))
        {
            return false;
        }
        runtime.scanlineCapacity = runtime.worldScanlineSharedBuffer.capacity();

        if (scanlineCount == 0u)
        {
            runtime.worldScanlineCudaBuffer.reset();
            runtime.worldScanlineBridge.reset();
            CruComputeBindScanlinesDevice(runtime.engine, nullptr, 0u,
                                          runtime.component.lineLength);
            return true;
        }

        if (!runtime.worldScanlineCudaBuffer.importFromSharedExportBuffer(
                runtime.worldScanlineSharedBuffer))
        {
            return false;
        }
        if (!runtime.worldScanlineBridge.isInitialized() &&
            !runtime.worldScanlineBridge.initialize(computeBackend.renderDevice,
                                                    "CRESSimNeo.Ultrasound.ScanlineBridge"))
        {
            return false;
        }
        if (!runtime.worldScanlineBridge.bindSharedBuffer(runtime.worldScanlineSharedBuffer))
        {
            return false;
        }

        CruComputeBindScanlinesDevice(
            runtime.engine,
            static_cast<const CruPackedScanline *>(runtime.worldScanlineCudaBuffer.devicePointer()),
            scanlineCount, runtime.component.lineLength);
        return true;
    }

    bool updateProbeScanlines(const gpu::GpuComputeBackendContext &computeBackend,
                              const common::PoseBufferView &poseView, ProbeRuntime &runtime)
    {
        if (!scanlinePassState.initialized || runtime.localScanlineTemplateBuffer == nullptr ||
            runtime.worldScanlineSharedBuffer.buffer() == nullptr)
        {
            return false;
        }
        if (runtime.entityPoseSlot == kInvalidScenePoseSlot ||
            runtime.entityPoseSlot >= poseView.count || poseView.positionsBuffer == nullptr ||
            poseView.orientationsBuffer == nullptr)
        {
            return false;
        }

        UltrasoundScanlineUpdateConstants constants{};
        constants.entityPoseSlot = runtime.entityPoseSlot;
        constants.scanlineCount  = runtime.component.numScanlines;
        if (!writeBuffer(computeBackend.computeContext, scanlinePassState.constantsBuffer,
                         &constants, sizeof(constants)))
        {
            return false;
        }

        const std::array bindings{
            gpu::GpuBufferBinding{"UltrasoundScanlineUpdateConstantsBuffer",
                                  scanlinePassState.constantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_EntityPositions", poseView.positionsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_EntityOrientations", poseView.orientationsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_LocalScanlines", runtime.localScanlineTemplateBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_WorldScanlinesRW", runtime.worldScanlineSharedBuffer.buffer(),
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!scanlinePassState.pass.dispatch(
                computeBackend.computeContext, 0u, bindings,
                dispatchGroupCount(runtime.component.numScanlines,
                                   kUltrasoundScanlineThreadGroupSize)))
        {
            return false;
        }

        return runtime.worldScanlineBridge.synchronizeFromDeviceContext(
            computeBackend.computeContext);
    }

    bool ensureImagePassInitialized(gpu::GpuDevice &device,
                                    const gpu::GpuGraphicsBackendContext &graphicsBackend)
    {
        if (imagePassState.initialized)
        {
            return true;
        }

        gpu::ShaderLibrary shaderLibrary(device.shaderSourceDirectory());
        Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
        if (streamFactory == nullptr)
        {
            return false;
        }

        const Diligent::Uint64 graphicsContextMask =
            gpu::contextMaskForId(graphicsBackend.contextId);
        if (!imagePassState.maxReducePass.initialize(device, streamFactory, graphicsContextMask,
                                                     kUltrasoundMaxReducePassDefinition) ||
            !imagePassState.maxFinalizePass.initialize(device, streamFactory, graphicsContextMask,
                                                       kUltrasoundMaxFinalizePassDefinition) ||
            !imagePassState.linearImagePass.initialize(device, streamFactory, graphicsContextMask,
                                                       kUltrasoundImagePassDefinition) ||
            !imagePassState.layeredLinearImagePass.initialize(
                device, streamFactory, graphicsContextMask,
                kUltrasoundLayeredImagePassDefinition) ||
            !imagePassState.curvilinearImagePass.initialize(
                device, streamFactory, graphicsContextMask,
                kUltrasoundCurvilinearImagePassDefinition) ||
            !imagePassState.layeredCurvilinearImagePass.initialize(
                device, streamFactory, graphicsContextMask,
                kUltrasoundLayeredCurvilinearImagePassDefinition))
        {
            return false;
        }

        Diligent::BufferDesc constantsDesc{};
        constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
        constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
        constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
        constantsDesc.ImmediateContextMask = graphicsContextMask;

        constantsDesc.Name = "CRESSimNeo.Ultrasound.ReductionConstants";
        constantsDesc.Size = sizeof(UltrasoundReductionConstants);
        graphicsBackend.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                                   &imagePassState.reductionConstantsBuffer);
        if (imagePassState.reductionConstantsBuffer == nullptr)
        {
            return false;
        }

        constantsDesc.Name = "CRESSimNeo.Ultrasound.ImageConstants";
        constantsDesc.Size = sizeof(UltrasoundImageConstants);
        graphicsBackend.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                                   &imagePassState.imageConstantsBuffer);
        if (imagePassState.imageConstantsBuffer == nullptr)
        {
            return false;
        }

        imagePassState.initialized = true;
        return true;
    }

    bool ensureImageReductionBuffers(Diligent::IRenderDevice *renderDevice,
                                     Diligent::Uint64 graphicsContextMask,
                                     const std::uint32_t totalSamples)
    {
        const std::uint32_t groupCount = std::max<std::uint32_t>(
            1u, dispatchGroupCount(totalSamples, kUltrasoundReductionThreadGroupSize));

        if (!gpu::detail::ensureStructuredBufferCapacity(
                renderDevice, "CRESSimNeo.Ultrasound.GroupMax", sizeof(float), groupCount,
                groupCount, Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, graphicsContextMask,
                imagePassState.groupMaxBuffer, imagePassState.groupMaxCapacity))
        {
            return false;
        }

        std::uint32_t finalMaxCapacity = 0u;
        return gpu::detail::ensureStructuredBufferCapacity(
            renderDevice, "CRESSimNeo.Ultrasound.FinalMax", sizeof(float), 1u, 1u,
            Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, graphicsContextMask,
            imagePassState.finalMaxBuffer, finalMaxCapacity);
    }

    void publishProbeResult(World &world, const common::EntityId probeEntityId,
                            const ProbeRuntime &runtime, const bool completed,
                            const std::uint64_t totalScattererCount = 0u,
                            const std::uint64_t frameIndex          = 0u) const
    {
        UltrasoundProbeResult result{};
        result.prepared            = true;
        result.completed           = completed;
        result.numScanlines        = runtime.rfLayout.numScanlines;
        result.samplesPerScanline  = runtime.rfLayout.samplesPerLine;
        result.totalScattererCount = totalScattererCount;
        result.imageWidth          = runtime.imageWidth;
        result.imageHeight         = runtime.imageHeight;
        result.imageBinding        = runtime.imageBinding;
        if (completed)
        {
            result.completedFrameIndex = frameIndex;
        }
        world.setUltrasoundProbeResult(probeEntityId, result);
    }

    bool renderProbeImage(gpu::GpuDevice &device,
                          const gpu::GpuGraphicsBackendContext &graphicsBackend,
                          ProbeRuntime &runtime)
    {
        if (!runtime.renderer.enabled || !runtime.rfSharedBuffer.buffer() ||
            !device.renderTargetSystem().isValidRenderTarget(runtime.imageBinding.target))
        {
            return false;
        }

        Diligent::ITexture *imageTexture = nullptr;
        if (!device.renderTargetSystem().tryGetRenderTargetColorTexture(runtime.imageBinding.target,
                                                                        imageTexture) ||
            imageTexture == nullptr)
        {
            return false;
        }

        Diligent::ITextureView *imageUav              = nullptr;
        const Diligent::TextureDesc &imageTextureDesc = imageTexture->GetDesc();
        imageUav = imageTexture->GetDefaultView(Diligent::TEXTURE_VIEW_UNORDERED_ACCESS);
        if (imageUav == nullptr)
        {
            return false;
        }
        const bool layeredOutput = imageTextureDesc.Type == Diligent::RESOURCE_DIM_TEX_2D_ARRAY;
        if (layeredOutput && runtime.imageBinding.layerCount != 1u)
        {
            return false;
        }

        const Diligent::Uint64 graphicsContextMask =
            gpu::contextMaskForId(graphicsBackend.contextId);
        if (!ensureImageReductionBuffers(graphicsBackend.renderDevice, graphicsContextMask,
                                         runtime.rfLayout.totalSamples))
        {
            return false;
        }

        if (!runtime.renderer.useFixedMaxNormalization)
        {
            const std::uint32_t groupCount = std::max<std::uint32_t>(
                1u, dispatchGroupCount(runtime.rfLayout.totalSamples,
                                       kUltrasoundReductionThreadGroupSize));

            UltrasoundReductionConstants reductionConstants{};
            reductionConstants.dataLength = runtime.rfLayout.totalSamples;

            void *mappedConstants = nullptr;
            graphicsBackend.graphicsContext->MapBuffer(imagePassState.reductionConstantsBuffer,
                                                       Diligent::MAP_WRITE,
                                                       Diligent::MAP_FLAG_DISCARD, mappedConstants);
            if (mappedConstants == nullptr)
            {
                return false;
            }
            std::memcpy(mappedConstants, &reductionConstants, sizeof(reductionConstants));
            graphicsBackend.graphicsContext->UnmapBuffer(imagePassState.reductionConstantsBuffer,
                                                         Diligent::MAP_WRITE);

            const std::array maxReduceBindings{
                gpu::GpuBufferBinding{"UltrasoundReductionConstants",
                                      imagePassState.reductionConstantsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_RfData", runtime.rfSharedBuffer.buffer(),
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_GroupMaxRW", imagePassState.groupMaxBuffer,
                                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            };

            if (!imagePassState.maxReducePass.dispatch(graphicsBackend.graphicsContext, 0u,
                                                       maxReduceBindings, groupCount, 1u, 1u))
            {
                return false;
            }

            reductionConstants.dataLength = groupCount;
            mappedConstants               = nullptr;
            graphicsBackend.graphicsContext->MapBuffer(imagePassState.reductionConstantsBuffer,
                                                       Diligent::MAP_WRITE,
                                                       Diligent::MAP_FLAG_DISCARD, mappedConstants);
            if (mappedConstants == nullptr)
            {
                return false;
            }
            std::memcpy(mappedConstants, &reductionConstants, sizeof(reductionConstants));
            graphicsBackend.graphicsContext->UnmapBuffer(imagePassState.reductionConstantsBuffer,
                                                         Diligent::MAP_WRITE);

            const std::array maxFinalizeBindings{
                gpu::GpuBufferBinding{"UltrasoundReductionConstants",
                                      imagePassState.reductionConstantsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_GroupMax", imagePassState.groupMaxBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_FinalMaxRW", imagePassState.finalMaxBuffer,
                                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            };
            if (!imagePassState.maxFinalizePass.dispatch(graphicsBackend.graphicsContext, 0u,
                                                         maxFinalizeBindings, 1u, 1u, 1u))
            {
                return false;
            }
        }

        UltrasoundImageConstants imageConstants{};
        const UltrasoundImageSize imageSize =
            computeImageSize(runtime.component, runtime.renderer, runtime.rfLayout);
        imageConstants.numScanlines   = runtime.rfLayout.numScanlines;
        imageConstants.samplesPerLine = runtime.rfLayout.samplesPerLine;
        imageConstants.imageWidth     = runtime.imageWidth;
        imageConstants.imageHeight    = runtime.imageHeight;
        imageConstants.outputLayer    = layeredOutput ? runtime.imageBinding.firstLayer : 0u;
        imageConstants.fixedMaxSignal = std::max(runtime.renderer.fixedMaxSignal, 1.0e-6f);
        imageConstants.useFixedMaxNormalization =
            runtime.renderer.useFixedMaxNormalization ? 1u : 0u;
        imageConstants.sectorAngleRadians = imageSize.sectorAngleRadians;
        imageConstants.probeRadiusPixels  = imageSize.probeRadiusPixels;

        void *mappedImageConstants = nullptr;
        graphicsBackend.graphicsContext->MapBuffer(imagePassState.imageConstantsBuffer,
                                                   Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                                                   mappedImageConstants);
        if (mappedImageConstants == nullptr)
        {
            return false;
        }
        std::memcpy(mappedImageConstants, &imageConstants, sizeof(imageConstants));
        graphicsBackend.graphicsContext->UnmapBuffer(imagePassState.imageConstantsBuffer,
                                                     Diligent::MAP_WRITE);

        const std::array imageBufferBindings{
            gpu::GpuBufferBinding{"UltrasoundImageConstants", imagePassState.imageConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RfData", runtime.rfSharedBuffer.buffer(),
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_FinalMax", imagePassState.finalMaxBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        };
        const std::array imageTextureBindings{
            gpu::GpuTextureBinding{"g_OutputImageRW", imageUav},
        };

        gpu::GpuComputePass &imagePass =
            runtime.component.geometry == UltrasoundProbeComponent::Geometry::Curvilinear
                ? (layeredOutput ? imagePassState.layeredCurvilinearImagePass
                                 : imagePassState.curvilinearImagePass)
                : (layeredOutput ? imagePassState.layeredLinearImagePass
                                 : imagePassState.linearImagePass);

        return imagePass.dispatchResources(
            graphicsBackend.graphicsContext, 0u, imageBufferBindings, imageTextureBindings,
            dispatchGroupCount(runtime.imageWidth, kUltrasoundImageThreadGroupSizeX),
            dispatchGroupCount(runtime.imageHeight, kUltrasoundImageThreadGroupSizeY), 1u);
    }

#endif
#if !CRESSIM_NEO_HAS_ULTRASOUND
    void destroyProbeImageTarget(gpu::GpuRenderTargetSystem &renderTargetSystem,
                                 ProbeRuntime &runtime)
    {
        (void)renderTargetSystem;
        (void)runtime;
    }
#endif
};

UltrasoundSystem::UltrasoundSystem(gpu::GpuDevice &device, physics::PhysicsSolver &physicsSolver)
    : mDevice(device), mPhysicsSolver(physicsSolver), mImpl(std::make_unique<Impl>())
{
}

UltrasoundSystem::~UltrasoundSystem()
{
    shutdown();
}

bool UltrasoundSystem::computeProbeLayout(const UltrasoundProbeComponent &probeComponent,
                                          const UltrasoundRendererComponent &rendererComponent,
                                          UltrasoundProbeLayout &outLayout) const
{
    return computeUltrasoundProbeLayout(probeComponent, rendererComponent, outLayout);
}

bool UltrasoundSystem::initialize()
{
    shutdown();
    mImpl        = std::make_unique<Impl>();
    mInitialized = true;
    return true;
}

void UltrasoundSystem::shutdown()
{
    if (mImpl)
    {
        for (auto &[entityId, runtime] : mImpl->probeRuntimes)
        {
            (void)entityId;
            mImpl->destroyProbeImageTarget(mDevice.renderTargetSystem(), runtime);
        }
        mImpl->probeRuntimes.clear();
        mImpl->environmentRuntimes.clear();
        mImpl->bridge.reset();
    }
    mInitialized = false;
}

bool UltrasoundSystem::prepare(World &world)
{
    if (!mInitialized)
    {
        return false;
    }

#if !CRESSIM_NEO_HAS_ULTRASOUND
    (void)world;
    return true;
#else
    if (mImpl->disabled)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext graphicsBackend{};
    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsBackend) ||
        graphicsBackend.renderDevice == nullptr)
    {
        return true;
    }
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr)
    {
        return true;
    }

    const auto &probeComponents               = world.ultrasoundProbeComponents();
    const auto &rendererComponents            = world.ultrasoundRendererComponents();
    const auto &sourceComponents              = world.ultrasoundScattererSourceComponents();
    const physics::PhysicsWorld &physicsWorld = world.physicsWorld();

    for (auto it = mImpl->probeRuntimes.begin(); it != mImpl->probeRuntimes.end();)
    {
        if (probeComponents.find(it->first) == probeComponents.end() ||
            !probeComponents.at(it->first).enabled)
        {
            world.clearUltrasoundProbeResult(it->first);
            mImpl->destroyProbeImageTarget(mDevice.renderTargetSystem(), it->second);
            it = mImpl->probeRuntimes.erase(it);
            continue;
        }
        ++it;
    }

    std::unordered_set<std::uint32_t> validSourceEnvironments;
    for (const auto &[sourceEntityId, sourceComponent] : sourceComponents)
    {
        if (!sourceComponent.enabled)
        {
            continue;
        }

        const physics::SoftBodyState *softBody = physicsWorld.tryGetSoftBody(sourceEntityId);
        if (softBody == nullptr || !softBody->simulated || softBody->particleCount == 0u)
        {
            continue;
        }

        const auto *authoredRanges = world.tryGetUltrasoundScattererAmplitudeRanges(sourceEntityId);
        if (authoredRanges == nullptr || authoredRanges->size() != softBody->particleCount)
        {
            continue;
        }

        validSourceEnvironments.insert(world.entityEnvironment(sourceEntityId));
    }

    for (const auto &[probeEntityId, probeComponent] : probeComponents)
    {
        if (!probeComponent.enabled)
        {
            world.clearUltrasoundProbeResult(probeEntityId);
            continue;
        }

        const std::uint32_t envIndex = world.entityEnvironment(probeEntityId);
        if (validSourceEnvironments.find(envIndex) == validSourceEnvironments.end())
        {
            world.clearUltrasoundProbeResult(probeEntityId);
            auto runtimeIt = mImpl->probeRuntimes.find(probeEntityId);
            if (runtimeIt != mImpl->probeRuntimes.end())
            {
                mImpl->destroyProbeImageTarget(mDevice.renderTargetSystem(), runtimeIt->second);
                mImpl->probeRuntimes.erase(runtimeIt);
            }
            continue;
        }

        Impl::ProbeRuntime &runtime = mImpl->probeRuntimes[probeEntityId];
        const auto rendererIt       = rendererComponents.find(probeEntityId);
        const UltrasoundRendererComponent rendererComponent = rendererIt != rendererComponents.end()
                                                                  ? rendererIt->second
                                                                  : disabledRendererComponent();
        bool rebuild = runtime.engine == nullptr || runtime.envIndex != envIndex ||
                       !equalsProbeComponent(runtime.component, probeComponent) ||
                       !equalsRendererComponent(runtime.renderer, rendererComponent);
        if (rebuild && !mImpl->rebuildProbeRuntime(mDevice, graphicsBackend, world, computeBackend,
                                                   probeEntityId, runtime, envIndex, probeComponent,
                                                   rendererComponent, 0u, false))
        {
            continue;
        }

        const CruRfLayout refreshedLayout = CruComputeGetRfLayout(runtime.engine);
        if (!equalsRfLayout(runtime.rfLayout, refreshedLayout) &&
            !mImpl->ensureProbeRfBuffer(runtime, graphicsBackend.renderDevice,
                                        gpu::contextMaskForId(graphicsBackend.contextId),
                                        refreshedLayout))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " could not resize RF output buffer.");
            mImpl->clearPublishedProbeResult(world, probeEntityId);
            continue;
        }
        runtime.rfLayout = refreshedLayout;

        if (!mImpl->ensureProbeImageTarget(mDevice.renderTargetSystem(), runtime))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " could not create image target.");
            mImpl->clearPublishedProbeResult(world, probeEntityId);
            continue;
        }
        mImpl->publishProbeResult(world, probeEntityId, runtime, false);
    }

    return true;
#endif
}

bool UltrasoundSystem::execute(const common::FrameContext &frameContext, World &world)
{
    if (!mInitialized)
    {
        return false;
    }

#if !CRESSIM_NEO_HAS_ULTRASOUND
    (void)frameContext;
    (void)world;
    return true;
#else
    if (mImpl->disabled)
    {
        return false;
    }

    const auto *sharedBuffer = mPhysicsSolver.softPositionsInvMassSharedBuffer();
    if (sharedBuffer == nullptr || !sharedBuffer->usesNativeSharedAllocation())
    {
        return true;
    }

    gpu::GpuGraphicsBackendContext graphicsBackend{};
    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetGraphicsBackendContext(graphicsBackend) ||
        graphicsBackend.renderDevice == nullptr || graphicsBackend.graphicsContext == nullptr)
    {
        return true;
    }
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr)
    {
        return true;
    }

    if (!mImpl->ensureImagePassInitialized(mDevice, graphicsBackend))
    {
        CRESSIM_LOG_WARNING("UltrasoundSystem: failed to initialize image compute passes.");
        mImpl->disabled = true;
        return false;
    }
    if (!mImpl->ensureScanlinePassInitialized(mDevice, computeBackend))
    {
        CRESSIM_LOG_WARNING("UltrasoundSystem: failed to initialize scanline compute pass.");
        mImpl->disabled = true;
        return false;
    }

    if (!mImpl->bridge.isInitialized() &&
        !mImpl->bridge.initialize(computeBackend.renderDevice,
                                  "CRESSimNeo.Engine.UltrasoundSystem"))
    {
        CRESSIM_LOG_WARNING("UltrasoundSystem: failed to initialize CUDA interop bridge.");
        mImpl->disabled = true;
        return false;
    }

    if (!mImpl->bridge.bindSharedBuffer(*sharedBuffer) ||
        !mImpl->bridge.synchronizeFromDeviceContext(computeBackend.computeContext))
    {
        CRESSIM_LOG_WARNING("UltrasoundSystem: failed to synchronize shared soft positions.");
        mImpl->disabled = true;
        return false;
    }

    const auto &probeComponents                        = world.ultrasoundProbeComponents();
    const auto &rendererComponents                     = world.ultrasoundRendererComponents();
    const graphics::GpuEntitySceneView &gpuEntityScene = world.gpuEntityScene();
    auto *sharedBase = static_cast<std::byte *>(mImpl->bridge.devicePointer());
    if (sharedBase == nullptr)
    {
        return true;
    }

    const auto &sourceComponents                   = world.ultrasoundScattererSourceComponents();
    const physics::PhysicsWorld &physicsWorld      = world.physicsWorld();
    const std::uint64_t amplitudeAuthoringRevision = world.ultrasoundScattererAmplitudeRevision();

    std::unordered_map<
        std::uint32_t,
        std::vector<std::pair<const physics::SoftBodyState *, UltrasoundScattererSourceComponent>>>
        sourcesByEnv;
    for (const auto &[sourceEntityId, sourceComponent] : sourceComponents)
    {
        if (!sourceComponent.enabled)
        {
            continue;
        }
        const physics::SoftBodyState *softBody = physicsWorld.tryGetSoftBody(sourceEntityId);
        if (softBody == nullptr || !softBody->simulated || softBody->particleCount == 0u)
        {
            continue;
        }
        sourcesByEnv[world.entityEnvironment(sourceEntityId)].emplace_back(softBody,
                                                                           sourceComponent);
    }

    for (auto it = mImpl->environmentRuntimes.begin(); it != mImpl->environmentRuntimes.end();)
    {
        if (sourcesByEnv.find(it->first) == sourcesByEnv.end())
        {
            it = mImpl->environmentRuntimes.erase(it);
        }
        else
        {
            ++it;
        }
    }

    for (auto &[envIndex, sources] : sourcesByEnv)
    {
        struct PendingSource
        {
            const physics::SoftBodyState *softBody = nullptr;
            UltrasoundScattererSourceComponent component{};
            const std::vector<UltrasoundAmplitudeRange> *amplitudeRanges = nullptr;
        };

        std::vector<PendingSource> validSources;
        validSources.reserve(sources.size());
        for (const auto &[softBody, sourceComponent] : sources)
        {
            const auto *authoredRanges =
                world.tryGetUltrasoundScattererAmplitudeRanges(softBody->entityId);
            if (authoredRanges == nullptr || authoredRanges->size() != softBody->particleCount)
            {
                continue;
            }
            validSources.push_back(PendingSource{softBody, sourceComponent, authoredRanges});
        }

        Impl::EnvironmentRuntime &envRuntime = mImpl->environmentRuntimes[envIndex];
        const bool needsScattererStorage     = !validSources.empty();
        bool rebuild = (needsScattererStorage && envRuntime.scatterersDevice == nullptr) ||
                       envRuntime.softTopologyRevision != physicsWorld.softBodyTopologyRevision() ||
                       envRuntime.amplitudeAuthoringRevision != amplitudeAuthoringRevision ||
                       envRuntime.sourceOrder.size() != validSources.size();
        if (!rebuild)
        {
            for (std::size_t i = 0; i < validSources.size(); ++i)
            {
                if (envRuntime.sourceOrder[i] != validSources[i].softBody->entityId ||
                    !equalsSourceComponent(envRuntime.bindings[i].component,
                                           validSources[i].component) ||
                    envRuntime.bindings[i].particleOffset !=
                        validSources[i].softBody->particleOffset ||
                    envRuntime.bindings[i].particleCount != validSources[i].softBody->particleCount)
                {
                    rebuild = true;
                    break;
                }
            }
        }

        if (rebuild)
        {
            envRuntime.reset();
            envRuntime.softTopologyRevision       = physicsWorld.softBodyTopologyRevision();
            envRuntime.amplitudeAuthoringRevision = amplitudeAuthoringRevision;

            if (validSources.empty())
            {
                continue;
            }

            std::vector<std::vector<UltrasoundAmplitudeRange>> amplitudeRanges;
            amplitudeRanges.reserve(validSources.size());
            std::vector<CruBounds3> sourceBounds;
            sourceBounds.reserve(validSources.size());
            for (const PendingSource &source : validSources)
            {
                const float pointDistance =
                    resolvePointDistance(*source.softBody, source.component);
                sourceBounds.push_back(computeBounds(*source.softBody, pointDistance));
                amplitudeRanges.emplace_back(source.amplitudeRanges->begin(),
                                             source.amplitudeRanges->end());
            }

            for (std::size_t i = 0; i < validSources.size(); ++i)
            {
                const PendingSource &source = validSources[i];
                const float pointDistance =
                    resolvePointDistance(*source.softBody, source.component);
                CruGenerateScatterersFromPointsConfig config{};
                config.bounds          = sourceBounds[i];
                config.density         = std::max(source.component.density, 1.0f);
                config.seed            = 1337u + static_cast<std::uint64_t>(i);
                config.threadsPerBlock = 128;
                config.pointDistance   = pointDistance;

                void *generatedScatterers    = nullptr;
                void *neighborIndices        = nullptr;
                void *neighborWeights        = nullptr;
                std::uint64_t scattererCount = 0u;
                void *pointPointer =
                    sharedBase + static_cast<std::size_t>(source.softBody->particleOffset) *
                                     sizeof(Diligent::float4);
                CruExtGenerateScatterersFromPoints(pointPointer, amplitudeRanges[i].data(),
                                                   static_cast<int>(source.softBody->particleCount),
                                                   &config, &generatedScatterers, &neighborIndices,
                                                   &neighborWeights, &scattererCount);

                Impl::SourceBinding binding{};
                binding.sourceEntityId      = source.softBody->entityId;
                binding.particleOffset      = source.softBody->particleOffset;
                binding.particleCount       = source.softBody->particleCount;
                binding.component           = source.component;
                binding.scattererCount      = scattererCount;
                binding.neighborIndices     = neighborIndices;
                binding.neighborWeights     = neighborWeights;
                binding.generatedScatterers = generatedScatterers;
                envRuntime.sourceOrder.push_back(source.softBody->entityId);
                envRuntime.bindings.push_back(std::move(binding));
                envRuntime.totalScattererCount += scattererCount;
            }

            envRuntime.scatterersDevice =
                CruAllocateCudaBuffer(envRuntime.totalScattererCount * sizeof(Diligent::float4));
            auto *envScatterersBase = static_cast<std::byte *>(envRuntime.scatterersDevice);
            if (envRuntime.totalScattererCount > 0u && envScatterersBase == nullptr)
            {
                envRuntime.reset();
                continue;
            }

            bool buildFailed                  = false;
            std::uint32_t nextScattererOffset = 0u;
            for (Impl::SourceBinding &binding : envRuntime.bindings)
            {
                binding.scattererOffset = nextScattererOffset;
                nextScattererOffset += static_cast<std::uint32_t>(binding.scattererCount);
                binding.tracker = CruExtCreateDeformationTracker();
                if (binding.tracker == nullptr)
                {
                    buildFailed = true;
                    break;
                }

                void *pointPointer = sharedBase + static_cast<std::size_t>(binding.particleOffset) *
                                                      sizeof(Diligent::float4);
                void *scattererPointer =
                    envScatterersBase +
                    static_cast<std::size_t>(binding.scattererOffset) * sizeof(Diligent::float4);
                if (binding.generatedScatterers != nullptr)
                {
                    CruCopyDeviceToDevice(scattererPointer, binding.generatedScatterers,
                                          binding.scattererCount * sizeof(Diligent::float4));
                    CruFreeCudaBuffer(binding.generatedScatterers);
                    binding.generatedScatterers = nullptr;
                }
                CruExtDeformationTrackerSetInitialPoints(binding.tracker, pointPointer,
                                                         static_cast<int>(binding.particleCount));
                CruExtDeformationTrackerSetCurrentPoints(binding.tracker, pointPointer,
                                                         static_cast<int>(binding.particleCount));
                CruExtDeformationTrackerSetScatterers(
                    binding.tracker, scattererPointer, binding.neighborIndices,
                    binding.neighborWeights, binding.scattererCount);
            }
            if (buildFailed)
            {
                envRuntime.reset();
                continue;
            }

            CRESSIM_LOG_INFO("UltrasoundSystem: configured environment ", envIndex, " with ",
                             envRuntime.bindings.size(), " soft-body bindings and ",
                             envRuntime.totalScattererCount, " total scatterers.");
        }
    }

    for (auto &[envIndex, envRuntime] : mImpl->environmentRuntimes)
    {
        (void)envIndex;
        for (Impl::SourceBinding &binding : envRuntime.bindings)
        {
            if (binding.tracker == nullptr)
            {
                continue;
            }

            void *pointPointer = sharedBase + static_cast<std::size_t>(binding.particleOffset) *
                                                  sizeof(Diligent::float4);
            CruExtDeformationTrackerSetCurrentPoints(binding.tracker, pointPointer,
                                                     static_cast<int>(binding.particleCount));
            CruExtDeformationTrackerUpdateScatterers(binding.tracker);
        }
    }

    for (const auto &[probeEntityId, probeComponent] : probeComponents)
    {
        if (!probeComponent.enabled)
        {
            continue;
        }

        const std::uint32_t envIndex = world.entityEnvironment(probeEntityId);
        auto envIt                   = mImpl->environmentRuntimes.find(envIndex);
        if (envIt == mImpl->environmentRuntimes.end() || envIt->second.totalScattererCount == 0u)
        {
            continue;
        }

        auto runtimeIt = mImpl->probeRuntimes.find(probeEntityId);
        if (runtimeIt == mImpl->probeRuntimes.end())
        {
            continue;
        }

        const Impl::EnvironmentRuntime &envRuntime = envIt->second;
        Impl::ProbeRuntime &runtime                = runtimeIt->second;
        const auto rendererIt                      = rendererComponents.find(probeEntityId);
        const UltrasoundRendererComponent rendererComponent = rendererIt != rendererComponents.end()
                                                                  ? rendererIt->second
                                                                  : disabledRendererComponent();
        const std::uint32_t currentPoseSlot                 = world.entityPoseSlot(probeEntityId);
        const bool rebuild = runtime.engine == nullptr || runtime.envIndex != envIndex ||
                             runtime.boundScattererCount != envRuntime.totalScattererCount ||
                             runtime.entityPoseSlot != currentPoseSlot ||
                             !equalsProbeComponent(runtime.component, probeComponent) ||
                             !equalsRendererComponent(runtime.renderer, rendererComponent);
        if (rebuild &&
            !mImpl->rebuildProbeRuntime(mDevice, graphicsBackend, world, computeBackend,
                                        probeEntityId, runtime, envIndex, probeComponent,
                                        rendererComponent, envRuntime.totalScattererCount, true))
        {
            continue;
        }

        if (!mImpl->updateProbeScanlines(computeBackend, gpuEntityScene.poses, runtime))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " could not update live scanlines.");
            mImpl->publishProbeResult(world, probeEntityId, runtime, false);
            continue;
        }

        auto *envScatterersBase = static_cast<std::byte *>(envRuntime.scatterersDevice);
        if (envScatterersBase == nullptr)
        {
            continue;
        }
        CruComputeBindScatterersDevice(runtime.engine, envScatterersBase,
                                       envRuntime.totalScattererCount);

        const CruRfLayout refreshedLayout = CruComputeGetRfLayout(runtime.engine);
        if (!equalsRfLayout(runtime.rfLayout, refreshedLayout) &&
            !mImpl->ensureProbeRfBuffer(runtime, graphicsBackend.renderDevice,
                                        gpu::contextMaskForId(graphicsBackend.contextId),
                                        refreshedLayout))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " could not resize RF output buffer.");
            mImpl->clearPublishedProbeResult(world, probeEntityId);
            continue;
        }
        runtime.rfLayout = refreshedLayout;

        if (!mImpl->ensureProbeImageTarget(mDevice.renderTargetSystem(), runtime))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " could not create image target.");
            mImpl->clearPublishedProbeResult(world, probeEntityId);
            continue;
        }

        bool useGpuCompletionWait = mImpl->ensureProbeCompletionSemaphoreInitialized(
                                        runtime, graphicsBackend.renderDevice, probeEntityId) &&
                                    runtime.completionSemaphore.cudaSemaphoreHandle() != nullptr;
        const std::uint64_t completionFenceValue =
            useGpuCompletionWait ? runtime.nextCompletionFenceValue++ : 0u;
        if (useGpuCompletionWait)
        {
            CruComputeSetCompletionCudaSemaphore(runtime.engine,
                                                 runtime.completionSemaphore.cudaSemaphoreHandle(),
                                                 completionFenceValue);
        }
        else
        {
            CruComputeSetCompletionCudaSemaphore(runtime.engine, nullptr, 0u);
        }

        if (!CruComputeSimulate(runtime.engine))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " simulation could not start.");
            mImpl->publishProbeResult(world, probeEntityId, runtime, false);
            continue;
        }
        if (useGpuCompletionWait)
        {
            if (!runtime.completionSemaphore.waitOnDeviceContext(graphicsBackend.graphicsContext,
                                                                 completionFenceValue))
            {
                CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                    " could not queue GPU completion wait.");
                mImpl->publishProbeResult(world, probeEntityId, runtime, false);
                continue;
            }
        }
        else if (!CruComputeSynchronize(runtime.engine))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " simulation could not complete.");
            mImpl->publishProbeResult(world, probeEntityId, runtime, false);
            continue;
        }

        bool imageValid = false;
        if (runtime.renderer.enabled)
        {
            if (mImpl->renderProbeImage(mDevice, graphicsBackend, runtime))
            {
                imageValid = true;
            }
        }
        mImpl->publishProbeResult(world, probeEntityId, runtime, imageValid,
                                  envRuntime.totalScattererCount, frameContext.frameIndex);
    }

    return true;
#endif
}

} // namespace cressim::neo::engine
