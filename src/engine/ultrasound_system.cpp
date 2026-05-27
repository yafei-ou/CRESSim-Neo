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
#include <limits>
#include <memory>
#include <unordered_map>
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

float computeAutoDensity(const physics::SoftBodyState &body)
{
    if (body.restPositions.empty())
    {
        return 1.0f;
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

    const float extentX = std::max(maximum.x - minimum.x, 1.0e-4f);
    const float extentY = std::max(maximum.y - minimum.y, 1.0e-4f);
    const float extentZ = std::max(maximum.z - minimum.z, 1.0e-4f);
    const float volume  = extentX * extentY * extentZ;
    return std::max(4096.0f / volume, 1.0f);
}

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

bool equalsProbeComponent(const UltrasoundProbeComponent &lhs,
                          const UltrasoundProbeComponent &rhs) noexcept
{
    return lhs.enabled == rhs.enabled && lhs.numScanlines == rhs.numScanlines &&
           lhs.lineLength == rhs.lineLength && lhs.scanlineSpacing == rhs.scanlineSpacing &&
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
           lhs.enablePhaseDelay == rhs.enablePhaseDelay && lhs.imageEnabled == rhs.imageEnabled &&
           lhs.imageBaseHeight == rhs.imageBaseHeight &&
           lhs.imageUseFixedMaxNormalization == rhs.imageUseFixedMaxNormalization &&
           lhs.imageFixedMaxSignal == rhs.imageFixedMaxSignal &&
           lhs.imageDynamicRangeDb == rhs.imageDynamicRangeDb;
}

bool equalsSourceComponent(const UltrasoundScattererSourceComponent &lhs,
                           const UltrasoundScattererSourceComponent &rhs) noexcept
{
    return lhs.enabled == rhs.enabled && lhs.density == rhs.density &&
           lhs.amplitudeMin == rhs.amplitudeMin && lhs.amplitudeMax == rhs.amplitudeMax &&
           lhs.pointDistanceOverride == rhs.pointDistanceOverride;
}

bool equalsRfLayout(const CruRfLayout &lhs, const CruRfLayout &rhs) noexcept
{
    return lhs.numScanlines == rhs.numScanlines &&
           lhs.requiredTimeSamples == rhs.requiredTimeSamples &&
           lhs.timeSamples == rhs.timeSamples &&
           lhs.delayCompensationSamples == rhs.delayCompensationSamples &&
           lhs.samplesPerLine == rhs.samplesPerLine && lhs.totalSamples == rhs.totalSamples;
}

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

struct UltrasoundAmplitudeRange
{
    float minimum = 0.0f;
    float maximum = 1.0f;
};

struct UltrasoundImageSize
{
    std::uint32_t width  = 1u;
    std::uint32_t height = 1u;
};

UltrasoundImageSize computeImageSize(const UltrasoundProbeComponent &probe) noexcept
{
    const std::uint32_t height = std::max(probe.imageBaseHeight, 1u);
    const std::uint32_t lateralIntervals =
        probe.numScanlines > 0u ? std::max(probe.numScanlines - 1u, 1u) : 1u;
    const float lateralSpan =
        static_cast<float>(lateralIntervals) * std::max(probe.scanlineSpacing, 1.0e-4f);
    const float aspect = lateralSpan / std::max(probe.lineLength, 1.0e-4f);
    const std::uint32_t width =
        std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(std::floor(height * aspect + 0.5f)));
    return UltrasoundImageSize{width, height};
}

std::uint32_t dispatchGroupCount(std::uint32_t threadCount, std::uint32_t groupSize) noexcept
{
    return groupSize == 0u ? 0u : (threadCount + groupSize - 1u) / groupSize;
}

#if CRESSIM_NEO_HAS_ULTRASOUND
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
    float lineLength                       = 1.0f;
    float scanlineSpacing                  = 1.0f;
    float dynamicRangeDb                   = 60.0f;
    float fixedMaxSignal                   = 1.0f;
    std::uint32_t useFixedMaxNormalization = 0u;
    std::uint32_t padding0                 = 0u;
    std::uint32_t padding1                 = 0u;
    std::uint32_t padding2                 = 0u;
};

constexpr std::uint32_t kUltrasoundReductionThreadGroupSize = 256u;
constexpr std::uint32_t kUltrasoundImageThreadGroupSizeX    = 8u;
constexpr std::uint32_t kUltrasoundImageThreadGroupSizeY    = 8u;

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
    "graphics/ultrasound_rf_image.cs.hlsl", "CRESSimNeo.Ultrasound.Image.CS",
    "CRESSimNeo.Ultrasound.Image.PSO",      kUltrasoundImageVars,
    std::size(kUltrasoundImageVars),
};
#endif

} // namespace

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
        std::uint64_t softTopologyRevision = 0u;
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
            totalScattererCount = 0u;
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
        UltrasoundProbeComponent component{};
        std::uint32_t envIndex            = 0u;
        std::uint64_t boundScattererCount = 0u;

#if CRESSIM_NEO_HAS_ULTRASOUND
        CruComputeHandle engine           = nullptr;
        ScanSequenceHandle sequence       = nullptr;
        ExcitationSignalHandle excitation = nullptr;
        BeamProfileHandle beamProfile     = nullptr;
        gpu::SharedExportBuffer rfSharedBuffer{};
        gpu::CudaSharedBuffer rfCudaBuffer{};
        std::uint32_t rfCapacitySamples = 0u;
        CruRfLayout rfLayout{};
        gpu::GpuRenderTargetHandle imageTarget{};
        std::uint32_t imageWidth  = 0u;
        std::uint32_t imageHeight = 0u;
        std::vector<ScanlineHandle> scanlines{};
#endif

        ProbeRuntime()                                = default;
        ProbeRuntime(const ProbeRuntime &)            = delete;
        ProbeRuntime &operator=(const ProbeRuntime &) = delete;
        ProbeRuntime(ProbeRuntime &&)                 = delete;
        ProbeRuntime &operator=(ProbeRuntime &&)      = delete;

        void reset()
        {
            boundScattererCount = 0u;
#if CRESSIM_NEO_HAS_ULTRASOUND
            if (engine != nullptr)
            {
                CruComputeDestroy(engine);
                engine = nullptr;
            }
            rfCudaBuffer.reset();
            rfSharedBuffer.reset();
            rfCapacitySamples = 0u;
            rfLayout          = CruRfLayout{};
            imageTarget       = {};
            imageWidth        = 0u;
            imageHeight       = 0u;
            for (ScanlineHandle scanline : scanlines)
            {
                if (scanline != nullptr)
                {
                    CruReleaseScanline(scanline);
                }
            }
            scanlines.clear();
            if (sequence != nullptr)
            {
                CruReleaseScanSequence(sequence);
                sequence = nullptr;
            }
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
        gpu::GpuComputePass imagePass;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> reductionConstantsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> imageConstantsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> groupMaxBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> finalMaxBuffer;
        std::uint32_t groupMaxCapacity = 0u;
        bool initialized               = false;
    };

    gpu::CudaSharedBufferBridge bridge;
    bool disabled = false;
    std::unordered_map<std::uint32_t, EnvironmentRuntime> environmentRuntimes{};
    std::unordered_map<common::EntityId, ProbeRuntime> probeRuntimes{};
    ImagePassState imagePassState{};

#if CRESSIM_NEO_HAS_ULTRASOUND
    void destroyProbeImageTarget(gpu::GpuRenderTargetSystem &renderTargetSystem,
                                 ProbeRuntime &runtime)
    {
        if (renderTargetSystem.isValidRenderTarget(runtime.imageTarget))
        {
            renderTargetSystem.destroyRenderTarget(runtime.imageTarget);
        }
        runtime.imageTarget = {};
        runtime.imageWidth  = 0u;
        runtime.imageHeight = 0u;
    }

    void resetProbeRuntime(gpu::GpuRenderTargetSystem &renderTargetSystem, ProbeRuntime &runtime)
    {
        destroyProbeImageTarget(renderTargetSystem, runtime);
        runtime.reset();
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
        if (!runtime.component.imageEnabled)
        {
            destroyProbeImageTarget(renderTargetSystem, runtime);
            return true;
        }

        const UltrasoundImageSize size = computeImageSize(runtime.component);
        if (renderTargetSystem.isValidRenderTarget(runtime.imageTarget) &&
            runtime.imageWidth == size.width && runtime.imageHeight == size.height)
        {
            return true;
        }

        destroyProbeImageTarget(renderTargetSystem, runtime);

        gpu::GpuRenderTargetDesc desc{};
        desc.width            = size.width;
        desc.height           = size.height;
        desc.arraySize        = 1u;
        desc.color            = true;
        desc.depth            = false;
        desc.layeredRendering = false;
        desc.shaderReadable   = true;
        desc.unorderedAccess  = true;
        desc.colorFormat      = Diligent::TEX_FORMAT_RGBA8_UNORM;
        desc.debugName        = "CRESSimNeo.Ultrasound.Image";

        runtime.imageTarget = renderTargetSystem.createRenderTarget(desc);
        if (!renderTargetSystem.isValidRenderTarget(runtime.imageTarget))
        {
            runtime.imageTarget = {};
            return false;
        }

        runtime.imageWidth  = size.width;
        runtime.imageHeight = size.height;
        return true;
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
            !imagePassState.imagePass.initialize(device, streamFactory, graphicsContextMask,
                                                 kUltrasoundImagePassDefinition))
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

    bool renderProbeImage(gpu::GpuDevice &device,
                          const gpu::GpuGraphicsBackendContext &graphicsBackend,
                          ProbeRuntime &runtime)
    {
        if (!runtime.component.imageEnabled || !runtime.rfSharedBuffer.buffer() ||
            !device.renderTargetSystem().isValidRenderTarget(runtime.imageTarget))
        {
            return false;
        }

        Diligent::ITexture *imageTexture = nullptr;
        if (!device.renderTargetSystem().tryGetRenderTargetColorTexture(runtime.imageTarget,
                                                                        imageTexture) ||
            imageTexture == nullptr)
        {
            return false;
        }

        Diligent::ITextureView *imageUav =
            imageTexture->GetDefaultView(Diligent::TEXTURE_VIEW_UNORDERED_ACCESS);
        if (imageUav == nullptr)
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

        if (!runtime.component.imageUseFixedMaxNormalization)
        {
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

            const std::uint32_t groupCount = std::max<std::uint32_t>(
                1u, dispatchGroupCount(runtime.rfLayout.totalSamples,
                                       kUltrasoundReductionThreadGroupSize));
            if (!imagePassState.maxReducePass.dispatch(graphicsBackend.graphicsContext, 0u,
                                                       maxReduceBindings, groupCount, 1u, 1u))
            {
                return false;
            }

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
        imageConstants.numScanlines    = runtime.rfLayout.numScanlines;
        imageConstants.samplesPerLine  = runtime.rfLayout.samplesPerLine;
        imageConstants.imageWidth      = runtime.imageWidth;
        imageConstants.imageHeight     = runtime.imageHeight;
        imageConstants.lineLength      = std::max(runtime.component.lineLength, 1.0e-4f);
        imageConstants.scanlineSpacing = std::max(runtime.component.scanlineSpacing, 1.0e-4f);
        imageConstants.dynamicRangeDb  = std::max(runtime.component.imageDynamicRangeDb, 1.0f);
        imageConstants.fixedMaxSignal  = std::max(runtime.component.imageFixedMaxSignal, 1.0e-6f);
        imageConstants.useFixedMaxNormalization =
            runtime.component.imageUseFixedMaxNormalization ? 1u : 0u;

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

        return imagePassState.imagePass.dispatchResources(
            graphicsBackend.graphicsContext, 0u, imageBufferBindings, imageTextureBindings,
            dispatchGroupCount(runtime.imageWidth, kUltrasoundImageThreadGroupSizeX),
            dispatchGroupCount(runtime.imageHeight, kUltrasoundImageThreadGroupSizeY), 1u);
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

bool UltrasoundSystem::tick(const common::FrameContext &frameContext, World &world)
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

    if (!mImpl->bridge.isInitialized() &&
        !mImpl->bridge.initializeForVulkan(computeBackend.renderDevice,
                                           "CRESSimNeo.Engine.UltrasoundSystem"))
    {
        CRESSIM_LOG_WARNING("UltrasoundSystem: failed to initialize CUDA interop bridge.");
        mImpl->disabled = true;
        return false;
    }

    if (!mImpl->bridge.bindSharedBuffer(*sharedBuffer) ||
        !mImpl->bridge.synchronizeFromDeviceContext(computeBackend.computeContext) ||
        !mImpl->bridge.synchronizeStream())
    {
        CRESSIM_LOG_WARNING("UltrasoundSystem: failed to synchronize shared soft positions.");
        mImpl->disabled = true;
        return false;
    }

    const auto &probeComponents               = world.ultrasoundProbeComponents();
    const auto &sourceComponents              = world.ultrasoundScattererSourceComponents();
    const physics::PhysicsWorld &physicsWorld = world.physicsWorld();
    auto *sharedBase = static_cast<std::byte *>(mImpl->bridge.devicePointer());
    if (sharedBase == nullptr)
    {
        return true;
    }

    for (auto it = mImpl->probeRuntimes.begin(); it != mImpl->probeRuntimes.end();)
    {
        if (probeComponents.find(it->first) == probeComponents.end())
        {
            world.clearUltrasoundProbeResult(it->first);
            mImpl->destroyProbeImageTarget(mDevice.renderTargetSystem(), it->second);
            it = mImpl->probeRuntimes.erase(it);
        }
        else if (!probeComponents.at(it->first).enabled)
        {
            world.clearUltrasoundProbeResult(it->first);
            mImpl->destroyProbeImageTarget(mDevice.renderTargetSystem(), it->second);
            it = mImpl->probeRuntimes.erase(it);
        }
        else
        {
            ++it;
        }
    }

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
        Impl::EnvironmentRuntime &envRuntime = mImpl->environmentRuntimes[envIndex];
        bool rebuild = envRuntime.scatterersDevice == nullptr ||
                       envRuntime.softTopologyRevision != physicsWorld.softBodyTopologyRevision() ||
                       envRuntime.sourceOrder.size() != sources.size();
        if (!rebuild)
        {
            for (std::size_t i = 0; i < sources.size(); ++i)
            {
                if (envRuntime.sourceOrder[i] != sources[i].first->entityId ||
                    !equalsSourceComponent(envRuntime.bindings[i].component, sources[i].second) ||
                    envRuntime.bindings[i].particleOffset != sources[i].first->particleOffset ||
                    envRuntime.bindings[i].particleCount != sources[i].first->particleCount)
                {
                    rebuild = true;
                    break;
                }
            }
        }

        if (rebuild)
        {
            envRuntime.reset();
            envRuntime.softTopologyRevision = physicsWorld.softBodyTopologyRevision();

            std::vector<std::vector<UltrasoundAmplitudeRange>> amplitudeRanges;
            amplitudeRanges.reserve(sources.size());
            std::vector<CruBounds3> sourceBounds;
            sourceBounds.reserve(sources.size());
            for (const auto &[softBody, sourceComponent] : sources)
            {
                const float pointDistance = resolvePointDistance(*softBody, sourceComponent);
                sourceBounds.push_back(computeBounds(*softBody, pointDistance));
                amplitudeRanges.emplace_back(
                    softBody->particleCount,
                    UltrasoundAmplitudeRange{sourceComponent.amplitudeMin,
                                             sourceComponent.amplitudeMax});
            }

            for (std::size_t i = 0; i < sources.size(); ++i)
            {
                const auto &[softBody, sourceComponent] = sources[i];
                const float pointDistance = resolvePointDistance(*softBody, sourceComponent);
                CruGenerateScatterersFromPointsConfig config{};
                config.bounds  = sourceBounds[i];
                config.density = sourceComponent.density > 0.0f ? sourceComponent.density
                                                                : computeAutoDensity(*softBody);
                config.seed    = 1337u + static_cast<std::uint64_t>(i);
                config.threadsPerBlock = 128;
                config.pointDistance   = pointDistance;

                void *generatedScatterers    = nullptr;
                void *neighborIndices        = nullptr;
                void *neighborWeights        = nullptr;
                std::uint64_t scattererCount = 0u;
                void *pointPointer =
                    sharedBase +
                    static_cast<std::size_t>(softBody->particleOffset) * sizeof(Diligent::float4);
                CruExtGenerateScatterersFromPoints(pointPointer, amplitudeRanges[i].data(),
                                                   static_cast<int>(softBody->particleCount),
                                                   &config, &generatedScatterers, &neighborIndices,
                                                   &neighborWeights, &scattererCount);

                Impl::SourceBinding binding{};
                binding.sourceEntityId      = softBody->entityId;
                binding.particleOffset      = softBody->particleOffset;
                binding.particleCount       = softBody->particleCount;
                binding.component           = sourceComponent;
                binding.scattererCount      = scattererCount;
                binding.neighborIndices     = neighborIndices;
                binding.neighborWeights     = neighborWeights;
                binding.generatedScatterers = generatedScatterers;
                envRuntime.sourceOrder.push_back(softBody->entityId);
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
                CruExtDeformationTrackerSetPoints(binding.tracker, pointPointer,
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

        for (Impl::SourceBinding &binding : envRuntime.bindings)
        {
            void *pointPointer = sharedBase + static_cast<std::size_t>(binding.particleOffset) *
                                                  sizeof(Diligent::float4);
            CruExtDeformationTrackerSetPoints(binding.tracker, pointPointer,
                                              static_cast<int>(binding.particleCount));
            CruExtDeformationTrackerUpdateScatterers(binding.tracker);
        }
    }

    for (const auto &[probeEntityId, probeComponent] : probeComponents)
    {
        if (!probeComponent.enabled)
        {
            world.clearUltrasoundProbeResult(probeEntityId);
            continue;
        }

        const std::uint32_t envIndex = world.entityEnvironment(probeEntityId);
        auto envIt                   = mImpl->environmentRuntimes.find(envIndex);
        if (envIt == mImpl->environmentRuntimes.end() || envIt->second.totalScattererCount == 0u)
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

        const Impl::EnvironmentRuntime &envRuntime = envIt->second;
        const auto probeTransform =
            world.tryGetTransform(probeEntityId).value_or(TransformComponent{});
        Impl::ProbeRuntime &runtime = mImpl->probeRuntimes[probeEntityId];
        bool rebuild = runtime.engine == nullptr || runtime.envIndex != envIndex ||
                       runtime.boundScattererCount != envRuntime.totalScattererCount ||
                       !equalsProbeComponent(runtime.component, probeComponent);
        if (rebuild)
        {
            mImpl->resetProbeRuntime(mDevice.renderTargetSystem(), runtime);
            runtime.component           = probeComponent;
            runtime.envIndex            = envIndex;
            runtime.boundScattererCount = envRuntime.totalScattererCount;

            runtime.engine = CruComputeCreate();
            if (runtime.engine == nullptr)
            {
                CRESSIM_LOG_WARNING("UltrasoundSystem: failed to create simulation engine.");
                continue;
            }

            const float effectiveSoundSpeed =
                probeComponent.soundSpeed * std::max(probeComponent.worldUnitsPerMeter, 1.0f);
            int radialDecimation = static_cast<int>(std::max(probeComponent.radialDecimation, 1u));
            int threadsPerBlock  = static_cast<int>(std::max(probeComponent.threadsPerBlock, 1u));
            int cudaNumStreams   = static_cast<int>(std::max(probeComponent.cudaNumStreams, 1u));
            int numTimeSamples   = static_cast<int>(probeComponent.numTimeSamples);
            if (numTimeSamples <= 0)
            {
                const float requiredMaxTime =
                    2.0f * probeComponent.lineLength / effectiveSoundSpeed;
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
            runtime.beamProfile = CruCreateGaussianBeamProfile(
                runtime.component.beamSigmaLateral, runtime.component.beamSigmaElevational);
            runtime.sequence = CruCreateScanSequence(runtime.component.lineLength);
            if (runtime.excitation == nullptr || runtime.beamProfile == nullptr ||
                runtime.sequence == nullptr)
            {
                mImpl->resetProbeRuntime(mDevice.renderTargetSystem(), runtime);
                continue;
            }
            CruComputeSetExcitation(runtime.engine, runtime.excitation);
            CruComputeSetBeamProfile(runtime.engine, runtime.beamProfile);

            const Diligent::float3 originCenter = probeTransform.worldTransform.position;
            const Diligent::float3 directionVec =
                probeTransform.worldTransform.rotation.RotateVector(
                    Diligent::float3{0.0f, 0.0f, 1.0f});
            const Diligent::float3 lateralVec = probeTransform.worldTransform.rotation.RotateVector(
                Diligent::float3{1.0f, 0.0f, 0.0f});
            bool buildFailed = false;
            runtime.scanlines.reserve(runtime.component.numScanlines);
            for (std::uint32_t i = 0u; i < runtime.component.numScanlines; ++i)
            {
                const float centeredIndex =
                    static_cast<float>(i) -
                    0.5f * static_cast<float>(runtime.component.numScanlines - 1u);
                const Diligent::float3 originPos =
                    originCenter + lateralVec * (centeredIndex * runtime.component.scanlineSpacing);
                const float origin[3]    = {originPos.x, originPos.y, originPos.z};
                const float direction[3] = {directionVec.x, directionVec.y, directionVec.z};
                const float lateral[3]   = {lateralVec.x, lateralVec.y, lateralVec.z};
                ScanlineHandle scanline  = CruCreateScanline(origin, direction, lateral);
                if (scanline == nullptr)
                {
                    buildFailed = true;
                    break;
                }
                runtime.scanlines.push_back(scanline);
                CruScanSequenceAddScanline(runtime.sequence, scanline);
            }
            if (buildFailed)
            {
                mImpl->resetProbeRuntime(mDevice.renderTargetSystem(), runtime);
                continue;
            }
            CruComputeSetScanSequence(runtime.engine, runtime.sequence);
            runtime.rfLayout = CruComputeGetRfLayout(runtime.engine);
            if (!mImpl->ensureProbeRfBuffer(runtime, graphicsBackend.renderDevice,
                                            gpu::contextMaskForId(graphicsBackend.contextId),
                                            runtime.rfLayout) ||
                !mImpl->ensureProbeImageTarget(mDevice.renderTargetSystem(), runtime))
            {
                mImpl->resetProbeRuntime(mDevice.renderTargetSystem(), runtime);
                continue;
            }
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
            continue;
        }
        runtime.rfLayout = refreshedLayout;

        if (!mImpl->ensureProbeImageTarget(mDevice.renderTargetSystem(), runtime))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " could not create image target.");
            continue;
        }

        if (!CruComputeSimulate(runtime.engine) || !CruComputeSynchronize(runtime.engine))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " simulation could not start.");
            continue;
        }

        UltrasoundProbeResult result{};
        result.frameIndex          = frameContext.frameIndex;
        result.valid               = true;
        result.numScanlines        = runtime.rfLayout.numScanlines;
        result.samplesPerScanline  = runtime.rfLayout.samplesPerLine;
        result.totalScattererCount = envRuntime.totalScattererCount;
        if (runtime.component.imageEnabled)
        {
            if (mImpl->renderProbeImage(mDevice, graphicsBackend, runtime))
            {
                result.imageValid  = true;
                result.imageWidth  = runtime.imageWidth;
                result.imageHeight = runtime.imageHeight;
                result.imageTarget = runtime.imageTarget;
            }
        }
        world.setUltrasoundProbeResult(probeEntityId, result);
    }

    return true;
#endif
}

} // namespace cressim::neo::engine
