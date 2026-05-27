#include "engine/ultrasound_system.h"

#include "common/logger.h"
#include "engine/components.h"
#include "engine/world.h"
#include "gpu/cuda_interop.h"
#include "gpu/gpu_device.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
           lhs.enablePhaseDelay == rhs.enablePhaseDelay &&
           lhs.cpuReadbackEnabled == rhs.cpuReadbackEnabled;
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
        void *rfLinesDevice               = nullptr;
        std::uint32_t rfCapacitySamples   = 0u;
        CruRfLayout rfLayout{};
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
            if (rfLinesDevice != nullptr)
            {
                CruFreeCudaBuffer(rfLinesDevice);
                rfLinesDevice = nullptr;
            }
            rfCapacitySamples = 0u;
            rfLayout          = CruRfLayout{};
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

    gpu::CudaSharedBufferBridge bridge;
    bool disabled = false;
    std::unordered_map<std::uint32_t, EnvironmentRuntime> environmentRuntimes{};
    std::unordered_map<common::EntityId, ProbeRuntime> probeRuntimes{};

#if CRESSIM_NEO_HAS_ULTRASOUND
    bool ensureProbeRfBuffer(ProbeRuntime &runtime, const CruRfLayout &layout)
    {
        if (layout.totalSamples == 0u)
        {
            if (runtime.rfLinesDevice != nullptr)
            {
                CruFreeCudaBuffer(runtime.rfLinesDevice);
                runtime.rfLinesDevice = nullptr;
            }
            runtime.rfCapacitySamples = 0u;
            CruComputeBindRfLinesDevice(runtime.engine, nullptr, 0u);
            runtime.rfLayout = layout;
            return true;
        }

        if (runtime.rfLinesDevice == nullptr || runtime.rfCapacitySamples != layout.totalSamples)
        {
            if (runtime.rfLinesDevice != nullptr)
            {
                CruFreeCudaBuffer(runtime.rfLinesDevice);
                runtime.rfLinesDevice = nullptr;
            }
            runtime.rfLinesDevice = CruAllocateCudaBuffer(
                static_cast<std::uint64_t>(layout.totalSamples) * sizeof(Diligent::float2));
            if (runtime.rfLinesDevice == nullptr)
            {
                runtime.rfCapacitySamples = 0u;
                return false;
            }
            runtime.rfCapacitySamples = layout.totalSamples;
        }

        CruComputeBindRfLinesDevice(runtime.engine, runtime.rfLinesDevice,
                                    runtime.rfCapacitySamples);
        runtime.rfLayout = layout;
        return true;
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

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr)
    {
        return true;
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
            it = mImpl->probeRuntimes.erase(it);
        }
        else if (!probeComponents.at(it->first).enabled)
        {
            world.clearUltrasoundProbeResult(it->first);
            it = mImpl->probeRuntimes.erase(it);
        }
        else
        {
            ++it;
        }
    }

    for (auto it = mImpl->environmentRuntimes.begin(); it != mImpl->environmentRuntimes.end();)
    {
        bool hasProbeInEnv = false;
        for (const auto &[probeEntityId, probeComponent] : probeComponents)
        {
            if (probeComponent.enabled && world.entityEnvironment(probeEntityId) == it->first)
            {
                hasProbeInEnv = true;
                break;
            }
        }
        if (!hasProbeInEnv)
        {
            it = mImpl->environmentRuntimes.erase(it);
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
            runtime.reset();
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
                runtime.reset();
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
                runtime.reset();
                continue;
            }
            CruComputeSetScanSequence(runtime.engine, runtime.sequence);
            runtime.rfLayout = CruComputeGetRfLayout(runtime.engine);
            if (!mImpl->ensureProbeRfBuffer(runtime, runtime.rfLayout))
            {
                runtime.reset();
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
            !mImpl->ensureProbeRfBuffer(runtime, refreshedLayout))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " could not resize RF output buffer.");
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
        if (runtime.component.cpuReadbackEnabled)
        {
            const std::uint32_t dataSize = runtime.rfLayout.totalSamples;
            result.numScanlines          = runtime.rfLayout.numScanlines;
            result.samplesPerScanline    = runtime.rfLayout.samplesPerLine;
            result.rfSamples.resize(dataSize);
            CruCopyDeviceToHost(result.rfSamples.data(), runtime.rfLinesDevice,
                                static_cast<uint64_t>(dataSize) * sizeof(UltrasoundRfSample));

            if (!result.rfSamples.empty())
            {
                CRESSIM_LOG_INFO(
                    "UltrasoundSystem: probe ", probeEntityId, " RF readback ok at frame ",
                    frameContext.frameIndex, " (scanlines=", result.numScanlines,
                    ", samplesPerScanline=", result.samplesPerScanline,
                    ", totalSamples=", result.rfSamples.size(),
                    ", scatterers=", result.totalScattererCount, "). Preview: [0]=(",
                    result.rfSamples[0].real, ", ", result.rfSamples[0].imag, "), [1]=(",
                    (result.rfSamples.size() > 1u ? result.rfSamples[1].real : 0.0f), ", ",
                    (result.rfSamples.size() > 1u ? result.rfSamples[1].imag : 0.0f), "), [2]=(",
                    (result.rfSamples.size() > 2u ? result.rfSamples[2].real : 0.0f), ", ",
                    (result.rfSamples.size() > 2u ? result.rfSamples[2].imag : 0.0f), ").");
            }
        }
        world.setUltrasoundProbeResult(probeEntityId, result);
    }

    return true;
#endif
}

} // namespace cressim::neo::engine
