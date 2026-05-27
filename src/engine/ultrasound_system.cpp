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
#include "cru_c_core.h"
#include "cru_c_extension.h"
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
           lhs.soundSpeed == rhs.soundSpeed &&
           lhs.worldUnitsPerMeter == rhs.worldUnitsPerMeter &&
           lhs.noiseAmplitude == rhs.noiseAmplitude &&
           lhs.samplingFrequency == rhs.samplingFrequency &&
           lhs.demodulationFrequency == rhs.demodulationFrequency &&
           lhs.centerFrequency == rhs.centerFrequency &&
           lhs.fractionalBandwidth == rhs.fractionalBandwidth &&
           lhs.beamSigmaLateral == rhs.beamSigmaLateral &&
           lhs.beamSigmaElevational == rhs.beamSigmaElevational &&
           lhs.radialDecimation == rhs.radialDecimation &&
           lhs.threadsPerBlock == rhs.threadsPerBlock &&
           lhs.cudaNumStreams == rhs.cudaNumStreams &&
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
        common::EntityId sourceEntityId  = common::kInvalidEntityId;
        std::uint32_t particleOffset     = 0u;
        std::uint32_t particleCount      = 0u;
        std::uint32_t scattererOffset    = 0u;
        std::uint64_t scattererCount     = 0u;
        UltrasoundScattererSourceComponent component{};

        SourceBinding() = default;
        SourceBinding(const SourceBinding &) = delete;
        SourceBinding &operator=(const SourceBinding &) = delete;

        SourceBinding(SourceBinding &&other) noexcept
            : tracker(other.tracker),
              neighborIndices(other.neighborIndices),
              neighborWeights(other.neighborWeights),
              generatedScatterers(other.generatedScatterers),
              sourceEntityId(other.sourceEntityId),
              particleOffset(other.particleOffset),
              particleCount(other.particleCount),
              scattererOffset(other.scattererOffset),
              scattererCount(other.scattererCount),
              component(other.component)
        {
            other.tracker            = nullptr;
            other.neighborIndices    = nullptr;
            other.neighborWeights    = nullptr;
            other.generatedScatterers = nullptr;
            other.sourceEntityId     = common::kInvalidEntityId;
            other.particleOffset     = 0u;
            other.particleCount      = 0u;
            other.scattererOffset    = 0u;
            other.scattererCount     = 0u;
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
                FreeCudaBuffer(neighborIndices);
                neighborIndices = nullptr;
            }
            if (neighborWeights != nullptr)
            {
                FreeCudaBuffer(neighborWeights);
                neighborWeights = nullptr;
            }
            if (generatedScatterers != nullptr)
            {
                FreeCudaBuffer(generatedScatterers);
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

    struct ProbeRuntime
    {
        common::EntityId probeEntityId = common::kInvalidEntityId;
        UltrasoundProbeComponent component{};
        std::uint64_t softTopologyRevision = 0u;
        std::vector<common::EntityId> sourceOrder{};
        std::vector<SourceBinding> bindings{};

#if CRESSIM_NEO_HAS_ULTRASOUND
        SimulationEngineHandle engine     = nullptr;
        ScanSequenceHandle sequence       = nullptr;
        ExcitationSignalHandle excitation = nullptr;
        BeamProfileHandle beamProfile     = nullptr;
        std::vector<ScanlineHandle> scanlines{};
#endif

        ProbeRuntime() = default;
        ProbeRuntime(const ProbeRuntime &) = delete;
        ProbeRuntime &operator=(const ProbeRuntime &) = delete;
        ProbeRuntime(ProbeRuntime &&) = delete;
        ProbeRuntime &operator=(ProbeRuntime &&) = delete;

        void reset()
        {
            for (SourceBinding &binding : bindings)
            {
                binding.reset();
            }
            bindings.clear();
            sourceOrder.clear();
#if CRESSIM_NEO_HAS_ULTRASOUND
            if (engine != nullptr)
            {
                CruReleaseSimulationEngine(engine);
                engine = nullptr;
            }
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
    std::unordered_map<common::EntityId, ProbeRuntime> runtimes{};
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
        mImpl->runtimes.clear();
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

    const auto &probeComponents  = world.ultrasoundProbeComponents();
    const auto &sourceComponents = world.ultrasoundScattererSourceComponents();
    const physics::PhysicsWorld &physicsWorld = world.physicsWorld();
    auto *sharedBase = static_cast<std::byte *>(mImpl->bridge.devicePointer());
    if (sharedBase == nullptr)
    {
        return true;
    }

    for (auto it = mImpl->runtimes.begin(); it != mImpl->runtimes.end();)
    {
        if (probeComponents.find(it->first) == probeComponents.end())
        {
            world.clearUltrasoundProbeResult(it->first);
            it = mImpl->runtimes.erase(it);
        }
        else
        {
            ++it;
        }
    }

    for (const auto &[probeEntityId, probeComponent] : probeComponents)
    {
        if (!probeComponent.enabled)
        {
            world.clearUltrasoundProbeResult(probeEntityId);
            continue;
        }

        const auto probeTransform = world.tryGetTransform(probeEntityId).value_or(
            TransformComponent{});
        const std::uint32_t envIndex = world.entityEnvironment(probeEntityId);

        std::vector<std::pair<const physics::SoftBodyState *,
                              UltrasoundScattererSourceComponent>>
            sources;
        for (const auto &[sourceEntityId, sourceComponent] : sourceComponents)
        {
            if (!sourceComponent.enabled || world.entityEnvironment(sourceEntityId) != envIndex)
            {
                continue;
            }

            const physics::SoftBodyState *softBody =
                physicsWorld.tryGetSoftBody(sourceEntityId);
            if (softBody == nullptr || !softBody->simulated || softBody->particleCount == 0u)
            {
                continue;
            }
            sources.emplace_back(softBody, sourceComponent);
        }

        if (sources.empty())
        {
            world.clearUltrasoundProbeResult(probeEntityId);
            continue;
        }

        Impl::ProbeRuntime &runtime = mImpl->runtimes[probeEntityId];
        bool rebuild = runtime.engine == nullptr ||
                       !equalsProbeComponent(runtime.component, probeComponent) ||
                       runtime.softTopologyRevision != physicsWorld.softBodyTopologyRevision() ||
                       runtime.sourceOrder.size() != sources.size();
        if (!rebuild)
        {
            for (std::size_t i = 0; i < sources.size(); ++i)
            {
                if (runtime.sourceOrder[i] != sources[i].first->entityId ||
                    !equalsSourceComponent(runtime.bindings[i].component, sources[i].second) ||
                    runtime.bindings[i].particleOffset != sources[i].first->particleOffset ||
                    runtime.bindings[i].particleCount != sources[i].first->particleCount)
                {
                    rebuild = true;
                    break;
                }
            }
        }

        if (rebuild)
        {
            runtime.reset();
            runtime.probeEntityId        = probeEntityId;
            runtime.component            = probeComponent;
            runtime.softTopologyRevision = physicsWorld.softBodyTopologyRevision();

            runtime.engine = CruCreateSimulationEngine();
            if (runtime.engine == nullptr)
            {
                CRESSIM_LOG_WARNING("UltrasoundSystem: failed to create simulation engine.");
                continue;
            }

            std::uint64_t totalScatterers = 0u;
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

            int maxNumScatterers = 0;
            for (std::size_t i = 0; i < sources.size(); ++i)
            {
                const auto &[softBody, sourceComponent] = sources[i];
                const float pointDistance = resolvePointDistance(*softBody, sourceComponent);
                CruGenerateScatterersFromPointsConfig config{};
                config.bounds          = sourceBounds[i];
                config.density         = sourceComponent.density > 0.0f
                                             ? sourceComponent.density
                                             : computeAutoDensity(*softBody);
                config.seed            = 1337u + static_cast<std::uint64_t>(i);
                config.threadsPerBlock = static_cast<int>(probeComponent.threadsPerBlock);
                config.pointDistance   = pointDistance;

                void *generatedScatterers = nullptr;
                void *neighborIndices     = nullptr;
                void *neighborWeights     = nullptr;
                std::uint64_t scattererCount = 0u;
                void *pointPointer = sharedBase +
                                     static_cast<std::size_t>(softBody->particleOffset) *
                                         sizeof(Diligent::float4);
                CruExtGenerateScatterersFromPoints(
                    pointPointer, amplitudeRanges[i].data(),
                    static_cast<int>(softBody->particleCount), &config, &generatedScatterers,
                    &neighborIndices, &neighborWeights, &scattererCount);

                Impl::SourceBinding binding{};
                binding.sourceEntityId = softBody->entityId;
                binding.particleOffset = softBody->particleOffset;
                binding.particleCount  = softBody->particleCount;
                binding.component      = sourceComponent;
                binding.scattererCount = scattererCount;
                binding.neighborIndices = neighborIndices;
                binding.neighborWeights = neighborWeights;
                binding.generatedScatterers = generatedScatterers;
                runtime.sourceOrder.push_back(softBody->entityId);
                runtime.bindings.push_back(std::move(binding));
                totalScatterers += scattererCount;
            }

            maxNumScatterers = static_cast<int>(std::min<std::uint64_t>(
                totalScatterers, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
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
                const auto requiredNumSamples = static_cast<std::uint32_t>(std::floor(
                    probeComponent.samplingFrequency * requiredMaxTime + 0.5f));
                numTimeSamples = static_cast<int>(
                    std::max<std::uint32_t>(16384u, nextPowerOfTwo(requiredNumSamples + 2048u)));
            }

            CruSimulationEngineSetParameter(runtime.engine,
                                            SimulationEngineParamType_eMaxNumScatterers,
                                            &maxNumScatterers);
            CruSimulationEngineSetParameter(runtime.engine, SimulationEngineParamType_eSoundSpeed,
                                            &effectiveSoundSpeed);
            CruSimulationEngineSetParameter(runtime.engine,
                                            SimulationEngineParamType_eNoiseAmplitude,
                                            &runtime.component.noiseAmplitude);
            CruSimulationEngineSetParameter(runtime.engine,
                                            SimulationEngineParamType_eUseArcProjection,
                                            &runtime.component.useArcProjection);
            CruSimulationEngineSetParameter(runtime.engine,
                                            SimulationEngineParamType_eRadialDecimation,
                                            &radialDecimation);
            CruSimulationEngineSetParameter(runtime.engine,
                                            SimulationEngineParamType_eEnablePhaseDelay,
                                            &runtime.component.enablePhaseDelay);
            CruSimulationEngineSetParameter(runtime.engine,
                                            SimulationEngineParamType_eNumTimeSamples,
                                            &numTimeSamples);
            CruSimulationEngineSetParameter(runtime.engine,
                                            SimulationEngineParamType_eThreadsPerBlock,
                                            &threadsPerBlock);
            CruSimulationEngineSetParameter(runtime.engine,
                                            SimulationEngineParamType_eCudaNumStreams,
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
                CRESSIM_LOG_WARNING("UltrasoundSystem: failed to create probe resources.");
                runtime.reset();
                continue;
            }

            CruSimulationEngineSetExcitation(runtime.engine, runtime.excitation);
            CruSimulationEngineSetBeamProfile(runtime.engine, runtime.beamProfile);

            const Diligent::float3 originCenter = probeTransform.worldTransform.position;
            const Diligent::float3 directionVec = probeTransform.worldTransform.rotation.RotateVector(
                Diligent::float3{0.0f, 0.0f, 1.0f});
            const Diligent::float3 lateralVec = probeTransform.worldTransform.rotation.RotateVector(
                Diligent::float3{1.0f, 0.0f, 0.0f});
            runtime.scanlines.reserve(runtime.component.numScanlines);
            bool buildFailed = false;
            for (std::uint32_t i = 0u; i < runtime.component.numScanlines; ++i)
            {
                const float centeredIndex =
                    static_cast<float>(i) -
                    0.5f * static_cast<float>(runtime.component.numScanlines - 1u);
                const Diligent::float3 originPos =
                    originCenter + lateralVec * (centeredIndex * runtime.component.scanlineSpacing);
                const float origin[3] = {originPos.x, originPos.y, originPos.z};
                const float direction[3] = {directionVec.x, directionVec.y, directionVec.z};
                const float lateral[3] = {lateralVec.x, lateralVec.y, lateralVec.z};
                ScanlineHandle scanline = CruCreateScanline(origin, direction, lateral);
                if (scanline == nullptr)
                {
                    CRESSIM_LOG_WARNING("UltrasoundSystem: failed to create scanline.");
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

            CruSimulationEngineSetScanSequence(runtime.engine, runtime.sequence);
            CruSimulationEngineClearAndResetScatterers(runtime.engine, maxNumScatterers);
            auto *engineScatterersBase =
                static_cast<std::byte *>(CruSimulationEngineGetScatterersDevice(runtime.engine));
            if (engineScatterersBase == nullptr)
            {
                CRESSIM_LOG_WARNING("UltrasoundSystem: engine scatterer buffer unavailable.");
                runtime.reset();
                continue;
            }

            buildFailed = false;
            for (Impl::SourceBinding &binding : runtime.bindings)
            {
                unsigned int scattererOffset = 0u;
                if (!CruSimulationEngineAllocateScatterers(
                        runtime.engine, static_cast<unsigned int>(binding.scattererCount),
                        &scattererOffset))
                {
                    CRESSIM_LOG_WARNING("UltrasoundSystem: failed to allocate scatterer slice.");
                    buildFailed = true;
                    break;
                }

                binding.scattererOffset = scattererOffset;
                binding.tracker         = CruExtCreateDeformationTracker();
                if (binding.tracker == nullptr)
                {
                    buildFailed = true;
                    break;
                }

                void *pointPointer = sharedBase +
                                     static_cast<std::size_t>(binding.particleOffset) *
                                         sizeof(Diligent::float4);
                void *scattererPointer = engineScatterersBase +
                                         static_cast<std::size_t>(scattererOffset) *
                                             sizeof(Diligent::float4);
                if (binding.generatedScatterers != nullptr)
                {
                    FastCopyDeviceToDevice(scattererPointer, binding.generatedScatterers,
                                           binding.scattererCount * sizeof(Diligent::float4));
                    FreeCudaBuffer(binding.generatedScatterers);
                    binding.generatedScatterers = nullptr;
                }
                CruExtDeformationTrackerSetPoints(binding.tracker, pointPointer,
                                                  static_cast<int>(binding.particleCount));
                CruExtDeformationTrackerSetScatterers(binding.tracker, scattererPointer,
                                                      binding.neighborIndices,
                                                      binding.neighborWeights,
                                                      binding.scattererCount);
            }
            if (buildFailed)
            {
                runtime.reset();
                continue;
            }

            CRESSIM_LOG_INFO("UltrasoundSystem: configured probe ", probeEntityId, " with ",
                             runtime.bindings.size(), " soft-body bindings and ",
                             totalScatterers, " total scatterers.");
        }

        for (Impl::SourceBinding &binding : runtime.bindings)
        {
            void *pointPointer = sharedBase +
                                 static_cast<std::size_t>(binding.particleOffset) *
                                     sizeof(Diligent::float4);
            CruExtDeformationTrackerSetPoints(binding.tracker, pointPointer,
                                              static_cast<int>(binding.particleCount));
            CruExtDeformationTrackerUpdateScatterers(binding.tracker);
        }

        if (!CruSimulationEngineSimulate(runtime.engine))
        {
            CRESSIM_LOG_WARNING("UltrasoundSystem: probe ", probeEntityId,
                                " simulation could not start.");
            continue;
        }

        UltrasoundProbeResult result{};
        result.frameIndex = frameContext.frameIndex;
        result.valid      = true;
        result.numScanlines = runtime.component.numScanlines;
        for (const Impl::SourceBinding &binding : runtime.bindings)
        {
            result.totalScattererCount += binding.scattererCount;
        }

        if (runtime.component.cpuReadbackEnabled)
        {
            const std::uint32_t dataSize = CruSimulationEngineFetchResults(runtime.engine);
            result.rfSamples.resize(dataSize);
            CruSimulationEngineCopyRfLines(runtime.engine, result.rfSamples.data(), dataSize);
            result.samplesPerScanline =
                runtime.component.numScanlines > 0u ? dataSize / runtime.component.numScanlines : 0u;

            if (!result.rfSamples.empty())
            {
                CRESSIM_LOG_INFO("UltrasoundSystem: probe ", probeEntityId,
                                 " RF readback ok at frame ", frameContext.frameIndex,
                                 " (scanlines=", result.numScanlines,
                                 ", samplesPerScanline=", result.samplesPerScanline,
                                 ", totalSamples=", result.rfSamples.size(),
                                 ", scatterers=", result.totalScattererCount, "). Preview: [0]=(",
                                 result.rfSamples[0].real, ", ", result.rfSamples[0].imag,
                                 "), [1]=(",
                                 (result.rfSamples.size() > 1u ? result.rfSamples[1].real : 0.0f),
                                 ", ",
                                 (result.rfSamples.size() > 1u ? result.rfSamples[1].imag : 0.0f),
                                 "), [2]=(",
                                 (result.rfSamples.size() > 2u ? result.rfSamples[2].real : 0.0f),
                                 ", ",
                                 (result.rfSamples.size() > 2u ? result.rfSamples[2].imag : 0.0f),
                                 ").");
            }
        }

        world.setUltrasoundProbeResult(probeEntityId, result);
    }

    return true;
#endif
}

} // namespace cressim::neo::engine
