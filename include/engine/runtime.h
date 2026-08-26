#ifndef CRESSIM_NEO_ENGINE_RUNTIME_H
#define CRESSIM_NEO_ENGINE_RUNTIME_H

#include "common/frame_context.h"
#include "engine/constraint_layout_mapping.h"
#include "engine/custom_compute.h"
#include "engine/export.h"
#include "engine/joint_layout_mapping.h"
#include "engine/particle_layout_mapping.h"
#include "engine/rigid_layout_mapping.h"
#include "engine/shared_buffer.h"
#include "engine/world.h"
#include "gpu/gpu_device.h"
#include "graphics/render_resource_manager.h"
#include "graphics/renderer.h"
#include "physics/physics_solver.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cressim::neo::engine
{

struct UltrasoundProbeComponent;
struct UltrasoundRendererComponent;
struct UltrasoundProbeLayout;

/// @brief Configuration descriptor for initializing the CRESSim-Neo engine runtime.
struct RuntimeConfig
{
    gpu::GpuDeviceDesc gpuDeviceDesc{};       ///< Desired GPU device configuration.
    common::SceneLayoutDesc sceneLayout{};    ///< Scene layout capacity settings.
    graphics::RendererDesc rendererDesc{};    ///< Graphics renderer parameters.
    physics::PhysicsSolverDesc physicsDesc{}; ///< Physics solver parameters.
};

/// @brief Information structure holding engine version and optional feature support flags.
struct RuntimeInfo
{
    std::string engineVersion;                ///< Full semver engine version string.
    std::uint32_t engineVersionMajor = 0u;    ///< Major version number.
    std::uint32_t engineVersionMinor = 0u;    ///< Minor version number.
    std::uint32_t engineVersionPatch = 0u;    ///< Patch version number.
    bool cudaInteropSupported        = false; ///< True if CUDA interop is enabled and available.
    bool ultrasoundSupported = false; ///< True if CRESSim-Ultrasound integration is available.
};

/// @brief Main engine runtime coordinator managing GPU devices, physics solvers, graphics, and
/// custom compute passes.
class CRESSIM_NEO_ENGINE_API Runtime
{
public:
    /// @brief Constructs an uninitialized runtime.
    Runtime();

    /// @brief Shuts down the runtime if necessary.
    ~Runtime();

    Runtime(const Runtime &)            = delete;
    Runtime &operator=(const Runtime &) = delete;
    Runtime(Runtime &&)                 = delete;
    Runtime &operator=(Runtime &&)      = delete;

    /// @brief Initializes the engine runtime with the specified configuration parameters.
    /// @param config Runtime configuration options including GPU, physics, and renderer settings.
    /// @return True if initialization succeeds; false otherwise.
    bool initialize(const RuntimeConfig &config = RuntimeConfig{});

    /// @brief Shuts down the engine runtime and releases allocated GPU/physics resources.
    void shutdown();

    /// @brief Prepares authored world and render state for GPU upload.
    ///
    /// Call uploadWorld() after this method and before stepPhysics() or custom-compute execution.
    /// Does nothing before initialize().
    void prepare();

    /// @brief Uploads the prepared physics and render scene state to GPU resources.
    /// @return True if all required uploads succeed; false before initialize().
    bool uploadWorld();

    /// @brief Advances physics using the supplied frame context.
    ///
    /// Requires a successful uploadWorld() call after the most recent prepare().
    /// @return True if the physics step succeeds.
    bool stepPhysics(const common::FrameContext &frameContext);

    /// @brief Executes simulation-driven sensors, including ultrasound when configured.
    /// @return False before initialize() or if the ultrasound system reports failure.
    bool stepSimulationSensors(const common::FrameContext &frameContext);

    /// @brief Updates the GPU scene and renders visual sensors for the supplied frame.
    /// @note Does nothing before initialize().
    void stepVisualSensors(const common::FrameContext &frameContext);

    /// @brief Ends the active device frame and submits queued presentation/readback work.
    /// @note Does nothing before initialize() or when no device frame is active.
    void endFrame(const common::FrameContext &frameContext);

    /// @brief Returns the runtime-owned world.
    World &getWorld() noexcept;

    /// @brief Returns the runtime-owned world.
    const World &getWorld() const noexcept;

    /// @brief Returns the GPU device, or nullptr before initialization and after shutdown.
    gpu::GpuDevice *getGpuDevice() noexcept;

    /// @brief Returns the GPU device, or nullptr before initialization and after shutdown.
    const gpu::GpuDevice *getGpuDevice() const noexcept;

    /// @brief Returns the physics solver, or nullptr before initialization and after shutdown.
    physics::PhysicsSolver *getPhysicsSolver() noexcept;

    /// @brief Returns the physics solver, or nullptr before initialization and after shutdown.
    const physics::PhysicsSolver *getPhysicsSolver() const noexcept;

    /// @brief Sets gravity for the initialized physics solver.
    /// @param gravity Gravity acceleration vector.
    void setGravity(const Diligent::float3 &gravity) noexcept;

    /// @brief Returns statistics from the most recent visual-sensor render.
    const graphics::RenderStats &lastRenderStats() const noexcept;

    /// @brief Sets options applied to subsequent visual-sensor renders.
    /// @param options Per-frame rendering options.
    void setRenderFrameOptions(const graphics::RenderFrameOptions &options) noexcept;

    /// @brief Returns options applied to visual-sensor renders.
    const graphics::RenderFrameOptions &renderFrameOptions() const noexcept;

    /// @brief Returns the runtime-owned render resource manager.
    graphics::RenderResourceManager &getResources() noexcept;

    /// @brief Returns the runtime-owned render resource manager.
    const graphics::RenderResourceManager &getResources() const noexcept;

    /// @brief Returns engine version and optional feature support information.
    RuntimeInfo getInfo() const noexcept;

    /// Creates an engine-owned structured GPU buffer that can be bound in custom compute.
    /// When CUDA interop is available and the allocation is exportable, the same buffer can
    /// also be imported into CUDA/Torch through DLPack.
    SharedBufferHandle createSharedBuffer(const SharedBufferDesc &desc);

    /// Destroys the runtime handle for a shared buffer. Exported DLPack tensors may keep the
    /// underlying storage alive until their consumer releases it.
    bool destroySharedBuffer(SharedBufferHandle handle);

    /// Lists currently registered shared-buffer handles and their metadata.
    std::vector<SharedBufferInfo> listSharedBuffers() const;

    /// Queries metadata for one shared buffer handle.
    bool tryGetSharedBufferInfo(SharedBufferHandle handle, SharedBufferInfo &outInfo) const;

    /// Returns the CUDA-facing device pointer view for one shared buffer when CUDA interop is
    /// available and this buffer was successfully imported into CUDA.
    bool tryGetSharedBufferCudaView(SharedBufferHandle handle, SharedBufferCudaView &outView) const;

    /// Inserts a GPU-side wait so subsequent CUDA/Torch work sees prior Vulkan/D3D compute writes.
    /// Fails when the shared buffer is not imported into CUDA.
    bool syncSharedBufferToCuda(SharedBufferHandle handle);

    /// Inserts a GPU-side wait so subsequent Vulkan/D3D compute work sees prior CUDA/Torch writes.
    /// Fails when the shared buffer is not imported into CUDA.
    bool syncSharedBufferFromCuda(SharedBufferHandle handle);

    /// Retains the storage backing a shared buffer independently of its runtime handle.
    /// Keep the returned lease alive while an external consumer uses an exported buffer view.
    SharedBufferLease retainSharedBufferLease(SharedBufferHandle handle) const;

    /// @brief Returns the current prepared rigid-body/collider slot mapping derived from authored
    /// state. This is valid after prepare() and does not require uploadWorld(). The returned
    /// layoutRevision is a prepared host-side slot-layout invalidation key and is separate from the
    /// live GPU custom-compute bindingGeneration exposed after uploadWorld().
    /// @param outMapping Receives an empty mapping when the runtime is uninitialized.
    /// @return False when the runtime or physics solver is unavailable.
    bool tryGetPreparedRigidLayoutMapping(RigidLayoutMapping &outMapping) const;

    /// @brief Returns the current prepared rigid-adjacent constraint mapping derived from authored
    /// state. This is valid after prepare() and does not require uploadWorld(). The returned
    /// layoutRevision is a prepared host-side slot-layout invalidation key and is separate from the
    /// live GPU custom-compute bindingGeneration exposed after uploadWorld().
    /// @param outMapping Receives an empty mapping when the runtime is uninitialized.
    /// @return False when the runtime or physics solver is unavailable.
    bool tryGetPreparedConstraintLayoutMapping(ConstraintLayoutMapping &outMapping) const;

    /// @brief Returns the current prepared particle/deformable slot mapping derived from authored
    /// state. This is valid after prepare() and does not require uploadWorld(). The returned
    /// layoutRevision is a prepared host-side slot-layout invalidation key and is separate from the
    /// live GPU custom-compute bindingGeneration exposed after uploadWorld().
    /// @param outMapping Receives an empty mapping when the runtime is uninitialized.
    /// @return False when the runtime or physics solver is unavailable.
    bool tryGetPreparedParticleLayoutMapping(ParticleLayoutMapping &outMapping) const;

    /// @brief Returns the current prepared rigid-joint slot mapping derived from authored state for
    /// ball, hinge, spherical, and slider joints. This is valid after prepare() and does not
    /// require uploadWorld(). The returned layoutRevision is a prepared host-side slot-layout
    /// invalidation key and is separate from the live GPU custom-compute bindingGeneration
    /// exposed after uploadWorld().
    /// @param outMapping Receives an empty mapping when the runtime is uninitialized.
    /// @return False when the runtime or physics solver is unavailable.
    bool tryGetPreparedJointLayoutMapping(JointLayoutMapping &outMapping) const;

    /// @brief Computes an ultrasound output layout for probe and renderer components.
    /// @param probeComponent Ultrasound probe configuration.
    /// @param rendererComponent Ultrasound renderer configuration.
    /// @param outLayout Output layout to populate.
    /// @return True if layout computation succeeds.
    bool computeUltrasoundProbeLayout(const UltrasoundProbeComponent &probeComponent,
                                      const UltrasoundRendererComponent &rendererComponent,
                                      UltrasoundProbeLayout &outLayout) const;

    /// @brief Lists custom-compute resources registered for the uploaded world.
    /// @return Resource descriptors, or an empty vector when unavailable.
    std::vector<CustomComputeResourceDesc> listCustomComputeResources();

    /// @brief Compiles and registers a custom compute pass for the uploaded world.
    /// @param desc Compute-pass configuration.
    /// @return Registered pass handle, or an invalid handle if creation fails.
    CustomComputePassHandle createCustomComputePass(const CustomComputePassDesc &desc);

    /// @brief Updates a custom compute pass's constant-buffer data.
    /// @param handle Registered compute-pass handle.
    /// @param data Replacement constant-buffer bytes.
    /// @return False for an invalid handle, a pass without constants, or an oversized payload.
    bool updateCustomComputePassConstants(CustomComputePassHandle handle,
                                          const std::vector<std::uint8_t> &data);

    /// @brief Executes a registered custom compute pass for the uploaded world.
    /// @param handle Registered compute-pass handle.
    /// @return False if the pass or its required resources are unavailable or changed.
    bool executeCustomComputePass(CustomComputePassHandle handle);

    /// @brief Destroys a registered custom compute pass.
    /// @param handle Registered compute-pass handle.
    /// @return True if the handle was registered.
    bool destroyCustomComputePass(CustomComputePassHandle handle);

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RUNTIME_H
