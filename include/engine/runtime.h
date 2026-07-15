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

struct RuntimeConfig
{
    gpu::GpuDeviceDesc gpuDeviceDesc{};
    common::SceneLayoutDesc sceneLayout{};
    graphics::RendererDesc rendererDesc{};
    physics::PhysicsSolverDesc physicsDesc{};
};

struct RuntimeInfo
{
    std::string engineVersion;
    std::uint32_t engineVersionMajor = 0u;
    std::uint32_t engineVersionMinor = 0u;
    std::uint32_t engineVersionPatch = 0u;
    bool cudaInteropSupported        = false;
    bool ultrasoundSupported         = false;
};

class CRESSIM_NEO_ENGINE_API Runtime
{
public:
    Runtime();
    ~Runtime();

    Runtime(const Runtime &)            = delete;
    Runtime &operator=(const Runtime &) = delete;
    Runtime(Runtime &&)                 = delete;
    Runtime &operator=(Runtime &&)      = delete;

    bool initialize(const RuntimeConfig &config = RuntimeConfig{});
    void shutdown();

    // Staged entry points require the caller to explicitly prepare authored state and
    // upload it before physics/custom-compute execution.
    void prepare();
    bool uploadWorld();
    bool stepPhysics(const common::FrameContext &frameContext);
    bool stepSimulationSensors(const common::FrameContext &frameContext);
    void stepVisualSensors(const common::FrameContext &frameContext);
    void endFrame(const common::FrameContext &frameContext);

    World &getWorld() noexcept;
    const World &getWorld() const noexcept;

    gpu::GpuDevice *getGpuDevice() noexcept;
    const gpu::GpuDevice *getGpuDevice() const noexcept;
    physics::PhysicsSolver *getPhysicsSolver() noexcept;
    const physics::PhysicsSolver *getPhysicsSolver() const noexcept;
    const graphics::RenderStats &lastRenderStats() const noexcept;
    void setRenderFrameOptions(const graphics::RenderFrameOptions &options) noexcept;
    const graphics::RenderFrameOptions &renderFrameOptions() const noexcept;

    graphics::RenderResourceManager &getResources() noexcept;
    const graphics::RenderResourceManager &getResources() const noexcept;
    RuntimeInfo getInfo() const noexcept;

    // Creates an engine-owned structured GPU buffer that can be bound in custom compute.
    // When CUDA interop is available and the allocation is exportable, the same buffer can
    // also be imported into CUDA/Torch through DLPack.
    SharedBufferHandle createSharedBuffer(const SharedBufferDesc &desc);

    // Destroys the runtime handle for a shared buffer. Exported DLPack tensors may keep the
    // underlying storage alive until their consumer releases it.
    bool destroySharedBuffer(SharedBufferHandle handle);

    // Lists currently registered shared-buffer handles and their metadata.
    std::vector<SharedBufferInfo> listSharedBuffers() const;

    // Queries metadata for one shared buffer handle.
    bool tryGetSharedBufferInfo(SharedBufferHandle handle, SharedBufferInfo &outInfo) const;

    // Returns the CUDA-facing device pointer view for one shared buffer when CUDA interop is
    // available and this buffer was successfully imported into CUDA.
    bool tryGetSharedBufferCudaView(SharedBufferHandle handle, SharedBufferCudaView &outView) const;

    // Inserts a GPU-side wait so subsequent CUDA/Torch work sees prior Vulkan/D3D compute writes.
    // Fails when the shared buffer is not imported into CUDA.
    bool syncSharedBufferToCuda(SharedBufferHandle handle);

    // Inserts a GPU-side wait so subsequent Vulkan/D3D compute work sees prior CUDA/Torch writes.
    // Fails when the shared buffer is not imported into CUDA.
    bool syncSharedBufferFromCuda(SharedBufferHandle handle);

    // Retains the storage backing a shared buffer independently of its runtime handle.
    // Keep the returned lease alive while an external consumer uses an exported buffer view.
    SharedBufferLease retainSharedBufferLease(SharedBufferHandle handle) const;

    // Returns the current prepared rigid-body/collider slot mapping derived from authored state.
    // This is valid after prepare() and does not require uploadWorld().
    // The returned layoutRevision is a prepared host-side slot-layout invalidation key and is
    // separate from the live GPU custom-compute bindingGeneration exposed after uploadWorld().
    bool tryGetPreparedRigidLayoutMapping(RigidLayoutMapping &outMapping) const;

    // Returns the current prepared rigid-adjacent constraint mapping derived from authored state.
    // This is valid after prepare() and does not require uploadWorld().
    // The returned layoutRevision is a prepared host-side slot-layout invalidation key and is
    // separate from the live GPU custom-compute bindingGeneration exposed after uploadWorld().
    bool tryGetPreparedConstraintLayoutMapping(ConstraintLayoutMapping &outMapping) const;

    // Returns the current prepared particle/deformable slot mapping derived from authored state.
    // This is valid after prepare() and does not require uploadWorld().
    // The returned layoutRevision is a prepared host-side slot-layout invalidation key and is
    // separate from the live GPU custom-compute bindingGeneration exposed after uploadWorld().
    bool tryGetPreparedParticleLayoutMapping(ParticleLayoutMapping &outMapping) const;

    // Returns the current prepared rigid-joint slot mapping derived from authored state for
    // ball, hinge, spherical, and slider joints. This is valid after prepare() and does not
    // require uploadWorld(). The returned layoutRevision is a prepared host-side slot-layout
    // invalidation key and is separate from the live GPU custom-compute bindingGeneration
    // exposed after uploadWorld().
    bool tryGetPreparedJointLayoutMapping(JointLayoutMapping &outMapping) const;

    bool computeUltrasoundProbeLayout(const UltrasoundProbeComponent &probeComponent,
                                      const UltrasoundRendererComponent &rendererComponent,
                                      UltrasoundProbeLayout &outLayout) const;

    std::vector<CustomComputeResourceDesc> listCustomComputeResources();
    CustomComputePassHandle createCustomComputePass(const CustomComputePassDesc &desc);
    bool updateCustomComputePassConstants(CustomComputePassHandle handle,
                                          const std::vector<std::uint8_t> &data);
    bool executeCustomComputePass(CustomComputePassHandle handle);
    bool destroyCustomComputePass(CustomComputePassHandle handle);

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RUNTIME_H
