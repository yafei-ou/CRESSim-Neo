#ifndef CRESSIM_NEO_ENGINE_RUNTIME_H
#define CRESSIM_NEO_ENGINE_RUNTIME_H

#include "common/frame_context.h"
#include "engine/custom_compute.h"
#include "engine/export.h"
#include "engine/render_scene_uploader.h"
#include "engine/shared_buffer.h"
#include "engine/ultrasound_system.h"
#include "engine/world.h"
#include "gpu/gpu_device.h"
#include "graphics/render_resource_manager.h"
#include "graphics/renderer.h"
#include "physics/physics_solver.h"

#include <memory>

namespace cressim::neo::engine
{

class CustomComputeService;
class SharedBufferService;

struct RuntimeConfig
{
    gpu::GpuDeviceDesc gpuDeviceDesc{};
    common::SceneLayoutDesc sceneLayout{};
    graphics::RendererDesc rendererDesc{};
    physics::PhysicsSolverDesc physicsDesc{};
};

class CRESSIM_NEO_ENGINE_API Runtime
{
public:
    Runtime();
    ~Runtime();

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
    SharedBufferHandle createSharedBuffer(const SharedBufferDesc &desc);
    bool destroySharedBuffer(SharedBufferHandle handle);
    std::vector<SharedBufferInfo> listSharedBuffers() const;
    bool tryGetSharedBufferInfo(SharedBufferHandle handle, SharedBufferInfo &outInfo) const;
    bool tryGetSharedBufferCudaView(SharedBufferHandle handle, SharedBufferCudaView &outView) const;
    bool syncSharedBufferToCuda(SharedBufferHandle handle);
    bool syncSharedBufferFromCuda(SharedBufferHandle handle);
    std::vector<CustomComputeResourceDesc> listCustomComputeResources();
    CustomComputePassHandle createCustomComputePass(const CustomComputePassDesc &desc);
    bool updateCustomComputePassConstants(CustomComputePassHandle handle,
                                          const std::vector<std::uint8_t> &data);
    bool executeCustomComputePass(CustomComputePassHandle handle);
    bool destroyCustomComputePass(CustomComputePassHandle handle);

private:
    bool mInitialized = false;
    std::unique_ptr<gpu::GpuDevice> mGpuDevice;
    std::unique_ptr<RenderSceneUploader> mRenderSceneUploader;
    std::unique_ptr<physics::PhysicsSolver> mPhysicsSolver;
    std::unique_ptr<UltrasoundSystem> mUltrasoundSystem;
    std::unique_ptr<graphics::Renderer> mRenderer;
    std::unique_ptr<CustomComputeService> mCustomComputeService;
    std::unique_ptr<SharedBufferService> mSharedBufferService;
    graphics::RenderFrameOptions mRenderFrameOptions{};
    graphics::RenderStats mLastRenderStats{};
    World mWorld;
    graphics::RenderResourceManager mResources;
    common::FrameContext mLastFrameContext{};
    bool mDeviceFrameActive = false;
    bool mWorldUploaded     = false;
    bool mHasPhysicsState   = false;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RUNTIME_H
