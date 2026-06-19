#ifndef CRESSIM_NEO_ENGINE_RUNTIME_H
#define CRESSIM_NEO_ENGINE_RUNTIME_H

#include "common/frame_context.h"
#include "engine/export.h"
#include "engine/render_scene_uploader.h"
#include "engine/ultrasound_system.h"
#include "engine/world.h"
#include "gpu/gpu_device.h"
#include "graphics/render_resource_manager.h"
#include "graphics/renderer.h"
#include "physics/physics_solver.h"

#include <memory>

namespace cressim::neo::engine
{

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
    ~Runtime();

    bool initialize(const RuntimeConfig &config = RuntimeConfig{});
    void shutdown();

    // Staged entry points require the caller to keep authored state prepared.
    // Only the legacy tick() path performs implicit prepare().
    void prepare();
    bool stepPhysics(const common::FrameContext &frameContext);
    bool stepSimulationSensors(const common::FrameContext &frameContext);
    void stepVisualSensors(const common::FrameContext &frameContext);
    // Flushes pending simulation-sensor work when no visual step is issued.
    void flushSimulationSensors();

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

private:
    bool mInitialized = false;
    std::unique_ptr<gpu::GpuDevice> mGpuDevice;
    std::unique_ptr<RenderSceneUploader> mRenderSceneUploader;
    std::unique_ptr<physics::PhysicsSolver> mPhysicsSolver;
    std::unique_ptr<UltrasoundSystem> mUltrasoundSystem;
    std::unique_ptr<graphics::Renderer> mRenderer;
    graphics::RenderFrameOptions mRenderFrameOptions{};
    graphics::RenderStats mLastRenderStats{};
    World mWorld;
    graphics::RenderResourceManager mResources;
    common::FrameContext mLastFrameContext{};
    bool mFrameBoundaryPending = false;
    bool mHasPhysicsState      = false;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RUNTIME_H
