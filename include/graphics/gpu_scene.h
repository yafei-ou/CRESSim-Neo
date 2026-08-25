#ifndef CRESSIM_NEO_GRAPHICS_GPU_SCENE_H
#define CRESSIM_NEO_GRAPHICS_GPU_SCENE_H

#include "common/flags.h"
#include "common/scene_primitives.h"
#include "graphics/gpu_scene_pipeline_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <array>
#include <cstdint>

/// @file gpu_scene.h
/// @brief GPU scene buffers, camera/light input layouts, culling metadata, and vertex binding structures.

namespace cressim::neo::graphics
{

/// @brief Physical light source category for GPU shading and shadow casting.
enum class GpuLightType : std::uint32_t
{
    Directional = 0u, ///< Distant directional sun light with cascaded shadow maps.
    Point       = 1u, ///< Omnidirectional point light source with spherical attenuation.
    Spot        = 2u, ///< Cone-constrained spotlight source with projected shadow map.
};

/// @brief Raw GPU uniform input buffer layout representing a camera entity.
struct GpuCameraInput
{
    Diligent::float4 position{};                                   ///< Camera world position (xyz).
    Diligent::float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};          ///< Camera world orientation quaternion (xyzw).
    Diligent::float4 projectionParams{60.0f, 0.01f, 1000.0f, 0.0f};///< FOV (x), near plane (y), far plane (z), reserved (w).
    Diligent::float4 viewportAndOutputSize{0.0f, 0.0f, 1.0f, 1.0f};///< Viewport origin (xy) and extent (zw).
    std::uint32_t envIndex       = 0u;                             ///< Environment index.
    std::uint32_t cameraSlot     = 0u;                             ///< Environment local camera slot.
    std::uint32_t active         = 0u;                             ///< Active state flag (0 or 1).
    std::uint32_t entityPoseSlot = kInvalidGpuSceneIndex;          ///< Index into GPU global entity pose buffer.
};

/// @brief Raw GPU uniform input buffer layout representing a light source.
struct GpuLightInput
{
    Diligent::float4 positionRange{};      ///< World position (xyz) and max attenuation range (w).
    Diligent::float4 directionIntensity{};  ///< Emission direction (xyz) and radiant intensity (w).
    Diligent::float4 color{};              ///< Linear radiant flux color (RGB).
    Diligent::float4 spotAngles{};         ///< Spot angles: cos(inner) in x, cos(outer) in y.
    float shadowDistance       = 0.0f;     ///< Shadow projection distance.
    float shadowFadeDistance   = 0.0f;     ///< Shadow smooth fade distance.
    float shadowBias           = 0.0015f;  ///< Constant shadow depth bias.
    float shadowPadding0       = 0.0f;     ///< Reserved padding.
    std::uint32_t envIndex     = 0u;       ///< Environment index.
    std::uint32_t lightSlot    = 0u;       ///< Local environment light slot.
    std::uint32_t type         = 0u;       ///< GpuLightType enum integer value.
    std::uint32_t active       = 0u;       ///< Active flag (1 = active, 0 = inactive).
    std::uint32_t castsShadows = 0u;       ///< Shadow casting flag (0 or 1).
    std::uint32_t reserved0    = 0u;       ///< Reserved padding.
    std::uint32_t reserved1    = 0u;       ///< Reserved padding.
    std::uint32_t reserved2    = 0u;       ///< Reserved padding.
};

/// @brief Pre-culled selection of active local (point/spot) lights affecting a camera's view.
struct GpuLocalLightSelection
{
    std::uint32_t localLightCount         = 0u; ///< Total active local lights.
    std::uint32_t shadowedLocalLightCount = 0u; ///< Shadowed local spotlights.
    std::uint32_t shadowedPointLightCount = 0u; ///< Shadowed point lights.
    std::uint32_t reserved0               = 0u; ///< Reserved padding.
    std::array<std::uint32_t, kForwardLocalLightCap> lightIndices{}; ///< Array of active light indices in the global light buffer.
};

/// @brief Bitmask flags describing renderable object visibility and pipeline behavior.
enum class GpuRenderableFlags : std::uint32_t
{
    None         = 0u,       ///< Inactive or non-rendering object.
    Active       = 1u << 0u, ///< Object is active and candidate for rendering.
    Opaque       = 1u << 1u, ///< Object draws in the opaque pass.
    ShadowCaster = 1u << 2u, ///< Object casts shadows into directional/local shadow maps.
};
CRESSIM_NEO_DEFINE_ENUM_FLAGS(GpuRenderableFlags)

/// @brief Deformation binding classification for GPU skinning and soft-body animation.
enum class GpuRenderableDeformableType : std::uint32_t
{
    None     = 0u, ///< Rigid transform only; no GPU deformation.
    SoftBody = 1u, ///< Volumetric 3D soft-body particle skinning.
    Curve    = 2u, ///< 1D elastic strand / curve deformation.
};

/// @brief GPU per-renderable object descriptor consumed by compute culling and vertex shaders.
struct GpuRenderableMetadata
{
    std::uint32_t flags             = static_cast<std::uint32_t>(GpuRenderableFlags::None); ///< GpuRenderableFlags bitmask.
    std::uint32_t deformVertexBase  = 0u;                  ///< Base vertex offset in deformed position buffer.
    std::uint32_t deformNormalBase  = 0u;                  ///< Base vertex offset in deformed normal buffer.
    std::uint32_t deformableIndex   = 0xffffffffu;         ///< Index into soft-body / strand physics buffer.
    std::uint32_t deformVertexCount = 0u;                  ///< Number of deformable vertices.
    std::uint32_t deformableType    = 0u;                  ///< GpuRenderableDeformableType value.
    std::uint32_t segmentationId    = 0u;                  ///< Segmentation class ID.
    std::uint32_t entityPoseSlot    = kInvalidGpuSceneIndex;///< Index in global entity pose buffer.
    Diligent::float4 localBoundsMin{};                     ///< Local bounding box minimum (xyz).
    Diligent::float4 localBoundsMax{};                     ///< Local bounding box maximum (xyz).
};
static_assert(sizeof(GpuRenderableMetadata) == 64u);

/// @brief Skinning weights and barycentric particle indices binding a soft-body surface vertex to simulation nodes.
struct GpuSoftBodyVertexBinding
{
    Diligent::uint4 particleIndices{0u, 0u, 0u, 0u}; ///< Indices of up to 4 influencing simulation particles.
    Diligent::float4 weights{1.0f, 0.0f, 0.0f, 0.0f};///< Normalized barycentric skinning weights.
};
static_assert(sizeof(GpuSoftBodyVertexBinding) == 32u);

/// @brief Mapping linking a renderable object to its slots in indirect draw argument command buffers.
struct GpuRenderableQueueInfo
{
    std::uint32_t opaqueCommandIndex      = 0xffffffffu; ///< Index of opaque draw command.
    std::uint32_t shadowCommandBaseIndex  = 0xffffffffu; ///< Base index of directional shadow draw command.
    std::uint32_t localShadowCommandIndex = 0xffffffffu; ///< Index of local shadow draw command.
    std::uint32_t reserved0               = 0u;          ///< Reserved padding.
};

/// @brief Structured collection of GPU buffer views representing the complete scene state on device memory.
struct GpuEntitySceneView
{
    common::SceneLayoutDesc layout{};                                    ///< Multi-environment scene layout limits.
    common::PoseBufferView poses{};                                      ///< Entity transform pose buffer view.
    Diligent::IBuffer *renderableMetadataBuffer           = nullptr;     ///< GPU buffer holding GpuRenderableMetadata array.
    Diligent::IBuffer *renderableQueueInfoBuffer          = nullptr;     ///< GPU buffer holding GpuRenderableQueueInfo array.
    Diligent::IBuffer *renderableVisibilityFlagsBuffer    = nullptr;     ///< GPU buffer holding per-camera visibility bitmasks.
    Diligent::IBuffer *renderableShadowCascadeMasksBuffer = nullptr;     ///< GPU buffer holding shadow cascade visibility masks.
    Diligent::IBuffer *cameraInputsBuffer                 = nullptr;     ///< GPU buffer holding GpuCameraInput array.
    Diligent::IBuffer *preparedCamerasBuffer              = nullptr;     ///< GPU buffer holding GpuPreparedCamera uniforms.
    Diligent::IBuffer *lightInputsBuffer                  = nullptr;     ///< GPU buffer holding GpuLightInput array.
    Diligent::IBuffer *localLightSelectionBuffer          = nullptr;     ///< GPU buffer holding GpuLocalLightSelection array.
    Diligent::IBuffer *softBodyVertexBindingBuffer        = nullptr;     ///< GPU buffer holding GpuSoftBodyVertexBinding array.
    std::uint32_t entityCount                             = 0;           ///< Total active entities.
    std::uint32_t renderableCount                         = 0;           ///< Total renderable objects.
    std::uint32_t cameraCount                             = 0;           ///< Total cameras.
    std::uint32_t lightCount                              = 0;           ///< Total lights.
    std::uint64_t bindingGeneration                       = 0;           ///< Monotonic revision counter for buffer bindings.
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_GPU_SCENE_H
