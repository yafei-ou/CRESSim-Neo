#ifndef CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI
#define CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI

struct RenderableMetadata
{
    uint flags;
    uint softBodyAttachmentBase;
    uint softBodyParticleOffset;
    uint softBodyAttachmentCount;
    float4 localBoundsMin;
    float4 localBoundsMax;
};

struct SoftBodyVertexAttachment
{
    uint4 particleIndices;
    float4 barycentricAndOffset;
};

struct PreparedCamera
{
    float4x4 viewMatrix;
    float4x4 viewProjectionMatrix;
    float4x4 lightViewProjectionMatrices[4];
    float4 cameraPosition;
    float4 cascadeSplits;
    float2 mainShadowTexelSize;
    float mainShadowCascadeCount;
    float mainShadowFadeDistance;
    uint envIndex;
    uint active;
    uint objectRangeStart;
    uint objectRangeCount;
    uint visibilityDataOffset;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct RenderableQueueInfo
{
    uint opaqueCommandIndex;
    uint shadowCommandBaseIndex;
    uint localShadowCommandIndex;
    uint reserved0;
};

static const uint CRESSIM_LIGHT_TYPE_DIRECTIONAL = 0u;
static const uint CRESSIM_LIGHT_TYPE_POINT = 1u;
static const uint CRESSIM_LIGHT_TYPE_SPOT = 2u;

// Forward-path main directional light selection is explicit slot 0 per environment.
struct LightInput
{
    float4 positionRange;
    float4 directionIntensity;
    float4 color;
    float4 spotAngles;
    float shadowDistance;
    float shadowFadeDistance;
    float shadowBias;
    float shadowPadding0;
    uint envIndex;
    uint lightSlot;
    uint type;
    uint active;
    uint castsShadows;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct LocalLightSelection
{
    uint localLightCount;
    uint shadowedLocalLightCount;
    uint shadowedPointLightCount;
    uint reserved0;
    uint lightIndices[8];
};

struct LightShadowAssignment
{
    uint shadowMode;
    uint shadowViewIndex;
    uint reserved0;
    uint reserved1;
};

struct LocalShadowView
{
    float4x4 lightViewProjectionMatrices[6];
    float4 lightPositionRange;
    float4 lightDirection;
    float2 shadowTexelSize;
    float shadowNearPlane;
    float shadowFarPlane;
    uint lightIndex;
    uint envIndex;
    uint firstLayer;
    uint layerCount;
    uint lightType;
    uint active;
    uint reserved0;
    uint reserved1;
};

struct BatchCameraMetadata
{
    uint globalCameraIndex;
    uint envIndex;
    uint mainLightIndex;
    uint colorLayer;
    uint shadowLayer;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct VisiblePairInstance
{
    uint objectIndex;
    uint batchCameraIndex;
    uint bucketIndex;
    uint shadowSubviewIndex;
};

struct EnvironmentIblLookupEntry
{
    uint sliceIndex;
    uint enabled;
    float intensity;
    float reserved0;
};

#include "include/structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_EntityOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_EntityScales);
CRESSIM_STRUCTURED_BUFFER(RenderableMetadata, g_RenderableMetadata);
CRESSIM_STRUCTURED_BUFFER(RenderableQueueInfo, g_RenderableQueueInfo);
CRESSIM_STRUCTURED_BUFFER(uint, g_RenderableVisibilityFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_RenderableShadowCascadeMasks);
CRESSIM_STRUCTURED_BUFFER(PreparedCamera, g_PreparedCameras);
CRESSIM_STRUCTURED_BUFFER(LightInput, g_LightInputs);
CRESSIM_STRUCTURED_BUFFER(LocalLightSelection, g_LocalLightSelections);
CRESSIM_STRUCTURED_BUFFER(LightShadowAssignment, g_LightShadowAssignments);
CRESSIM_STRUCTURED_BUFFER(LocalShadowView, g_LocalShadowViews);
CRESSIM_STRUCTURED_BUFFER(BatchCameraMetadata, g_BatchCameras);
CRESSIM_STRUCTURED_BUFFER(uint, g_VisibleObjectIndices);
CRESSIM_STRUCTURED_BUFFER(VisiblePairInstance, g_VisiblePairs);
#if defined(CRESSIM_PROGRAM_FAMILY_SOFT_BODY)
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositions);
CRESSIM_STRUCTURED_BUFFER(SoftBodyVertexAttachment, g_SoftBodyVertexAttachments);
#endif
#if defined(CRESSIM_IBL_DIFFUSE_ONLY) || defined(CRESSIM_IBL_FULL)
CRESSIM_STRUCTURED_BUFFER(EnvironmentIblLookupEntry, g_EnvironmentIblLookup);
#endif

static const uint CRESSIM_RENDERABLE_FLAG_ACTIVE = 1u << 0u;
static const uint CRESSIM_RENDERABLE_FLAG_SHADOW_CASTER = 1u << 2u;
static const uint CRESSIM_INVALID_GPU_SCENE_INDEX = 0xffffffffu;
static const uint CRESSIM_INVALID_BATCH_CAMERA_LAYER = 0xffffffffu;
static const uint CRESSIM_INVALID_ATTACHMENT_BASE = 0xffffffffu;
static const uint CRESSIM_FORWARD_LOCAL_LIGHT_CAP = 8u;
static const uint CRESSIM_SHADOWED_LOCAL_LIGHT_CAP = 4u;
static const uint CRESSIM_SHADOWED_POINT_LIGHT_CAP = 1u;

float3 quaternionRotateVector(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

void loadRenderablePose(uint instanceIndex, out bool isValid, out float3 position,
                        out float4 orientation, out float3 scale)
{
    isValid = false;
    position = float3(0.0, 0.0, 0.0);
    orientation = float4(0.0, 0.0, 0.0, 1.0);
    scale = float3(1.0, 1.0, 1.0);

    RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, instanceIndex);
    if ((metadata.flags & CRESSIM_RENDERABLE_FLAG_ACTIVE) == 0u)
    {
        return;
    }

    position = CRESSIM_SB_REF(g_EntityPositions, instanceIndex).xyz;
    orientation = normalize(CRESSIM_SB_LOAD(g_EntityOrientations, instanceIndex));
    scale = CRESSIM_SB_REF(g_EntityScales, instanceIndex).xyz;
    isValid = true;
}

#endif // CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI
