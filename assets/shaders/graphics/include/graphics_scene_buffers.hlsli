#ifndef CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI
#define CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI

struct RenderableMetadata
{
    uint flags;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    float4 localBoundsMin;
    float4 localBoundsMax;
};

struct PreparedCamera
{
    float4x4 viewMatrix;
    float4x4 viewProjectionMatrix;
    float4x4 lightViewProjectionMatrices[4];
    float4 cameraPosition;
    float4 cascadeSplits;
    float4 shadowParams;
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
    uint reserved0;
    uint reserved1;
};

struct VisiblePairInstance
{
    uint objectIndex;
    uint cameraIndex;
    uint cameraLayer;
    uint bucketIndex;
};

StructuredBuffer<float4> g_EntityPositions;
StructuredBuffer<float4> g_EntityOrientations;
StructuredBuffer<float4> g_EntityScales;
StructuredBuffer<RenderableMetadata> g_RenderableMetadata;
StructuredBuffer<RenderableQueueInfo> g_RenderableQueueInfo;
StructuredBuffer<uint> g_RenderableVisibilityFlags;
StructuredBuffer<uint> g_RenderableShadowCascadeMasks;
StructuredBuffer<PreparedCamera> g_PreparedCameras;
StructuredBuffer<uint> g_VisibleObjectIndices;
StructuredBuffer<VisiblePairInstance> g_VisiblePairs;

static const uint CRESSIM_RENDERABLE_FLAG_ACTIVE = 1u << 0u;
static const uint CRESSIM_RENDERABLE_FLAG_SHADOW_CASTER = 1u << 2u;

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

    RenderableMetadata metadata = g_RenderableMetadata[instanceIndex];
    if ((metadata.flags & CRESSIM_RENDERABLE_FLAG_ACTIVE) == 0u)
    {
        return;
    }

    position = g_EntityPositions[instanceIndex].xyz;
    orientation = normalize(g_EntityOrientations[instanceIndex]);
    scale = g_EntityScales[instanceIndex].xyz;
    isValid = true;
}

#endif // CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI
