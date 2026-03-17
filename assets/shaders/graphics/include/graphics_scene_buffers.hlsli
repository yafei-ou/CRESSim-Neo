#ifndef CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI
#define CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI

struct RenderableMetadata
{
    uint entityPoseIndex;
    uint envIndex;
    uint flags;
    uint reserved;
    float4 localBoundsMin;
    float4 localBoundsMax;
};

StructuredBuffer<float4> g_EntityPositions;
StructuredBuffer<float4> g_EntityOrientations;
StructuredBuffer<float4> g_EntityScales;
StructuredBuffer<RenderableMetadata> g_RenderableMetadata;
StructuredBuffer<uint> g_RenderableVisibilityFlags;
StructuredBuffer<uint> g_RenderableShadowCascadeMasks;

static const uint CRESSIM_RENDERABLE_FLAG_GPU_POSE = 1u << 3u;

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
    if ((metadata.flags & CRESSIM_RENDERABLE_FLAG_GPU_POSE) == 0u ||
        metadata.entityPoseIndex == 0xffffffffu)
    {
        return;
    }

    position = g_EntityPositions[metadata.entityPoseIndex].xyz;
    orientation = normalize(g_EntityOrientations[metadata.entityPoseIndex]);
    scale = g_EntityScales[metadata.entityPoseIndex].xyz;
    isValid = true;
}

#endif // CRESSIM_NEO_GRAPHICS_SCENE_BUFFERS_HLSLI
