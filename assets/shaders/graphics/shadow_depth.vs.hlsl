#include "include/graphics/graphics_per_object.hlsli"
#include "include/graphics/graphics_scene_buffers.hlsli"
#if !defined(CRESSIM_CAMERA_DEPTH_PASS)
#include "include/graphics/graphics_shadow_constants.hlsli"
#endif

struct VSInput
{
    float3 Position : ATTRIB0;
    float3 Normal : ATTRIB1;
    float2 TexCoord : ATTRIB2;
};

struct VSOutput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD7;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
#if defined(CRESSIM_CAMERA_SEGMENTATION_PASS)
    nointerpolation uint SegmentationId : TEXCOORD0;
#endif
};

void main(in VSInput In, out VSOutput Out, uint instanceId : SV_InstanceID
#if defined(CRESSIM_PROGRAM_FAMILY_SOFT_BODY) || defined(CRESSIM_PROGRAM_FAMILY_CURVE)
    , uint vertexId : SV_VertexID
#endif
)
{
    Out.TexCoord = In.TexCoord;
#if defined(CRESSIM_CAMERA_DEPTH_PASS)
    const VisiblePairInstance pair = CRESSIM_SB_LOAD(g_VisiblePairs, g_DrawListOffset + instanceId);
    const BatchCameraMetadata batchCamera = CRESSIM_SB_LOAD(g_BatchCameras, pair.batchCameraIndex);
    const uint objectIndex = pair.objectIndex;
    const uint cameraIndex = batchCamera.globalCameraIndex;
    const uint colorLayer = batchCamera.colorLayer;
    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, cameraIndex);
    const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, objectIndex);

    bool poseValid = false;
    float3 position = float3(0.0, 0.0, 0.0);
    float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
    float3 scale = float3(1.0, 1.0, 1.0);
    loadRenderablePose(objectIndex, poseValid, position, orientation, scale);

    const uint localObjectIndex = objectIndex - preparedCamera.objectRangeStart;
    const uint visibilityIndex = preparedCamera.visibilityDataOffset + localObjectIndex;
    if (!poseValid || preparedCamera.active == 0u ||
        CRESSIM_SB_LOAD(g_RenderableVisibilityFlags, visibilityIndex) == 0u)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
#if MANUAL_LAYER_EXPORT
        Out.Layer = colorLayer;
#endif
#if defined(CRESSIM_CAMERA_SEGMENTATION_PASS)
        Out.SegmentationId = 0u;
#endif
        Out.TexCoord = In.TexCoord;
        return;
    }

    float4 worldPos = float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
#if defined(CRESSIM_PROGRAM_FAMILY_SOFT_BODY)
    if (metadata.deformVertexBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
        vertexId < metadata.deformVertexCount)
    {
        const float3 deformedPos =
            CRESSIM_SB_LOAD(g_SoftBodyRenderPositions, metadata.deformVertexBase + vertexId).xyz;
        worldPos = float4(deformedPos, 1.0);
    }
#elif defined(CRESSIM_PROGRAM_FAMILY_CURVE)
    if (metadata.deformVertexBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
        vertexId < metadata.deformVertexCount)
    {
        worldPos =
            float4(CRESSIM_SB_LOAD(g_CurveRenderPositions, metadata.deformVertexBase + vertexId).xyz,
                   1.0);
    }
#endif

    Out.Position = mul(worldPos, preparedCamera.viewProjectionMatrix);
#if MANUAL_LAYER_EXPORT
    Out.Layer = colorLayer;
#endif
#if defined(CRESSIM_CAMERA_SEGMENTATION_PASS)
    Out.SegmentationId = metadata.segmentationId;
#endif
    Out.TexCoord = In.TexCoord;
    return;
#else

    uint objectIndex = g_InstanceIndex;
    uint cameraIndex = 0u;
    uint shadowLayer = 0u;
    const bool localShadowPass = g_ShadowPassMode != 0u;
    if (g_UseDrawListBuffer != 0u)
    {
        if (localShadowPass)
        {
            const VisiblePairInstance pair = CRESSIM_SB_LOAD(g_VisiblePairs, instanceId);
            objectIndex = pair.objectIndex;
            const LocalShadowView shadowView = CRESSIM_SB_LOAD(g_LocalShadowViews, pair.batchCameraIndex);
            const uint localMatrixIndex = pair.shadowSubviewIndex;

            bool poseValid = false;
            float3 position = float3(0.0, 0.0, 0.0);
            float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
            float3 scale = float3(1.0, 1.0, 1.0);
            loadRenderablePose(objectIndex, poseValid, position, orientation, scale);
            if (!poseValid || shadowView.active == 0u || localMatrixIndex >= shadowView.layerCount)
            {
                Out.Position = float4(2.0, 2.0, 2.0, 1.0);
#if MANUAL_LAYER_EXPORT
                Out.Layer = 0u;
#endif
                return;
            }

            float4 worldPos = float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
#if defined(CRESSIM_PROGRAM_FAMILY_SOFT_BODY)
            const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, objectIndex);
            if (metadata.deformVertexBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
                vertexId < metadata.deformVertexCount)
            {
                const float3 deformedPos =
                    CRESSIM_SB_LOAD(g_SoftBodyRenderPositions, metadata.deformVertexBase + vertexId).xyz;
                worldPos = float4(deformedPos, 1.0);
            }
#elif defined(CRESSIM_PROGRAM_FAMILY_CURVE)
            const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, objectIndex);
            if (metadata.deformVertexBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
                vertexId < metadata.deformVertexCount)
            {
                worldPos = float4(CRESSIM_SB_LOAD(g_CurveRenderPositions,
                                                  metadata.deformVertexBase + vertexId).xyz,
                                  1.0);
            }
#endif
            Out.Position = mul(worldPos, shadowView.lightViewProjectionMatrices[localMatrixIndex]);
#if MANUAL_LAYER_EXPORT
            Out.Layer = shadowView.firstLayer + localMatrixIndex;
#endif
            return;
        }

        const VisiblePairInstance pair = CRESSIM_SB_LOAD(g_VisiblePairs, g_DrawListOffset + instanceId);
        objectIndex = pair.objectIndex;
        const BatchCameraMetadata batchCamera = CRESSIM_SB_LOAD(g_BatchCameras, pair.batchCameraIndex);
        cameraIndex = batchCamera.globalCameraIndex;
        shadowLayer = batchCamera.shadowLayer;
    }
    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, cameraIndex);
    bool poseValid = false;
    float3 position = float3(0.0, 0.0, 0.0);
    float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
    float3 scale = float3(1.0, 1.0, 1.0);
    loadRenderablePose(objectIndex, poseValid, position, orientation, scale);
    const uint localObjectIndex = objectIndex - preparedCamera.objectRangeStart;
    const uint shadowMask = CRESSIM_SB_LOAD(
        g_RenderableShadowCascadeMasks, preparedCamera.visibilityDataOffset + localObjectIndex);
    if (!poseValid || preparedCamera.active == 0u ||
        shadowLayer == CRESSIM_INVALID_BATCH_CAMERA_LAYER ||
        ((shadowMask & (1u << g_CascadeIndex)) == 0u))
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
#if MANUAL_LAYER_EXPORT
        Out.Layer = shadowLayer;
#endif
        return;
    }
    float4 worldPos = float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
#if defined(CRESSIM_PROGRAM_FAMILY_SOFT_BODY)
    const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, objectIndex);
    if (metadata.deformVertexBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
        vertexId < metadata.deformVertexCount)
    {
        const float3 deformedPos =
            CRESSIM_SB_LOAD(g_SoftBodyRenderPositions, metadata.deformVertexBase + vertexId).xyz;
        worldPos = float4(deformedPos, 1.0);
    }
#elif defined(CRESSIM_PROGRAM_FAMILY_CURVE)
    const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, objectIndex);
    if (metadata.deformVertexBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
        vertexId < metadata.deformVertexCount)
    {
        worldPos =
            float4(CRESSIM_SB_LOAD(g_CurveRenderPositions, metadata.deformVertexBase + vertexId).xyz,
                   1.0);
    }
#endif
    Out.Position = mul(worldPos, preparedCamera.lightViewProjectionMatrices[g_CascadeIndex]);
#if MANUAL_LAYER_EXPORT
    Out.Layer = shadowLayer;
#endif
#endif
}
