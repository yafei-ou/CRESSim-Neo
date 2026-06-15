#include "include/graphics/graphics_forward_constants.hlsli"
#include "include/graphics/graphics_scene_buffers.hlsli"

struct VSInput
{
    float3 Position : ATTRIB0;
    float3 Normal : ATTRIB1;
    float2 TexCoord : ATTRIB2;
    float4 Tangent : ATTRIB3;
};

struct VSOutput
{
    float4 Position : SV_Position;
    float3 WorldPos : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    float2 TexCoord : TEXCOORD2;
    float4 WorldTangent : TEXCOORD3;
    nointerpolation uint CameraIndex : TEXCOORD4;
    nointerpolation uint MainLightIndex : TEXCOORD5;
    nointerpolation uint ShadowLayer : TEXCOORD6;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
};

void main(in VSInput In, out VSOutput Out, uint instanceId : SV_InstanceID
#if defined(CRESSIM_PROGRAM_FAMILY_SOFT_BODY) || defined(CRESSIM_PROGRAM_FAMILY_CURVE)
    , uint vertexId : SV_VertexID
#endif
)
{
    uint objectIndex = g_InstanceIndex;
    uint cameraIndex = g_CurrentCameraIndex;
    uint colorLayer = 0u;
    uint shadowLayer = CRESSIM_INVALID_BATCH_CAMERA_LAYER;
    uint mainLightIndex = CRESSIM_INVALID_GPU_SCENE_INDEX;
    if (g_UseDrawListBuffer != 0u)
    {
        const VisiblePairInstance pair = CRESSIM_SB_LOAD(g_VisiblePairs, g_DrawListOffset + instanceId);
        const BatchCameraMetadata batchCamera = CRESSIM_SB_LOAD(g_BatchCameras, pair.batchCameraIndex);
        objectIndex = pair.objectIndex;
        cameraIndex = batchCamera.globalCameraIndex;
        colorLayer = batchCamera.colorLayer;
        shadowLayer = batchCamera.shadowLayer;
        mainLightIndex = batchCamera.mainLightIndex;
    }
    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, cameraIndex);
    bool poseValid = false;
    float3 position = float3(0.0, 0.0, 0.0);
    float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
    float3 scale = float3(1.0, 1.0, 1.0);
    loadRenderablePose(objectIndex, poseValid, position, orientation, scale);
    const uint localObjectIndex = objectIndex - preparedCamera.objectRangeStart;
    const uint visibilityIndex = preparedCamera.visibilityDataOffset + localObjectIndex;
    if (!poseValid || CRESSIM_SB_LOAD(g_RenderableVisibilityFlags, visibilityIndex) == 0u ||
        preparedCamera.active == 0u)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.WorldPos = float3(0.0, 0.0, 0.0);
        Out.WorldNormal = float3(0.0, 1.0, 0.0);
        Out.TexCoord = In.TexCoord;
        Out.WorldTangent = float4(1.0, 0.0, 0.0, 1.0);
        Out.CameraIndex = cameraIndex;
        Out.MainLightIndex = mainLightIndex;
        Out.ShadowLayer = shadowLayer;
#if MANUAL_LAYER_EXPORT
        Out.Layer = colorLayer;
#endif
        return;
    }

    float4 worldPos = float4(0.0, 0.0, 0.0, 1.0);
    float3 worldNormal = float3(0.0, 1.0, 0.0);
    float3 worldTangent = float3(1.0, 0.0, 0.0);
    float transformSign = 1.0;
#if defined(CRESSIM_PROGRAM_FAMILY_SOFT_BODY)
    const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, objectIndex);
    if (metadata.deformVertexBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
        metadata.deformNormalBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
        metadata.deformVertexCount > 0u && vertexId < metadata.deformVertexCount)
    {
        const float3 deformedPos =
            CRESSIM_SB_LOAD(g_SoftBodyRenderPositions, metadata.deformVertexBase + vertexId).xyz;
        worldPos = float4(deformedPos, 1.0);
        worldNormal =
            normalize(CRESSIM_SB_LOAD(g_SoftBodyVertexNormals, metadata.deformNormalBase + vertexId).xyz);
        const float3 tangentCandidate = In.Tangent.xyz - worldNormal * dot(worldNormal, In.Tangent.xyz);
        worldTangent = normalize(dot(tangentCandidate, tangentCandidate) > 1e-6
                                     ? tangentCandidate
                                     : cross(abs(worldNormal.y) < 0.99 ? float3(0.0, 1.0, 0.0)
                                                                       : float3(1.0, 0.0, 0.0),
                                             worldNormal));
    }
    else
    {
        worldPos = float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
        float3 safeScale = max(abs(scale), float3(1e-6, 1e-6, 1e-6));
        worldNormal = normalize(quaternionRotateVector(orientation, In.Normal / safeScale));
        worldTangent = quaternionRotateVector(orientation, In.Tangent.xyz * scale);
        worldTangent = normalize(worldTangent - worldNormal * dot(worldNormal, worldTangent));
        transformSign = (scale.x * scale.y * scale.z) < 0.0 ? -1.0 : 1.0;
    }
#elif defined(CRESSIM_PROGRAM_FAMILY_CURVE)
    const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, objectIndex);
    if (metadata.deformVertexBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
        metadata.deformNormalBase != CRESSIM_INVALID_DEFORM_VERTEX_BASE &&
        metadata.deformVertexCount > 0u && vertexId < metadata.deformVertexCount)
    {
        worldPos = float4(CRESSIM_SB_LOAD(g_CurveRenderPositions, metadata.deformVertexBase + vertexId).xyz, 1.0);
        worldNormal =
            normalize(CRESSIM_SB_LOAD(g_CurveRenderNormals, metadata.deformNormalBase + vertexId).xyz);
        const float3 tangentCandidate = In.Tangent.xyz - worldNormal * dot(worldNormal, In.Tangent.xyz);
        worldTangent = normalize(dot(tangentCandidate, tangentCandidate) > 1e-6
                                     ? tangentCandidate
                                     : cross(abs(worldNormal.y) < 0.99 ? float3(0.0, 1.0, 0.0)
                                                                       : float3(1.0, 0.0, 0.0),
                                             worldNormal));
    }
    else
    {
        worldPos = float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
        float3 safeScale = max(abs(scale), float3(1e-6, 1e-6, 1e-6));
        worldNormal = normalize(quaternionRotateVector(orientation, In.Normal / safeScale));
        worldTangent = quaternionRotateVector(orientation, In.Tangent.xyz * scale);
        worldTangent = normalize(worldTangent - worldNormal * dot(worldNormal, worldTangent));
        transformSign = (scale.x * scale.y * scale.z) < 0.0 ? -1.0 : 1.0;
    }
#else
    worldPos = float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
    float3 safeScale = max(abs(scale), float3(1e-6, 1e-6, 1e-6));
    worldNormal = normalize(quaternionRotateVector(orientation, In.Normal / safeScale));
    worldTangent = quaternionRotateVector(orientation, In.Tangent.xyz * scale);
    worldTangent = normalize(worldTangent - worldNormal * dot(worldNormal, worldTangent));
    transformSign = (scale.x * scale.y * scale.z) < 0.0 ? -1.0 : 1.0;
#endif

    Out.Position = mul(worldPos, preparedCamera.viewProjectionMatrix);
    Out.WorldPos = worldPos.xyz;
    Out.WorldNormal = worldNormal;
    Out.TexCoord = In.TexCoord;
    Out.WorldTangent = float4(worldTangent, In.Tangent.w * transformSign);
    Out.CameraIndex = cameraIndex;
    Out.MainLightIndex = mainLightIndex;
    Out.ShadowLayer = shadowLayer;
#if MANUAL_LAYER_EXPORT
    Out.Layer = colorLayer;
#endif
}
