#include "include/graphics/graphics_scene_buffers.hlsli"

struct CameraInput
{
    float4 position;
    float4 orientation;
    float4 projectionParams;
    float4 viewportAndOutputSize;
    uint envIndex;
    uint cameraSlot;
    uint active;
    uint reserved;
};

cbuffer GraphicsFluidColor
{
    uint4 g_FluidColorParams;
};

#define g_FluidColorCameraIndex g_FluidColorParams.x
#define g_FluidColorDepthLayer g_FluidColorParams.y
#define g_FluidColorEnvIndex g_FluidColorParams.z

static const uint CRESSIM_PARTICLE_KIND_FLUID = 1u;
static const uint CRESSIM_PARTICLE_OWNER_FLUID_BODY = 2u;

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleEnvironmentIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidVisuals);

struct VSOutput
{
    float4 Position : SV_Position;
    float3 ViewPos : TEXCOORD0;
    float2 Uv : TEXCOORD1;
    float4 Color : TEXCOORD2;
    float DepthTolerance : TEXCOORD3;
};

float degreesToRadians(float degrees)
{
    return degrees * 0.017453292519943295f;
}

float computeEffectiveViewportAspect(float4 viewportAndOutputSize)
{
    const float viewportWidth = clamp(viewportAndOutputSize.x, 0.0, 1.0);
    const float viewportHeight = clamp(viewportAndOutputSize.y, 0.0, 1.0);
    const float outputWidth = max(viewportAndOutputSize.z, 1.0);
    const float outputHeight = max(viewportAndOutputSize.w, 1.0);
    const float effectiveWidth = outputWidth * max(viewportWidth, 1.0e-5);
    const float effectiveHeight = outputHeight * max(viewportHeight, 1.0e-5);
    return max(effectiveWidth / max(effectiveHeight, 1.0e-5), 1.0e-5);
}

float4x4 buildProjectionMatrix(float verticalFovDegrees, float aspect, float nearClip, float farClip)
{
    const float fovRadians = max(degreesToRadians(verticalFovDegrees), degreesToRadians(1.0));
    const float yScale = 1.0 / tan(0.5 * fovRadians);
    const float xScale = yScale / max(aspect, 1e-5);
    const float zScale = farClip / max(farClip - nearClip, 1e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1e-5);

    return float4x4(
        xScale, 0.0,    0.0,       0.0,
        0.0,    yScale, 0.0,       0.0,
        0.0,    0.0,    zScale,    1.0,
        0.0,    0.0,    zTranslate, 0.0);
}

float2 quadCornerForVertex(uint triangleVertexIndex)
{
    if (triangleVertexIndex == 0u) return float2(-1.0, -1.0);
    if (triangleVertexIndex == 1u) return float2(1.0, -1.0);
    if (triangleVertexIndex == 2u) return float2(-1.0, 1.0);
    if (triangleVertexIndex == 3u) return float2(-1.0, 1.0);
    if (triangleVertexIndex == 4u) return float2(1.0, -1.0);
    return float2(1.0, 1.0);
}

void main(uint vertexId : SV_VertexID, out VSOutput Out)
{
    const uint particleIndex = vertexId / 6u;
    const uint triangleVertexIndex = vertexId % 6u;
    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, g_FluidColorCameraIndex);
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_FluidColorCameraIndex);
    const float4 particlePositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float particleRadius = CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex);
    const uint particleEnvIndex = CRESSIM_SB_LOAD(g_ParticleEnvironmentIndices, particleIndex);
    const uint particleKind = CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex);
    const uint ownerType = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleIndex);
    const uint ownerIndex = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, particleIndex);

    if (preparedCamera.active == 0u || cameraInput.active == 0u ||
        particleEnvIndex != g_FluidColorEnvIndex || particleKind != CRESSIM_PARTICLE_KIND_FLUID ||
        ownerType != CRESSIM_PARTICLE_OWNER_FLUID_BODY)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.ViewPos = float3(0.0, 0.0, 1.0);
        Out.Uv = float2(0.0, 0.0);
        Out.Color = 0.0;
        Out.DepthTolerance = 0.02;
        return;
    }

    const float4 worldPos = float4(particlePositionInvMass.xyz, 1.0);
    const float3 particleViewPos = mul(worldPos, preparedCamera.viewMatrix).xyz;
    const float2 quadCoord = quadCornerForVertex(triangleVertexIndex);
    const float scale = max(particleRadius, 1.0e-4) * 1.5;
    const float3 viewPos = particleViewPos + float3(quadCoord * scale, 0.0);
    const float aspect = computeEffectiveViewportAspect(cameraInput.viewportAndOutputSize);
    const float4x4 projectionMatrix =
        buildProjectionMatrix(cameraInput.projectionParams.x, aspect,
                              cameraInput.projectionParams.y, cameraInput.projectionParams.z);

    Out.Position = mul(float4(viewPos, 1.0), projectionMatrix);
    Out.ViewPos = viewPos;
    Out.Uv = quadCoord * 0.5 + 0.5;
    Out.Color = CRESSIM_SB_LOAD(g_FluidVisuals, ownerIndex);
    // Let the color pass cover a slightly wider depth band than the exact billboard
    // depth so edge pixels survive the filtered surface match.
    Out.DepthTolerance = max(scale * 1.35, 0.04);
}
