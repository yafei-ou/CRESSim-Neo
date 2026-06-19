cbuffer GraphicsDebugParticles
{
    float4 g_DebugParticleColor;
    float4 g_DebugParticleStaticColor;
    float4 g_DebugParticleEdgeColor;
    float4 g_DebugParticleEdgeHighStrainColor;
    float4 g_DebugParticleEdgeDamagedColor;
    float4 g_DebugParticleEdgeDisabledColor;
    uint4 g_DebugParticleParams;
    uint4 g_DebugShapeParams;
    float4 g_DebugParticleMisc;
};

struct PSInput
{
    float4 Position : SV_Position;
    nointerpolation float4 Color : TEXCOORD0;
};

float4 main(in PSInput In) : SV_Target
{
    return In.Color;
}
