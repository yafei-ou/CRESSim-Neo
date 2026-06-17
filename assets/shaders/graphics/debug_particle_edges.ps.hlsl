cbuffer GraphicsDebugParticles
{
    float4 g_DebugParticleColor;
    float4 g_DebugParticleStaticColor;
    float4 g_DebugParticleEdgeColor;
    uint4 g_DebugParticleParams;
    float4 g_DebugParticleMisc;
};

float4 main() : SV_Target
{
    return g_DebugParticleEdgeColor;
}
