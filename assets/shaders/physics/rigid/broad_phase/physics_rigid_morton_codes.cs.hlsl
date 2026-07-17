#include "physics/physics_rigid_broad_phase_build_constants.hlsli"
#include "physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseElement, g_BroadPhaseElements);
CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseExtent, g_GlobalExtent);
CRESSIM_RW_STRUCTURED_BUFFER(GpuMortonCodeElement, g_MortonCodes);

uint ExpandBits(uint value)
{
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

uint Morton3D(float x, float y, float z)
{
    x = clamp(x * 1024.0, 0.0, 1023.0);
    y = clamp(y * 1024.0, 0.0, 1023.0);
    z = clamp(z * 1024.0, 0.0, 1023.0);
    const uint xx = ExpandBits((uint)x);
    const uint yy = ExpandBits((uint)y);
    const uint zz = ExpandBits((uint)z);
    return xx * 4u + yy * 2u + zz;
}

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= elementCount)
    {
        return;
    }

    const GpuBroadPhaseElement element = CRESSIM_SB_LOAD(g_BroadPhaseElements, index);
    const GpuBroadPhaseExtent globalExtent = CRESSIM_SB_LOAD(g_GlobalExtent, 0);
    const float3 minExtent = globalExtent.minBounds.xyz;
    const float3 maxExtent = globalExtent.maxBounds.xyz;
    const float3 extentSize = max(maxExtent - minExtent, float3(1.0e-5, 1.0e-5, 1.0e-5));
    const float3 center =
        0.5 * (float3(element.aabbMinX, element.aabbMinY, element.aabbMinZ) +
               float3(element.aabbMaxX, element.aabbMaxY, element.aabbMaxZ));
    const float3 mappedCenter = saturate((center - minExtent) / extentSize);

    GpuMortonCodeElement morton;
    morton.mortonCode = Morton3D(mappedCenter.x, mappedCenter.y, mappedCenter.z);
    morton.elementIdx = index;
    CRESSIM_SB_STORE(g_MortonCodes, index, morton);
}
