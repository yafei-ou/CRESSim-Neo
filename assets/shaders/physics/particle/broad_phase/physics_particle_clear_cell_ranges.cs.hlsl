#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= particleCellRangeCapacity)
    {
        return;
    }

    GpuParticleCellRange range;
    range.cellKey = kInvalidIndex;
    range.startIndex = 0u;
    range.endIndex = 0u;
    range.reserved0 = 0u;
    CRESSIM_SB_STORE(g_ParticleCellRanges, idx, range);
}
