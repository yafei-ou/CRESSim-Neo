static const uint kMaxPreparedScanLevels = 8u;
static const uint kScanThreadGroupSize = 64u;

CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
CRESSIM_RW_STRUCTURED_BUFFER(uint4, g_ScanConstantsRW);
CRESSIM_RW_STRUCTURED_BUFFER(uint4, g_ScanIndirectArgsRW);

#ifndef CRESSIM_PARTICLE_CONTACT_SCAN_COUNT_EXPR
#    error "CRESSIM_PARTICLE_CONTACT_SCAN_COUNT_EXPR must be defined before including this file."
#endif

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    uint count = (CRESSIM_PARTICLE_CONTACT_SCAN_COUNT_EXPR);
    [loop]
    for (uint level = 0u; level < kMaxPreparedScanLevels; ++level)
    {
        const uint dispatchGroupCount =
            (count == 0u) ? 1u : ((count + kScanThreadGroupSize - 1u) / kScanThreadGroupSize);
        const uint nextCount = (count == 0u) ? 0u : dispatchGroupCount;
        const uint hasParentOffsets = nextCount > 1u ? 1u : 0u;
        CRESSIM_SB_STORE(g_ScanConstantsRW, level, uint4(count, hasParentOffsets, 0u, 0u));

        CRESSIM_SB_STORE(g_ScanIndirectArgsRW, level, uint4(dispatchGroupCount, 1u, 1u, 0u));

        count = hasParentOffsets != 0u ? nextCount : 0u;
    }
}

#undef CRESSIM_PARTICLE_CONTACT_SCAN_COUNT_EXPR
