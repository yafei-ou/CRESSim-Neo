#include "physics_atomic_float.hlsli"
#include "physics_math.hlsli"
#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_solver_config.hlsli"

// Experimental all-pairs strand capsule self-collision. This intentionally avoids the production
// particle spatial hash so the feature remains isolated and easy to remove while its behavior is
// evaluated. The CPU dispatches N*N threads and this shader retains one unordered pair per thread.

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleStrandIds);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegment, g_StrandSegments);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);

static const uint kInvalidStrandId = 0xffffffffu;

void ClosestSegmentPoints(float3 a0, float3 a1, float3 b0, float3 b1, out float s, out float t,
                          out float3 pointA, out float3 pointB)
{
    const float3 d1 = a1 - a0;
    const float3 d2 = b1 - b0;
    const float3 r = a0 - b0;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);

    s = 0.0;
    t = 0.0;
    if (a <= kEpsilon && e <= kEpsilon)
    {
        pointA = a0;
        pointB = b0;
        return;
    }
    if (a <= kEpsilon)
    {
        t = saturate(f / e);
    }
    else
    {
        const float c = dot(d1, r);
        if (e <= kEpsilon)
        {
            s = saturate(-c / a);
        }
        else
        {
            const float b = dot(d1, d2);
            const float denominator = a * e - b * b;
            if (abs(denominator) > kEpsilon)
            {
                s = saturate((b * f - c * e) / denominator);
            }

            const float tNumerator = b * s + f;
            if (tNumerator < 0.0)
            {
                t = 0.0;
                s = saturate(-c / a);
            }
            else if (tNumerator > e)
            {
                t = 1.0;
                s = saturate((b - c) / a);
            }
            else
            {
                t = tNumerator / e;
            }
        }
    }

    pointA = a0 + d1 * s;
    pointB = b0 + d2 * t;
}

float3 ContactNormal(float3 pointA, float3 pointB, float3 fallbackA, float3 fallbackB,
                     float3 directionA, float3 directionB)
{
    const float3 delta = pointB - pointA;
    const float distanceSq = dot(delta, delta);
    if (distanceSq > kEpsilon)
    {
        return delta * rsqrt(distanceSq);
    }

    const float3 fallbackDelta = fallbackB - fallbackA;
    const float fallbackDistanceSq = dot(fallbackDelta, fallbackDelta);
    if (fallbackDistanceSq > kEpsilon)
    {
        return fallbackDelta * rsqrt(fallbackDistanceSq);
    }
    return SafeNormalize(cross(directionA, directionB), float3(0.0, 1.0, 0.0));
}

bool FindSweptContact(float3 previousA0, float3 previousA1, float3 currentA0, float3 currentA1,
                      float3 previousB0, float3 previousB1, float3 currentB0, float3 currentB1,
                      float targetDistance, uint maxIterations, out float contactS,
                      out float contactT, out float3 contactNormal)
{
    contactS = 0.0;
    contactT = 0.0;
    contactNormal = float3(0.0, 1.0, 0.0);

    const float motionA = max(length(currentA0 - previousA0), length(currentA1 - previousA1));
    const float motionB = max(length(currentB0 - previousB0), length(currentB1 - previousB1));
    const float motionBound = motionA + motionB;
    if (motionBound <= kEpsilon || maxIterations == 0u)
    {
        return false;
    }

    const float contactEpsilon = max(1.0e-7, targetDistance * 1.0e-5);
    float tau = 0.0;
    [loop]
    for (uint iteration = 0u; iteration < maxIterations; ++iteration)
    {
        const float3 a0 = lerp(previousA0, currentA0, tau);
        const float3 a1 = lerp(previousA1, currentA1, tau);
        const float3 b0 = lerp(previousB0, currentB0, tau);
        const float3 b1 = lerp(previousB1, currentB1, tau);
        float3 pointA;
        float3 pointB;
        ClosestSegmentPoints(a0, a1, b0, b1, contactS, contactT, pointA, pointB);
        const float distance = length(pointB - pointA);
        if (distance <= targetDistance + contactEpsilon)
        {
            const float3 previousPointA = lerp(previousA0, previousA1, contactS);
            const float3 previousPointB = lerp(previousB0, previousB1, contactT);
            contactNormal = ContactNormal(pointA, pointB, previousPointA, previousPointB,
                                          a1 - a0, b1 - b0);
            return true;
        }

        const float deltaTau = 0.9 * (distance - targetDistance) / motionBound;
        if (deltaTau <= 1.0e-6)
        {
            return false;
        }
        tau += deltaTau;
        if (tau > 1.0)
        {
            return false;
        }
    }
    return false;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    const uint segmentCount = strandSegmentCount;
    if (segmentCount < 2u || pairIndex >= segmentCount * segmentCount)
    {
        return;
    }

    const uint segmentIndexA = pairIndex / segmentCount;
    const uint segmentIndexB = pairIndex - segmentIndexA * segmentCount;
    if (segmentIndexB <= segmentIndexA || segmentIndexB - segmentIndexA <= reserved0)
    {
        return;
    }

    const GpuStrandSegment segmentA = CRESSIM_SB_LOAD(g_StrandSegments, segmentIndexA);
    const GpuStrandSegment segmentB = CRESSIM_SB_LOAD(g_StrandSegments, segmentIndexB);
    const uint strandIdA = CRESSIM_SB_LOAD(g_ParticleStrandIds, segmentA.particleA);
    const uint strandIdB = CRESSIM_SB_LOAD(g_ParticleStrandIds, segmentB.particleA);
    if (strandIdA == kInvalidStrandId || strandIdA != strandIdB)
    {
        return;
    }

    const float4 stateA0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segmentA.particleA);
    const float4 stateA1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segmentA.particleB);
    const float4 stateB0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segmentB.particleA);
    const float4 stateB1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segmentB.particleB);
    const float radiusA = max(CRESSIM_SB_LOAD(g_ParticleRadii, segmentA.particleA),
                              CRESSIM_SB_LOAD(g_ParticleRadii, segmentA.particleB));
    const float radiusB = max(CRESSIM_SB_LOAD(g_ParticleRadii, segmentB.particleA),
                              CRESSIM_SB_LOAD(g_ParticleRadii, segmentB.particleB));
    const float targetDistance = radiusA + radiusB;
    if (targetDistance <= 0.0)
    {
        return;
    }

    float s;
    float t;
    float3 pointA;
    float3 pointB;
    ClosestSegmentPoints(stateA0.xyz, stateA1.xyz, stateB0.xyz, stateB1.xyz, s, t, pointA,
                         pointB);
    const float3 currentDelta = pointB - pointA;
    const float currentDistance = length(currentDelta);
    float penetration = targetDistance - currentDistance;
    float3 normal = ContactNormal(pointA, pointB, pointA, pointB,
                                  stateA1.xyz - stateA0.xyz, stateB1.xyz - stateB0.xyz);

    if (penetration <= kContactSlop)
    {
        const float3 previousA0 =
            CRESSIM_SB_LOAD(g_ParticlePreviousPositions, segmentA.particleA).xyz;
        const float3 previousA1 =
            CRESSIM_SB_LOAD(g_ParticlePreviousPositions, segmentA.particleB).xyz;
        const float3 previousB0 =
            CRESSIM_SB_LOAD(g_ParticlePreviousPositions, segmentB.particleA).xyz;
        const float3 previousB1 =
            CRESSIM_SB_LOAD(g_ParticlePreviousPositions, segmentB.particleB).xyz;
        if (!FindSweptContact(previousA0, previousA1, stateA0.xyz, stateA1.xyz, previousB0,
                              previousB1, stateB0.xyz, stateB1.xyz, targetDistance, reserved1, s,
                              t, normal))
        {
            return;
        }

        pointA = lerp(stateA0.xyz, stateA1.xyz, s);
        pointB = lerp(stateB0.xyz, stateB1.xyz, t);
        penetration = targetDistance - dot(pointB - pointA, normal);
        if (penetration <= kContactSlop)
        {
            return;
        }
    }

    const float weightA0 = 1.0 - s;
    const float weightA1 = s;
    const float weightB0 = 1.0 - t;
    const float weightB1 = t;
    const float inverseMass = stateA0.w * weightA0 * weightA0 +
                              stateA1.w * weightA1 * weightA1 +
                              stateB0.w * weightB0 * weightB0 +
                              stateB1.w * weightB1 * weightB1;
    if (inverseMass <= kEpsilon)
    {
        return;
    }

    const float correctionDistance =
        min(penetration - kContactSlop, kSoftMaxCorrectionPerIter) * kSoftContactRelaxation;
    const float lambda = correctionDistance / inverseMass;
    CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, segmentA.particleA,
                                  -normal * (stateA0.w * weightA0 * lambda));
    CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, segmentA.particleB,
                                  -normal * (stateA1.w * weightA1 * lambda));
    CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, segmentB.particleA,
                                  normal * (stateB0.w * weightB0 * lambda));
    CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, segmentB.particleB,
                                  normal * (stateB1.w * weightB1 * lambda));
}
