#include "physics/physics_rigid_dispatch_constants.hlsli"
#include "physics/physics_atomic_float.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"
#include "physics/rigid/physics_rigid_joint_solver_shared.hlsli"
#include "physics/rigid/physics_rigid_solver_shared.hlsli"

static const float kCableRelaxation = 0.95;
static const float kMaxCableTranslationCorrection = 0.025;
static const float kMaxCableAngularCorrection = 0.15;
static const uint kMaxCableRoutePoints = 64u;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuRoutedCableConstraint, g_RoutedCableConstraints);
CRESSIM_STRUCTURED_BUFFER(GpuRoutedCableRoutePoint, g_RoutedCableRoutePoints);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_RoutedCableLambdas);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint cableIndex = dispatchThreadID.x;
    if (cableIndex >= reserved0)
    {
        return;
    }

    const GpuRoutedCableConstraint cable = CRESSIM_SB_LOAD(g_RoutedCableConstraints, cableIndex);
    if (cable.enabled == 0u)
    {
        CRESSIM_SB_STORE(g_RoutedCableLambdas, cableIndex, 0.0);
        return;
    }
    if (cable.routePointCount < 2u || cable.routePointCount > kMaxCableRoutePoints)
    {
        return;
    }

    uint bodyIndices[kMaxCableRoutePoints];
    float3 worldPoints[kMaxCableRoutePoints];
    float3 worldLeverArms[kMaxCableRoutePoints];
    float4 orientations[kMaxCableRoutePoints];
    float invMasses[kMaxCableRoutePoints];
    float3 inverseInertias[kMaxCableRoutePoints];
    uint bodyTypes[kMaxCableRoutePoints];
    float totalLength = 0.0;

    [loop] for (uint i = 0u; i < cable.routePointCount; ++i)
    {
        const GpuRoutedCableRoutePoint routePoint =
            CRESSIM_SB_LOAD(g_RoutedCableRoutePoints, cable.routePointStart + i);
        const uint rigidBodyIndex = routePoint.rigidBodyIndex;
        if (rigidBodyIndex >= rigidBodyCount)
        {
            return;
        }

        const float4 posInvMass = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex);
        const float4 q = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndex));
        const uint bodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndex);
        const float invMass = bodyType == kRigidBodyTypeDynamic ? posInvMass.w : 0.0;
        float3 invInertia = CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndex).xyz;
        if (invMass == 0.0)
        {
            invInertia = 0.0;
        }

        bodyIndices[i] = rigidBodyIndex;
        orientations[i] = q;
        bodyTypes[i] = bodyType;
        invMasses[i] = invMass;
        inverseInertias[i] = invInertia;
        const float3 worldLeverArm = QuaternionRotate(q, routePoint.localGuideOffset.xyz);
        worldLeverArms[i] = worldLeverArm;
        worldPoints[i] = posInvMass.xyz + worldLeverArm;

        if (i > 0u)
        {
            totalLength += length(worldPoints[i] - worldPoints[i - 1u]);
        }
    }

    const float constraint = totalLength - cable.targetLength;
    if (cable.tensionOnly != 0u && constraint <= 0.0)
    {
        return;
    }

    float3 guideGradients[kMaxCableRoutePoints];
    float denominator = 0.0;
    [loop] for (uint i = 0u; i < cable.routePointCount; ++i)
    {
        float3 gradient = 0.0;
        if (i > 0u)
        {
            const float3 deltaPrev = worldPoints[i] - worldPoints[i - 1u];
            const float prevLengthSq = dot(deltaPrev, deltaPrev);
            if (prevLengthSq > kEpsilon)
            {
                gradient += deltaPrev * rsqrt(prevLengthSq);
            }
        }
        if (i + 1u < cable.routePointCount)
        {
            const float3 deltaNext = worldPoints[i + 1u] - worldPoints[i];
            const float nextLengthSq = dot(deltaNext, deltaNext);
            if (nextLengthSq > kEpsilon)
            {
                gradient -= deltaNext * rsqrt(nextLengthSq);
            }
        }

        guideGradients[i] = gradient;
        if (dot(gradient, gradient) <= kEpsilon)
        {
            continue;
        }

        const float invMass = invMasses[i];
        if (invMass == 0.0)
        {
            continue;
        }

        const float3 angularJacobian = cross(worldLeverArms[i], gradient);
        denominator += invMass * dot(gradient, gradient) +
                       dot(angularJacobian,
                           MultiplyWorldInverseInertia(inverseInertias[i], orientations[i],
                                                       angularJacobian));
    }

    const float alpha = cable.compliance / max(dt * dt, kEpsilon);
    denominator += alpha;
    if (denominator <= kEpsilon)
    {
        return;
    }

    const float lambda = CRESSIM_SB_LOAD(g_RoutedCableLambdas, cableIndex);
    const float deltaLambda = -(constraint + alpha * lambda) / denominator;
    CRESSIM_SB_STORE(g_RoutedCableLambdas, cableIndex, lambda + deltaLambda);

    [loop] for (uint i = 0u; i < cable.routePointCount; ++i)
    {
        const float invMass = invMasses[i];
        if (bodyTypes[i] != kRigidBodyTypeDynamic || invMass == 0.0)
        {
            continue;
        }

        const float3 gradient = guideGradients[i];
        if (dot(gradient, gradient) <= kEpsilon)
        {
            continue;
        }

        const float3 linearImpulse = gradient * deltaLambda * kCableRelaxation;
        const float3 angularImpulse =
            cross(worldLeverArms[i], gradient) * deltaLambda * kCableRelaxation;
        const float3 translation = linearImpulse * invMass;
        const float3 rotation =
            MultiplyWorldInverseInertia(inverseInertias[i], orientations[i], angularImpulse);
        const float correctionScale = ComputeCorrectionLimitScale(
            translation, rotation, 0.0, 0.0, kMaxCableTranslationCorrection,
            kMaxCableAngularCorrection);

        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyIndices[i],
                                      translation * correctionScale);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyIndices[i],
                                      rotation * correctionScale);
    }
}
