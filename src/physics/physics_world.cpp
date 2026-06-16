#include "physics/physics_world.h"

#include "common/logger.h"
#include "common/math_utils_runtime.h"
#include "physics/particle_phase.h"
#include "physics/soft_body_authoring.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uint32_t kBroadPhaseContributionNone   = 0u;
constexpr std::uint32_t kBroadPhaseContributionMoving = 1u;
constexpr std::uint32_t kBroadPhaseContributionStatic = 2u;

std::uint32_t encodeSuturingFloat(float value) noexcept
{
    std::uint32_t encoded = 0u;
    static_assert(sizeof(encoded) == sizeof(value));
    std::memcpy(&encoded, &value, sizeof(encoded));
    return encoded;
}

Diligent::float4 toPositionInvMass(const RigidBodyState &state)
{
    return Diligent::float4{state.position.x, state.position.y, state.position.z,
                            state.inverseMass};
}

Diligent::float4 toOrientation(const RigidBodyState &state)
{
    return Diligent::float4{state.rotation.q.x, state.rotation.q.y, state.rotation.q.z,
                            state.rotation.q.w};
}

Diligent::float4 toLinearVelocity(const RigidBodyState &state)
{
    return Diligent::float4{state.linearVelocity.x, state.linearVelocity.y, state.linearVelocity.z,
                            0.0f};
}

Diligent::float4 toScale(const RigidBodyState &state)
{
    return Diligent::float4{state.scale.x, state.scale.y, state.scale.z, 0.0f};
}

Diligent::float4 toAngularVelocity(const RigidBodyState &state)
{
    return Diligent::float4{state.angularVelocity.x, state.angularVelocity.y,
                            state.angularVelocity.z, 0.0f};
}

Diligent::float4 toInverseInertiaLocal(const RigidBodyState &state)
{
    return Diligent::float4{state.inverseInertiaLocal.x, state.inverseInertiaLocal.y,
                            state.inverseInertiaLocal.z, 0.0f};
}

Diligent::float4 toKinematicTargetPosition(const RigidBodyState &state)
{
    return Diligent::float4{state.kinematicTargetPosition.x, state.kinematicTargetPosition.y,
                            state.kinematicTargetPosition.z, 0.0f};
}

Diligent::float4 toKinematicTargetOrientation(const RigidBodyState &state)
{
    return Diligent::float4{state.kinematicTargetRotation.q.x, state.kinematicTargetRotation.q.y,
                            state.kinematicTargetRotation.q.z, state.kinematicTargetRotation.q.w};
}

Diligent::float4 toColliderLocalPosition(const ColliderState &state)
{
    return Diligent::float4{state.localPosition.x, state.localPosition.y, state.localPosition.z,
                            0.0f};
}

Diligent::float4 toColliderLocalOrientation(const ColliderState &state)
{
    return Diligent::float4{state.localRotation.q.x, state.localRotation.q.y,
                            state.localRotation.q.z, state.localRotation.q.w};
}

Diligent::float4 toColliderMaterial(const ColliderState &state)
{
    return Diligent::float4{state.friction, state.restitution, 0.0f, state.staticFriction};
}

Diligent::float4 toParticleContactMaterial(const ParticleContactMaterialDesc &material)
{
    return Diligent::float4{material.friction, material.restitution, material.damping,
                            material.staticFriction};
}

constexpr float kFluidKernelPi                  = 3.14159265358979323846f;
constexpr float kFluidContactDistanceMultiplier = 2.0f / 0.6f;

float fluidSpikyWeight(float distance, float supportRadius) noexcept
{
    const float oneMinusQ = 1.0f - distance / supportRadius;
    const float k = 15.0f / (kFluidKernelPi * supportRadius * supportRadius * supportRadius);
    return k * oneMinusQ * oneMinusQ;
}

float fluidSpikyWeightDerivative(float distance, float supportRadius) noexcept
{
    const float oneMinusQ = 1.0f - distance / supportRadius;
    const float k = 15.0f / (kFluidKernelPi * supportRadius * supportRadius * supportRadius);
    return -k * 2.0f * oneMinusQ / supportRadius;
}

std::vector<std::uint32_t> validateStaticParticleIndices(const std::vector<std::uint32_t> &indices,
                                                         const std::uint32_t particleCount,
                                                         std::string &errorMessage)
{
    std::vector<std::uint32_t> result = indices;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    for (const std::uint32_t index : result)
    {
        if (index >= particleCount)
        {
            std::ostringstream stream;
            stream << "Static particle index " << index << " is out of range for " << particleCount
                   << " particles.";
            errorMessage = stream.str();
            return {};
        }
    }

    errorMessage.clear();
    return result;
}

std::uint32_t tightPack3D(float radius, float separation,
                          std::array<Diligent::float3, 2048> &points) noexcept
{
    const int dim        = static_cast<int>(std::ceil(radius / separation));
    std::uint32_t count  = 0u;
    const float rowScale = std::sqrt(0.75f) * separation;

    for (int z = -dim; z <= dim; ++z)
    {
        for (int y = -dim; y <= dim; ++y)
        {
            for (int x = -dim; x <= dim; ++x)
            {
                const float xpos = x * separation + (((y + z) & 1) != 0 ? separation * 0.5f : 0.0f);
                const Diligent::float3 sample{xpos, y * rowScale, z * rowScale};
                const float magnitudeSq =
                    sample.x * sample.x + sample.y * sample.y + sample.z * sample.z;
                if (magnitudeSq == 0.0f)
                {
                    continue;
                }

                if (std::sqrt(magnitudeSq) <= radius)
                {
                    if (count == points.size())
                    {
                        return count;
                    }
                    points[count++] = sample;
                }
            }
        }
    }

    return count;
}

void calculateFluidSolveScales(float restDistance, float supportRadius, float &restDensity,
                               float &densityConstraintScale,
                               float &surfaceConstraintScale) noexcept
{
    std::array<Diligent::float3, 2048> samples{};
    const std::uint32_t count = tightPack3D(supportRadius, restDistance, samples);

    restDensity            = 0.0f;
    densityConstraintScale = 0.0f;
    float a                = 0.0f;
    float b                = 0.0f;

    for (std::uint32_t i = 0u; i < count; ++i)
    {
        const Diligent::float3 sample = samples[i];
        const float distance =
            std::sqrt(sample.x * sample.x + sample.y * sample.y + sample.z * sample.z);

        restDensity += fluidSpikyWeight(distance, supportRadius);
        const float derivative = fluidSpikyWeightDerivative(distance, supportRadius);
        densityConstraintScale += derivative * derivative;

        if (sample.y <= 0.0f && distance > 0.0f)
        {
            const float cosTheta = sample.y / distance;
            a += derivative * cosTheta;
            b -= distance * cosTheta;
        }
    }

    if (restDensity <= 1.0e-6f)
    {
        restDensity = 1.0f;
    }
    if (densityConstraintScale <= 1.0e-6f)
    {
        densityConstraintScale = 1.0f;
    }
    surfaceConstraintScale =
        (std::abs(b) > 1.0e-6f && std::abs(a) > 1.0e-6f) ? std::abs(a / b) : 1.0f;
}

FluidMaterialGpu toFluidSolverMaterial(const FluidMaterialDesc &material,
                                       float particleRadius) noexcept
{
    const float fluidRestDistance = std::max(2.0f * particleRadius, 1.0e-4f);
    // PhysX-like shared density model: support/contact distance is derived from spacing
    // rather than taken directly from the authored material.
    const float smoothingRadius =
        std::max(kFluidContactDistanceMultiplier * particleRadius, fluidRestDistance);
    float restDensity            = 0.0f;
    float densityConstraintScale = 0.0f;
    float surfaceConstraintScale = 0.0f;
    calculateFluidSolveScales(fluidRestDistance, smoothingRadius, restDensity,
                              densityConstraintScale, surfaceConstraintScale);

    const float invRestDensity = 1.0f / restDensity;
    const float restRatio      = std::clamp(fluidRestDistance / smoothingRadius, 1.0e-3f, 0.999f);
    const float restRatioSq    = restRatio * restRatio;
    const float cohesion1      = -(1.0f + restRatio) / restRatioSq;
    const float cohesion2      = (restRatioSq + restRatio + 1.0f) / restRatioSq;

    return FluidMaterialGpu{restDensity,
                            invRestDensity,
                            smoothingRadius,
                            1.0f / std::max(densityConstraintScale, 1.0e-6f),
                            material.viscosity * invRestDensity,
                            material.cohesion * smoothingRadius,
                            cohesion1,
                            cohesion2,
                            (material.surfaceTension * invRestDensity) / surfaceConstraintScale,
                            material.vorticityConfinement * invRestDensity,
                            material.gravityScale,
                            material.cflCoefficient * smoothingRadius};
}

bool nearlyEqual(float a, float b, float epsilon = 1.0e-4f) noexcept
{
    const float scale = std::max({1.0f, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= epsilon * scale;
}

void normalizeParticleContactMaterial(ParticleContactMaterialDesc &material) noexcept
{
    material.friction       = std::max(material.friction, 0.0f);
    material.staticFriction = material.staticFriction < 0.0f
                                  ? material.friction
                                  : std::max(material.staticFriction, 0.0f);
    material.restitution    = std::clamp(material.restitution, 0.0f, 1.0f);
    material.damping        = std::max(material.damping, 0.0f);
}

void normalizeFluidMaterial(FluidMaterialDesc &material) noexcept
{
    normalizeParticleContactMaterial(material.contact);
    material.viscosity            = std::max(material.viscosity, 0.0f);
    material.cohesion             = std::max(material.cohesion, 0.0f);
    material.surfaceTension       = std::max(material.surfaceTension, 0.0f);
    material.vorticityConfinement = std::max(material.vorticityConfinement, 0.0f);
    material.gravityScale         = std::max(material.gravityScale, 0.0f);
    material.cflCoefficient       = std::max(material.cflCoefficient, 0.0f);
}

std::uint32_t findOrAppendParticleContactMaterial(std::vector<Diligent::float4> &materials,
                                                  const Diligent::float4 &material) noexcept
{
    for (std::uint32_t materialIndex = 0u;
         materialIndex < static_cast<std::uint32_t>(materials.size()); ++materialIndex)
    {
        if (materials[materialIndex] == material)
        {
            return materialIndex;
        }
    }

    materials.push_back(material);
    return static_cast<std::uint32_t>(materials.size() - 1u);
}

std::uint32_t findOrAppendFluidMaterial(std::vector<FluidMaterialGpu> &materials,
                                        const FluidMaterialGpu &material) noexcept
{
    for (std::uint32_t materialIndex = 0u;
         materialIndex < static_cast<std::uint32_t>(materials.size()); ++materialIndex)
    {
        if (materials[materialIndex] == material)
        {
            return materialIndex;
        }
    }

    materials.push_back(material);
    return static_cast<std::uint32_t>(materials.size() - 1u);
}

Diligent::float4 toFloat4(const Diligent::float3 &value, float w = 0.0f) noexcept
{
    return Diligent::float4{value.x, value.y, value.z, w};
}

Diligent::float3 safeNormalize(const Diligent::float3 &value,
                               const Diligent::float3 &fallback) noexcept
{
    const float lengthSq = Diligent::dot(value, value);
    if (lengthSq <= 1.0e-8f)
    {
        return fallback;
    }
    return value / std::sqrt(lengthSq);
}

Diligent::float3 quaternionRotate(const Diligent::QuaternionF &q,
                                  const Diligent::float3 &v) noexcept
{
    const Diligent::float3 qv{q.q.x, q.q.y, q.q.z};
    const Diligent::float3 t = 2.0f * Diligent::cross(qv, v);
    return v + q.q.w * t + Diligent::cross(qv, t);
}

Diligent::float3 quaternionInverseRotate(const Diligent::QuaternionF &q,
                                         const Diligent::float3 &v) noexcept
{
    const Diligent::QuaternionF conjugate{-q.q.x, -q.q.y, -q.q.z, q.q.w};
    return quaternionRotate(conjugate, v);
}

Diligent::QuaternionF quaternionConjugate(const Diligent::QuaternionF &q) noexcept
{
    return Diligent::QuaternionF{-q.q.x, -q.q.y, -q.q.z, q.q.w};
}

Diligent::QuaternionF quaternionMultiply(const Diligent::QuaternionF &a,
                                         const Diligent::QuaternionF &b) noexcept
{
    return Diligent::QuaternionF{a.q.w * b.q.x + a.q.x * b.q.w + a.q.y * b.q.z - a.q.z * b.q.y,
                                 a.q.w * b.q.y - a.q.x * b.q.z + a.q.y * b.q.w + a.q.z * b.q.x,
                                 a.q.w * b.q.z + a.q.x * b.q.y - a.q.y * b.q.x + a.q.z * b.q.w,
                                 a.q.w * b.q.w - a.q.x * b.q.x - a.q.y * b.q.y - a.q.z * b.q.z};
}

Diligent::QuaternionF normalizeQuaternion(const Diligent::QuaternionF &q) noexcept
{
    return common::runtime_math::normalizeQuaternion(q);
}

Diligent::QuaternionF quaternionFromBasis(const Diligent::float3 &xAxis,
                                          const Diligent::float3 &yAxis,
                                          const Diligent::float3 &zAxis) noexcept
{
    const float m00 = xAxis.x;
    const float m01 = yAxis.x;
    const float m02 = zAxis.x;
    const float m10 = xAxis.y;
    const float m11 = yAxis.y;
    const float m12 = zAxis.y;
    const float m20 = xAxis.z;
    const float m21 = yAxis.z;
    const float m22 = zAxis.z;

    const float trace = m00 + m11 + m22;
    Diligent::QuaternionF result{};
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        result.q.w    = 0.25f * s;
        result.q.x    = (m21 - m12) / s;
        result.q.y    = (m02 - m20) / s;
        result.q.z    = (m10 - m01) / s;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        result.q.w    = (m21 - m12) / s;
        result.q.x    = 0.25f * s;
        result.q.y    = (m01 + m10) / s;
        result.q.z    = (m02 + m20) / s;
    }
    else if (m11 > m22)
    {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        result.q.w    = (m02 - m20) / s;
        result.q.x    = (m01 + m10) / s;
        result.q.y    = 0.25f * s;
        result.q.z    = (m12 + m21) / s;
    }
    else
    {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        result.q.w    = (m10 - m01) / s;
        result.q.x    = (m02 + m20) / s;
        result.q.y    = (m12 + m21) / s;
        result.q.z    = 0.25f * s;
    }

    return normalizeQuaternion(result);
}

Diligent::float3 defaultRootMaterialNormal(const Diligent::float3 &segmentDirection,
                                           const Diligent::float3 &authoredNormal) noexcept
{
    const Diligent::float3 direction = safeNormalize(segmentDirection, Diligent::float3{1.0f, 0.0f, 0.0f});
    Diligent::float3 normal          = authoredNormal;
    if (Diligent::dot(normal, normal) <= 1.0e-8f)
    {
        normal = Diligent::float3{0.0f, 1.0f, 0.0f};
    }

    normal = normal - direction * Diligent::dot(normal, direction);
    if (Diligent::dot(normal, normal) <= 1.0e-8f)
    {
        Diligent::float3 fallback{0.0f, 1.0f, 0.0f};
        if (std::abs(Diligent::dot(direction, fallback)) > 0.9f)
        {
            fallback = Diligent::float3{1.0f, 0.0f, 0.0f};
        }
        normal = fallback - direction * Diligent::dot(fallback, direction);
    }

    return safeNormalize(normal, Diligent::float3{0.0f, 1.0f, 0.0f});
}

Diligent::float3 parallelTransportNormal(const Diligent::float3 &previousTangent,
                                         const Diligent::float3 &nextTangent,
                                         const Diligent::float3 &previousNormal) noexcept
{
    Diligent::float3 transported = previousNormal - nextTangent * Diligent::dot(previousNormal, nextTangent);
    if (Diligent::dot(transported, transported) <= 1.0e-8f)
    {
        transported = defaultRootMaterialNormal(nextTangent, previousNormal);
    }
    return safeNormalize(transported, defaultRootMaterialNormal(nextTangent, previousNormal));
}

Diligent::QuaternionF segmentOrientationFromTangentNormal(const Diligent::float3 &tangent,
                                                          const Diligent::float3 &materialNormal) noexcept
{
    const Diligent::float3 xAxis = safeNormalize(tangent, Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 yAxis =
        safeNormalize(materialNormal - xAxis * Diligent::dot(materialNormal, xAxis),
                      defaultRootMaterialNormal(xAxis, materialNormal));
    const Diligent::float3 zAxis = safeNormalize(Diligent::cross(xAxis, yAxis),
                                                 Diligent::float3{0.0f, 0.0f, 1.0f});
    return quaternionFromBasis(xAxis, yAxis, zAxis);
}

void computeMatrixQ(const Diligent::QuaternionF &q, float outQ[4][4]) noexcept
{
    outQ[0][0] = q.q.w;
    outQ[0][1] = -q.q.x;
    outQ[0][2] = -q.q.y;
    outQ[0][3] = -q.q.z;
    outQ[1][0] = q.q.x;
    outQ[1][1] = q.q.w;
    outQ[1][2] = -q.q.z;
    outQ[1][3] = q.q.y;
    outQ[2][0] = q.q.y;
    outQ[2][1] = q.q.z;
    outQ[2][2] = q.q.w;
    outQ[2][3] = -q.q.x;
    outQ[3][0] = q.q.z;
    outQ[3][1] = -q.q.y;
    outQ[3][2] = q.q.x;
    outQ[3][3] = q.q.w;
}

void computeMatrixQHat(const Diligent::QuaternionF &q, float outQ[4][4]) noexcept
{
    outQ[0][0] = q.q.w;
    outQ[0][1] = -q.q.x;
    outQ[0][2] = -q.q.y;
    outQ[0][3] = -q.q.z;
    outQ[1][0] = q.q.x;
    outQ[1][1] = q.q.w;
    outQ[1][2] = q.q.z;
    outQ[1][3] = -q.q.y;
    outQ[2][0] = q.q.y;
    outQ[2][1] = -q.q.z;
    outQ[2][2] = q.q.w;
    outQ[2][3] = q.q.x;
    outQ[3][0] = q.q.z;
    outQ[3][1] = q.q.y;
    outQ[3][2] = -q.q.x;
    outQ[3][3] = q.q.w;
}

void computeProjectionRowsFromLocalFrames(const Diligent::QuaternionF &localRotationA,
                                          const Diligent::QuaternionF &localRotationB,
                                          std::uint32_t rowStart, std::uint32_t rowCount,
                                          Diligent::float4 *rowsOut) noexcept
{
    const Diligent::QuaternionF q00 = quaternionConjugate(localRotationA);
    const Diligent::QuaternionF q10 = quaternionConjugate(localRotationB);

    float Q[4][4];
    float QHat[4][4];
    computeMatrixQ(q00, Q);
    computeMatrixQHat(q10, QHat);

    for (std::uint32_t row = 0u; row < rowCount; ++row)
    {
        const std::uint32_t srcRow = rowStart + row;
        Diligent::float4 outRow{};
        for (std::uint32_t col = 0u; col < 4u; ++col)
        {
            outRow[col] = QHat[0][srcRow] * Q[0][col] + QHat[1][srcRow] * Q[1][col] +
                          QHat[2][srcRow] * Q[2][col] + QHat[3][srcRow] * Q[3][col];
        }
        rowsOut[row] = outRow;
    }
}

} // namespace

namespace
{

void enqueueDirtyIndex(std::uint32_t index, std::vector<std::uint32_t> &dirtyIndices,
                       std::vector<std::uint8_t> &dirtyBits) noexcept
{
    if (index >= dirtyBits.size())
    {
        dirtyBits.resize(index + 1u, 0u);
    }
    if (dirtyBits[index] != 0u)
    {
        return;
    }

    dirtyBits[index] = 1u;
    dirtyIndices.push_back(index);
}

void clearDirtyIndices(std::vector<std::uint32_t> &dirtyIndices,
                       std::vector<std::uint8_t> &dirtyBits) noexcept
{
    for (const std::uint32_t index : dirtyIndices)
    {
        if (index < dirtyBits.size())
        {
            dirtyBits[index] = 0u;
        }
    }
    dirtyIndices.clear();
}

} // namespace

namespace
{

bool equalSoftBodyRegularGridSource(const SoftBodyRegularGridSource &lhs,
                                    const SoftBodyRegularGridSource &rhs) noexcept
{
    return lhs.size == rhs.size && lhs.targetParticleSpacing == rhs.targetParticleSpacing &&
           lhs.staticParticleIndices == rhs.staticParticleIndices;
}

bool equalSoftBodyTetMeshSource(const SoftBodyTetMeshSource &lhs,
                                const SoftBodyTetMeshSource &rhs) noexcept
{
    return lhs.objectSpaceRestPositions == rhs.objectSpaceRestPositions &&
           lhs.tetVertexIndices == rhs.tetVertexIndices &&
           lhs.staticParticleIndices == rhs.staticParticleIndices;
}

bool equalSoftBodyTetGenSource(const SoftBodyTetGenSource &lhs,
                               const SoftBodyTetGenSource &rhs) noexcept
{
    return lhs.nodeFile == rhs.nodeFile && lhs.eleFile == rhs.eleFile &&
           lhs.staticParticleIndices == rhs.staticParticleIndices;
}

bool equalSoftBodyMeshfreeParticleSource(const SoftBodyMeshfreeParticleSource &lhs,
                                         const SoftBodyMeshfreeParticleSource &rhs) noexcept
{
    return lhs.particleRestPositions == rhs.particleRestPositions &&
           lhs.surfaceRestPositions == rhs.surfaceRestPositions &&
           lhs.surfaceNormals == rhs.surfaceNormals &&
           lhs.surfaceTriangles == rhs.surfaceTriangles &&
           lhs.staticParticleIndices == rhs.staticParticleIndices &&
           lhs.neighbourCount == rhs.neighbourCount;
}

bool equalSoftBodySourceDesc(const SoftBodySourceDesc &lhs, const SoftBodySourceDesc &rhs) noexcept
{
    if (lhs.kind != rhs.kind)
    {
        return false;
    }

    switch (lhs.kind)
    {
    case SoftBodySourceKind::RegularGrid:
        return equalSoftBodyRegularGridSource(lhs.regularGrid, rhs.regularGrid);
    case SoftBodySourceKind::TetMesh:
        return equalSoftBodyTetMeshSource(lhs.tetMesh, rhs.tetMesh);
    case SoftBodySourceKind::TetGenFiles:
        return equalSoftBodyTetGenSource(lhs.tetGen, rhs.tetGen);
    case SoftBodySourceKind::MeshfreeParticles:
        return equalSoftBodyMeshfreeParticleSource(lhs.meshfreeParticles, rhs.meshfreeParticles);
    }

    return false;
}

bool equalFluidSourceDesc(const FluidSourceDesc &lhs, const FluidSourceDesc &rhs) noexcept
{
    if (lhs.kind != rhs.kind)
    {
        return false;
    }

    switch (lhs.kind)
    {
    case FluidSourceKind::RegularGrid:
        return lhs.regularGrid.size == rhs.regularGrid.size &&
               lhs.regularGrid.targetParticleSpacing == rhs.regularGrid.targetParticleSpacing;
    }

    return false;
}

} // namespace

void PhysicsWorld::clear()
{
    mRigidBodies.clear();
    mColliders.clear();
    mBodyColliderMapping.clear();
    mRigidJointScene.clear();
    mJointCollisionSuppression.clear();
    mEntityToRigidBodyIndex.clear();
    mRigidBodyIdToIndex.clear();
    mColliderIdToIndex.clear();
    mEntityToColliderIds.clear();
    mEntityToSoftBodyIndex.clear();
    mEntityToStrandIndex.clear();
    mEntityToFluidIndex.clear();
    mParticleSequenceIdToIndex.clear();
    mParticleConstraintIdToIndex.clear();
    mRigidParticleAttachmentConstraintIdToIndex.clear();
    mStrandRigidAttachmentConstraintIdToIndex.clear();
    mRigidDistanceConstraintIdToIndex.clear();
    mRoutedCableConstraintIdToIndex.clear();
    mParticleCollisionFilterIdToIndex.clear();
    mSuturingSequenceIdToIndex.clear();
    mTetGenMeshCache.clear();
    mRigidBodySnapshot.clear();
    mColliderSnapshot.clear();
    mSoftBodySnapshot.clear();
    mStrandSnapshot.clear();
    mFluidSnapshot.clear();
    mParticleSequenceSnapshot.clear();
    mParticleDistanceConstraintSnapshot.clear();
    mRigidParticleAttachmentConstraintSnapshot.clear();
    mStrandRigidAttachmentConstraintSnapshot.clear();
    mRigidDistanceConstraintSnapshot.clear();
    mRoutedCableConstraintSnapshot.clear();
    mParticleCollisionFilterSnapshot.clear();
    mSuturingSequenceSnapshot.clear();
    mBallJointSnapshot.clear();
    mHingeJointSnapshot.clear();
    mSliderJointSnapshot.clear();
    mSoftBodyDerivedCaches.clear();
    mStrandDerivedCaches.clear();
    mFluidDerivedCaches.clear();
    mSuturingPairs.clear();
    mReservedSuturingPathHeaders = 0u;
    mReservedSuturingPathNodes   = 0u;
    mParticles.clear();
    mParticleContactMaterials.clear();
    mFluidMaterials.clear();
    mSoftEdges.clear();
    mSoftBends.clear();
    mSoftTets.clear();
    mStrandSegments.clear();
    mStrandJoints.clear();
    mStrandDistanceConstraints.clear();
    mStrandSegmentStates.clear();
    mRigidParticleAttachments.clear();
    mStrandRigidAttachments.clear();
    mRigidDistanceConstraints.clear();
    mRoutedCableConstraints.clear();
    mRoutedCableRoutePoints.clear();
    mSoftRenderData.clear();
    mCurveRenderData.clear();
    mRigidBodyDirtyIndices.clear();
    mColliderDirtyIndices.clear();
    mRigidBodyDirtyBits.clear();
    mColliderDirtyBits.clear();
    mRigidBodyCountDirty            = true;
    mColliderCountDirty             = true;
    mFullRigidBodyUploadRequired    = true;
    mFullColliderUploadRequired     = true;
    mBodyColliderMappingDirty       = true;
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    mSoftBodyDerivedStateDirty      = true;
    mStaticBroadPhaseDirty          = true;
    mActiveMovingColliderCount      = 0u;
    mStaticColliderCount            = 0u;
    mParticleGridCellSize           = 0.1f;
    mSoftBodyBoundsChunkCount       = 0u;
    mNextRigidBodyId                = 1u;
    mNextColliderId                 = 1u;
    mNextBallJointId                = 1u;
    mNextHingeJointId               = 1u;
    mNextSliderJointId              = 1u;
    mNextParticleSequenceId         = 1u;
    mNextParticleConstraintId       = 1u;
    mNextRigidParticleAttachmentConstraintId = 1u;
    mNextStrandRigidAttachmentConstraintId = 1u;
    mNextRigidDistanceConstraintId  = 1u;
    mNextRoutedCableConstraintId    = 1u;
    mNextParticleCollisionFilterId  = 1u;
    mNextSuturingSequenceId         = 1u;
    ++mRigidBodyTopologyRevision;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mRigidJointTopologyRevision;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mRigidParticleAttachmentRevision;
    ++mRigidParticleAttachmentTopologyRevision;
    ++mStrandRigidAttachmentRevision;
    ++mStrandRigidAttachmentTopologyRevision;
    ++mRigidDistanceConstraintRevision;
    ++mRigidDistanceConstraintTopologyRevision;
    ++mRoutedCableRevision;
    ++mRoutedCableTopologyRevision;
    ++mAuthoredRevision;
    ++mSimulationRevision;
}

RigidBodyState &PhysicsWorld::upsertRigidBody(const RigidBodyState &state)
{
    RigidBodyState normalizedState = state;
    normalizeRigidBodyState(normalizedState);

    auto it = mEntityToRigidBodyIndex.find(normalizedState.entityId);
    if (it == mEntityToRigidBodyIndex.end())
    {
        normalizedState.rigidBodyId = mNextRigidBodyId++;

        const std::uint32_t index = static_cast<std::uint32_t>(mRigidBodies.size());
        mEntityToRigidBodyIndex.emplace(normalizedState.entityId, index);
        mRigidBodyIdToIndex.emplace(normalizedState.rigidBodyId, index);
        mRigidBodySnapshot.push_back(normalizedState);
        mRigidBodies.rigidBodyIds.push_back(normalizedState.rigidBodyId);
        mRigidBodies.entityIds.push_back(normalizedState.entityId);
        mRigidBodies.positionsInvMass.push_back(toPositionInvMass(normalizedState));
        mRigidBodies.orientations.push_back(toOrientation(normalizedState));
        mRigidBodies.scales.push_back(toScale(normalizedState));
        mRigidBodies.linearVelocities.push_back(toLinearVelocity(normalizedState));
        mRigidBodies.angularVelocities.push_back(toAngularVelocity(normalizedState));
        mRigidBodies.inverseInertiaLocal.push_back(toInverseInertiaLocal(normalizedState));
        mRigidBodies.bodyTypes.push_back(static_cast<std::uint32_t>(normalizedState.bodyType));
        mRigidBodies.kinematicTargetPositions.push_back(toKinematicTargetPosition(normalizedState));
        mRigidBodies.kinematicTargetOrientations.push_back(
            toKinematicTargetOrientation(normalizedState));
        mRigidBodies.kinematicTargetFlags.push_back(normalizedState.kinematicTargetEnabled ? 1u
                                                                                           : 0u);
        mRigidBodies.proxyParticleContactMaterials.push_back(
            normalizedState.proxyParticleContactMaterial);
        markRigidBodyDirty(index);
        markRigidBodyCountDirty();
        mBodyColliderMappingDirty       = true;
        mJointCollisionSuppressionDirty = true;
        mStaticBroadPhaseDirty          = mStaticBroadPhaseDirty || isStaticBody(normalizedState);
        if (!normalizedState.proxyParticleLocalPositions.empty())
        {
            mSoftBodyDerivedStateDirty = true;
            ++mSoftParticleRevision;
            ++mSoftGpuTopologyRevision;
        }
        ++mRigidBodyTopologyRevision;
        ++mAuthoredRevision;
        return mRigidBodySnapshot.back();
    }

    const std::uint32_t index          = it->second;
    normalizedState.rigidBodyId        = mRigidBodySnapshot[index].rigidBodyId;
    const RigidBodyState previousState = mRigidBodySnapshot[index];
    if (previousState.bodyType != normalizedState.bodyType)
    {
        const std::uint32_t enabledColliderCount =
            enabledColliderCountForEntity(previousState.entityId);
        if (isStaticBody(previousState))
        {
            mStaticColliderCount -= enabledColliderCount;
        }
        else if (isMovingBody(previousState))
        {
            mActiveMovingColliderCount -= enabledColliderCount;
        }

        if (isStaticBody(normalizedState))
        {
            mStaticColliderCount += enabledColliderCount;
        }
        else if (isMovingBody(normalizedState))
        {
            mActiveMovingColliderCount += enabledColliderCount;
        }
    }
    writeRigidBodySoAAt(mRigidBodies, index, normalizedState);
    mRigidBodySnapshot[index] = normalizedState;
    markRigidBodyDirty(index);
    const bool proxyMaterialChanged = previousState.proxyParticleMaterial.friction !=
                                          normalizedState.proxyParticleMaterial.friction ||
                                      previousState.proxyParticleMaterial.restitution !=
                                          normalizedState.proxyParticleMaterial.restitution ||
                                      previousState.proxyParticleMaterial.damping !=
                                          normalizedState.proxyParticleMaterial.damping ||
                                      previousState.proxyParticleMaterial.staticFriction !=
                                          normalizedState.proxyParticleMaterial.staticFriction;
    const bool proxyLayoutChanged =
        previousState.proxyParticleLocalPositions != normalizedState.proxyParticleLocalPositions ||
        previousState.proxyParticleRadius != normalizedState.proxyParticleRadius ||
        proxyMaterialChanged ||
        previousState.proxyCollisionLayer != normalizedState.proxyCollisionLayer ||
        previousState.proxyCollisionMask != normalizedState.proxyCollisionMask ||
        previousState.scale != normalizedState.scale ||
        previousState.environmentIndex != normalizedState.environmentIndex ||
        previousState.suturingEnabled != normalizedState.suturingEnabled ||
        previousState.needleTipProxyIndex != normalizedState.needleTipProxyIndex;
    if ((!previousState.proxyParticleLocalPositions.empty() ||
         !normalizedState.proxyParticleLocalPositions.empty()) &&
        proxyLayoutChanged)
    {
        mSoftBodyDerivedStateDirty = true;
        ++mSoftParticleRevision;
        ++mSoftGpuTopologyRevision;
    }
    if (previousState.environmentIndex != normalizedState.environmentIndex)
    {
        auto colliderHandlesIt = mEntityToColliderIds.find(previousState.entityId);
        if (colliderHandlesIt != mEntityToColliderIds.end())
        {
            for (const ColliderId colliderId : colliderHandlesIt->second)
            {
                const auto colliderIt = mColliderIdToIndex.find(colliderId);
                if (colliderIt != mColliderIdToIndex.end())
                {
                    const std::uint32_t colliderIndex = colliderIt->second;
                    if (colliderIndex < mColliders.environmentIndices.size())
                    {
                        mColliders.environmentIndices[colliderIndex] =
                            normalizedState.environmentIndex;
                        markColliderDirty(colliderIndex);
                    }
                }
            }
        }
        mRigidJointSceneDirty           = true;
        mJointCollisionSuppressionDirty = true;
        ++mRigidJointSceneRevision;
        ++mRigidJointModeRevision;
        ++mRigidJointTopologyRevision;
    }
    mStaticBroadPhaseDirty =
        mStaticBroadPhaseDirty || staticBodyPoseChanged(previousState, normalizedState);
    ++mAuthoredRevision;
    return mRigidBodySnapshot[index];
}

bool PhysicsWorld::removeRigidBody(common::EntityId entityId)
{
    const auto it = mEntityToRigidBodyIndex.find(entityId);
    if (it == mEntityToRigidBodyIndex.end())
    {
        return false;
    }

    removeCollidersForEntity(entityId);

    const std::uint32_t index       = it->second;
    const std::uint32_t last        = static_cast<std::uint32_t>(mRigidBodies.size() - 1u);
    const RigidBodyId removedBodyId = mRigidBodySnapshot[index].rigidBodyId;
    const bool removedStatic        = isStaticBody(mRigidBodySnapshot[index]);
    const bool removedProxyParticles =
        !mRigidBodySnapshot[index].proxyParticleLocalPositions.empty();
    bool movedProxyParticles = false;

    if (index != last)
    {
        movedProxyParticles       = !mRigidBodySnapshot[last].proxyParticleLocalPositions.empty();
        mRigidBodySnapshot[index] = mRigidBodySnapshot[last];
        mRigidBodies.rigidBodyIds[index]             = mRigidBodies.rigidBodyIds[last];
        mRigidBodies.entityIds[index]                = mRigidBodies.entityIds[last];
        mRigidBodies.positionsInvMass[index]         = mRigidBodies.positionsInvMass[last];
        mRigidBodies.orientations[index]             = mRigidBodies.orientations[last];
        mRigidBodies.scales[index]                   = mRigidBodies.scales[last];
        mRigidBodies.linearVelocities[index]         = mRigidBodies.linearVelocities[last];
        mRigidBodies.angularVelocities[index]        = mRigidBodies.angularVelocities[last];
        mRigidBodies.inverseInertiaLocal[index]      = mRigidBodies.inverseInertiaLocal[last];
        mRigidBodies.bodyTypes[index]                = mRigidBodies.bodyTypes[last];
        mRigidBodies.kinematicTargetPositions[index] = mRigidBodies.kinematicTargetPositions[last];
        mRigidBodies.kinematicTargetOrientations[index] =
            mRigidBodies.kinematicTargetOrientations[last];
        mRigidBodies.kinematicTargetFlags[index] = mRigidBodies.kinematicTargetFlags[last];
        mEntityToRigidBodyIndex[mRigidBodies.entityIds[index]] = index;
        mRigidBodyIdToIndex[mRigidBodies.rigidBodyIds[index]]  = index;
        markRigidBodyDirty(index);
    }

    mEntityToRigidBodyIndex.erase(it);
    mRigidBodyIdToIndex.erase(removedBodyId);
    mRigidBodySnapshot.pop_back();
    mRigidBodies.rigidBodyIds.pop_back();
    mRigidBodies.entityIds.pop_back();
    mRigidBodies.positionsInvMass.pop_back();
    mRigidBodies.orientations.pop_back();
    mRigidBodies.scales.pop_back();
    mRigidBodies.linearVelocities.pop_back();
    mRigidBodies.angularVelocities.pop_back();
    mRigidBodies.inverseInertiaLocal.pop_back();
    mRigidBodies.bodyTypes.pop_back();
    mRigidBodies.kinematicTargetPositions.pop_back();
    mRigidBodies.kinematicTargetOrientations.pop_back();
    mRigidBodies.kinematicTargetFlags.pop_back();
    mRigidBodies.proxyParticleContactMaterials.pop_back();

    markRigidBodyCountDirty();
    markColliderCountDirty(true);
    mBodyColliderMappingDirty = true;
    mStaticBroadPhaseDirty    = mStaticBroadPhaseDirty || removedStatic;
    if (removedProxyParticles || movedProxyParticles)
    {
        mSoftBodyDerivedStateDirty = true;
        ++mSoftParticleRevision;
        ++mSoftGpuTopologyRevision;
    }
    pruneRigidJointsForBody(removedBodyId);
    ++mRigidBodyTopologyRevision;
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mRigidJointTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

void PhysicsWorld::upsertCollider(const ColliderState &state)
{
    RigidBodyId ownerRigidBodyId = state.ownerRigidBodyId;
    std::uint32_t ownerBodyIndex = 0xffffffffu;

    if (ownerRigidBodyId != kInvalidRigidBodyId)
    {
        const auto bodyIt = mRigidBodyIdToIndex.find(ownerRigidBodyId);
        if (bodyIt != mRigidBodyIdToIndex.end())
        {
            ownerBodyIndex = bodyIt->second;
        }
    }

    if (ownerBodyIndex == 0xffffffffu)
    {
        const auto bodyIt = mEntityToRigidBodyIndex.find(state.entityId);
        if (bodyIt == mEntityToRigidBodyIndex.end())
        {
            removeCollider(state.colliderId);
            return;
        }
        ownerBodyIndex   = bodyIt->second;
        ownerRigidBodyId = mRigidBodySnapshot[ownerBodyIndex].rigidBodyId;
    }

    ColliderState normalizedState = state;
    normalizeColliderState(normalizedState);
    normalizedState.entityId         = mRigidBodySnapshot[ownerBodyIndex].entityId;
    normalizedState.ownerRigidBodyId = ownerRigidBodyId;
    if (normalizedState.colliderId == kInvalidColliderId)
    {
        normalizedState.colliderId = mNextColliderId++;
    }

    const bool ownerIsStatic = isStaticBody(mRigidBodySnapshot[ownerBodyIndex]);
    const auto colliderIt    = mColliderIdToIndex.find(normalizedState.colliderId);
    if (colliderIt == mColliderIdToIndex.end())
    {
        const std::uint32_t colliderIndex = static_cast<std::uint32_t>(mColliders.size());
        mColliderSnapshot.push_back(normalizedState);
        writeColliderSoAAt(mColliders, colliderIndex, normalizedState, ownerBodyIndex,
                           mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
        mColliderIdToIndex.emplace(normalizedState.colliderId, colliderIndex);
        auto &entityColliderIds = mEntityToColliderIds[normalizedState.entityId];
        entityColliderIds.push_back(normalizedState.colliderId);
        const std::uint32_t contribution =
            colliderBroadPhaseContribution(normalizedState, &mRigidBodySnapshot[ownerBodyIndex]);
        if (contribution == kBroadPhaseContributionStatic)
        {
            ++mStaticColliderCount;
        }
        else if (contribution == kBroadPhaseContributionMoving)
        {
            ++mActiveMovingColliderCount;
        }
        markColliderDirty(colliderIndex);
        markColliderCountDirty();
        mBodyColliderMappingDirty = true;
        mStaticBroadPhaseDirty    = mStaticBroadPhaseDirty || ownerIsStatic;
        ++mAuthoredRevision;
        return;
    }

    const std::uint32_t colliderIndex        = colliderIt->second;
    const ColliderState previousState        = mColliderSnapshot[colliderIndex];
    const std::uint32_t previousContribution = broadPhaseContributionForCollider(previousState);
    const std::uint32_t newContribution =
        colliderBroadPhaseContribution(normalizedState, &mRigidBodySnapshot[ownerBodyIndex]);
    const bool staticContributionChanged = previousContribution == kBroadPhaseContributionStatic ||
                                           newContribution == kBroadPhaseContributionStatic;
    if (previousContribution != newContribution)
    {
        if (previousContribution == kBroadPhaseContributionStatic)
        {
            --mStaticColliderCount;
        }
        else if (previousContribution == kBroadPhaseContributionMoving)
        {
            --mActiveMovingColliderCount;
        }

        if (newContribution == kBroadPhaseContributionStatic)
        {
            ++mStaticColliderCount;
        }
        else if (newContribution == kBroadPhaseContributionMoving)
        {
            ++mActiveMovingColliderCount;
        }
    }
    writeColliderSoAAt(mColliders, colliderIndex, normalizedState, ownerBodyIndex,
                       mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
    mColliderSnapshot[colliderIndex] = normalizedState;
    markColliderDirty(colliderIndex);
    if (previousState.ownerRigidBodyId != normalizedState.ownerRigidBodyId)
    {
        mBodyColliderMappingDirty = true;
    }
    const bool staticColliderShapeChanged =
        previousState.shapeType != normalizedState.shapeType ||
        previousState.shapeParams.x != normalizedState.shapeParams.x ||
        previousState.shapeParams.y != normalizedState.shapeParams.y ||
        previousState.shapeParams.z != normalizedState.shapeParams.z ||
        previousState.shapeParams.w != normalizedState.shapeParams.w ||
        previousState.localPosition.x != normalizedState.localPosition.x ||
        previousState.localPosition.y != normalizedState.localPosition.y ||
        previousState.localPosition.z != normalizedState.localPosition.z ||
        previousState.localRotation.q.x != normalizedState.localRotation.q.x ||
        previousState.localRotation.q.y != normalizedState.localRotation.q.y ||
        previousState.localRotation.q.z != normalizedState.localRotation.q.z ||
        previousState.localRotation.q.w != normalizedState.localRotation.q.w ||
        previousState.enabled != normalizedState.enabled;
    if (staticContributionChanged || ((previousContribution == kBroadPhaseContributionStatic ||
                                       newContribution == kBroadPhaseContributionStatic) &&
                                      staticColliderShapeChanged))
    {
        mStaticBroadPhaseDirty = true;
    }
    ++mAuthoredRevision;
}

bool PhysicsWorld::removeCollider(ColliderId colliderId)
{
    const auto it = mColliderIdToIndex.find(colliderId);
    if (it == mColliderIdToIndex.end())
    {
        return false;
    }

    bool removedStaticOwner            = false;
    const std::uint32_t ownerBodyIndex = mColliders.ownerRigidBodyIndices[it->second];
    if (ownerBodyIndex != 0xffffffffu && ownerBodyIndex < rigidBodyCount())
    {
        removedStaticOwner = isStaticBody(mRigidBodySnapshot[ownerBodyIndex]);
    }
    removeColliderAtIndex(it->second);
    markColliderCountDirty();
    mBodyColliderMappingDirty = true;
    mStaticBroadPhaseDirty    = mStaticBroadPhaseDirty || removedStaticOwner;
    ++mAuthoredRevision;
    return true;
}

void PhysicsWorld::replaceColliders(common::EntityId entityId,
                                    const std::vector<ColliderState> &colliders)
{
    removeCollidersForEntity(entityId);

    const auto bodyIt = mEntityToRigidBodyIndex.find(entityId);
    if (bodyIt == mEntityToRigidBodyIndex.end() || colliders.empty())
    {
        markColliderCountDirty(true);
        mBodyColliderMappingDirty = true;
        ++mAuthoredRevision;
        return;
    }

    const std::uint32_t ownerBodyIndex = bodyIt->second;
    const RigidBodyId ownerRigidBodyId = mRigidBodySnapshot[ownerBodyIndex].rigidBodyId;

    auto &entityColliderIds = mEntityToColliderIds[entityId];
    entityColliderIds.reserve(colliders.size());

    for (ColliderState collider : colliders)
    {
        normalizeColliderState(collider);
        collider.colliderId       = mNextColliderId++;
        collider.entityId         = entityId;
        collider.ownerRigidBodyId = ownerRigidBodyId;

        const std::uint32_t colliderIndex = static_cast<std::uint32_t>(mColliders.size());
        mColliderSnapshot.push_back(collider);
        writeColliderSoAAt(mColliders, colliderIndex, collider, ownerBodyIndex,
                           mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
        mColliderIdToIndex.emplace(collider.colliderId, colliderIndex);
        entityColliderIds.push_back(collider.colliderId);
        const std::uint32_t contribution =
            colliderBroadPhaseContribution(collider, &mRigidBodySnapshot[ownerBodyIndex]);
        if (contribution == kBroadPhaseContributionStatic)
        {
            ++mStaticColliderCount;
        }
        else if (contribution == kBroadPhaseContributionMoving)
        {
            ++mActiveMovingColliderCount;
        }
    }

    markColliderCountDirty(true);
    mBodyColliderMappingDirty = true;
    mStaticBroadPhaseDirty    = true;
    ++mAuthoredRevision;
}

bool PhysicsWorld::upsertSoftBody(const SoftBodyState &state)
{
    SoftBodyState normalizedState = state;
    normalizeSoftBodyState(normalizedState);

    const auto it                      = mEntityToSoftBodyIndex.find(normalizedState.entityId);
    const SoftBodyState *previousState = nullptr;
    if (it != mEntityToSoftBodyIndex.end())
    {
        previousState = &mSoftBodySnapshot[it->second];
    }

    if (previousState != nullptr && classifySoftBodyChange(*previousState, normalizedState) ==
                                        SoftBodyChangeKind::RuntimePropertiesOnly)
    {
        applySoftBodyRuntimeProperties(it->second, normalizedState);
        ++mSoftParticleRevision;
        ++mAuthoredRevision;
        return true;
    }

    SoftBodyDerivedCache derivedCache;
    if (!prepareSoftBodyStateForInsert(normalizedState, previousState, derivedCache))
    {
        return false;
    }

    if (it == mEntityToSoftBodyIndex.end())
    {
        mEntityToSoftBodyIndex.emplace(normalizedState.entityId,
                                       static_cast<std::uint32_t>(mSoftBodySnapshot.size()));
        mSoftBodySnapshot.push_back(normalizedState);
        mSoftBodyDerivedCaches.push_back(std::move(derivedCache));
    }
    else
    {
        mSoftBodySnapshot[it->second]      = normalizedState;
        mSoftBodyDerivedCaches[it->second] = std::move(derivedCache);
    }

    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::upsertStrand(const StrandState &state)
{
    StrandState normalizedState = state;
    normalizeStrandState(normalizedState);

    const auto it = mEntityToStrandIndex.find(normalizedState.entityId);
    const StrandState *previousState =
        it != mEntityToStrandIndex.end() ? &mStrandSnapshot[it->second] : nullptr;
    if (previousState != nullptr && classifyStrandChange(*previousState, normalizedState) ==
                                        StrandChangeKind::RuntimePropertiesOnly)
    {
        applyStrandRuntimeProperties(it->second, normalizedState);
        ++mSoftParticleRevision;
        ++mAuthoredRevision;
        return true;
    }

    StrandDerivedCache derivedCache;
    if (!prepareStrandStateForInsert(normalizedState, derivedCache))
    {
        return false;
    }

    if (it == mEntityToStrandIndex.end())
    {
        mEntityToStrandIndex.emplace(normalizedState.entityId,
                                     static_cast<std::uint32_t>(mStrandSnapshot.size()));
        mStrandSnapshot.push_back(normalizedState);
        mStrandDerivedCaches.push_back(std::move(derivedCache));
    }
    else
    {
        mStrandSnapshot[it->second]      = normalizedState;
        mStrandDerivedCaches[it->second] = std::move(derivedCache);
    }

    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::upsertFluid(const FluidState &state)
{
    FluidState normalizedState = state;
    normalizeFluidState(normalizedState);

    const auto it = mEntityToFluidIndex.find(normalizedState.entityId);
    const FluidState *previousState =
        it != mEntityToFluidIndex.end() ? &mFluidSnapshot[it->second] : nullptr;
    if (!validateFluidMaterialCompatibility(normalizedState, previousState))
    {
        return false;
    }

    FluidDerivedCache derivedCache;
    if (!prepareFluidStateForInsert(normalizedState, derivedCache))
    {
        return false;
    }

    if (it == mEntityToFluidIndex.end())
    {
        mEntityToFluidIndex.emplace(normalizedState.entityId,
                                    static_cast<std::uint32_t>(mFluidSnapshot.size()));
        mFluidSnapshot.push_back(normalizedState);
        mFluidDerivedCaches.push_back(std::move(derivedCache));
    }
    else
    {
        mFluidSnapshot[it->second]      = normalizedState;
        mFluidDerivedCaches[it->second] = std::move(derivedCache);
    }
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

AuthoredParticleDistanceConstraintState &PhysicsWorld::upsertParticleDistanceConstraint(
    const AuthoredParticleDistanceConstraintState &state)
{
    AuthoredParticleDistanceConstraintState normalizedState = state;
    normalizedState.restLength = std::max(normalizedState.restLength, 0.0f);
    normalizedState.compliance = std::max(normalizedState.compliance, 0.0f);

    auto it = mParticleConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidParticleConstraintId ||
        it == mParticleConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mNextParticleConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mParticleDistanceConstraintSnapshot.size());
        mParticleConstraintIdToIndex.emplace(normalizedState.constraintId, index);
        mParticleDistanceConstraintSnapshot.push_back(normalizedState);
        mSoftBodyDerivedStateDirty = true;
        ++mSoftBodyTopologyRevision;
        ++mSoftGpuTopologyRevision;
        ++mAuthoredRevision;
        return mParticleDistanceConstraintSnapshot.back();
    }

    mParticleDistanceConstraintSnapshot[it->second] = normalizedState;
    mSoftBodyDerivedStateDirty                      = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return mParticleDistanceConstraintSnapshot[it->second];
}

AuthoredRigidParticleAttachmentConstraintState &PhysicsWorld::upsertRigidParticleAttachmentConstraint(
    const AuthoredRigidParticleAttachmentConstraintState &state)
{
    AuthoredRigidParticleAttachmentConstraintState normalizedState = state;
    normalizedState.compliance = std::max(normalizedState.compliance, 0.0f);

    auto it = mRigidParticleAttachmentConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidRigidParticleAttachmentConstraintId ||
        it == mRigidParticleAttachmentConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mNextRigidParticleAttachmentConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mRigidParticleAttachmentConstraintSnapshot.size());
        mRigidParticleAttachmentConstraintIdToIndex.emplace(normalizedState.constraintId, index);
        mRigidParticleAttachmentConstraintSnapshot.push_back(normalizedState);
        mSoftBodyDerivedStateDirty = true;
        ++mRigidParticleAttachmentRevision;
        ++mRigidParticleAttachmentTopologyRevision;
        ++mAuthoredRevision;
        return mRigidParticleAttachmentConstraintSnapshot.back();
    }

    const AuthoredRigidParticleAttachmentConstraintState &previous =
        mRigidParticleAttachmentConstraintSnapshot[it->second];
    const bool topologyChanged =
        previous.particle.entityId != normalizedState.particle.entityId ||
        previous.particle.type != normalizedState.particle.type ||
        previous.particle.localParticleIndex != normalizedState.particle.localParticleIndex ||
        previous.rigidBodyEntityId != normalizedState.rigidBodyEntityId;
    mRigidParticleAttachmentConstraintSnapshot[it->second] = normalizedState;
    mSoftBodyDerivedStateDirty                             = true;
    ++mRigidParticleAttachmentRevision;
    if (topologyChanged)
    {
        ++mRigidParticleAttachmentTopologyRevision;
    }
    ++mAuthoredRevision;
    return mRigidParticleAttachmentConstraintSnapshot[it->second];
}

AuthoredStrandRigidAttachmentConstraintState &PhysicsWorld::upsertStrandRigidAttachmentConstraint(
    const AuthoredStrandRigidAttachmentConstraintState &state)
{
    AuthoredStrandRigidAttachmentConstraintState normalizedState = state;
    normalizedState.segmentT = std::clamp(normalizedState.segmentT, 0.0f, 1.0f);
    normalizedState.localRotation =
        common::runtime_math::normalizeQuaternion(normalizedState.localRotation);
    normalizedState.translationCompliance = std::max(normalizedState.translationCompliance, 0.0f);
    normalizedState.rotationCompliance    = std::max(normalizedState.rotationCompliance, 0.0f);

    auto it = mStrandRigidAttachmentConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidStrandRigidAttachmentConstraintId ||
        it == mStrandRigidAttachmentConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mNextStrandRigidAttachmentConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mStrandRigidAttachmentConstraintSnapshot.size());
        mStrandRigidAttachmentConstraintIdToIndex.emplace(normalizedState.constraintId, index);
        mStrandRigidAttachmentConstraintSnapshot.push_back(normalizedState);
        mSoftBodyDerivedStateDirty = true;
        ++mSoftGpuTopologyRevision;
        ++mStrandRigidAttachmentRevision;
        ++mStrandRigidAttachmentTopologyRevision;
        ++mAuthoredRevision;
        return mStrandRigidAttachmentConstraintSnapshot.back();
    }

    const AuthoredStrandRigidAttachmentConstraintState &previous =
        mStrandRigidAttachmentConstraintSnapshot[it->second];
    const bool topologyChanged = previous.strandEntityId != normalizedState.strandEntityId ||
                                 previous.localSegmentIndex != normalizedState.localSegmentIndex ||
                                 previous.rigidBodyEntityId != normalizedState.rigidBodyEntityId;
    mStrandRigidAttachmentConstraintSnapshot[it->second] = normalizedState;
    mSoftBodyDerivedStateDirty                           = true;
    ++mSoftGpuTopologyRevision;
    ++mStrandRigidAttachmentRevision;
    if (topologyChanged)
    {
        ++mStrandRigidAttachmentTopologyRevision;
    }
    ++mAuthoredRevision;
    return mStrandRigidAttachmentConstraintSnapshot[it->second];
}

AuthoredRigidDistanceConstraintState &PhysicsWorld::upsertRigidDistanceConstraint(
    const AuthoredRigidDistanceConstraintState &state)
{
    AuthoredRigidDistanceConstraintState normalizedState = state;
    normalizedState.restDistance = std::max(normalizedState.restDistance, 0.0f);
    normalizedState.compliance   = std::max(normalizedState.compliance, 0.0f);

    auto it = mRigidDistanceConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidRigidDistanceConstraintId ||
        it == mRigidDistanceConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mNextRigidDistanceConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mRigidDistanceConstraintSnapshot.size());
        mRigidDistanceConstraintIdToIndex.emplace(normalizedState.constraintId, index);
        mRigidDistanceConstraintSnapshot.push_back(normalizedState);
        mSoftBodyDerivedStateDirty = true;
        ++mRigidDistanceConstraintRevision;
        ++mRigidDistanceConstraintTopologyRevision;
        ++mAuthoredRevision;
        return mRigidDistanceConstraintSnapshot.back();
    }

    const AuthoredRigidDistanceConstraintState &previous =
        mRigidDistanceConstraintSnapshot[it->second];
    const bool topologyChanged = previous.entityA != normalizedState.entityA ||
                                 previous.entityB != normalizedState.entityB ||
                                 previous.localAnchorA != normalizedState.localAnchorA ||
                                 previous.localAnchorB != normalizedState.localAnchorB;
    mRigidDistanceConstraintSnapshot[it->second] = normalizedState;
    mSoftBodyDerivedStateDirty                   = true;
    ++mRigidDistanceConstraintRevision;
    if (topologyChanged)
    {
        ++mRigidDistanceConstraintTopologyRevision;
    }
    ++mAuthoredRevision;
    return mRigidDistanceConstraintSnapshot[it->second];
}

AuthoredRoutedCableConstraintState &PhysicsWorld::upsertRoutedCableConstraint(
    const AuthoredRoutedCableConstraintState &state)
{
    AuthoredRoutedCableConstraintState normalizedState = state;
    normalizedState.targetLength = std::max(normalizedState.targetLength, 0.0f);
    normalizedState.compliance   = std::max(normalizedState.compliance, 0.0f);

    auto it = mRoutedCableConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidRoutedCableConstraintId ||
        it == mRoutedCableConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mNextRoutedCableConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mRoutedCableConstraintSnapshot.size());
        mRoutedCableConstraintIdToIndex.emplace(normalizedState.constraintId, index);
        mRoutedCableConstraintSnapshot.push_back(normalizedState);
        mSoftBodyDerivedStateDirty = true;
        ++mRoutedCableRevision;
        ++mRoutedCableTopologyRevision;
        ++mAuthoredRevision;
        return mRoutedCableConstraintSnapshot.back();
    }

    const bool topologyChanged =
        mRoutedCableConstraintSnapshot[it->second].routePoints != normalizedState.routePoints;
    mRoutedCableConstraintSnapshot[it->second] = normalizedState;
    mSoftBodyDerivedStateDirty                 = true;
    ++mRoutedCableRevision;
    if (topologyChanged)
    {
        ++mRoutedCableTopologyRevision;
    }
    ++mAuthoredRevision;
    return mRoutedCableConstraintSnapshot[it->second];
}

AuthoredParticleCollisionFilterState &PhysicsWorld::upsertParticleCollisionFilter(
    const AuthoredParticleCollisionFilterState &state)
{
    AuthoredParticleCollisionFilterState normalizedState = state;
    if (normalizedState.collisionLayer == 0u)
    {
        normalizedState.collisionLayer = 1u;
    }

    auto it = mParticleCollisionFilterIdToIndex.find(normalizedState.filterId);
    if (normalizedState.filterId == kInvalidParticleCollisionFilterId ||
        it == mParticleCollisionFilterIdToIndex.end())
    {
        normalizedState.filterId = mNextParticleCollisionFilterId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mParticleCollisionFilterSnapshot.size());
        mParticleCollisionFilterIdToIndex.emplace(normalizedState.filterId, index);
        mParticleCollisionFilterSnapshot.push_back(normalizedState);
        mSoftBodyDerivedStateDirty = true;
        ++mSoftParticleRevision;
        ++mAuthoredRevision;
        return mParticleCollisionFilterSnapshot.back();
    }

    mParticleCollisionFilterSnapshot[it->second] = normalizedState;
    mSoftBodyDerivedStateDirty                   = true;
    ++mSoftParticleRevision;
    ++mAuthoredRevision;
    return mParticleCollisionFilterSnapshot[it->second];
}

AuthoredSuturingSequenceState &PhysicsWorld::upsertSuturingSequence(
    const AuthoredSuturingSequenceState &state)
{
    AuthoredSuturingSequenceState normalizedState = state;
    normalizedState.pathNodeSpacing               = std::max(normalizedState.pathNodeSpacing, 0.0f);
    normalizedState.needleTangentialDrag = std::max(normalizedState.needleTangentialDrag, 0.0f);
    normalizedState.threadTangentialDrag = std::max(normalizedState.threadTangentialDrag, 0.0f);
    if (normalizedState.entries.empty())
    {
        normalizedState.tipEntryIndex = 0u;
    }
    else
    {
        normalizedState.tipEntryIndex = std::min<std::uint32_t>(
            normalizedState.tipEntryIndex,
            static_cast<std::uint32_t>(normalizedState.entries.size() - 1u));
    }

    auto it = mSuturingSequenceIdToIndex.find(normalizedState.sequenceId);
    if (normalizedState.sequenceId == kInvalidSuturingSequenceId ||
        it == mSuturingSequenceIdToIndex.end())
    {
        normalizedState.sequenceId = mNextSuturingSequenceId++;
        const std::uint32_t index  = static_cast<std::uint32_t>(mSuturingSequenceSnapshot.size());
        mSuturingSequenceIdToIndex.emplace(normalizedState.sequenceId, index);
        mSuturingSequenceSnapshot.push_back(normalizedState);
        mSoftBodyDerivedStateDirty = true;
        ++mSoftBodyTopologyRevision;
        ++mSoftGpuTopologyRevision;
        ++mAuthoredRevision;
        return mSuturingSequenceSnapshot.back();
    }

    mSuturingSequenceSnapshot[it->second] = normalizedState;
    mSoftBodyDerivedStateDirty            = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return mSuturingSequenceSnapshot[it->second];
}

AuthoredParticleSequenceState &PhysicsWorld::upsertParticleSequence(
    const AuthoredParticleSequenceState &state)
{
    AuthoredParticleSequenceState normalizedState = state;
    auto it = mParticleSequenceIdToIndex.find(normalizedState.sequenceId);
    if (normalizedState.sequenceId == kInvalidParticleSequenceId ||
        it == mParticleSequenceIdToIndex.end())
    {
        normalizedState.sequenceId = mNextParticleSequenceId++;
        const std::uint32_t index  = static_cast<std::uint32_t>(mParticleSequenceSnapshot.size());
        mParticleSequenceIdToIndex.emplace(normalizedState.sequenceId, index);
        mParticleSequenceSnapshot.push_back(normalizedState);
        ++mAuthoredRevision;
        return mParticleSequenceSnapshot.back();
    }

    mParticleSequenceSnapshot[it->second] = normalizedState;
    ++mAuthoredRevision;
    return mParticleSequenceSnapshot[it->second];
}

bool PhysicsWorld::removeSoftBody(common::EntityId entityId)
{
    const auto it = mEntityToSoftBodyIndex.find(entityId);
    if (it == mEntityToSoftBodyIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mSoftBodySnapshot.size() - 1u);
    if (index != last)
    {
        mSoftBodySnapshot[index]      = mSoftBodySnapshot[last];
        mSoftBodyDerivedCaches[index] = std::move(mSoftBodyDerivedCaches[last]);
        mEntityToSoftBodyIndex[mSoftBodySnapshot[index].entityId] = index;
    }
    mSoftBodySnapshot.pop_back();
    mSoftBodyDerivedCaches.pop_back();
    mEntityToSoftBodyIndex.erase(it);
    mTetGenMeshCache.erase(entityId);
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeStrand(common::EntityId entityId)
{
    const auto it = mEntityToStrandIndex.find(entityId);
    if (it == mEntityToStrandIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mStrandSnapshot.size() - 1u);
    if (index != last)
    {
        mStrandSnapshot[index]      = mStrandSnapshot[last];
        mStrandDerivedCaches[index] = std::move(mStrandDerivedCaches[last]);
        mEntityToStrandIndex[mStrandSnapshot[index].entityId] = index;
    }
    mStrandSnapshot.pop_back();
    mStrandDerivedCaches.pop_back();
    mEntityToStrandIndex.erase(it);
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeFluid(common::EntityId entityId)
{
    const auto it = mEntityToFluidIndex.find(entityId);
    if (it == mEntityToFluidIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mFluidSnapshot.size() - 1u);
    if (index != last)
    {
        mFluidSnapshot[index]                               = mFluidSnapshot[last];
        mFluidDerivedCaches[index]                          = std::move(mFluidDerivedCaches[last]);
        mEntityToFluidIndex[mFluidSnapshot[index].entityId] = index;
    }
    mFluidSnapshot.pop_back();
    mFluidDerivedCaches.pop_back();
    mEntityToFluidIndex.erase(it);
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftParticleRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeParticleDistanceConstraint(ParticleConstraintId constraintId)
{
    const auto it = mParticleConstraintIdToIndex.find(constraintId);
    if (it == mParticleConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mParticleDistanceConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mParticleDistanceConstraintSnapshot[index] = mParticleDistanceConstraintSnapshot[last];
        mParticleConstraintIdToIndex[mParticleDistanceConstraintSnapshot[index].constraintId] =
            index;
    }

    mParticleConstraintIdToIndex.erase(it);
    mParticleDistanceConstraintSnapshot.pop_back();
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeRigidParticleAttachmentConstraint(
    RigidParticleAttachmentConstraintId constraintId)
{
    const auto it = mRigidParticleAttachmentConstraintIdToIndex.find(constraintId);
    if (it == mRigidParticleAttachmentConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mRigidParticleAttachmentConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mRigidParticleAttachmentConstraintSnapshot[index] =
            mRigidParticleAttachmentConstraintSnapshot[last];
        mRigidParticleAttachmentConstraintIdToIndex
            [mRigidParticleAttachmentConstraintSnapshot[index].constraintId] = index;
    }

    mRigidParticleAttachmentConstraintIdToIndex.erase(it);
    mRigidParticleAttachmentConstraintSnapshot.pop_back();
    mSoftBodyDerivedStateDirty = true;
    ++mRigidParticleAttachmentRevision;
    ++mRigidParticleAttachmentTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeStrandRigidAttachmentConstraint(
    StrandRigidAttachmentConstraintId constraintId)
{
    const auto it = mStrandRigidAttachmentConstraintIdToIndex.find(constraintId);
    if (it == mStrandRigidAttachmentConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mStrandRigidAttachmentConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mStrandRigidAttachmentConstraintSnapshot[index] =
            mStrandRigidAttachmentConstraintSnapshot[last];
        mStrandRigidAttachmentConstraintIdToIndex
            [mStrandRigidAttachmentConstraintSnapshot[index].constraintId] = index;
    }

    mStrandRigidAttachmentConstraintIdToIndex.erase(it);
    mStrandRigidAttachmentConstraintSnapshot.pop_back();
    mSoftBodyDerivedStateDirty = true;
    ++mSoftGpuTopologyRevision;
    ++mStrandRigidAttachmentRevision;
    ++mStrandRigidAttachmentTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeRigidDistanceConstraint(RigidDistanceConstraintId constraintId)
{
    const auto it = mRigidDistanceConstraintIdToIndex.find(constraintId);
    if (it == mRigidDistanceConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mRigidDistanceConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mRigidDistanceConstraintSnapshot[index] = mRigidDistanceConstraintSnapshot[last];
        mRigidDistanceConstraintIdToIndex[mRigidDistanceConstraintSnapshot[index].constraintId] =
            index;
    }

    mRigidDistanceConstraintIdToIndex.erase(it);
    mRigidDistanceConstraintSnapshot.pop_back();
    mSoftBodyDerivedStateDirty = true;
    ++mRigidDistanceConstraintRevision;
    ++mRigidDistanceConstraintTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeRoutedCableConstraint(RoutedCableConstraintId constraintId)
{
    const auto it = mRoutedCableConstraintIdToIndex.find(constraintId);
    if (it == mRoutedCableConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mRoutedCableConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mRoutedCableConstraintSnapshot[index] = mRoutedCableConstraintSnapshot[last];
        mRoutedCableConstraintIdToIndex[mRoutedCableConstraintSnapshot[index].constraintId] = index;
    }

    mRoutedCableConstraintIdToIndex.erase(it);
    mRoutedCableConstraintSnapshot.pop_back();
    mSoftBodyDerivedStateDirty = true;
    ++mRoutedCableRevision;
    ++mRoutedCableTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeParticleSequence(ParticleSequenceId sequenceId)
{
    const auto it = mParticleSequenceIdToIndex.find(sequenceId);
    if (it == mParticleSequenceIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mParticleSequenceSnapshot.size() - 1u);
    if (index != last)
    {
        mParticleSequenceSnapshot[index] = mParticleSequenceSnapshot[last];
        mParticleSequenceIdToIndex[mParticleSequenceSnapshot[index].sequenceId] = index;
    }

    mParticleSequenceIdToIndex.erase(it);
    mParticleSequenceSnapshot.pop_back();
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeParticleCollisionFilter(ParticleCollisionFilterId filterId)
{
    const auto it = mParticleCollisionFilterIdToIndex.find(filterId);
    if (it == mParticleCollisionFilterIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mParticleCollisionFilterSnapshot.size() - 1u);
    if (index != last)
    {
        mParticleCollisionFilterSnapshot[index] = mParticleCollisionFilterSnapshot[last];
        mParticleCollisionFilterIdToIndex[mParticleCollisionFilterSnapshot[index].filterId] = index;
    }

    mParticleCollisionFilterIdToIndex.erase(it);
    mParticleCollisionFilterSnapshot.pop_back();
    mSoftBodyDerivedStateDirty = true;
    ++mSoftParticleRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeSuturingSequence(SuturingSequenceId sequenceId)
{
    const auto it = mSuturingSequenceIdToIndex.find(sequenceId);
    if (it == mSuturingSequenceIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mSuturingSequenceSnapshot.size() - 1u);
    if (index != last)
    {
        mSuturingSequenceSnapshot[index] = mSuturingSequenceSnapshot[last];
        mSuturingSequenceIdToIndex[mSuturingSequenceSnapshot[index].sequenceId] = index;
    }

    mSuturingSequenceIdToIndex.erase(it);
    mSuturingSequenceSnapshot.pop_back();
    mSoftBodyDerivedStateDirty = true;
    ++mSoftBodyTopologyRevision;
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
    return true;
}

bool PhysicsWorld::upsertBallJoint(const BallJointState &state)
{
    BallJointState normalized = state;
    if (normalized.bodyA == kInvalidRigidBodyId || normalized.bodyB == kInvalidRigidBodyId ||
        normalized.bodyA == normalized.bodyB)
    {
        return false;
    }
    if (normalized.jointId == kInvalidBallJointId)
    {
        normalized.jointId = mNextBallJointId++;
    }
    const auto bodyAIt = mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mRigidBodyIdToIndex.end() || bodyBIt == mRigidBodyIdToIndex.end())
    {
        return false;
    }
    if (mRigidBodySnapshot[bodyAIt->second].environmentIndex !=
        mRigidBodySnapshot[bodyBIt->second].environmentIndex)
    {
        return false;
    }

    auto it             = std::find_if(mBallJointSnapshot.begin(), mBallJointSnapshot.end(),
                                       [&](const BallJointState &existing)
                                       { return existing.jointId == normalized.jointId; });
    const bool inserted = it == mBallJointSnapshot.end();
    if (it == mBallJointSnapshot.end())
    {
        mBallJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    applyRigidJointChange(classifyBallJointChange(inserted));
    return true;
}

bool PhysicsWorld::upsertHingeJoint(const HingeJointState &state)
{
    HingeJointState normalized = state;
    if (normalized.bodyA == kInvalidRigidBodyId || normalized.bodyB == kInvalidRigidBodyId ||
        normalized.bodyA == normalized.bodyB)
    {
        return false;
    }
    normalized.localRotationA =
        common::runtime_math::normalizeQuaternion(normalized.localRotationA);
    normalized.localRotationB =
        common::runtime_math::normalizeQuaternion(normalized.localRotationB);
    normalized.driveTargetAngle =
        std::remainder(normalized.driveTargetAngle, 2.0f * Diligent::PI_F);
    if (!normalized.limitEnabled)
    {
        normalized.limitMin = 0.0f;
        normalized.limitMax = 0.0f;
    }
    normalized.constraintCompliance = std::max(normalized.constraintCompliance, 0.0f);
    normalized.driveCompliance      = std::max(normalized.driveCompliance, 0.0f);
    if (normalized.jointId == kInvalidHingeJointId)
    {
        normalized.jointId = mNextHingeJointId++;
    }
    const auto bodyAIt = mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mRigidBodyIdToIndex.end() || bodyBIt == mRigidBodyIdToIndex.end())
    {
        return false;
    }
    if (mRigidBodySnapshot[bodyAIt->second].environmentIndex !=
        mRigidBodySnapshot[bodyBIt->second].environmentIndex)
    {
        return false;
    }

    auto it             = std::find_if(mHingeJointSnapshot.begin(), mHingeJointSnapshot.end(),
                                       [&](const HingeJointState &existing)
                                       { return existing.jointId == normalized.jointId; });
    const bool inserted = it == mHingeJointSnapshot.end();
    const HingeJointState previousState = inserted ? HingeJointState{} : *it;
    if (it == mHingeJointSnapshot.end())
    {
        mHingeJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    applyRigidJointChange(classifyHingeJointChange(previousState, normalized, inserted));
    return true;
}

bool PhysicsWorld::upsertSliderJoint(const SliderJointState &state)
{
    SliderJointState normalized = state;
    if (normalized.bodyA == kInvalidRigidBodyId || normalized.bodyB == kInvalidRigidBodyId ||
        normalized.bodyA == normalized.bodyB)
    {
        return false;
    }
    normalized.localRotationA =
        common::runtime_math::normalizeQuaternion(normalized.localRotationA);
    normalized.localRotationB =
        common::runtime_math::normalizeQuaternion(normalized.localRotationB);
    if (normalized.limitEnabled && normalized.limitMin > normalized.limitMax)
    {
        std::swap(normalized.limitMin, normalized.limitMax);
    }
    if (!normalized.limitEnabled)
    {
        normalized.limitMin = 0.0f;
        normalized.limitMax = 0.0f;
    }
    normalized.constraintCompliance = std::max(normalized.constraintCompliance, 0.0f);
    normalized.driveCompliance      = std::max(normalized.driveCompliance, 0.0f);

    const auto bodyAIt = mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mRigidBodyIdToIndex.end() || bodyBIt == mRigidBodyIdToIndex.end())
    {
        return false;
    }
    const RigidBodyState &bodyA = mRigidBodySnapshot[bodyAIt->second];
    const RigidBodyState &bodyB = mRigidBodySnapshot[bodyBIt->second];
    if (bodyA.environmentIndex != bodyB.environmentIndex)
    {
        return false;
    }
    const float anchorAMagSq = Diligent::dot(normalized.localAnchorA, normalized.localAnchorA);
    const float anchorBMagSq = Diligent::dot(normalized.localAnchorB, normalized.localAnchorB);
    if (anchorAMagSq <= 1.0e-8f && anchorBMagSq <= 1.0e-8f)
    {
        const Diligent::float3 worldAnchor = bodyB.position;
        normalized.localAnchorA =
            quaternionInverseRotate(bodyA.rotation, worldAnchor - bodyA.position);
        normalized.localAnchorB =
            quaternionInverseRotate(bodyB.rotation, worldAnchor - bodyB.position);
    }
    if (normalized.jointId == kInvalidSliderJointId)
    {
        normalized.jointId = mNextSliderJointId++;
    }

    auto it             = std::find_if(mSliderJointSnapshot.begin(), mSliderJointSnapshot.end(),
                                       [&](const SliderJointState &existing)
                                       { return existing.jointId == normalized.jointId; });
    const bool inserted = it == mSliderJointSnapshot.end();
    const SliderJointState previousState = inserted ? SliderJointState{} : *it;
    if (it == mSliderJointSnapshot.end())
    {
        mSliderJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    applyRigidJointChange(classifySliderJointChange(previousState, normalized, inserted));
    return true;
}

bool PhysicsWorld::removeBallJoint(BallJointId jointId)
{
    auto it = std::find_if(mBallJointSnapshot.begin(), mBallJointSnapshot.end(),
                           [&](const BallJointState &state) { return state.jointId == jointId; });
    if (it != mBallJointSnapshot.end())
    {
        mBallJointSnapshot.erase(it);
        applyRigidJointChange(RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

bool PhysicsWorld::removeHingeJoint(HingeJointId jointId)
{
    auto it = std::find_if(mHingeJointSnapshot.begin(), mHingeJointSnapshot.end(),
                           [&](const HingeJointState &state) { return state.jointId == jointId; });
    if (it != mHingeJointSnapshot.end())
    {
        mHingeJointSnapshot.erase(it);
        applyRigidJointChange(RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

bool PhysicsWorld::removeSliderJoint(SliderJointId jointId)
{
    auto it = std::find_if(mSliderJointSnapshot.begin(), mSliderJointSnapshot.end(),
                           [&](const SliderJointState &state) { return state.jointId == jointId; });
    if (it != mSliderJointSnapshot.end())
    {
        mSliderJointSnapshot.erase(it);
        applyRigidJointChange(RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

RigidBodyState *PhysicsWorld::tryGetRigidBody(common::EntityId entityId)
{
    const auto it = mEntityToRigidBodyIndex.find(entityId);
    return it == mEntityToRigidBodyIndex.end() ? nullptr : &mRigidBodySnapshot[it->second];
}

const RigidBodyState *PhysicsWorld::tryGetRigidBody(common::EntityId entityId) const
{
    const auto it = mEntityToRigidBodyIndex.find(entityId);
    return it == mEntityToRigidBodyIndex.end() ? nullptr : &mRigidBodySnapshot[it->second];
}

const ColliderState *PhysicsWorld::tryGetCollider(ColliderId colliderId) const
{
    const auto it = mColliderIdToIndex.find(colliderId);
    return it == mColliderIdToIndex.end() ? nullptr : &mColliderSnapshot[it->second];
}

const BallJointState *PhysicsWorld::tryGetBallJoint(BallJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mBallJointSnapshot.begin(), mBallJointSnapshot.end(),
                     [&](const BallJointState &state) { return state.jointId == jointId; });
    return it == mBallJointSnapshot.end() ? nullptr : &(*it);
}

const HingeJointState *PhysicsWorld::tryGetHingeJoint(HingeJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mHingeJointSnapshot.begin(), mHingeJointSnapshot.end(),
                     [&](const HingeJointState &state) { return state.jointId == jointId; });
    return it == mHingeJointSnapshot.end() ? nullptr : &(*it);
}

const SliderJointState *PhysicsWorld::tryGetSliderJoint(SliderJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mSliderJointSnapshot.begin(), mSliderJointSnapshot.end(),
                     [&](const SliderJointState &state) { return state.jointId == jointId; });
    return it == mSliderJointSnapshot.end() ? nullptr : &(*it);
}

SoftBodyState *PhysicsWorld::tryGetSoftBody(common::EntityId entityId)
{
    const auto it = mEntityToSoftBodyIndex.find(entityId);
    return it == mEntityToSoftBodyIndex.end() ? nullptr : &mSoftBodySnapshot[it->second];
}

const SoftBodyState *PhysicsWorld::tryGetSoftBody(common::EntityId entityId) const
{
    const auto it = mEntityToSoftBodyIndex.find(entityId);
    return it == mEntityToSoftBodyIndex.end() ? nullptr : &mSoftBodySnapshot[it->second];
}

StrandState *PhysicsWorld::tryGetStrand(common::EntityId entityId)
{
    const auto it = mEntityToStrandIndex.find(entityId);
    return it == mEntityToStrandIndex.end() ? nullptr : &mStrandSnapshot[it->second];
}

const StrandState *PhysicsWorld::tryGetStrand(common::EntityId entityId) const
{
    const auto it = mEntityToStrandIndex.find(entityId);
    return it == mEntityToStrandIndex.end() ? nullptr : &mStrandSnapshot[it->second];
}

bool PhysicsWorld::tryGetSoftBodyAuthoringRestPositions(
    common::EntityId entityId, std::vector<Diligent::float3> &outRestPositions) const
{
    outRestPositions.clear();

    const SoftBodyState *softBody = tryGetSoftBody(entityId);
    if (softBody == nullptr)
    {
        return false;
    }

    if (!softBody->restPositions.empty())
    {
        outRestPositions = softBody->restPositions;
        return true;
    }

    ResolvedSoftBodyTopology topology{};
    std::string errorMessage;
    TetMeshData tetGenMeshCache{};
    const TetGenMeshCache *tetGenCache = tryGetTetGenMeshCache(entityId);
    const TetMeshData *cacheView       = nullptr;
    if (tetGenCache != nullptr)
    {
        tetGenMeshCache.objectSpaceRestPositions = tetGenCache->objectSpaceRestPositions;
        tetGenMeshCache.tetVertexIndices         = tetGenCache->tetVertexIndices;
        cacheView                                = &tetGenMeshCache;
    }

    if (!resolveSoftBodyTopology(*softBody, cacheView, topology, errorMessage))
    {
        CRESSIM_LOG_ERROR("Failed to resolve authored soft-body particles for entity ", entityId,
                          ": ", errorMessage);
        return false;
    }

    outRestPositions = std::move(topology.restPositions);
    return true;
}

FluidState *PhysicsWorld::tryGetFluid(common::EntityId entityId)
{
    const auto it = mEntityToFluidIndex.find(entityId);
    return it == mEntityToFluidIndex.end() ? nullptr : &mFluidSnapshot[it->second];
}

const FluidState *PhysicsWorld::tryGetFluid(common::EntityId entityId) const
{
    const auto it = mEntityToFluidIndex.find(entityId);
    return it == mEntityToFluidIndex.end() ? nullptr : &mFluidSnapshot[it->second];
}

AuthoredParticleSequenceState *PhysicsWorld::tryGetParticleSequence(ParticleSequenceId sequenceId)
{
    const auto it = mParticleSequenceIdToIndex.find(sequenceId);
    return it == mParticleSequenceIdToIndex.end() ? nullptr
                                                  : &mParticleSequenceSnapshot[it->second];
}

const AuthoredParticleSequenceState *PhysicsWorld::tryGetParticleSequence(
    ParticleSequenceId sequenceId) const
{
    const auto it = mParticleSequenceIdToIndex.find(sequenceId);
    return it == mParticleSequenceIdToIndex.end() ? nullptr
                                                  : &mParticleSequenceSnapshot[it->second];
}

AuthoredParticleDistanceConstraintState *PhysicsWorld::tryGetParticleDistanceConstraint(
    ParticleConstraintId constraintId)
{
    const auto it = mParticleConstraintIdToIndex.find(constraintId);
    return it == mParticleConstraintIdToIndex.end()
               ? nullptr
               : &mParticleDistanceConstraintSnapshot[it->second];
}

const AuthoredParticleDistanceConstraintState *PhysicsWorld::tryGetParticleDistanceConstraint(
    ParticleConstraintId constraintId) const
{
    const auto it = mParticleConstraintIdToIndex.find(constraintId);
    return it == mParticleConstraintIdToIndex.end()
               ? nullptr
               : &mParticleDistanceConstraintSnapshot[it->second];
}

AuthoredRigidParticleAttachmentConstraintState *
PhysicsWorld::tryGetRigidParticleAttachmentConstraint(
    RigidParticleAttachmentConstraintId constraintId)
{
    const auto it = mRigidParticleAttachmentConstraintIdToIndex.find(constraintId);
    return it == mRigidParticleAttachmentConstraintIdToIndex.end()
               ? nullptr
               : &mRigidParticleAttachmentConstraintSnapshot[it->second];
}

const AuthoredRigidParticleAttachmentConstraintState *
PhysicsWorld::tryGetRigidParticleAttachmentConstraint(
    RigidParticleAttachmentConstraintId constraintId) const
{
    const auto it = mRigidParticleAttachmentConstraintIdToIndex.find(constraintId);
    return it == mRigidParticleAttachmentConstraintIdToIndex.end()
               ? nullptr
               : &mRigidParticleAttachmentConstraintSnapshot[it->second];
}

AuthoredStrandRigidAttachmentConstraintState *
PhysicsWorld::tryGetStrandRigidAttachmentConstraint(
    StrandRigidAttachmentConstraintId constraintId)
{
    const auto it = mStrandRigidAttachmentConstraintIdToIndex.find(constraintId);
    return it == mStrandRigidAttachmentConstraintIdToIndex.end()
               ? nullptr
               : &mStrandRigidAttachmentConstraintSnapshot[it->second];
}

const AuthoredStrandRigidAttachmentConstraintState *
PhysicsWorld::tryGetStrandRigidAttachmentConstraint(
    StrandRigidAttachmentConstraintId constraintId) const
{
    const auto it = mStrandRigidAttachmentConstraintIdToIndex.find(constraintId);
    return it == mStrandRigidAttachmentConstraintIdToIndex.end()
               ? nullptr
               : &mStrandRigidAttachmentConstraintSnapshot[it->second];
}

AuthoredRigidDistanceConstraintState *PhysicsWorld::tryGetRigidDistanceConstraint(
    RigidDistanceConstraintId constraintId)
{
    const auto it = mRigidDistanceConstraintIdToIndex.find(constraintId);
    return it == mRigidDistanceConstraintIdToIndex.end()
               ? nullptr
               : &mRigidDistanceConstraintSnapshot[it->second];
}

const AuthoredRigidDistanceConstraintState *PhysicsWorld::tryGetRigidDistanceConstraint(
    RigidDistanceConstraintId constraintId) const
{
    const auto it = mRigidDistanceConstraintIdToIndex.find(constraintId);
    return it == mRigidDistanceConstraintIdToIndex.end()
               ? nullptr
               : &mRigidDistanceConstraintSnapshot[it->second];
}

AuthoredRoutedCableConstraintState *PhysicsWorld::tryGetRoutedCableConstraint(
    RoutedCableConstraintId constraintId)
{
    const auto it = mRoutedCableConstraintIdToIndex.find(constraintId);
    return it == mRoutedCableConstraintIdToIndex.end()
               ? nullptr
               : &mRoutedCableConstraintSnapshot[it->second];
}

const AuthoredRoutedCableConstraintState *PhysicsWorld::tryGetRoutedCableConstraint(
    RoutedCableConstraintId constraintId) const
{
    const auto it = mRoutedCableConstraintIdToIndex.find(constraintId);
    return it == mRoutedCableConstraintIdToIndex.end()
               ? nullptr
               : &mRoutedCableConstraintSnapshot[it->second];
}

AuthoredParticleCollisionFilterState *PhysicsWorld::tryGetParticleCollisionFilter(
    ParticleCollisionFilterId filterId)
{
    const auto it = mParticleCollisionFilterIdToIndex.find(filterId);
    return it == mParticleCollisionFilterIdToIndex.end()
               ? nullptr
               : &mParticleCollisionFilterSnapshot[it->second];
}

const AuthoredParticleCollisionFilterState *PhysicsWorld::tryGetParticleCollisionFilter(
    ParticleCollisionFilterId filterId) const
{
    const auto it = mParticleCollisionFilterIdToIndex.find(filterId);
    return it == mParticleCollisionFilterIdToIndex.end()
               ? nullptr
               : &mParticleCollisionFilterSnapshot[it->second];
}

AuthoredSuturingSequenceState *PhysicsWorld::tryGetSuturingSequence(SuturingSequenceId sequenceId)
{
    const auto it = mSuturingSequenceIdToIndex.find(sequenceId);
    return it == mSuturingSequenceIdToIndex.end() ? nullptr
                                                  : &mSuturingSequenceSnapshot[it->second];
}

const AuthoredSuturingSequenceState *PhysicsWorld::tryGetSuturingSequence(
    SuturingSequenceId sequenceId) const
{
    const auto it = mSuturingSequenceIdToIndex.find(sequenceId);
    return it == mSuturingSequenceIdToIndex.end() ? nullptr
                                                  : &mSuturingSequenceSnapshot[it->second];
}

const std::vector<RigidBodyState> &PhysicsWorld::rigidBodySnapshot() const noexcept
{
    return mRigidBodySnapshot;
}

const std::vector<ColliderState> &PhysicsWorld::colliderSnapshot() const noexcept
{
    return mColliderSnapshot;
}

const std::vector<SoftBodyState> &PhysicsWorld::softBodySnapshot() const noexcept
{
    return mSoftBodySnapshot;
}

const std::vector<StrandState> &PhysicsWorld::strandSnapshot() const noexcept
{
    return mStrandSnapshot;
}

const std::vector<FluidState> &PhysicsWorld::fluidSnapshot() const noexcept
{
    return mFluidSnapshot;
}

const std::vector<AuthoredParticleSequenceState> &PhysicsWorld::particleSequenceSnapshot()
    const noexcept
{
    return mParticleSequenceSnapshot;
}

const std::vector<AuthoredParticleDistanceConstraintState> &PhysicsWorld::
    particleDistanceConstraintSnapshot() const noexcept
{
    return mParticleDistanceConstraintSnapshot;
}

const std::vector<AuthoredRigidParticleAttachmentConstraintState> &PhysicsWorld::
    rigidParticleAttachmentConstraintSnapshot() const noexcept
{
    return mRigidParticleAttachmentConstraintSnapshot;
}

const std::vector<AuthoredStrandRigidAttachmentConstraintState> &PhysicsWorld::
    strandRigidAttachmentConstraintSnapshot() const noexcept
{
    return mStrandRigidAttachmentConstraintSnapshot;
}

const std::vector<AuthoredRigidDistanceConstraintState> &PhysicsWorld::
    rigidDistanceConstraintSnapshot() const noexcept
{
    return mRigidDistanceConstraintSnapshot;
}

const std::vector<AuthoredRoutedCableConstraintState> &PhysicsWorld::routedCableConstraintSnapshot()
    const noexcept
{
    return mRoutedCableConstraintSnapshot;
}

const std::vector<AuthoredParticleCollisionFilterState> &PhysicsWorld::
    particleCollisionFilterSnapshot() const noexcept
{
    return mParticleCollisionFilterSnapshot;
}

const std::vector<AuthoredSuturingSequenceState> &PhysicsWorld::suturingSequenceSnapshot()
    const noexcept
{
    return mSuturingSequenceSnapshot;
}

const std::vector<BallJointState> &PhysicsWorld::ballJointSnapshot() const noexcept
{
    return mBallJointSnapshot;
}

const std::vector<HingeJointState> &PhysicsWorld::hingeJointSnapshot() const noexcept
{
    return mHingeJointSnapshot;
}

const std::vector<SliderJointState> &PhysicsWorld::sliderJointSnapshot() const noexcept
{
    return mSliderJointSnapshot;
}

const RigidBodySoAHost &PhysicsWorld::rigidBodySoA() const noexcept
{
    return mRigidBodies;
}

const ColliderSoAHost &PhysicsWorld::colliderSoA() const noexcept
{
    ensureDerivedStateUpToDate();
    return mColliders;
}

const BodyColliderMappingHost &PhysicsWorld::bodyColliderMapping() const noexcept
{
    ensureDerivedStateUpToDate();
    return mBodyColliderMapping;
}

const RigidJointSceneHost &PhysicsWorld::rigidJointScene() const noexcept
{
    ensureDerivedStateUpToDate();
    return mRigidJointScene;
}

const JointCollisionSuppressionHost &PhysicsWorld::jointCollisionSuppression() const noexcept
{
    ensureDerivedStateUpToDate();
    return mJointCollisionSuppression;
}

const ParticleSoAHost &PhysicsWorld::particles() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mParticles;
}

const std::vector<Diligent::float4> &PhysicsWorld::particleContactMaterials() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mParticleContactMaterials;
}

const std::vector<FluidMaterialGpu> &PhysicsWorld::fluidMaterials() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mFluidMaterials;
}

const std::vector<DeformableDistanceConstraint> &PhysicsWorld::distanceConstraints() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mSoftEdges;
}

const std::vector<DeformableBendConstraint> &PhysicsWorld::bendConstraints() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mSoftBends;
}

const std::vector<DeformableVolumeConstraint> &PhysicsWorld::volumeConstraints() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mSoftTets;
}

const std::vector<SoftEdge> &PhysicsWorld::softEdges() const noexcept
{
    return distanceConstraints();
}

const std::vector<SoftBend> &PhysicsWorld::softBends() const noexcept
{
    return bendConstraints();
}

const std::vector<SoftTet> &PhysicsWorld::softTets() const noexcept
{
    return volumeConstraints();
}

const std::vector<StrandSegmentConstraint> &PhysicsWorld::strandSegments() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mStrandSegments;
}

const std::vector<StrandJointConstraint> &PhysicsWorld::strandJoints() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mStrandJoints;
}

const std::vector<StrandDistanceConstraint> &PhysicsWorld::strandDistanceConstraints() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mStrandDistanceConstraints;
}

const std::vector<StrandSegmentState> &PhysicsWorld::strandSegmentStates() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mStrandSegmentStates;
}

const std::vector<RigidParticleAttachmentConstraint> &PhysicsWorld::rigidParticleAttachments()
    const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mRigidParticleAttachments;
}

const std::vector<StrandRigidAttachmentConstraint> &PhysicsWorld::strandRigidAttachments()
    const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mStrandRigidAttachments;
}

const std::vector<RigidDistanceConstraint> &PhysicsWorld::rigidDistanceConstraints() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mRigidDistanceConstraints;
}

const std::vector<RoutedCableConstraint> &PhysicsWorld::routedCableConstraints() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mRoutedCableConstraints;
}

const std::vector<RoutedCableRoutePoint> &PhysicsWorld::routedCableRoutePoints() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mRoutedCableRoutePoints;
}

const std::vector<StrandSoftSuturingPair> &PhysicsWorld::suturingPairs() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mSuturingPairs;
}

const std::vector<std::uint32_t> &PhysicsWorld::suturingParticleIndices() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mParticles.suturingParticleIndices;
}

const SoftRenderDataHost &PhysicsWorld::softRenderData() const noexcept
{
    return mSoftRenderData;
}

void PhysicsWorld::setSoftRenderData(const SoftRenderDataHost &data)
{
    mSoftRenderData = data;
    recomputeSoftBodyBoundsChunkCount();
    ++mSoftGpuTopologyRevision;
    ++mAuthoredRevision;
}

const CurveRenderDataHost &PhysicsWorld::curveRenderData() const noexcept
{
    return mCurveRenderData;
}

void PhysicsWorld::setCurveRenderData(const CurveRenderDataHost &data)
{
    mCurveRenderData = data;
    ++mCurveRenderRevision;
}

void PhysicsWorld::ensureDerivedStateUpToDate() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    if (mBodyColliderMappingDirty)
    {
        rebuildBodyColliderMapping();
        mBodyColliderMappingDirty = false;
    }
    if (mRigidJointSceneDirty)
    {
        rebuildRigidJointScene();
        mRigidJointSceneDirty = false;
    }
    if (mJointCollisionSuppressionDirty)
    {
        rebuildJointCollisionSuppression();
        mJointCollisionSuppressionDirty = false;
    }
}

void PhysicsWorld::ensureSoftBodyDerivedStateUpToDate() noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        rebuildSoftBodyDerivedState();
    }
}

const std::vector<std::uint32_t> &PhysicsWorld::rigidBodyDirtyIndices() const noexcept
{
    return mRigidBodyDirtyIndices;
}

const std::vector<std::uint32_t> &PhysicsWorld::colliderDirtyIndices() const noexcept
{
    return mColliderDirtyIndices;
}

std::uint32_t PhysicsWorld::rigidBodyCount() const noexcept
{
    return static_cast<std::uint32_t>(mRigidBodies.size());
}

std::uint32_t PhysicsWorld::colliderCount() const noexcept
{
    return static_cast<std::uint32_t>(mColliders.size());
}

std::uint32_t PhysicsWorld::softBodyCount() const noexcept
{
    return static_cast<std::uint32_t>(mSoftBodySnapshot.size());
}

std::uint32_t PhysicsWorld::strandCount() const noexcept
{
    return static_cast<std::uint32_t>(mStrandSnapshot.size());
}

std::uint32_t PhysicsWorld::fluidCount() const noexcept
{
    return static_cast<std::uint32_t>(mFluidSnapshot.size());
}

bool PhysicsWorld::rigidBodyCountDirty() const noexcept
{
    return mRigidBodyCountDirty;
}

bool PhysicsWorld::colliderCountDirty() const noexcept
{
    return mColliderCountDirty;
}

bool PhysicsWorld::fullRigidBodyUploadRequired() const noexcept
{
    return mFullRigidBodyUploadRequired;
}

bool PhysicsWorld::fullColliderUploadRequired() const noexcept
{
    return mFullColliderUploadRequired;
}

void PhysicsWorld::clearRigidBodyUploadState() noexcept
{
    clearDirtyIndices(mRigidBodyDirtyIndices, mRigidBodyDirtyBits);
    mRigidBodyCountDirty         = false;
    mFullRigidBodyUploadRequired = false;
}

void PhysicsWorld::clearColliderUploadState() noexcept
{
    clearDirtyIndices(mColliderDirtyIndices, mColliderDirtyBits);
    mColliderCountDirty         = false;
    mFullColliderUploadRequired = false;
}

bool PhysicsWorld::staticBroadPhaseDirty() const noexcept
{
    return mStaticBroadPhaseDirty;
}

void PhysicsWorld::clearStaticBroadPhaseDirty() noexcept
{
    mStaticBroadPhaseDirty = false;
}

std::uint32_t PhysicsWorld::activeMovingColliderCount() const noexcept
{
    return mActiveMovingColliderCount;
}

std::uint32_t PhysicsWorld::staticColliderCount() const noexcept
{
    return mStaticColliderCount;
}

float PhysicsWorld::particleGridCellSize() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mParticleGridCellSize;
}

std::uint32_t PhysicsWorld::softBodyBoundsChunkCount() const noexcept
{
    return mSoftBodyBoundsChunkCount;
}

std::uint32_t PhysicsWorld::maxSuturingPathsPerPair() const noexcept
{
    return mMaxSuturingPathsPerPair;
}

std::uint32_t PhysicsWorld::maxSuturingNodesPerPath() const noexcept
{
    return mMaxSuturingNodesPerPath;
}

std::uint32_t PhysicsWorld::suturingParticleCount() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return static_cast<std::uint32_t>(mParticles.suturingParticleIndices.size());
}

std::uint32_t PhysicsWorld::reservedSuturingPathHeaderCount() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mReservedSuturingPathHeaders;
}

std::uint32_t PhysicsWorld::reservedSuturingPathNodeCount() const noexcept
{
    if (mSoftBodyDerivedStateDirty)
    {
        const_cast<PhysicsWorld *>(this)->rebuildSoftBodyDerivedState();
    }
    return mReservedSuturingPathNodes;
}

void PhysicsWorld::integrateRigidBodiesCpu(float dt) noexcept
{
    if (mRigidBodies.empty())
    {
        return;
    }

    for (std::uint32_t i = 0; i < rigidBodyCount(); ++i)
    {
        Diligent::float4 &positionInvMass     = mRigidBodies.positionsInvMass[i];
        const Diligent::float4 linearVelocity = mRigidBodies.linearVelocities[i];
        positionInvMass.x += linearVelocity.x * dt;
        positionInvMass.y += linearVelocity.y * dt;
        positionInvMass.z += linearVelocity.z * dt;

        RigidBodyState &state = mRigidBodySnapshot[i];
        state.position.x      = positionInvMass.x;
        state.position.y      = positionInvMass.y;
        state.position.z      = positionInvMass.z;
    }

    markAllRigidBodiesDirty();
    ++mAuthoredRevision;
}

bool PhysicsWorld::syncRigidBodyStateFromSimulation(
    std::uint32_t index, const Diligent::float4 &positionInvMass,
    const Diligent::float4 &orientation, const Diligent::float4 &linearVelocity,
    const Diligent::float4 &angularVelocity) noexcept
{
    if (index >= rigidBodyCount())
    {
        return false;
    }

    mRigidBodies.positionsInvMass[index]  = positionInvMass;
    mRigidBodies.orientations[index]      = orientation;
    mRigidBodies.linearVelocities[index]  = linearVelocity;
    mRigidBodies.angularVelocities[index] = angularVelocity;
    RigidBodyState &state                 = mRigidBodySnapshot[index];
    state.position    = Diligent::float3{positionInvMass.x, positionInvMass.y, positionInvMass.z};
    state.inverseMass = positionInvMass.w;
    state.rotation =
        Diligent::QuaternionF{orientation.x, orientation.y, orientation.z, orientation.w};
    state.linearVelocity = Diligent::float3{linearVelocity.x, linearVelocity.y, linearVelocity.z};
    state.angularVelocity =
        Diligent::float3{angularVelocity.x, angularVelocity.y, angularVelocity.z};
    return true;
}

void PhysicsWorld::finalizeRigidBodyWriteback() noexcept
{
    if (mRigidBodies.empty())
    {
        return;
    }
    ++mSimulationRevision;
}

bool PhysicsWorld::syncParticleStateFromSimulation(std::uint32_t index,
                                                   const Diligent::float4 &positionInvMass,
                                                   const Diligent::float4 &previousPosition,
                                                   const Diligent::float4 &velocity) noexcept
{
    if (index >= mParticles.size())
    {
        return false;
    }

    mParticles.positionsInvMass[index]  = positionInvMass;
    mParticles.previousPositions[index] = previousPosition;
    mParticles.velocities[index]        = velocity;
    return true;
}

void PhysicsWorld::finalizeParticleWriteback() noexcept
{
    if (mParticles.empty())
    {
        return;
    }
    ++mSimulationRevision;
}

std::uint64_t PhysicsWorld::authoredRevision() const noexcept
{
    return mAuthoredRevision;
}

std::uint64_t PhysicsWorld::simulationRevision() const noexcept
{
    return mSimulationRevision;
}

std::uint64_t PhysicsWorld::rigidBodyTopologyRevision() const noexcept
{
    return mRigidBodyTopologyRevision;
}

std::uint64_t PhysicsWorld::rigidJointTopologyRevision() const noexcept
{
    return mRigidJointTopologyRevision;
}

std::uint64_t PhysicsWorld::rigidJointSceneRevision() const noexcept
{
    return mRigidJointSceneRevision;
}

std::uint64_t PhysicsWorld::rigidJointModeRevision() const noexcept
{
    return mRigidJointModeRevision;
}

std::uint64_t PhysicsWorld::softBodyTopologyRevision() const noexcept
{
    return mSoftBodyTopologyRevision;
}

std::uint64_t PhysicsWorld::softParticleRevision() const noexcept
{
    return mSoftParticleRevision;
}

std::uint64_t PhysicsWorld::softGpuTopologyRevision() const noexcept
{
    return mSoftGpuTopologyRevision;
}

std::uint64_t PhysicsWorld::rigidParticleAttachmentRevision() const noexcept
{
    return mRigidParticleAttachmentRevision;
}

std::uint64_t PhysicsWorld::rigidParticleAttachmentTopologyRevision() const noexcept
{
    return mRigidParticleAttachmentTopologyRevision;
}

std::uint64_t PhysicsWorld::strandRigidAttachmentRevision() const noexcept
{
    return mStrandRigidAttachmentRevision;
}

std::uint64_t PhysicsWorld::strandRigidAttachmentTopologyRevision() const noexcept
{
    return mStrandRigidAttachmentTopologyRevision;
}

std::uint64_t PhysicsWorld::rigidDistanceConstraintRevision() const noexcept
{
    return mRigidDistanceConstraintRevision;
}

std::uint64_t PhysicsWorld::rigidDistanceConstraintTopologyRevision() const noexcept
{
    return mRigidDistanceConstraintTopologyRevision;
}

std::uint64_t PhysicsWorld::routedCableRevision() const noexcept
{
    return mRoutedCableRevision;
}

std::uint64_t PhysicsWorld::routedCableTopologyRevision() const noexcept
{
    return mRoutedCableTopologyRevision;
}

std::uint64_t PhysicsWorld::curveRenderRevision() const noexcept
{
    return mCurveRenderRevision;
}

void PhysicsWorld::writeRigidBodySoAAt(RigidBodySoAHost &soa, std::uint32_t index,
                                       const RigidBodyState &state)
{
    soa.rigidBodyIds[index]                  = state.rigidBodyId;
    soa.entityIds[index]                     = state.entityId;
    soa.positionsInvMass[index]              = toPositionInvMass(state);
    soa.orientations[index]                  = toOrientation(state);
    soa.scales[index]                        = toScale(state);
    soa.linearVelocities[index]              = toLinearVelocity(state);
    soa.angularVelocities[index]             = toAngularVelocity(state);
    soa.inverseInertiaLocal[index]           = toInverseInertiaLocal(state);
    soa.bodyTypes[index]                     = static_cast<std::uint32_t>(state.bodyType);
    soa.kinematicTargetPositions[index]      = toKinematicTargetPosition(state);
    soa.kinematicTargetOrientations[index]   = toKinematicTargetOrientation(state);
    soa.kinematicTargetFlags[index]          = state.kinematicTargetEnabled ? 1u : 0u;
    soa.proxyParticleContactMaterials[index] = state.proxyParticleContactMaterial;
}

void PhysicsWorld::writeColliderSoAAt(ColliderSoAHost &soa, std::uint32_t index,
                                      const ColliderState &state, std::uint32_t ownerBodyIndex,
                                      std::uint32_t ownerEnvironmentIndex)
{
    if (index == soa.size())
    {
        soa.colliderIds.push_back(state.colliderId);
        soa.entityIds.push_back(state.entityId);
        soa.ownerRigidBodyIds.push_back(state.ownerRigidBodyId);
        soa.ownerRigidBodyIndices.push_back(ownerBodyIndex);
        soa.environmentIndices.push_back(ownerEnvironmentIndex);
        soa.shapeTypes.push_back(static_cast<std::uint32_t>(state.shapeType));
        soa.shapeParams.push_back(state.shapeParams);
        soa.localPositions.push_back(toColliderLocalPosition(state));
        soa.localOrientations.push_back(toColliderLocalOrientation(state));
        soa.enabledFlags.push_back(state.enabled ? 1u : 0u);
        soa.frictionRestitution.push_back(toColliderMaterial(state));
        soa.collisionLayers.push_back(state.collisionLayer);
        soa.collisionMasks.push_back(state.collisionMask);
        return;
    }

    soa.colliderIds[index]           = state.colliderId;
    soa.entityIds[index]             = state.entityId;
    soa.ownerRigidBodyIds[index]     = state.ownerRigidBodyId;
    soa.ownerRigidBodyIndices[index] = ownerBodyIndex;
    soa.environmentIndices[index]    = ownerEnvironmentIndex;
    soa.shapeTypes[index]            = static_cast<std::uint32_t>(state.shapeType);
    soa.shapeParams[index]           = state.shapeParams;
    soa.localPositions[index]        = toColliderLocalPosition(state);
    soa.localOrientations[index]     = toColliderLocalOrientation(state);
    soa.enabledFlags[index]          = state.enabled ? 1u : 0u;
    soa.frictionRestitution[index]   = toColliderMaterial(state);
    soa.collisionLayers[index]       = state.collisionLayer;
    soa.collisionMasks[index]        = state.collisionMask;
}

bool PhysicsWorld::isStaticBody(const RigidBodyState &state) noexcept
{
    return state.bodyType == RigidBodyType::Static;
}

bool PhysicsWorld::isMovingBody(const RigidBodyState &state) noexcept
{
    return state.bodyType == RigidBodyType::Kinematic || state.bodyType == RigidBodyType::Dynamic;
}

std::uint32_t PhysicsWorld::colliderBroadPhaseContribution(const ColliderState &collider,
                                                           const RigidBodyState *owner) noexcept
{
    if (!collider.enabled || owner == nullptr)
    {
        return kBroadPhaseContributionNone;
    }
    if (isStaticBody(*owner))
    {
        return kBroadPhaseContributionStatic;
    }
    if (isMovingBody(*owner))
    {
        return kBroadPhaseContributionMoving;
    }
    return kBroadPhaseContributionNone;
}

bool PhysicsWorld::staticBodyPoseChanged(const RigidBodyState &before,
                                         const RigidBodyState &after) noexcept
{
    if (isStaticBody(before) != isStaticBody(after))
    {
        return true;
    }
    if (!isStaticBody(before))
    {
        return false;
    }

    return before.position.x != after.position.x || before.position.y != after.position.y ||
           before.position.z != after.position.z || before.rotation.q.x != after.rotation.q.x ||
           before.rotation.q.y != after.rotation.q.y || before.rotation.q.z != after.rotation.q.z ||
           before.rotation.q.w != after.rotation.q.w || before.scale.x != after.scale.x ||
           before.scale.y != after.scale.y || before.scale.z != after.scale.z;
}

void PhysicsWorld::normalizeRigidBodyState(RigidBodyState &state) noexcept
{
    if (state.bodyType == RigidBodyType::Dynamic && state.inverseMass <= 0.0f)
    {
        state.inverseMass = 1.0f;
    }

    if (state.bodyType != RigidBodyType::Kinematic)
    {
        state.kinematicTargetPosition = state.position;
        state.kinematicTargetRotation = state.rotation;
        state.kinematicTargetEnabled  = false;
    }
    else if (!state.kinematicTargetEnabled)
    {
        state.kinematicTargetPosition = state.position;
        state.kinematicTargetRotation = state.rotation;
    }

    state.proxyParticleRadius = std::max(state.proxyParticleRadius, 0.0f);
    normalizeParticleContactMaterial(state.proxyParticleMaterial);
    if (state.proxyCollisionLayer == 0u)
    {
        state.proxyCollisionLayer = 1u;
    }
    if (state.proxyParticleLocalPositions.empty())
    {
        state.proxyParticleRadius = 0.0f;
        state.needleTipProxyIndex = 0u;
        state.suturingEnabled     = false;
    }
    else
    {
        state.needleTipProxyIndex = std::min<std::uint32_t>(
            state.needleTipProxyIndex,
            static_cast<std::uint32_t>(state.proxyParticleLocalPositions.size() - 1u));
    }
}

void PhysicsWorld::normalizeColliderState(ColliderState &state) noexcept
{
    state.friction = std::max(state.friction, 0.0f);
    state.staticFriction =
        state.staticFriction < 0.0f ? state.friction : std::max(state.staticFriction, 0.0f);
    state.restitution = std::clamp(state.restitution, 0.0f, 1.0f);
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

void PhysicsWorld::normalizeSoftBodyState(SoftBodyState &state) noexcept
{
    normalizeParticleContactMaterial(state.material.contact);
    state.particleMass     = std::max(state.particleMass, 1.0e-4f);
    state.particleRadius   = std::max(state.particleRadius, 1.0e-4f);
    state.edgeCompliance   = std::max(state.edgeCompliance, 0.0f);
    state.volumeCompliance = std::max(state.volumeCompliance, 0.0f);

    const auto clampScale = [](float value) -> float
    {
        const float sign = value < 0.0f ? -1.0f : 1.0f;
        return sign * std::max(std::abs(value), 1.0e-4f);
    };
    state.restTransform.scale.x = clampScale(state.restTransform.scale.x);
    state.restTransform.scale.y = clampScale(state.restTransform.scale.y);
    state.restTransform.scale.z = clampScale(state.restTransform.scale.z);

    if (state.source.kind == SoftBodySourceKind::RegularGrid)
    {
        state.source.regularGrid.size.x = std::max(state.source.regularGrid.size.x, 1.0e-4f);
        state.source.regularGrid.size.y = std::max(state.source.regularGrid.size.y, 1.0e-4f);
        state.source.regularGrid.size.z = std::max(state.source.regularGrid.size.z, 1.0e-4f);
        state.source.regularGrid.targetParticleSpacing =
            std::max(state.source.regularGrid.targetParticleSpacing, 1.0e-4f);
    }
    if (state.source.kind == SoftBodySourceKind::MeshfreeParticles)
    {
        state.source.meshfreeParticles.neighbourCount =
            std::max(state.source.meshfreeParticles.neighbourCount, 1u);
    }
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

void PhysicsWorld::normalizeStrandState(StrandState &state) noexcept
{
    normalizeParticleContactMaterial(state.material.contact);
    state.particleMass            = std::max(state.particleMass, 1.0e-4f);
    state.particleRadius          = std::max(state.particleRadius, 1.0e-4f);
    state.stretchShearCompliance  = std::max(state.stretchShearCompliance, 0.0f);
    state.bendCompliance          = std::max(state.bendCompliance, 0.0f);
    state.twistCompliance         = std::max(state.twistCompliance, 0.0f);
    if (state.distanceCompliance < 0.0f)
    {
        state.distanceCompliance = state.stretchShearCompliance;
    }
    state.distanceCompliance = std::max(state.distanceCompliance, 0.0f);
    state.pathNodeSpacing         = std::max(state.pathNodeSpacing, 1.0e-4f);
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

void PhysicsWorld::normalizeFluidState(FluidState &state) noexcept
{
    normalizeFluidMaterial(state.material);
    state.particleMass   = std::max(state.particleMass, 1.0e-4f);
    state.particleRadius = std::max(state.particleRadius, 1.0e-4f);
    state.visualColor.x  = std::clamp(state.visualColor.x, 0.0f, 1.0f);
    state.visualColor.y  = std::clamp(state.visualColor.y, 0.0f, 1.0f);
    state.visualColor.z  = std::clamp(state.visualColor.z, 0.0f, 1.0f);
    state.visualColor.w  = std::clamp(state.visualColor.w, 0.0f, 1.0f);

    const auto clampScale = [](float value) -> float
    {
        const float sign = value < 0.0f ? -1.0f : 1.0f;
        return sign * std::max(std::abs(value), 1.0e-4f);
    };
    state.restTransform.scale.x = clampScale(state.restTransform.scale.x);
    state.restTransform.scale.y = clampScale(state.restTransform.scale.y);
    state.restTransform.scale.z = clampScale(state.restTransform.scale.z);

    if (state.source.kind == FluidSourceKind::RegularGrid)
    {
        state.source.regularGrid.size.x = std::max(state.source.regularGrid.size.x, 1.0e-4f);
        state.source.regularGrid.size.y = std::max(state.source.regularGrid.size.y, 1.0e-4f);
        state.source.regularGrid.size.z = std::max(state.source.regularGrid.size.z, 1.0e-4f);
        state.source.regularGrid.targetParticleSpacing =
            std::max(state.source.regularGrid.targetParticleSpacing, 1.0e-4f);
    }
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

bool PhysicsWorld::validateFluidMaterialCompatibility(
    const FluidState &candidate, const FluidState *previousState) const noexcept
{
    for (const FluidState &existing : mFluidSnapshot)
    {
        if (previousState != nullptr && existing.entityId == previousState->entityId)
        {
            continue;
        }

        if (!nearlyEqual(existing.particleRadius, candidate.particleRadius))
        {
            CRESSIM_LOG_ERROR(
                "Fluid bodies must currently share particleRadius because the fluid solve "
                "model is spacing-derived. Entity ",
                candidate.entityId, " is incompatible with existing fluid entity ",
                existing.entityId, ".");
            return false;
        }
    }

    return true;
}

PhysicsWorld::SoftBodyChangeKind PhysicsWorld::classifySoftBodyChange(
    const SoftBodyState &previousState, const SoftBodyState &candidate) noexcept
{
    if (!equalSoftBodySourceDesc(previousState.source, candidate.source) ||
        previousState.restTransform != candidate.restTransform ||
        previousState.supportsSuturing != candidate.supportsSuturing)
    {
        return SoftBodyChangeKind::TopologyRebuild;
    }

    return SoftBodyChangeKind::RuntimePropertiesOnly;
}

PhysicsWorld::StrandChangeKind PhysicsWorld::classifyStrandChange(
    const StrandState &previousState, const StrandState &candidate) noexcept
{
    if (previousState.restPositions != candidate.restPositions ||
        previousState.staticParticleIndices != candidate.staticParticleIndices ||
        previousState.suturingEnabled != candidate.suturingEnabled ||
        previousState.pathNodeSpacing != candidate.pathNodeSpacing ||
        previousState.rootMaterialNormal.x != candidate.rootMaterialNormal.x ||
        previousState.rootMaterialNormal.y != candidate.rootMaterialNormal.y ||
        previousState.rootMaterialNormal.z != candidate.rootMaterialNormal.z)
    {
        return StrandChangeKind::TopologyRebuild;
    }

    return StrandChangeKind::RuntimePropertiesOnly;
}

PhysicsWorld::RigidJointChangeKind PhysicsWorld::classifyBallJointChange(bool inserted) noexcept
{
    return inserted ? RigidJointChangeKind::TopologyRebuild : RigidJointChangeKind::PayloadOnly;
}

PhysicsWorld::RigidJointChangeKind PhysicsWorld::classifyHingeJointChange(
    const HingeJointState &previousState, const HingeJointState &candidate, bool inserted) noexcept
{
    if (inserted)
    {
        return RigidJointChangeKind::TopologyRebuild;
    }

    if (previousState.enabled != candidate.enabled ||
        previousState.driveMode != candidate.driveMode)
    {
        return RigidJointChangeKind::ModeRebuild;
    }

    return RigidJointChangeKind::PayloadOnly;
}

PhysicsWorld::RigidJointChangeKind PhysicsWorld::classifySliderJointChange(
    const SliderJointState &previousState, const SliderJointState &candidate,
    bool inserted) noexcept
{
    if (inserted)
    {
        return RigidJointChangeKind::TopologyRebuild;
    }

    if (previousState.enabled != candidate.enabled ||
        previousState.driveMode != candidate.driveMode)
    {
        return RigidJointChangeKind::ModeRebuild;
    }

    return RigidJointChangeKind::PayloadOnly;
}

void PhysicsWorld::applySoftBodyRuntimeProperties(std::uint32_t index,
                                                  const SoftBodyState &normalizedState) noexcept
{
    if (index >= mSoftBodySnapshot.size())
    {
        return;
    }

    if (mSoftBodyDerivedStateDirty)
    {
        rebuildSoftBodyDerivedState();
    }

    if (index >= mSoftBodySnapshot.size() || index >= mSoftBodyDerivedCaches.size())
    {
        return;
    }

    SoftBodyState &softBody       = mSoftBodySnapshot[index];
    softBody.environmentIndex     = normalizedState.environmentIndex;
    softBody.collisionLayer       = normalizedState.collisionLayer;
    softBody.collisionMask        = normalizedState.collisionMask;
    softBody.material             = normalizedState.material;
    softBody.particleMass         = normalizedState.particleMass;
    softBody.particleRadius       = normalizedState.particleRadius;
    softBody.edgeCompliance       = normalizedState.edgeCompliance;
    softBody.volumeCompliance     = normalizedState.volumeCompliance;
    softBody.simulated            = normalizedState.simulated;
    softBody.selfCollisionEnabled = normalizedState.selfCollisionEnabled;
    softBody.supportsSuturing     = normalizedState.supportsSuturing;

    softBody.contactMaterialIndex = findOrAppendParticleContactMaterial(
        mParticleContactMaterials, toParticleContactMaterial(softBody.material.contact));

    const std::uint32_t phase         = packParticlePhase(index, softBody.selfCollisionEnabled);
    const std::uint32_t particleBegin = softBody.particleOffset;
    const std::uint32_t particleEnd   = std::min<std::uint32_t>(
        particleBegin + softBody.particleCount, static_cast<std::uint32_t>(mParticles.size()));
    for (std::uint32_t particleIndex = particleBegin; particleIndex < particleEnd; ++particleIndex)
    {
        mParticles.positionsInvMass[particleIndex].w =
            softBody.particleMass > 0.0f ? 1.0f / softBody.particleMass : 0.0f;
        mParticles.radii[particleIndex]                   = softBody.particleRadius;
        mParticles.environmentIndices[particleIndex]      = softBody.environmentIndex;
        mParticles.collisionLayers[particleIndex]         = softBody.collisionLayer;
        mParticles.collisionMasks[particleIndex]          = softBody.collisionMask;
        mParticles.phases[particleIndex]                  = phase;
        mParticles.particleMaterialIndices[particleIndex] = softBody.contactMaterialIndex;
    }

    const std::uint32_t edgeBegin = softBody.edgeOffset;
    const std::uint32_t edgeEnd   = std::min<std::uint32_t>(
        edgeBegin + softBody.edgeCount, static_cast<std::uint32_t>(mSoftEdges.size()));
    for (std::uint32_t edgeIndex = edgeBegin; edgeIndex < edgeEnd; ++edgeIndex)
    {
        mSoftEdges[edgeIndex].compliance = softBody.edgeCompliance;
    }

    const std::uint32_t tetBegin = softBody.tetOffset;
    const std::uint32_t tetEnd   = std::min<std::uint32_t>(
        tetBegin + softBody.tetCount, static_cast<std::uint32_t>(mSoftTets.size()));
    for (std::uint32_t tetIndex = tetBegin; tetIndex < tetEnd; ++tetIndex)
    {
        mSoftTets[tetIndex].compliance = softBody.volumeCompliance;
    }

    recomputeParticleGridCellSize();
}

void PhysicsWorld::applyStrandRuntimeProperties(std::uint32_t index,
                                                const StrandState &normalizedState) noexcept
{
    if (index >= mStrandSnapshot.size())
    {
        return;
    }

    if (mSoftBodyDerivedStateDirty)
    {
        rebuildSoftBodyDerivedState();
    }

    if (index >= mStrandSnapshot.size() || index >= mStrandDerivedCaches.size())
    {
        return;
    }

    StrandState &strand          = mStrandSnapshot[index];
    strand.environmentIndex      = normalizedState.environmentIndex;
    strand.collisionLayer        = normalizedState.collisionLayer;
    strand.collisionMask         = normalizedState.collisionMask;
    strand.material              = normalizedState.material;
    strand.particleMass          = normalizedState.particleMass;
    strand.particleRadius        = normalizedState.particleRadius;
    strand.stretchShearCompliance = normalizedState.stretchShearCompliance;
    strand.bendCompliance         = normalizedState.bendCompliance;
    strand.twistCompliance        = normalizedState.twistCompliance;
    strand.distanceCompliance     = normalizedState.distanceCompliance;
    strand.rootMaterialNormal     = normalizedState.rootMaterialNormal;
    strand.simulated              = normalizedState.simulated;
    strand.selfCollisionEnabled   = normalizedState.selfCollisionEnabled;
    strand.suturingEnabled        = normalizedState.suturingEnabled;
    strand.pathNodeSpacing        = normalizedState.pathNodeSpacing;
    strand.staticParticleIndices  = normalizedState.staticParticleIndices;
    strand.contactMaterialIndex   = findOrAppendParticleContactMaterial(
        mParticleContactMaterials, toParticleContactMaterial(strand.material.contact));

    const std::uint32_t particleBegin = strand.particleOffset;
    const std::uint32_t particleEnd   = std::min(particleBegin + strand.particleCount,
                                                 static_cast<std::uint32_t>(mParticles.size()));
    const std::uint32_t phase         = packParticlePhase(
        static_cast<std::uint32_t>(mSoftBodySnapshot.size()) + index, strand.selfCollisionEnabled);
    const float inverseMass = strand.particleMass > 0.0f ? 1.0f / strand.particleMass : 0.0f;
    std::unordered_set<std::uint32_t> staticParticles(strand.staticParticleIndices.begin(),
                                                      strand.staticParticleIndices.end());
    for (std::uint32_t particleIndex = particleBegin; particleIndex < particleEnd; ++particleIndex)
    {
        const std::uint32_t localIndex = particleIndex - particleBegin;
        mParticles.positionsInvMass[particleIndex].w =
            staticParticles.find(localIndex) != staticParticles.end() ? 0.0f : inverseMass;
        mParticles.radii[particleIndex]                   = strand.particleRadius;
        mParticles.environmentIndices[particleIndex]      = strand.environmentIndex;
        mParticles.particleMaterialIndices[particleIndex] = strand.contactMaterialIndex;
        mParticles.phases[particleIndex]                  = phase;
        mParticles.collisionLayers[particleIndex]         = strand.collisionLayer;
        mParticles.collisionMasks[particleIndex]          = strand.collisionMask;
        mParticles.strandRoles[particleIndex]             = static_cast<std::uint32_t>(
            strand.suturingEnabled ? (localIndex == 0u ? ParticleStrandRole::NeedleTip
                                                       : ParticleStrandRole::NeedleBody)
                                   : ParticleStrandRole::None);
    }

    const std::uint32_t segmentBegin = strand.segmentOffset;
    const std::uint32_t segmentEnd = std::min(segmentBegin + strand.segmentCount,
                                              static_cast<std::uint32_t>(mStrandSegments.size()));
    for (std::uint32_t segmentIndex = segmentBegin; segmentIndex < segmentEnd; ++segmentIndex)
    {
        mStrandSegments[segmentIndex].stretchShearCompliance = strand.stretchShearCompliance;
    }

    const std::uint32_t jointBegin = strand.jointOffset;
    const std::uint32_t jointEnd = std::min(jointBegin + strand.jointCount,
                                            static_cast<std::uint32_t>(mStrandJoints.size()));
    for (std::uint32_t jointIndex = jointBegin; jointIndex < jointEnd; ++jointIndex)
    {
        mStrandJoints[jointIndex].bendCompliance  = strand.bendCompliance;
        mStrandJoints[jointIndex].twistCompliance = strand.twistCompliance;
    }

    const std::uint32_t distanceBegin = strand.segmentOffset;
    const std::uint32_t distanceEnd = std::min(distanceBegin + strand.segmentCount,
                                               static_cast<std::uint32_t>(mStrandDistanceConstraints.size()));
    for (std::uint32_t distanceIndex = distanceBegin; distanceIndex < distanceEnd; ++distanceIndex)
    {
        mStrandDistanceConstraints[distanceIndex].distanceCompliance = strand.distanceCompliance;
    }

    recomputeParticleGridCellSize();
}

void PhysicsWorld::removeCollidersForEntity(common::EntityId entityId) noexcept
{
    const auto idsIt = mEntityToColliderIds.find(entityId);
    if (idsIt == mEntityToColliderIds.end())
    {
        return;
    }

    while (!idsIt->second.empty())
    {
        const ColliderId colliderId = idsIt->second.back();
        const auto colliderIt       = mColliderIdToIndex.find(colliderId);
        if (colliderIt == mColliderIdToIndex.end())
        {
            idsIt->second.pop_back();
            continue;
        }
        removeColliderAtIndex(colliderIt->second);
    }

    mEntityToColliderIds.erase(entityId);
}

void PhysicsWorld::removeColliderAtIndex(std::uint32_t index) noexcept
{
    if (index >= colliderCount())
    {
        return;
    }

    const std::uint32_t last                = colliderCount() - 1u;
    const ColliderState removed             = mColliderSnapshot[index];
    const std::uint32_t removedContribution = broadPhaseContributionForCollider(removed);
    if (removedContribution == kBroadPhaseContributionStatic)
    {
        --mStaticColliderCount;
    }
    else if (removedContribution == kBroadPhaseContributionMoving)
    {
        --mActiveMovingColliderCount;
    }

    auto removeHandleFromEntity = [&](common::EntityId entityId, ColliderId colliderId)
    {
        const auto handlesIt = mEntityToColliderIds.find(entityId);
        if (handlesIt == mEntityToColliderIds.end())
        {
            return;
        }
        auto &handles = handlesIt->second;
        handles.erase(std::remove(handles.begin(), handles.end(), colliderId), handles.end());
    };

    removeHandleFromEntity(removed.entityId, removed.colliderId);
    mColliderIdToIndex.erase(removed.colliderId);

    if (index != last)
    {
        const ColliderState moved               = mColliderSnapshot[last];
        mColliderSnapshot[index]                = moved;
        mColliders.colliderIds[index]           = mColliders.colliderIds[last];
        mColliders.entityIds[index]             = mColliders.entityIds[last];
        mColliders.ownerRigidBodyIds[index]     = mColliders.ownerRigidBodyIds[last];
        mColliders.ownerRigidBodyIndices[index] = mColliders.ownerRigidBodyIndices[last];
        mColliders.environmentIndices[index]    = mColliders.environmentIndices[last];
        mColliders.shapeTypes[index]            = mColliders.shapeTypes[last];
        mColliders.shapeParams[index]           = mColliders.shapeParams[last];
        mColliders.localPositions[index]        = mColliders.localPositions[last];
        mColliders.localOrientations[index]     = mColliders.localOrientations[last];
        mColliders.enabledFlags[index]          = mColliders.enabledFlags[last];
        mColliders.frictionRestitution[index]   = mColliders.frictionRestitution[last];
        mColliders.collisionLayers[index]       = mColliders.collisionLayers[last];
        mColliders.collisionMasks[index]        = mColliders.collisionMasks[last];
        mColliderIdToIndex[moved.colliderId]    = index;
        markColliderDirty(index);
    }

    mColliderSnapshot.pop_back();
    mColliders.colliderIds.pop_back();
    mColliders.entityIds.pop_back();
    mColliders.ownerRigidBodyIds.pop_back();
    mColliders.ownerRigidBodyIndices.pop_back();
    mColliders.environmentIndices.pop_back();
    mColliders.shapeTypes.pop_back();
    mColliders.shapeParams.pop_back();
    mColliders.localPositions.pop_back();
    mColliders.localOrientations.pop_back();
    mColliders.enabledFlags.pop_back();
    mColliders.frictionRestitution.pop_back();
    mColliders.collisionLayers.pop_back();
    mColliders.collisionMasks.pop_back();
}

std::uint32_t PhysicsWorld::broadPhaseContributionForCollider(
    const ColliderState &collider) const noexcept
{
    const auto bodyIt = mRigidBodyIdToIndex.find(collider.ownerRigidBodyId);
    const RigidBodyState *owner =
        bodyIt != mRigidBodyIdToIndex.end() ? &mRigidBodySnapshot[bodyIt->second] : nullptr;
    return colliderBroadPhaseContribution(collider, owner);
}

std::uint32_t PhysicsWorld::enabledColliderCountForEntity(common::EntityId entityId) const noexcept
{
    const auto handlesIt = mEntityToColliderIds.find(entityId);
    if (handlesIt == mEntityToColliderIds.end())
    {
        return 0u;
    }

    std::uint32_t count = 0u;
    for (const ColliderId colliderId : handlesIt->second)
    {
        const auto colliderIt = mColliderIdToIndex.find(colliderId);
        if (colliderIt == mColliderIdToIndex.end())
        {
            continue;
        }
        if (mColliderSnapshot[colliderIt->second].enabled)
        {
            ++count;
        }
    }
    return count;
}

bool PhysicsWorld::pruneRigidJointsForBody(RigidBodyId rigidBodyId) noexcept
{
    const auto removeByBody = [rigidBodyId](auto &container) noexcept
    {
        const auto originalSize = container.size();
        container.erase(
            std::remove_if(container.begin(), container.end(), [rigidBodyId](const auto &joint)
                           { return joint.bodyA == rigidBodyId || joint.bodyB == rigidBodyId; }),
            container.end());
        return container.size() != originalSize;
    };

    const bool removedBall   = removeByBody(mBallJointSnapshot);
    const bool removedHinge  = removeByBody(mHingeJointSnapshot);
    const bool removedSlider = removeByBody(mSliderJointSnapshot);
    return removedBall || removedHinge || removedSlider;
}

void PhysicsWorld::rebuildBodyColliderMapping() const noexcept
{
    mBodyColliderMapping.colliderOffsets.assign(rigidBodyCount(), 0u);
    mBodyColliderMapping.colliderCounts.assign(rigidBodyCount(), 0u);
    mBodyColliderMapping.colliderIndices.assign(colliderCount(), 0u);

    for (std::uint32_t colliderIndex = 0; colliderIndex < colliderCount(); ++colliderIndex)
    {
        const RigidBodyId ownerRigidBodyId = mColliderSnapshot[colliderIndex].ownerRigidBodyId;
        const auto bodyIt                  = mRigidBodyIdToIndex.find(ownerRigidBodyId);
        const std::uint32_t bodyIndex =
            bodyIt != mRigidBodyIdToIndex.end() ? bodyIt->second : 0xffffffffu;
        mColliders.ownerRigidBodyIndices[colliderIndex] = bodyIndex;
        mColliders.environmentIndices[colliderIndex] =
            bodyIndex != 0xffffffffu ? mRigidBodySnapshot[bodyIndex].environmentIndex : 0u;
        if (bodyIndex != 0xffffffffu)
        {
            ++mBodyColliderMapping.colliderCounts[bodyIndex];
        }
    }

    std::uint32_t runningOffset = 0u;
    for (std::uint32_t bodyIndex = 0; bodyIndex < rigidBodyCount(); ++bodyIndex)
    {
        mBodyColliderMapping.colliderOffsets[bodyIndex] = runningOffset;
        runningOffset += mBodyColliderMapping.colliderCounts[bodyIndex];
    }

    std::vector<std::uint32_t> cursor = mBodyColliderMapping.colliderOffsets;
    for (std::uint32_t colliderIndex = 0; colliderIndex < colliderCount(); ++colliderIndex)
    {
        const std::uint32_t bodyIndex = mColliders.ownerRigidBodyIndices[colliderIndex];
        if (bodyIndex == 0xffffffffu || bodyIndex >= rigidBodyCount())
        {
            continue;
        }
        mBodyColliderMapping.colliderIndices[cursor[bodyIndex]++] = colliderIndex;
    }
}

void PhysicsWorld::rebuildRigidJointScene() const noexcept
{
    auto *self = const_cast<PhysicsWorld *>(this);
    self->mRigidJointScene.clear();

    auto appendBall = [&](const BallJointState &joint)
    {
        const auto bodyAIt = self->mRigidBodyIdToIndex.find(joint.bodyA);
        const auto bodyBIt = self->mRigidBodyIdToIndex.find(joint.bodyB);
        if (bodyAIt == self->mRigidBodyIdToIndex.end() ||
            bodyBIt == self->mRigidBodyIdToIndex.end())
        {
            return;
        }
        const std::uint32_t bodyIndexA = bodyAIt->second;
        const std::uint32_t bodyIndexB = bodyBIt->second;
        if (self->mRigidBodySnapshot[bodyIndexA].environmentIndex !=
            self->mRigidBodySnapshot[bodyIndexB].environmentIndex)
        {
            return;
        }

        self->mRigidJointScene.ball.bodyIndicesA.push_back(bodyIndexA);
        self->mRigidJointScene.ball.bodyIndicesB.push_back(bodyIndexB);
        self->mRigidJointScene.ball.enabledFlags.push_back(joint.enabled ? 1u : 0u);
        self->mRigidJointScene.ball.localAnchorsA.push_back(toFloat4(joint.localAnchorA, 0.0f));
        self->mRigidJointScene.ball.localAnchorsB.push_back(toFloat4(joint.localAnchorB, 0.0f));
    };

    auto appendHinge = [&](const HingeJointState &joint)
    {
        const auto bodyAIt = self->mRigidBodyIdToIndex.find(joint.bodyA);
        const auto bodyBIt = self->mRigidBodyIdToIndex.find(joint.bodyB);
        if (bodyAIt == self->mRigidBodyIdToIndex.end() ||
            bodyBIt == self->mRigidBodyIdToIndex.end())
        {
            return;
        }
        const std::uint32_t bodyIndexA = bodyAIt->second;
        const std::uint32_t bodyIndexB = bodyBIt->second;
        if (self->mRigidBodySnapshot[bodyIndexA].environmentIndex !=
            self->mRigidBodySnapshot[bodyIndexB].environmentIndex)
        {
            return;
        }

        Diligent::float4 projectionRows[3];
        computeProjectionRowsFromLocalFrames(joint.localRotationA, joint.localRotationB, 1u, 3u,
                                             projectionRows);
        const Diligent::float3 axisA0 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{1.0f, 0.0f, 0.0f}),
            Diligent::float3{1.0f, 0.0f, 0.0f});

        self->mRigidJointScene.hinge.bodyIndicesA.push_back(bodyIndexA);
        self->mRigidJointScene.hinge.bodyIndicesB.push_back(bodyIndexB);
        self->mRigidJointScene.hinge.enabledFlags.push_back(joint.enabled ? 1u : 0u);
        self->mRigidJointScene.hinge.driveModes.push_back(
            static_cast<std::uint32_t>(joint.driveMode));
        self->mRigidJointScene.hinge.localAnchorsA.push_back(toFloat4(joint.localAnchorA, 0.0f));
        self->mRigidJointScene.hinge.localAnchorsB.push_back(toFloat4(joint.localAnchorB, 0.0f));
        self->mRigidJointScene.hinge.localAxesA0.push_back(toFloat4(axisA0, 0.0f));
        self->mRigidJointScene.hinge.limitEnabledFlags.push_back(joint.limitEnabled ? 1u : 0u);
        self->mRigidJointScene.hinge.limitMins.push_back(joint.limitMin);
        self->mRigidJointScene.hinge.limitMaxs.push_back(joint.limitMax);
        self->mRigidJointScene.hinge.constraintCompliances.push_back(joint.constraintCompliance);
        self->mRigidJointScene.hinge.driveCompliances.push_back(joint.driveCompliance);
        self->mRigidJointScene.hinge.driveTargetAngles.push_back(joint.driveTargetAngle);
        self->mRigidJointScene.hinge.driveTargetAngularVelocities.push_back(
            joint.driveTargetAngularVelocity);
        self->mRigidJointScene.hinge.projectionRow0.push_back(projectionRows[0]);
        self->mRigidJointScene.hinge.projectionRow1.push_back(projectionRows[1]);
        self->mRigidJointScene.hinge.projectionRow2.push_back(projectionRows[2]);
    };

    auto appendSlider = [&](const SliderJointState &joint)
    {
        const auto bodyAIt = self->mRigidBodyIdToIndex.find(joint.bodyA);
        const auto bodyBIt = self->mRigidBodyIdToIndex.find(joint.bodyB);
        if (bodyAIt == self->mRigidBodyIdToIndex.end() ||
            bodyBIt == self->mRigidBodyIdToIndex.end())
        {
            return;
        }
        const std::uint32_t bodyIndexA = bodyAIt->second;
        const std::uint32_t bodyIndexB = bodyBIt->second;
        if (self->mRigidBodySnapshot[bodyIndexA].environmentIndex !=
            self->mRigidBodySnapshot[bodyIndexB].environmentIndex)
        {
            return;
        }

        const Diligent::float3 axisA0 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{1.0f, 0.0f, 0.0f}),
            Diligent::float3{1.0f, 0.0f, 0.0f});
        const Diligent::float3 axisA1 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{0.0f, 1.0f, 0.0f}),
            Diligent::float3{0.0f, 1.0f, 0.0f});
        const Diligent::float3 axisA2 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{0.0f, 0.0f, 1.0f}),
            Diligent::float3{0.0f, 0.0f, 1.0f});
        const Diligent::float3 worldAnchorA =
            self->mRigidBodySnapshot[bodyIndexA].position +
            quaternionRotate(self->mRigidBodySnapshot[bodyIndexA].rotation, joint.localAnchorA);
        const Diligent::float3 worldAnchorB =
            self->mRigidBodySnapshot[bodyIndexB].position +
            quaternionRotate(self->mRigidBodySnapshot[bodyIndexB].rotation, joint.localAnchorB);
        const float driveRestOffset = Diligent::dot(worldAnchorA - worldAnchorB, axisA0);
        Diligent::float4 projectionRows[3];
        computeProjectionRowsFromLocalFrames(joint.localRotationA, joint.localRotationB, 1u, 3u,
                                             projectionRows);
        self->mRigidJointScene.slider.bodyIndicesA.push_back(bodyIndexA);
        self->mRigidJointScene.slider.bodyIndicesB.push_back(bodyIndexB);
        self->mRigidJointScene.slider.enabledFlags.push_back(joint.enabled ? 1u : 0u);
        self->mRigidJointScene.slider.driveModes.push_back(
            static_cast<std::uint32_t>(joint.driveMode));
        self->mRigidJointScene.slider.localAnchorsA.push_back(toFloat4(joint.localAnchorA, 0.0f));
        self->mRigidJointScene.slider.localAnchorsB.push_back(toFloat4(joint.localAnchorB, 0.0f));
        self->mRigidJointScene.slider.limitEnabledFlags.push_back(joint.limitEnabled ? 1u : 0u);
        self->mRigidJointScene.slider.limitMins.push_back(joint.limitMin);
        self->mRigidJointScene.slider.limitMaxs.push_back(joint.limitMax);
        self->mRigidJointScene.slider.constraintCompliances.push_back(joint.constraintCompliance);
        self->mRigidJointScene.slider.driveCompliances.push_back(joint.driveCompliance);
        self->mRigidJointScene.slider.driveTargetPositions.push_back(joint.driveTargetPosition);
        self->mRigidJointScene.slider.driveTargetVelocities.push_back(joint.driveTargetVelocity);
        self->mRigidJointScene.slider.driveRestOffsets.push_back(driveRestOffset);
        self->mRigidJointScene.slider.localAxesA0.push_back(toFloat4(axisA0, 0.0f));
        self->mRigidJointScene.slider.localAxesA1.push_back(toFloat4(axisA1, 0.0f));
        self->mRigidJointScene.slider.localAxesA2.push_back(toFloat4(axisA2, 0.0f));
        self->mRigidJointScene.slider.projectionRow0.push_back(projectionRows[0]);
        self->mRigidJointScene.slider.projectionRow1.push_back(projectionRows[1]);
        self->mRigidJointScene.slider.projectionRow2.push_back(projectionRows[2]);
    };

    for (const BallJointState &joint : self->mBallJointSnapshot)
    {
        appendBall(joint);
    }
    for (const HingeJointState &joint : self->mHingeJointSnapshot)
    {
        appendHinge(joint);
    }
    for (const SliderJointState &joint : self->mSliderJointSnapshot)
    {
        appendSlider(joint);
    }
}

void PhysicsWorld::rebuildJointCollisionSuppression() const noexcept
{
    auto *self = const_cast<PhysicsWorld *>(this);
    self->mJointCollisionSuppression.clear();
    self->mJointCollisionSuppression.neighborOffsets.assign(rigidBodyCount() + 1u, 0u);
    if (rigidBodyCount() == 0u)
    {
        return;
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> directedPairs;
    directedPairs.reserve(
        (mBallJointSnapshot.size() + mHingeJointSnapshot.size() + mSliderJointSnapshot.size()) *
        2u);

    const auto appendPair = [&](RigidBodyId bodyAId, RigidBodyId bodyBId)
    {
        const auto bodyAIt = self->mRigidBodyIdToIndex.find(bodyAId);
        const auto bodyBIt = self->mRigidBodyIdToIndex.find(bodyBId);
        if (bodyAIt == self->mRigidBodyIdToIndex.end() ||
            bodyBIt == self->mRigidBodyIdToIndex.end())
        {
            return;
        }

        const std::uint32_t bodyIndexA = bodyAIt->second;
        const std::uint32_t bodyIndexB = bodyBIt->second;
        if (bodyIndexA == bodyIndexB || self->mRigidBodySnapshot[bodyIndexA].environmentIndex !=
                                            self->mRigidBodySnapshot[bodyIndexB].environmentIndex)
        {
            return;
        }

        directedPairs.emplace_back(bodyIndexA, bodyIndexB);
        directedPairs.emplace_back(bodyIndexB, bodyIndexA);
    };

    for (const BallJointState &joint : mBallJointSnapshot)
    {
        if (joint.enabled && joint.suppressConnectedBodyCollisions)
        {
            appendPair(joint.bodyA, joint.bodyB);
        }
    }
    for (const HingeJointState &joint : mHingeJointSnapshot)
    {
        if (joint.enabled && joint.suppressConnectedBodyCollisions)
        {
            appendPair(joint.bodyA, joint.bodyB);
        }
    }
    for (const SliderJointState &joint : mSliderJointSnapshot)
    {
        if (joint.enabled && joint.suppressConnectedBodyCollisions)
        {
            appendPair(joint.bodyA, joint.bodyB);
        }
    }

    std::sort(directedPairs.begin(), directedPairs.end());
    directedPairs.erase(std::unique(directedPairs.begin(), directedPairs.end()),
                        directedPairs.end());

    for (const auto &[bodyIndexA, bodyIndexB] : directedPairs)
    {
        ++self->mJointCollisionSuppression.neighborOffsets[bodyIndexA + 1u];
        self->mJointCollisionSuppression.neighbors.push_back(bodyIndexB);
    }

    for (std::uint32_t bodyIndex = 0u; bodyIndex < rigidBodyCount(); ++bodyIndex)
    {
        self->mJointCollisionSuppression.neighborOffsets[bodyIndex + 1u] +=
            self->mJointCollisionSuppression.neighborOffsets[bodyIndex];
    }
}

void PhysicsWorld::rebuildSoftBodyDerivedState() noexcept
{
    mParticles.clear();
    mParticleContactMaterials.clear();
    mFluidMaterials.clear();
    mSoftEdges.clear();
    mSoftBends.clear();
    mSoftTets.clear();
    mStrandSegments.clear();
    mStrandJoints.clear();
    mStrandDistanceConstraints.clear();
    mStrandSegmentStates.clear();
    mRigidParticleAttachments.clear();
    mStrandRigidAttachments.clear();
    mRigidDistanceConstraints.clear();
    mRoutedCableConstraints.clear();
    mRoutedCableRoutePoints.clear();
    std::vector<std::vector<std::uint32_t>> adjacencyLists;

    const std::uint32_t softBodyPhaseGroupBase = 0u;
    const std::uint32_t strandPhaseGroupBase = static_cast<std::uint32_t>(mSoftBodySnapshot.size());
    const std::uint32_t fluidPhaseGroupBase =
        strandPhaseGroupBase + static_cast<std::uint32_t>(mStrandSnapshot.size());
    const std::uint32_t rigidProxyPhaseGroupBase =
        fluidPhaseGroupBase + static_cast<std::uint32_t>(mFluidSnapshot.size());
    const std::uint32_t rigidProxySuturingGroupBase =
        static_cast<std::uint32_t>(mStrandSnapshot.size());

    for (std::uint32_t rigidBodyIndex = 0u; rigidBodyIndex < mRigidBodySnapshot.size();
         ++rigidBodyIndex)
    {
        RigidBodyState &rigidBody = mRigidBodySnapshot[rigidBodyIndex];
        normalizeRigidBodyState(rigidBody);
        rigidBody.proxyParticleOffset          = static_cast<std::uint32_t>(mParticles.size());
        rigidBody.proxyParticleCount           = 0u;
        rigidBody.proxyParticleMaterialIndex   = 0u;
        rigidBody.proxyParticleContactMaterial = Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (rigidBodyIndex < mRigidBodies.proxyParticleContactMaterials.size())
        {
            if (mRigidBodies.proxyParticleContactMaterials[rigidBodyIndex] !=
                rigidBody.proxyParticleContactMaterial)
            {
                mRigidBodies.proxyParticleContactMaterials[rigidBodyIndex] =
                    rigidBody.proxyParticleContactMaterial;
                markRigidBodyDirty(rigidBodyIndex);
            }
        }

        if (rigidBody.proxyParticleLocalPositions.empty() || rigidBody.proxyParticleRadius <= 0.0f)
        {
            continue;
        }

        rigidBody.proxyParticleContactMaterial =
            toParticleContactMaterial(rigidBody.proxyParticleMaterial);
        rigidBody.proxyParticleMaterialIndex = findOrAppendParticleContactMaterial(
            mParticleContactMaterials, rigidBody.proxyParticleContactMaterial);
        if (rigidBodyIndex < mRigidBodies.proxyParticleContactMaterials.size())
        {
            if (mRigidBodies.proxyParticleContactMaterials[rigidBodyIndex] !=
                rigidBody.proxyParticleContactMaterial)
            {
                mRigidBodies.proxyParticleContactMaterials[rigidBodyIndex] =
                    rigidBody.proxyParticleContactMaterial;
                markRigidBodyDirty(rigidBodyIndex);
            }
        }

        for (std::uint32_t proxyLocalIndex = 0u;
             proxyLocalIndex < rigidBody.proxyParticleLocalPositions.size(); ++proxyLocalIndex)
        {
            const Diligent::float3 &localPosition =
                rigidBody.proxyParticleLocalPositions[proxyLocalIndex];
            const Diligent::float3 scaledLocal{localPosition.x * rigidBody.scale.x,
                                               localPosition.y * rigidBody.scale.y,
                                               localPosition.z * rigidBody.scale.z};
            const Diligent::float3 worldPosition =
                rigidBody.rotation.RotateVector(scaledLocal) + rigidBody.position;
            mParticles.positionsInvMass.push_back(
                Diligent::float4{worldPosition.x, worldPosition.y, worldPosition.z, 0.0f});
            mParticles.previousPositions.push_back(
                Diligent::float4{worldPosition.x, worldPosition.y, worldPosition.z, 0.0f});
            mParticles.velocities.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            mParticles.radii.push_back(rigidBody.proxyParticleRadius);
            mParticles.environmentIndices.push_back(rigidBody.environmentIndex);
            mParticles.particleKinds.push_back(static_cast<std::uint32_t>(ParticleKind::SoftSolid));
            mParticles.ownerTypes.push_back(
                static_cast<std::uint32_t>(ParticleOwnerType::RigidBody));
            mParticles.ownerIndices.push_back(rigidBodyIndex);
            mParticles.strandIds.push_back(rigidBody.suturingEnabled
                                               ? rigidProxySuturingGroupBase + rigidBodyIndex
                                               : 0xffffffffu);
            mParticles.strandOrders.push_back(rigidBody.suturingEnabled ? proxyLocalIndex
                                                                        : 0xffffffffu);
            mParticles.strandRoles.push_back(static_cast<std::uint32_t>(
                rigidBody.suturingEnabled ? (proxyLocalIndex == rigidBody.needleTipProxyIndex
                                                 ? ParticleStrandRole::NeedleTip
                                                 : ParticleStrandRole::NeedleBody)
                                          : ParticleStrandRole::None));
            mParticles.suturingNeighborLinks.push_back(Diligent::uint4{
                rigidBody.suturingEnabled && proxyLocalIndex > 0u
                    ? rigidBody.proxyParticleOffset + proxyLocalIndex - 1u
                    : kInvalidSuturingIndex,
                rigidBody.suturingEnabled &&
                        proxyLocalIndex + 1u <
                            static_cast<std::uint32_t>(rigidBody.proxyParticleLocalPositions.size())
                    ? rigidBody.proxyParticleOffset + proxyLocalIndex + 1u
                    : kInvalidSuturingIndex,
                0u, 0u});
            mParticles.owningSoftBodyIndices.push_back(0xffffffffu);
            mParticles.particleMaterialIndices.push_back(rigidBody.proxyParticleMaterialIndex);
            mParticles.fluidMaterialIndices.push_back(0xffffffffu);
            mParticles.phases.push_back(
                packParticlePhase(rigidProxyPhaseGroupBase + rigidBodyIndex, false));
            mParticles.collisionLayers.push_back(rigidBody.proxyCollisionLayer);
            mParticles.collisionMasks.push_back(rigidBody.proxyCollisionMask);
            mParticles.rigidProxyLocalPositions.push_back(
                Diligent::float4{scaledLocal.x, scaledLocal.y, scaledLocal.z, 0.0f});
            adjacencyLists.emplace_back();
            ++rigidBody.proxyParticleCount;
        }
    }

    for (std::uint32_t softBodyIndex = 0u; softBodyIndex < mSoftBodySnapshot.size();
         ++softBodyIndex)
    {
        SoftBodyState &softBody = mSoftBodySnapshot[softBodyIndex];
        normalizeSoftBodyState(softBody);
        softBody.contactMaterialIndex = findOrAppendParticleContactMaterial(
            mParticleContactMaterials, toParticleContactMaterial(softBody.material.contact));

        softBody.particleOffset = static_cast<std::uint32_t>(mParticles.size());
        softBody.edgeOffset     = static_cast<std::uint32_t>(mSoftEdges.size());
        softBody.tetOffset      = static_cast<std::uint32_t>(mSoftTets.size());

        if (softBodyIndex >= mSoftBodyDerivedCaches.size())
        {
            CRESSIM_LOG_ERROR("Failed to rebuild derived soft-body state for entity ",
                              softBody.entityId, ": missing cached resolved topology.");
            softBody.particleCount = 0u;
            softBody.edgeCount     = 0u;
            softBody.tetCount      = 0u;
            continue;
        }
        const SoftBodyDerivedCache &topology = mSoftBodyDerivedCaches[softBodyIndex];

        const float inverseMass =
            softBody.particleMass > 0.0f ? 1.0f / softBody.particleMass : 0.0f;
        std::unordered_set<std::uint32_t> staticParticles(topology.staticParticleIndices.begin(),
                                                          topology.staticParticleIndices.end());
        softBody.restPositions = topology.restPositions;
        softBody.boundaryFaces = topology.boundaryFaces;

        softBody.particleCount = static_cast<std::uint32_t>(topology.restPositions.size());
        for (std::uint32_t localParticleIndex = 0u; localParticleIndex < softBody.particleCount;
             ++localParticleIndex)
        {
            const Diligent::float3 &position = topology.restPositions[localParticleIndex];
            const float particleInvMass =
                staticParticles.find(localParticleIndex) != staticParticles.end() ? 0.0f
                                                                                  : inverseMass;

            mParticles.positionsInvMass.push_back(
                Diligent::float4{position.x, position.y, position.z, particleInvMass});
            mParticles.previousPositions.push_back(
                Diligent::float4{position.x, position.y, position.z, 0.0f});
            mParticles.velocities.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            mParticles.radii.push_back(softBody.particleRadius);
            mParticles.environmentIndices.push_back(softBody.environmentIndex);
            mParticles.particleKinds.push_back(static_cast<std::uint32_t>(ParticleKind::SoftSolid));
            mParticles.ownerTypes.push_back(
                static_cast<std::uint32_t>(ParticleOwnerType::SoftBody));
            mParticles.ownerIndices.push_back(softBodyIndex);
            mParticles.strandIds.push_back(0xffffffffu);
            mParticles.strandOrders.push_back(0xffffffffu);
            mParticles.strandRoles.push_back(static_cast<std::uint32_t>(ParticleStrandRole::None));
            mParticles.suturingNeighborLinks.push_back(
                Diligent::uint4{kInvalidSuturingIndex, kInvalidSuturingIndex, 0u, 0u});
            mParticles.owningSoftBodyIndices.push_back(softBodyIndex);
            mParticles.particleMaterialIndices.push_back(softBody.contactMaterialIndex);
            mParticles.fluidMaterialIndices.push_back(0xffffffffu);
            mParticles.phases.push_back(packParticlePhase(softBodyPhaseGroupBase + softBodyIndex,
                                                          softBody.selfCollisionEnabled));
            mParticles.collisionLayers.push_back(softBody.collisionLayer);
            mParticles.collisionMasks.push_back(softBody.collisionMask);
            mParticles.rigidProxyLocalPositions.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            adjacencyLists.emplace_back();
        }

        for (std::uint32_t localParticleIndex = 0u;
             localParticleIndex < static_cast<std::uint32_t>(topology.adjacencyLists.size());
             ++localParticleIndex)
        {
            auto &globalAdjacency = adjacencyLists[softBody.particleOffset + localParticleIndex];
            globalAdjacency.reserve(topology.adjacencyLists[localParticleIndex].size());
            for (const std::uint32_t localNeighbor : topology.adjacencyLists[localParticleIndex])
            {
                globalAdjacency.push_back(softBody.particleOffset + localNeighbor);
            }
        }

        for (const auto &edgeDesc : topology.edges)
        {
            const std::uint32_t globalA = softBody.particleOffset + edgeDesc[0];
            const std::uint32_t globalB = softBody.particleOffset + edgeDesc[1];
            const Diligent::float3 delta{
                topology.restPositions[edgeDesc[1]].x - topology.restPositions[edgeDesc[0]].x,
                topology.restPositions[edgeDesc[1]].y - topology.restPositions[edgeDesc[0]].y,
                topology.restPositions[edgeDesc[1]].z - topology.restPositions[edgeDesc[0]].z,
            };

            SoftEdge edge{};
            edge.particleA  = globalA;
            edge.particleB  = globalB;
            edge.restLength = std::sqrt(Diligent::dot(delta, delta));
            edge.compliance = softBody.edgeCompliance;
            mSoftEdges.push_back(edge);
        }

        for (const auto &tetDesc : topology.tets)
        {
            const Diligent::float3 &p0 = topology.restPositions[tetDesc[0]];
            const Diligent::float3 &p1 = topology.restPositions[tetDesc[1]];
            const Diligent::float3 &p2 = topology.restPositions[tetDesc[2]];
            const Diligent::float3 &p3 = topology.restPositions[tetDesc[3]];

            SoftTet tet{};
            tet.particleIndices = {
                softBody.particleOffset + tetDesc[0], softBody.particleOffset + tetDesc[1],
                softBody.particleOffset + tetDesc[2], softBody.particleOffset + tetDesc[3]};
            tet.restVolume =
                std::abs(Diligent::dot(Diligent::cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0f;
            tet.compliance = softBody.volumeCompliance;
            mSoftTets.push_back(tet);
        }

        softBody.edgeCount = static_cast<std::uint32_t>(mSoftEdges.size()) - softBody.edgeOffset;
        softBody.tetCount  = static_cast<std::uint32_t>(mSoftTets.size()) - softBody.tetOffset;
    }

    for (std::uint32_t strandIndex = 0u; strandIndex < mStrandSnapshot.size(); ++strandIndex)
    {
        StrandState &strand = mStrandSnapshot[strandIndex];
        normalizeStrandState(strand);
        strand.contactMaterialIndex = findOrAppendParticleContactMaterial(
            mParticleContactMaterials, toParticleContactMaterial(strand.material.contact));

        strand.particleOffset = static_cast<std::uint32_t>(mParticles.size());
        strand.segmentOffset  = static_cast<std::uint32_t>(mStrandSegments.size());
        strand.jointOffset    = static_cast<std::uint32_t>(mStrandJoints.size());

        if (strandIndex >= mStrandDerivedCaches.size())
        {
            strand.particleCount = 0u;
            strand.segmentCount  = 0u;
            strand.jointCount    = 0u;
            continue;
        }

        const StrandDerivedCache &topology = mStrandDerivedCaches[strandIndex];
        const float inverseMass = strand.particleMass > 0.0f ? 1.0f / strand.particleMass : 0.0f;
        std::unordered_set<std::uint32_t> staticParticles(topology.staticParticleIndices.begin(),
                                                          topology.staticParticleIndices.end());
        strand.restPositions = topology.restPositions;
        strand.particleCount = static_cast<std::uint32_t>(topology.restPositions.size());
        for (std::uint32_t localParticleIndex = 0u; localParticleIndex < strand.particleCount;
             ++localParticleIndex)
        {
            const Diligent::float3 &position = topology.restPositions[localParticleIndex];
            const float particleInvMass =
                staticParticles.find(localParticleIndex) != staticParticles.end() ? 0.0f
                                                                                  : inverseMass;
            mParticles.positionsInvMass.push_back(
                Diligent::float4{position.x, position.y, position.z, particleInvMass});
            mParticles.previousPositions.push_back(
                Diligent::float4{position.x, position.y, position.z, 0.0f});
            mParticles.velocities.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            mParticles.radii.push_back(strand.particleRadius);
            mParticles.environmentIndices.push_back(strand.environmentIndex);
            mParticles.particleKinds.push_back(static_cast<std::uint32_t>(ParticleKind::SoftSolid));
            mParticles.ownerTypes.push_back(static_cast<std::uint32_t>(ParticleOwnerType::Strand));
            mParticles.ownerIndices.push_back(strandIndex);
            mParticles.strandIds.push_back(strandIndex);
            mParticles.strandOrders.push_back(localParticleIndex);
            mParticles.strandRoles.push_back(static_cast<std::uint32_t>(
                strand.suturingEnabled ? (localParticleIndex == 0u ? ParticleStrandRole::NeedleTip
                                                                   : ParticleStrandRole::NeedleBody)
                                       : ParticleStrandRole::None));
            mParticles.suturingNeighborLinks.push_back(Diligent::uint4{
                strand.suturingEnabled && localParticleIndex > 0u
                    ? strand.particleOffset + localParticleIndex - 1u
                    : kInvalidSuturingIndex,
                strand.suturingEnabled && localParticleIndex + 1u < strand.particleCount
                    ? strand.particleOffset + localParticleIndex + 1u
                    : kInvalidSuturingIndex,
                0u, 0u});
            mParticles.owningSoftBodyIndices.push_back(0xffffffffu);
            mParticles.particleMaterialIndices.push_back(strand.contactMaterialIndex);
            mParticles.fluidMaterialIndices.push_back(0xffffffffu);
            mParticles.phases.push_back(
                packParticlePhase(strandPhaseGroupBase + strandIndex, strand.selfCollisionEnabled));
            mParticles.collisionLayers.push_back(strand.collisionLayer);
            mParticles.collisionMasks.push_back(strand.collisionMask);
            mParticles.rigidProxyLocalPositions.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            adjacencyLists.emplace_back();
        }

        for (std::uint32_t localParticleIndex = 0u;
             localParticleIndex < static_cast<std::uint32_t>(topology.adjacencyLists.size());
             ++localParticleIndex)
        {
            auto &globalAdjacency = adjacencyLists[strand.particleOffset + localParticleIndex];
            globalAdjacency.reserve(topology.adjacencyLists[localParticleIndex].size());
            for (const std::uint32_t localNeighbor : topology.adjacencyLists[localParticleIndex])
            {
                globalAdjacency.push_back(strand.particleOffset + localNeighbor);
            }
        }

        for (std::uint32_t segmentIndex = 0u;
             segmentIndex < static_cast<std::uint32_t>(topology.segments.size()); ++segmentIndex)
        {
            const auto &segmentDesc = topology.segments[segmentIndex];
            StrandSegmentConstraint segment{};
            segment.particleA              = strand.particleOffset + segmentDesc[0];
            segment.particleB              = strand.particleOffset + segmentDesc[1];
            segment.restLength             = topology.restSegmentLengths[segmentIndex];
            segment.stretchShearCompliance = strand.stretchShearCompliance;
            const Diligent::QuaternionF q  = topology.restSegmentOrientations[segmentIndex];
            segment.restOrientation        = Diligent::float4{q.q.x, q.q.y, q.q.z, q.q.w};
            mStrandSegments.push_back(segment);
            mStrandSegmentStates.push_back(StrandSegmentState{segment.restOrientation});

            StrandDistanceConstraint distanceConstraint{};
            distanceConstraint.particleA = segment.particleA;
            distanceConstraint.particleB = segment.particleB;
            distanceConstraint.restLength = segment.restLength;
            distanceConstraint.distanceCompliance = strand.distanceCompliance;
            mStrandDistanceConstraints.push_back(distanceConstraint);
        }

        for (std::uint32_t jointIndex = 0u;
             jointIndex < static_cast<std::uint32_t>(topology.joints.size()); ++jointIndex)
        {
            const auto &jointDesc = topology.joints[jointIndex];
            StrandJointConstraint joint{};
            joint.segmentA        = strand.segmentOffset + jointDesc[0];
            joint.segmentB        = strand.segmentOffset + jointDesc[1];
            joint.bendCompliance  = strand.bendCompliance;
            joint.twistCompliance = strand.twistCompliance;
            const Diligent::QuaternionF q = topology.restJointRelativeOrientations[jointIndex];
            joint.restRelativeOrientation = Diligent::float4{q.q.x, q.q.y, q.q.z, q.q.w};
            mStrandJoints.push_back(joint);
        }

        strand.segmentCount =
            static_cast<std::uint32_t>(mStrandSegments.size()) - strand.segmentOffset;
        strand.jointCount =
            static_cast<std::uint32_t>(mStrandJoints.size()) - strand.jointOffset;
    }

    for (std::uint32_t fluidIndex = 0u; fluidIndex < mFluidSnapshot.size(); ++fluidIndex)
    {
        FluidState &fluid = mFluidSnapshot[fluidIndex];
        normalizeFluidState(fluid);
        fluid.contactMaterialIndex = findOrAppendParticleContactMaterial(
            mParticleContactMaterials, toParticleContactMaterial(fluid.material.contact));
        fluid.fluidMaterialIndex = findOrAppendFluidMaterial(
            mFluidMaterials, toFluidSolverMaterial(fluid.material, fluid.particleRadius));
        fluid.particleOffset = static_cast<std::uint32_t>(mParticles.size());

        if (fluidIndex >= mFluidDerivedCaches.size())
        {
            fluid.particleCount = 0u;
            continue;
        }

        const FluidDerivedCache &cache = mFluidDerivedCaches[fluidIndex];
        fluid.restPositions            = cache.restPositions;
        fluid.particleCount            = static_cast<std::uint32_t>(cache.restPositions.size());
        const float inverseMass = fluid.particleMass > 0.0f ? 1.0f / fluid.particleMass : 0.0f;
        for (const Diligent::float3 &position : cache.restPositions)
        {
            mParticles.positionsInvMass.push_back(
                Diligent::float4{position.x, position.y, position.z, inverseMass});
            mParticles.previousPositions.push_back(
                Diligent::float4{position.x, position.y, position.z, 0.0f});
            mParticles.velocities.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            mParticles.radii.push_back(fluid.particleRadius);
            mParticles.environmentIndices.push_back(fluid.environmentIndex);
            mParticles.particleKinds.push_back(static_cast<std::uint32_t>(ParticleKind::Fluid));
            mParticles.ownerTypes.push_back(
                static_cast<std::uint32_t>(ParticleOwnerType::FluidBody));
            mParticles.ownerIndices.push_back(fluidIndex);
            mParticles.strandIds.push_back(0xffffffffu);
            mParticles.strandOrders.push_back(0xffffffffu);
            mParticles.strandRoles.push_back(static_cast<std::uint32_t>(ParticleStrandRole::None));
            mParticles.suturingNeighborLinks.push_back(
                Diligent::uint4{kInvalidSuturingIndex, kInvalidSuturingIndex, 0u, 0u});
            mParticles.owningSoftBodyIndices.push_back(0xffffffffu);
            mParticles.particleMaterialIndices.push_back(fluid.contactMaterialIndex);
            mParticles.fluidMaterialIndices.push_back(fluid.fluidMaterialIndex);
            mParticles.phases.push_back(packParticlePhase(fluidPhaseGroupBase + fluidIndex, true));
            mParticles.collisionLayers.push_back(fluid.collisionLayer);
            mParticles.collisionMasks.push_back(fluid.collisionMask);
            mParticles.rigidProxyLocalPositions.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
            adjacencyLists.emplace_back();
        }
    }

    const auto resolveParticleReference =
        [this](const AuthoredParticleReference &reference) -> std::optional<std::uint32_t>
    {
        switch (reference.type)
        {
        case AuthoredParticleReferenceType::SoftBodyParticle:
        {
            const SoftBodyState *softBody = tryGetSoftBody(reference.entityId);
            if (softBody == nullptr || reference.localParticleIndex >= softBody->particleCount)
            {
                return std::nullopt;
            }
            return softBody->particleOffset + reference.localParticleIndex;
        }
        case AuthoredParticleReferenceType::StrandParticle:
        {
            const StrandState *strand = tryGetStrand(reference.entityId);
            if (strand == nullptr || reference.localParticleIndex >= strand->particleCount)
            {
                return std::nullopt;
            }
            return strand->particleOffset + reference.localParticleIndex;
        }
        case AuthoredParticleReferenceType::RigidProxyParticle:
        {
            const RigidBodyState *rigidBody = tryGetRigidBody(reference.entityId);
            if (rigidBody == nullptr ||
                reference.localParticleIndex >= rigidBody->proxyParticleCount)
            {
                return std::nullopt;
            }
            return rigidBody->proxyParticleOffset + reference.localParticleIndex;
        }
        }

        return std::nullopt;
    };

    const auto resolveRigidBodyIndex =
        [this](common::EntityId entityId) -> std::optional<std::uint32_t>
    {
        const auto it = mEntityToRigidBodyIndex.find(entityId);
        return it == mEntityToRigidBodyIndex.end() ? std::nullopt : std::optional(it->second);
    };

    const auto resolveStrandSegmentIndex =
        [this](common::EntityId entityId, std::uint32_t localSegmentIndex)
        -> std::optional<std::uint32_t>
    {
        const StrandState *strand = tryGetStrand(entityId);
        if (strand == nullptr || localSegmentIndex >= strand->segmentCount)
        {
            return std::nullopt;
        }
        return strand->segmentOffset + localSegmentIndex;
    };

    for (const AuthoredParticleCollisionFilterState &authoredFilter :
         mParticleCollisionFilterSnapshot)
    {
        if (!authoredFilter.enabled)
        {
            continue;
        }

        const std::optional<std::uint32_t> particleIndex =
            resolveParticleReference(authoredFilter.particle);
        if (!particleIndex.has_value() || *particleIndex >= mParticles.collisionLayers.size() ||
            *particleIndex >= mParticles.collisionMasks.size())
        {
            continue;
        }

        mParticles.collisionLayers[*particleIndex] = authoredFilter.collisionLayer;
        mParticles.collisionMasks[*particleIndex]  = authoredFilter.collisionMask;
    }

    struct ResolvedSuturingSequence
    {
        const AuthoredSuturingSequenceState *authored = nullptr;
        std::vector<std::uint32_t> particleIndices{};
        std::uint32_t environmentIndex = 0u;
        std::uint32_t tipEntryIndex    = 0u;
        float pathNodeSpacing          = 0.0f;
        float needleTangentialDrag     = 0.0f;
        float threadTangentialDrag     = 0.0f;
        std::uint32_t groupId          = kInvalidSuturingIndex;
    };

    const std::uint32_t authoredSequenceSuturingGroupBase =
        rigidProxySuturingGroupBase + static_cast<std::uint32_t>(mRigidBodySnapshot.size());
    std::vector<ResolvedSuturingSequence> resolvedSequences{};
    resolvedSequences.reserve(mSuturingSequenceSnapshot.size());
    std::unordered_set<common::EntityId> sequenceRigidEntities{};
    std::unordered_set<common::EntityId> sequenceStrandEntities{};

    for (std::uint32_t sequenceIndex = 0u;
         sequenceIndex < static_cast<std::uint32_t>(mSuturingSequenceSnapshot.size());
         ++sequenceIndex)
    {
        const AuthoredSuturingSequenceState &sequence = mSuturingSequenceSnapshot[sequenceIndex];
        if (!sequence.enabled || sequence.entries.empty())
        {
            continue;
        }

        ResolvedSuturingSequence resolved{};
        resolved.authored      = &sequence;
        resolved.tipEntryIndex = std::min<std::uint32_t>(
            sequence.tipEntryIndex, static_cast<std::uint32_t>(sequence.entries.size() - 1u));
        resolved.groupId = authoredSequenceSuturingGroupBase + sequenceIndex;
        resolved.particleIndices.reserve(sequence.entries.size());

        bool validSequence = true;
        std::optional<std::uint32_t> environmentIndex;
        std::unordered_set<common::EntityId> resolvedRigidEntities{};
        std::unordered_set<common::EntityId> resolvedStrandEntities{};
        for (const AuthoredParticleReference &entry : sequence.entries)
        {
            if (entry.type == AuthoredParticleReferenceType::SoftBodyParticle)
            {
                validSequence = false;
                break;
            }

            const auto particleIndex = resolveParticleReference(entry);
            if (!particleIndex.has_value() ||
                *particleIndex >= mParticles.environmentIndices.size())
            {
                validSequence = false;
                break;
            }

            const std::uint32_t entryEnvironmentIndex =
                mParticles.environmentIndices[*particleIndex];
            if (!environmentIndex.has_value())
            {
                environmentIndex = entryEnvironmentIndex;
            }
            else if (*environmentIndex != entryEnvironmentIndex)
            {
                validSequence = false;
                break;
            }

            resolved.particleIndices.push_back(*particleIndex);
            if (entry.type == AuthoredParticleReferenceType::RigidProxyParticle)
            {
                resolvedRigidEntities.insert(entry.entityId);
            }
            else if (entry.type == AuthoredParticleReferenceType::StrandParticle)
            {
                resolvedStrandEntities.insert(entry.entityId);
            }
        }

        if (!validSequence || resolved.particleIndices.empty() ||
            resolved.tipEntryIndex >= resolved.particleIndices.size() ||
            !environmentIndex.has_value())
        {
            continue;
        }

        resolved.environmentIndex     = *environmentIndex;
        resolved.needleTangentialDrag = sequence.needleTangentialDrag;
        resolved.threadTangentialDrag = sequence.threadTangentialDrag;
        if (sequence.pathNodeSpacing > 0.0f)
        {
            resolved.pathNodeSpacing = sequence.pathNodeSpacing;
        }
        else
        {
            const AuthoredParticleReference &tipReference =
                sequence.entries[resolved.tipEntryIndex];
            if (tipReference.type == AuthoredParticleReferenceType::RigidProxyParticle)
            {
                const RigidBodyState *rigidBody = tryGetRigidBody(tipReference.entityId);
                resolved.pathNodeSpacing =
                    rigidBody != nullptr ? std::max(rigidBody->proxyParticleRadius * 1.5f, 1.0e-4f)
                                         : 0.2f;
            }
            else
            {
                const StrandState *strand = tryGetStrand(tipReference.entityId);
                resolved.pathNodeSpacing =
                    strand != nullptr ? std::max(strand->pathNodeSpacing, 1.0e-4f) : 0.2f;
            }
        }

        sequenceRigidEntities.insert(resolvedRigidEntities.begin(), resolvedRigidEntities.end());
        sequenceStrandEntities.insert(resolvedStrandEntities.begin(), resolvedStrandEntities.end());
        resolvedSequences.push_back(std::move(resolved));
    }

    for (std::uint32_t rigidBodyIndex = 0u; rigidBodyIndex < mRigidBodySnapshot.size();
         ++rigidBodyIndex)
    {
        const RigidBodyState &rigidBody = mRigidBodySnapshot[rigidBodyIndex];
        if (sequenceRigidEntities.find(rigidBody.entityId) == sequenceRigidEntities.end())
        {
            continue;
        }

        const std::uint32_t end =
            std::min(rigidBody.proxyParticleOffset + rigidBody.proxyParticleCount,
                     static_cast<std::uint32_t>(mParticles.strandIds.size()));
        for (std::uint32_t particleIndex = rigidBody.proxyParticleOffset; particleIndex < end;
             ++particleIndex)
        {
            mParticles.strandIds[particleIndex]    = kInvalidSuturingIndex;
            mParticles.strandOrders[particleIndex] = kInvalidSuturingIndex;
            mParticles.strandRoles[particleIndex] =
                static_cast<std::uint32_t>(ParticleStrandRole::None);
        }
    }

    for (std::uint32_t strandIndex = 0u; strandIndex < mStrandSnapshot.size(); ++strandIndex)
    {
        const StrandState &strand = mStrandSnapshot[strandIndex];
        if (sequenceStrandEntities.find(strand.entityId) == sequenceStrandEntities.end())
        {
            continue;
        }

        const std::uint32_t end = std::min(strand.particleOffset + strand.particleCount,
                                           static_cast<std::uint32_t>(mParticles.strandIds.size()));
        for (std::uint32_t particleIndex = strand.particleOffset; particleIndex < end;
             ++particleIndex)
        {
            mParticles.strandIds[particleIndex]    = kInvalidSuturingIndex;
            mParticles.strandOrders[particleIndex] = kInvalidSuturingIndex;
            mParticles.strandRoles[particleIndex] =
                static_cast<std::uint32_t>(ParticleStrandRole::None);
        }
    }

    for (const ResolvedSuturingSequence &sequence : resolvedSequences)
    {
        for (std::uint32_t entryIndex = 0u;
             entryIndex < static_cast<std::uint32_t>(sequence.particleIndices.size()); ++entryIndex)
        {
            const std::uint32_t particleIndex = sequence.particleIndices[entryIndex];
            if (particleIndex >= mParticles.strandIds.size())
            {
                continue;
            }

            mParticles.strandIds[particleIndex]    = sequence.groupId;
            mParticles.strandOrders[particleIndex] = entryIndex;
            const AuthoredParticleReference &entry = sequence.authored->entries[entryIndex];
            const ParticleStrandRole role =
                entryIndex == sequence.tipEntryIndex
                    ? ParticleStrandRole::NeedleTip
                    : (entry.type == AuthoredParticleReferenceType::RigidProxyParticle
                           ? ParticleStrandRole::NeedleBody
                           : ParticleStrandRole::Thread);
            mParticles.strandRoles[particleIndex]           = static_cast<std::uint32_t>(role);
            mParticles.suturingNeighborLinks[particleIndex] = Diligent::uint4{
                entryIndex > 0u ? sequence.particleIndices[entryIndex - 1u] : kInvalidSuturingIndex,
                entryIndex + 1u < sequence.particleIndices.size()
                    ? sequence.particleIndices[entryIndex + 1u]
                    : kInvalidSuturingIndex,
                0u, 0u};
        }
    }

    mParticles.suturingParticleIndices.clear();
    mParticles.suturingParticleIndices.reserve(mParticles.size());
    for (std::uint32_t particleIndex = 0u;
         particleIndex < static_cast<std::uint32_t>(mParticles.strandRoles.size()); ++particleIndex)
    {
        if (mParticles.strandRoles[particleIndex] ==
            static_cast<std::uint32_t>(ParticleStrandRole::None))
        {
            mParticles.suturingNeighborLinks[particleIndex].z = kInvalidSuturingIndex;
            mParticles.suturingNeighborLinks[particleIndex].w = 0u;
            continue;
        }

        mParticles.suturingNeighborLinks[particleIndex].z =
            static_cast<std::uint32_t>(mParticles.suturingParticleIndices.size());
        mParticles.suturingNeighborLinks[particleIndex].w = 0u;
        mParticles.suturingParticleIndices.push_back(particleIndex);
    }

    for (const AuthoredParticleDistanceConstraintState &constraint :
         mParticleDistanceConstraintSnapshot)
    {
        if (!constraint.enabled)
        {
            continue;
        }

        const auto particleA = resolveParticleReference(constraint.particleA);
        const auto particleB = resolveParticleReference(constraint.particleB);
        if (!particleA.has_value() || !particleB.has_value() || *particleA == *particleB)
        {
            continue;
        }

        if (*particleA >= mParticles.environmentIndices.size() ||
            *particleB >= mParticles.environmentIndices.size() ||
            mParticles.environmentIndices[*particleA] != mParticles.environmentIndices[*particleB])
        {
            continue;
        }

        DeformableDistanceConstraint resolved{};
        resolved.particleA  = *particleA;
        resolved.particleB  = *particleB;
        resolved.restLength = constraint.restLength;
        resolved.compliance = constraint.compliance;
        mSoftEdges.push_back(resolved);
    }

    for (const AuthoredRigidParticleAttachmentConstraintState &constraint :
         mRigidParticleAttachmentConstraintSnapshot)
    {
        if (!constraint.enabled)
        {
            continue;
        }

        const auto particleIndex = resolveParticleReference(constraint.particle);
        const std::optional<std::uint32_t> rigidBodyIndex =
            resolveRigidBodyIndex(constraint.rigidBodyEntityId);
        if (!particleIndex.has_value() || !rigidBodyIndex.has_value() ||
            *particleIndex >= mParticles.environmentIndices.size() ||
            *rigidBodyIndex >= mRigidBodySnapshot.size())
        {
            continue;
        }

        if (mParticles.environmentIndices[*particleIndex] !=
            mRigidBodySnapshot[*rigidBodyIndex].environmentIndex)
        {
            continue;
        }

        mRigidParticleAttachments.push_back(RigidParticleAttachmentConstraint{
            *particleIndex,
            *rigidBodyIndex,
            constraint.compliance,
            0u,
            Diligent::float4{constraint.localAnchor.x, constraint.localAnchor.y,
                             constraint.localAnchor.z, 0.0f},
        });
    }

    for (const AuthoredStrandRigidAttachmentConstraintState &constraint :
         mStrandRigidAttachmentConstraintSnapshot)
    {
        if (!constraint.enabled)
        {
            continue;
        }

        const std::optional<std::uint32_t> segmentIndex =
            resolveStrandSegmentIndex(constraint.strandEntityId, constraint.localSegmentIndex);
        const std::optional<std::uint32_t> rigidBodyIndex =
            resolveRigidBodyIndex(constraint.rigidBodyEntityId);
        if (!segmentIndex.has_value() || !rigidBodyIndex.has_value() ||
            *segmentIndex >= mStrandSegments.size() || *rigidBodyIndex >= mRigidBodySnapshot.size())
        {
            continue;
        }

        const StrandSegmentConstraint &segment = mStrandSegments[*segmentIndex];
        if (segment.particleA >= mParticles.environmentIndices.size() ||
            *rigidBodyIndex >= mRigidBodySnapshot.size())
        {
            continue;
        }

        if (mParticles.environmentIndices[segment.particleA] !=
            mRigidBodySnapshot[*rigidBodyIndex].environmentIndex)
        {
            continue;
        }

        const Diligent::QuaternionF localRotation =
            common::runtime_math::normalizeQuaternion(constraint.localRotation);
        mStrandRigidAttachments.push_back(StrandRigidAttachmentConstraint{
            *segmentIndex,
            *rigidBodyIndex,
            std::clamp(constraint.segmentT, 0.0f, 1.0f),
            constraint.translationCompliance,
            constraint.rotationCompliance,
            0u,
            0u,
            0u,
            Diligent::float4{constraint.localAnchor.x, constraint.localAnchor.y,
                             constraint.localAnchor.z, 0.0f},
            Diligent::float4{localRotation.q.x, localRotation.q.y, localRotation.q.z,
                             localRotation.q.w},
        });
    }

    for (const AuthoredRigidDistanceConstraintState &constraint :
         mRigidDistanceConstraintSnapshot)
    {
        if (!constraint.enabled)
        {
            continue;
        }

        const std::optional<std::uint32_t> rigidBodyIndexA = resolveRigidBodyIndex(constraint.entityA);
        const std::optional<std::uint32_t> rigidBodyIndexB = resolveRigidBodyIndex(constraint.entityB);
        if (!rigidBodyIndexA.has_value() || !rigidBodyIndexB.has_value() ||
            *rigidBodyIndexA == *rigidBodyIndexB ||
            *rigidBodyIndexA >= mRigidBodySnapshot.size() ||
            *rigidBodyIndexB >= mRigidBodySnapshot.size())
        {
            continue;
        }

        const RigidBodyState &bodyA = mRigidBodySnapshot[*rigidBodyIndexA];
        const RigidBodyState &bodyB = mRigidBodySnapshot[*rigidBodyIndexB];
        if (bodyA.environmentIndex != bodyB.environmentIndex)
        {
            continue;
        }

        mRigidDistanceConstraints.push_back(RigidDistanceConstraint{
            *rigidBodyIndexA,
            *rigidBodyIndexB,
            constraint.restDistance,
            constraint.compliance,
            Diligent::float4{constraint.localAnchorA.x, constraint.localAnchorA.y,
                             constraint.localAnchorA.z, 0.0f},
            Diligent::float4{constraint.localAnchorB.x, constraint.localAnchorB.y,
                             constraint.localAnchorB.z, 0.0f},
        });
    }

    for (const AuthoredRoutedCableConstraintState &constraint : mRoutedCableConstraintSnapshot)
    {
        if (!constraint.enabled || constraint.routePoints.size() < 2u)
        {
            continue;
        }

        std::optional<std::uint32_t> environmentIndex;
        std::uint32_t routePointStart =
            static_cast<std::uint32_t>(mRoutedCableRoutePoints.size());
        bool validRoute = true;
        std::optional<std::uint32_t> previousRigidBodyIndex;
        for (const AuthoredRoutedCableRoutePoint &routePoint : constraint.routePoints)
        {
            const std::optional<std::uint32_t> rigidBodyIndex =
                resolveRigidBodyIndex(routePoint.entityId);
            if (!rigidBodyIndex.has_value() || *rigidBodyIndex >= mRigidBodySnapshot.size())
            {
                validRoute = false;
                break;
            }

            const RigidBodyState &rigidBody = mRigidBodySnapshot[*rigidBodyIndex];
            if (!environmentIndex.has_value())
            {
                environmentIndex = rigidBody.environmentIndex;
            }
            else if (*environmentIndex != rigidBody.environmentIndex)
            {
                validRoute = false;
                break;
            }

            if (previousRigidBodyIndex.has_value() && *previousRigidBodyIndex == *rigidBodyIndex)
            {
                validRoute = false;
                break;
            }

            previousRigidBodyIndex = rigidBodyIndex;
            mRoutedCableRoutePoints.push_back(RoutedCableRoutePoint{
                *rigidBodyIndex, 0u, 0u, 0u,
                Diligent::float4{routePoint.localGuideOffset.x, routePoint.localGuideOffset.y,
                                 routePoint.localGuideOffset.z, 0.0f}});
        }

        if (!validRoute)
        {
            mRoutedCableRoutePoints.resize(routePointStart);
            continue;
        }

        mRoutedCableConstraints.push_back(RoutedCableConstraint{
            routePointStart,
            static_cast<std::uint32_t>(constraint.routePoints.size()),
            constraint.targetLength,
            constraint.compliance,
            constraint.tensionOnly ? 1u : 0u,
            0u,
            0u});
    }

    mSuturingPairs.clear();
    mReservedSuturingPathHeaders = 0u;
    mReservedSuturingPathNodes   = 0u;
    for (std::uint32_t strandIndex = 0u; strandIndex < mStrandSnapshot.size(); ++strandIndex)
    {
        const StrandState &strand = mStrandSnapshot[strandIndex];
        if (!strand.suturingEnabled || strand.particleCount == 0u ||
            sequenceStrandEntities.find(strand.entityId) != sequenceStrandEntities.end())
        {
            continue;
        }

        for (std::uint32_t softBodyIndex = 0u; softBodyIndex < mSoftBodySnapshot.size();
             ++softBodyIndex)
        {
            const SoftBodyState &softBody = mSoftBodySnapshot[softBodyIndex];
            if (!softBody.supportsSuturing || softBody.particleCount == 0u ||
                softBody.environmentIndex != strand.environmentIndex)
            {
                continue;
            }

            StrandSoftSuturingPair pair{};
            pair.suturingGroupId          = strandIndex;
            pair.softBodyIndex            = softBodyIndex;
            pair.strandParticleStart      = strand.particleOffset;
            pair.strandParticleCount      = strand.particleCount;
            pair.tipParticleIndex         = strand.particleOffset;
            pair.softTetStart             = softBody.tetOffset;
            pair.softTetCount             = softBody.tetCount;
            pair.pathStart                = mReservedSuturingPathHeaders;
            pair.pathCount                = mMaxSuturingPathsPerPair;
            pair.nodeStart                = mReservedSuturingPathNodes;
            pair.nodeCount                = mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
            pair.activePathIndex          = kInvalidSuturingIndex;
            pair.environmentIndex         = strand.environmentIndex;
            pair.pathNodeSpacing          = strand.pathNodeSpacing;
            pair.needleTangentialDragBits = encodeSuturingFloat(0.0f);
            pair.threadTangentialDragBits = encodeSuturingFloat(0.0f);
            mSuturingPairs.push_back(pair);

            mReservedSuturingPathHeaders += mMaxSuturingPathsPerPair;
            mReservedSuturingPathNodes += mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
        }
    }

    for (std::uint32_t rigidBodyIndex = 0u; rigidBodyIndex < mRigidBodySnapshot.size();
         ++rigidBodyIndex)
    {
        const RigidBodyState &rigidBody = mRigidBodySnapshot[rigidBodyIndex];
        if (!rigidBody.suturingEnabled || rigidBody.proxyParticleCount == 0u ||
            sequenceRigidEntities.find(rigidBody.entityId) != sequenceRigidEntities.end())
        {
            continue;
        }

        for (std::uint32_t softBodyIndex = 0u; softBodyIndex < mSoftBodySnapshot.size();
             ++softBodyIndex)
        {
            const SoftBodyState &softBody = mSoftBodySnapshot[softBodyIndex];
            if (!softBody.supportsSuturing || softBody.particleCount == 0u ||
                softBody.environmentIndex != rigidBody.environmentIndex)
            {
                continue;
            }

            StrandSoftSuturingPair pair{};
            pair.suturingGroupId     = rigidProxySuturingGroupBase + rigidBodyIndex;
            pair.softBodyIndex       = softBodyIndex;
            pair.strandParticleStart = rigidBody.proxyParticleOffset;
            pair.strandParticleCount = rigidBody.proxyParticleCount;
            pair.tipParticleIndex = rigidBody.proxyParticleOffset + rigidBody.needleTipProxyIndex;
            pair.softTetStart     = softBody.tetOffset;
            pair.softTetCount     = softBody.tetCount;
            pair.pathStart        = mReservedSuturingPathHeaders;
            pair.pathCount        = mMaxSuturingPathsPerPair;
            pair.nodeStart        = mReservedSuturingPathNodes;
            pair.nodeCount        = mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
            pair.activePathIndex  = kInvalidSuturingIndex;
            pair.environmentIndex = rigidBody.environmentIndex;
            pair.pathNodeSpacing  = std::max(rigidBody.proxyParticleRadius * 1.5f, 1.0e-4f);
            pair.needleTangentialDragBits = encodeSuturingFloat(0.0f);
            pair.threadTangentialDragBits = encodeSuturingFloat(0.0f);
            mSuturingPairs.push_back(pair);

            mReservedSuturingPathHeaders += mMaxSuturingPathsPerPair;
            mReservedSuturingPathNodes += mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
        }
    }

    for (const ResolvedSuturingSequence &sequence : resolvedSequences)
    {
        if (sequence.particleIndices.empty() ||
            sequence.tipEntryIndex >= sequence.particleIndices.size())
        {
            continue;
        }

        std::uint32_t particleStart = sequence.particleIndices[0];
        std::uint32_t particleEnd   = sequence.particleIndices[0];
        for (const std::uint32_t particleIndex : sequence.particleIndices)
        {
            particleStart = std::min(particleStart, particleIndex);
            particleEnd   = std::max(particleEnd, particleIndex);
        }

        for (std::uint32_t softBodyIndex = 0u; softBodyIndex < mSoftBodySnapshot.size();
             ++softBodyIndex)
        {
            const SoftBodyState &softBody = mSoftBodySnapshot[softBodyIndex];
            if (!softBody.supportsSuturing || softBody.particleCount == 0u ||
                softBody.environmentIndex != sequence.environmentIndex)
            {
                continue;
            }

            StrandSoftSuturingPair pair{};
            pair.suturingGroupId          = sequence.groupId;
            pair.softBodyIndex            = softBodyIndex;
            pair.strandParticleStart      = particleStart;
            pair.strandParticleCount      = particleEnd - particleStart + 1u;
            pair.tipParticleIndex         = sequence.particleIndices[sequence.tipEntryIndex];
            pair.softTetStart             = softBody.tetOffset;
            pair.softTetCount             = softBody.tetCount;
            pair.pathStart                = mReservedSuturingPathHeaders;
            pair.pathCount                = mMaxSuturingPathsPerPair;
            pair.nodeStart                = mReservedSuturingPathNodes;
            pair.nodeCount                = mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
            pair.activePathIndex          = kInvalidSuturingIndex;
            pair.environmentIndex         = sequence.environmentIndex;
            pair.pathNodeSpacing          = sequence.pathNodeSpacing;
            pair.needleTangentialDragBits = encodeSuturingFloat(sequence.needleTangentialDrag);
            pair.threadTangentialDragBits = encodeSuturingFloat(sequence.threadTangentialDrag);
            mSuturingPairs.push_back(pair);

            mReservedSuturingPathHeaders += mMaxSuturingPathsPerPair;
            mReservedSuturingPathNodes += mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
        }
    }

    mParticles.adjacencyOffsets.resize(adjacencyLists.size());
    mParticles.adjacencyCounts.resize(adjacencyLists.size());
    mParticles.adjacencyIndices.clear();
    for (std::uint32_t particleIndex = 0u;
         particleIndex < static_cast<std::uint32_t>(adjacencyLists.size()); ++particleIndex)
    {
        auto &neighbors = adjacencyLists[particleIndex];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        mParticles.adjacencyOffsets[particleIndex] =
            static_cast<std::uint32_t>(mParticles.adjacencyIndices.size());
        mParticles.adjacencyCounts[particleIndex] = static_cast<std::uint32_t>(neighbors.size());
        mParticles.adjacencyIndices.insert(mParticles.adjacencyIndices.end(), neighbors.begin(),
                                           neighbors.end());
    }

    recomputeParticleGridCellSize();
    mSoftBodyDerivedStateDirty = false;
}

void PhysicsWorld::recomputeParticleGridCellSize() noexcept
{
    float gridCellSize = 0.0f;
    for (std::size_t particleIndex = 0; particleIndex < mParticles.radii.size(); ++particleIndex)
    {
        gridCellSize = std::max(gridCellSize, mParticles.radii[particleIndex] * 2.0f);
        if (particleIndex < mParticles.fluidMaterialIndices.size() &&
            mParticles.fluidMaterialIndices[particleIndex] != 0xffffffffu &&
            mParticles.fluidMaterialIndices[particleIndex] < mFluidMaterials.size())
        {
            gridCellSize = std::max(
                gridCellSize,
                mFluidMaterials[mParticles.fluidMaterialIndices[particleIndex]].smoothingRadius);
        }
    }

    mParticleGridCellSize = std::max(gridCellSize, 0.1f);
}

void PhysicsWorld::recomputeSoftBodyBoundsChunkCount() noexcept
{
    mSoftBodyBoundsChunkCount = 0u;
    for (const Diligent::uint2 &range : mSoftRenderData.softBodyParticleRanges)
    {
        mSoftBodyBoundsChunkCount += (range.y + 64u - 1u) / 64u;
    }
}

bool PhysicsWorld::prepareFluidStateForInsert(const FluidState &candidate,
                                              FluidDerivedCache &derivedCache) noexcept
{
    if (candidate.source.kind != FluidSourceKind::RegularGrid)
    {
        return false;
    }

    const Diligent::float3 size = candidate.source.regularGrid.size;
    const float spacing         = candidate.source.regularGrid.targetParticleSpacing;
    const std::uint32_t nx      = std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>(std::floor(size.x / spacing + 0.5f)));
    const std::uint32_t ny = std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>(std::floor(size.y / spacing + 0.5f)));
    const std::uint32_t nz = std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>(std::floor(size.z / spacing + 0.5f)));

    derivedCache.restPositions.clear();
    derivedCache.restPositions.reserve(static_cast<std::size_t>(nx) * ny * nz);
    for (std::uint32_t z = 0u; z < nz; ++z)
    {
        for (std::uint32_t y = 0u; y < ny; ++y)
        {
            for (std::uint32_t x = 0u; x < nx; ++x)
            {
                const Diligent::float3 objectSpacePosition{
                    -size.x * 0.5f + (static_cast<float>(x) + 0.5f) * spacing,
                    -size.y * 0.5f + (static_cast<float>(y) + 0.5f) * spacing,
                    -size.z * 0.5f + (static_cast<float>(z) + 0.5f) * spacing};
                derivedCache.restPositions.push_back(common::runtime_math::applyTransform(
                    candidate.restTransform, objectSpacePosition));
            }
        }
    }
    return true;
}

bool PhysicsWorld::prepareStrandStateForInsert(const StrandState &candidate,
                                               StrandDerivedCache &derivedCache) noexcept
{
    derivedCache = {};
    if (candidate.restPositions.size() < 2u)
    {
        CRESSIM_LOG_ERROR("Failed to author strand for entity ", candidate.entityId,
                          ": strand must contain at least two rest positions.");
        return false;
    }

    derivedCache.restPositions = candidate.restPositions;
    derivedCache.adjacencyLists.resize(candidate.restPositions.size());
    derivedCache.incidentSegmentLists.resize(candidate.restPositions.size());
    derivedCache.incidentJointLists.resize(candidate.restPositions.size());

    const std::uint32_t particleCount =
        static_cast<std::uint32_t>(candidate.restPositions.size());
    derivedCache.segments.reserve(particleCount - 1u);
    derivedCache.restSegmentLengths.reserve(particleCount - 1u);
    derivedCache.restSegmentOrientations.reserve(particleCount - 1u);

    Diligent::float3 previousNormal{0.0f, 1.0f, 0.0f};
    Diligent::float3 previousTangent{1.0f, 0.0f, 0.0f};
    for (std::uint32_t i = 0u; i + 1u < particleCount; ++i)
    {
        derivedCache.segments.push_back({i, i + 1u});
        derivedCache.adjacencyLists[i].push_back(i + 1u);
        derivedCache.adjacencyLists[i + 1u].push_back(i);
        derivedCache.incidentSegmentLists[i].push_back(i);
        derivedCache.incidentSegmentLists[i + 1u].push_back(i);

        const Diligent::float3 delta = candidate.restPositions[i + 1u] - candidate.restPositions[i];
        const float restLength       = std::sqrt(Diligent::dot(delta, delta));
        if (restLength <= 1.0e-6f)
        {
            CRESSIM_LOG_ERROR("Failed to author strand for entity ", candidate.entityId,
                              ": consecutive rest positions must not be degenerate.");
            return false;
        }
        const Diligent::float3 tangent =
            safeNormalize(delta, Diligent::float3{1.0f, 0.0f, 0.0f});
        Diligent::float3 materialNormal =
            i == 0u ? defaultRootMaterialNormal(tangent, candidate.rootMaterialNormal)
                    : parallelTransportNormal(previousTangent, tangent, previousNormal);
        materialNormal = safeNormalize(materialNormal, Diligent::float3{0.0f, 1.0f, 0.0f});

        derivedCache.restSegmentLengths.push_back(restLength);
        derivedCache.restSegmentOrientations.push_back(
            segmentOrientationFromTangentNormal(tangent, materialNormal));
        previousNormal  = materialNormal;
        previousTangent = tangent;
    }

    derivedCache.joints.reserve(particleCount > 2u ? particleCount - 2u : 0u);
    derivedCache.restJointRelativeOrientations.reserve(derivedCache.joints.capacity());
    for (std::uint32_t i = 0u; i + 1u < static_cast<std::uint32_t>(derivedCache.segments.size());
         ++i)
    {
        derivedCache.joints.push_back({i, i + 1u});
        const std::uint32_t middleParticle = i + 1u;
        derivedCache.incidentJointLists[middleParticle].push_back(i);
        derivedCache.incidentJointLists[i].push_back(i);
        derivedCache.incidentJointLists[i + 2u].push_back(i);
        derivedCache.restJointRelativeOrientations.push_back(quaternionMultiply(
            quaternionConjugate(derivedCache.restSegmentOrientations[i]),
            derivedCache.restSegmentOrientations[i + 1u]));
    }

    std::string errorMessage;
    derivedCache.staticParticleIndices = validateStaticParticleIndices(
        candidate.staticParticleIndices, static_cast<std::uint32_t>(candidate.restPositions.size()),
        errorMessage);
    if (!errorMessage.empty())
    {
        CRESSIM_LOG_ERROR("Failed to author strand for entity ", candidate.entityId, ": ",
                          errorMessage);
        return false;
    }

    return true;
}

bool PhysicsWorld::prepareSoftBodyStateForInsert(const SoftBodyState &candidate,
                                                 const SoftBodyState *previousState,
                                                 SoftBodyDerivedCache &derivedCache) noexcept
{
    TetMeshData tetGenCache;
    TetGenMeshCache cacheEntry;
    const TetMeshData *tetGenCachePtr = nullptr;
    if (candidate.source.kind == SoftBodySourceKind::TetGenFiles)
    {
        bool needReload = true;
        if (previousState != nullptr &&
            previousState->source.kind == SoftBodySourceKind::TetGenFiles)
        {
            const auto &before = previousState->source.tetGen;
            const auto &after  = candidate.source.tetGen;
            if (before.nodeFile == after.nodeFile && before.eleFile == after.eleFile)
            {
                if (const TetGenMeshCache *existing = tryGetTetGenMeshCache(candidate.entityId))
                {
                    cacheEntry = *existing;
                    needReload = false;
                }
            }
        }

        std::string errorMessage;
        if (needReload)
        {
            if (!loadTetGenFiles(candidate.source.tetGen.nodeFile, candidate.source.tetGen.eleFile,
                                 tetGenCache, errorMessage))
            {
                CRESSIM_LOG_ERROR("Failed to author soft body for entity ", candidate.entityId,
                                  ": ", errorMessage);
                return false;
            }
            cacheEntry.nodeFile                 = candidate.source.tetGen.nodeFile;
            cacheEntry.eleFile                  = candidate.source.tetGen.eleFile;
            cacheEntry.objectSpaceRestPositions = tetGenCache.objectSpaceRestPositions;
            cacheEntry.tetVertexIndices         = tetGenCache.tetVertexIndices;
        }
        else
        {
            tetGenCache.objectSpaceRestPositions = cacheEntry.objectSpaceRestPositions;
            tetGenCache.tetVertexIndices         = cacheEntry.tetVertexIndices;
        }
        tetGenCachePtr = &tetGenCache;
    }

    std::string errorMessage;
    ResolvedSoftBodyTopology resolvedTopology;
    if (!resolveSoftBodyTopology(candidate, tetGenCachePtr, resolvedTopology, errorMessage))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body for entity ", candidate.entityId, ": ",
                          errorMessage);
        return false;
    }

    derivedCache.restPositions         = std::move(resolvedTopology.restPositions);
    derivedCache.edges                 = std::move(resolvedTopology.edges);
    derivedCache.tets                  = std::move(resolvedTopology.tets);
    derivedCache.boundaryFaces         = std::move(resolvedTopology.boundaryFaces);
    derivedCache.adjacencyLists        = std::move(resolvedTopology.adjacencyLists);
    derivedCache.staticParticleIndices = std::move(resolvedTopology.staticParticleIndices);
    if (candidate.supportsSuturing)
    {
        derivedCache.incidentTetLists.assign(derivedCache.restPositions.size(), {});
        for (std::uint32_t tetIndex = 0u;
             tetIndex < static_cast<std::uint32_t>(derivedCache.tets.size()); ++tetIndex)
        {
            const auto &tet = derivedCache.tets[tetIndex];
            for (std::uint32_t slot = 0u; slot < 4u; ++slot)
            {
                derivedCache.incidentTetLists[tet[slot]].push_back(tetIndex);
            }
        }
    }
    else
    {
        derivedCache.incidentTetLists.clear();
    }

    if (candidate.source.kind == SoftBodySourceKind::TetGenFiles)
    {
        mTetGenMeshCache[candidate.entityId] = std::move(cacheEntry);
    }
    else
    {
        mTetGenMeshCache.erase(candidate.entityId);
    }

    return true;
}

const PhysicsWorld::TetGenMeshCache *PhysicsWorld::tryGetTetGenMeshCache(
    common::EntityId entityId) const noexcept
{
    const auto it = mTetGenMeshCache.find(entityId);
    return it == mTetGenMeshCache.end() ? nullptr : &it->second;
}

void PhysicsWorld::markAllRigidBodiesDirty() noexcept
{
    mFullRigidBodyUploadRequired = true;
    mRigidBodyCountDirty         = true;
    mRigidBodyDirtyBits.assign(rigidBodyCount(), 1u);
    mRigidBodyDirtyIndices.resize(rigidBodyCount());
    for (std::uint32_t i = 0; i < rigidBodyCount(); ++i)
    {
        mRigidBodyDirtyIndices[i] = i;
    }
}

void PhysicsWorld::markAllCollidersDirty() noexcept
{
    mFullColliderUploadRequired = true;
    mColliderCountDirty         = true;
    mColliderDirtyBits.assign(colliderCount(), 1u);
    mColliderDirtyIndices.resize(colliderCount());
    for (std::uint32_t i = 0; i < colliderCount(); ++i)
    {
        mColliderDirtyIndices[i] = i;
    }
}

void PhysicsWorld::markRigidBodyDirty(std::uint32_t index) noexcept
{
    enqueueDirtyIndex(index, mRigidBodyDirtyIndices, mRigidBodyDirtyBits);
}

void PhysicsWorld::markColliderDirty(std::uint32_t index) noexcept
{
    enqueueDirtyIndex(index, mColliderDirtyIndices, mColliderDirtyBits);
}

void PhysicsWorld::markRigidBodyCountDirty(bool fullUploadRequired) noexcept
{
    mRigidBodyCountDirty         = true;
    mFullRigidBodyUploadRequired = mFullRigidBodyUploadRequired || fullUploadRequired;
}

void PhysicsWorld::markColliderCountDirty(bool fullUploadRequired) noexcept
{
    mColliderCountDirty         = true;
    mFullColliderUploadRequired = mFullColliderUploadRequired || fullUploadRequired;
}

void PhysicsWorld::applyRigidJointChange(RigidJointChangeKind changeKind) noexcept
{
    switch (changeKind)
    {
    case RigidJointChangeKind::PayloadOnly:
        markJointSceneDirty();
        return;
    case RigidJointChangeKind::ModeRebuild:
        markJointModeDirty();
        return;
    case RigidJointChangeKind::TopologyRebuild:
        markJointTopologyDirty();
        return;
    }
}

void PhysicsWorld::markJointSceneDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mAuthoredRevision;
}

void PhysicsWorld::markJointModeDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mAuthoredRevision;
}

void PhysicsWorld::markJointTopologyDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mRigidJointTopologyRevision;
    ++mAuthoredRevision;
}

} // namespace cressim::neo::physics
