#include "physics/physics_world.h"

#include "common/logger.h"
#include "common/math_utils_runtime.h"
#include "physics/particle_phase.h"
#include "physics/soft_body_authoring.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uint32_t kBroadPhaseContributionNone   = 0u;
constexpr std::uint32_t kBroadPhaseContributionMoving = 1u;
constexpr std::uint32_t kBroadPhaseContributionStatic = 2u;

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

template <typename T>
bool triviallyComparableVectorsDiffer(const std::vector<T> &lhs, const std::vector<T> &rhs) noexcept
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (lhs.size() != rhs.size())
    {
        return true;
    }
    if (lhs.empty())
    {
        return false;
    }
    return std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) != 0;
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
    const Diligent::float3 direction =
        safeNormalize(segmentDirection, Diligent::float3{1.0f, 0.0f, 0.0f});
    Diligent::float3 normal = authoredNormal;
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
    Diligent::float3 transported =
        previousNormal - nextTangent * Diligent::dot(previousNormal, nextTangent);
    if (Diligent::dot(transported, transported) <= 1.0e-8f)
    {
        transported = defaultRootMaterialNormal(nextTangent, previousNormal);
    }
    return safeNormalize(transported, defaultRootMaterialNormal(nextTangent, previousNormal));
}

Diligent::QuaternionF segmentOrientationFromTangentNormal(
    const Diligent::float3 &tangent, const Diligent::float3 &materialNormal) noexcept
{
    const Diligent::float3 xAxis = safeNormalize(tangent, Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 yAxis =
        safeNormalize(materialNormal - xAxis * Diligent::dot(materialNormal, xAxis),
                      defaultRootMaterialNormal(xAxis, materialNormal));
    const Diligent::float3 zAxis =
        safeNormalize(Diligent::cross(xAxis, yAxis), Diligent::float3{0.0f, 0.0f, 1.0f});
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

struct PhysicsWorld::Impl
{
    explicit Impl(PhysicsWorld &owner) : mOwner(&owner) {}

    // Helpers reuse public query semantics through the enclosing facade.
    PhysicsWorld *mOwner = nullptr;
    enum class SoftBodyChangeKind
    {
        RuntimePropertiesOnly,
        TopologyRebuild,
    };

    enum class StrandChangeKind
    {
        RuntimePropertiesOnly,
        TopologyRebuild,
    };

    enum class RigidJointChangeKind
    {
        PayloadOnly,
        ModeRebuild,
        TopologyRebuild,
    };

    struct SoftBodyDerivedCache
    {
        std::vector<Diligent::float3> restPositions;
        std::vector<std::array<std::uint32_t, 2>> edges;
        std::vector<std::array<std::uint32_t, 4>> tets;
        std::vector<Diligent::uint3> boundaryFaces;
        std::vector<std::vector<std::uint32_t>> adjacencyLists;
        std::vector<std::vector<std::uint32_t>> incidentTetLists;
        std::vector<std::uint32_t> staticParticleIndices;
    };

    struct FluidDerivedCache
    {
        std::vector<Diligent::float3> restPositions;
    };

    struct StrandDerivedCache
    {
        std::vector<Diligent::float3> restPositions;
        std::vector<std::array<std::uint32_t, 2>> segments;
        std::vector<std::array<std::uint32_t, 2>> joints;
        std::vector<std::vector<std::uint32_t>> adjacencyLists;
        std::vector<std::vector<std::uint32_t>> incidentSegmentLists;
        std::vector<std::vector<std::uint32_t>> incidentJointLists;
        std::vector<float> restSegmentLengths;
        std::vector<Diligent::QuaternionF> restSegmentOrientations;
        std::vector<Diligent::QuaternionF> restJointRelativeOrientations;
        std::vector<std::uint32_t> staticParticleIndices;
    };

    static void writeRigidBodySoAAt(RigidBodySoAHost &soa, std::uint32_t index,
                                    const RigidBodyState &state);
    static void writeColliderSoAAt(ColliderSoAHost &soa, std::uint32_t index,
                                   const ColliderState &state, std::uint32_t ownerBodyIndex,
                                   std::uint32_t ownerEnvironmentIndex);
    static bool isStaticBody(const RigidBodyState &state) noexcept;
    static bool staticBodyPoseChanged(const RigidBodyState &before,
                                      const RigidBodyState &after) noexcept;
    static bool isMovingBody(const RigidBodyState &state) noexcept;
    static void normalizeRigidBodyState(RigidBodyState &state) noexcept;
    static void normalizeColliderState(ColliderState &state) noexcept;
    static std::uint32_t colliderBroadPhaseContribution(const ColliderState &collider,
                                                        const RigidBodyState *owner) noexcept;

    void markRigidBodyDirty(std::uint32_t index) noexcept;
    void markColliderDirty(std::uint32_t index) noexcept;
    void markRigidBodyCountDirty(bool fullUploadRequired = false) noexcept;
    void markColliderCountDirty(bool fullUploadRequired = false) noexcept;
    void markJointSceneDirty() noexcept;
    void markJointModeDirty() noexcept;
    void markJointTopologyDirty() noexcept;
    void removeCollidersForEntity(common::EntityId entityId) noexcept;
    void removeColliderAtIndex(std::uint32_t index) noexcept;
    bool pruneRigidJointsForBody(RigidBodyId rigidBodyId) noexcept;
    void rebuildBodyColliderMapping() const noexcept;
    void rebuildRigidJointScene() const noexcept;
    void rebuildJointCollisionSuppression() const noexcept;
    void rebuildSoftParticleLayout() noexcept;
    void rebuildSoftConstraintData() noexcept;
    void rebuildSuturingData() noexcept;
    void rebuildResolvedRigidParticleAttachments() noexcept;
    void rebuildResolvedStrandRigidAttachments() noexcept;
    void rebuildResolvedRigidDistanceConstraints() noexcept;
    void rebuildResolvedRoutedCables() noexcept;
    void ensureRebuildDomainsUpToDate(PhysicsRebuildFlags flags) noexcept;
    void ensureResolvedConstraintStateUpToDate() noexcept;
    void invalidateSoftDerivedState() noexcept;
    void invalidateResolvedRigidParticleAttachments() noexcept;
    void invalidateResolvedStrandRigidAttachments(bool alsoSoftAdjacency) noexcept;
    void invalidateResolvedRigidDistanceConstraints() noexcept;
    void invalidateResolvedRoutedCables() noexcept;
    void invalidateAllRigidBodyDependentResolvedConstraints() noexcept;
    void markRebuildDirty(PhysicsRebuildFlags flags) noexcept;
    void clearRebuildDirty(PhysicsRebuildFlags flags) noexcept;
    bool isRebuildDirty(PhysicsRebuildFlags flags) const noexcept;
    void markAllRigidBodiesDirty() noexcept;
    void markAllCollidersDirty() noexcept;
    std::uint32_t broadPhaseContributionForCollider(const ColliderState &collider) const noexcept;
    std::uint32_t enabledColliderCountForEntity(common::EntityId entityId) const noexcept;
    std::optional<std::uint32_t> resolveParticleReference(
        const AuthoredParticleReference &reference) const noexcept;
    static void normalizeSoftBodyState(SoftBodyState &state) noexcept;
    static void normalizeStrandState(StrandState &state) noexcept;
    static void normalizeFluidState(FluidState &state) noexcept;
    bool validateFluidMaterialCompatibility(const FluidState &candidate,
                                            const FluidState *previousState) const noexcept;
    static SoftBodyChangeKind classifySoftBodyChange(const SoftBodyState &previousState,
                                                     const SoftBodyState &candidate) noexcept;
    static StrandChangeKind classifyStrandChange(const StrandState &previousState,
                                                 const StrandState &candidate) noexcept;
    static RigidJointChangeKind classifyBallJointChange(bool inserted) noexcept;
    static RigidJointChangeKind classifySphericalJointChange(
        const SphericalJointState &previousState, const SphericalJointState &candidate,
        bool inserted) noexcept;
    static RigidJointChangeKind classifyHingeJointChange(const HingeJointState &previousState,
                                                         const HingeJointState &candidate,
                                                         bool inserted) noexcept;
    static RigidJointChangeKind classifySliderJointChange(const SliderJointState &previousState,
                                                          const SliderJointState &candidate,
                                                          bool inserted) noexcept;
    void applyRigidJointChange(RigidJointChangeKind changeKind) noexcept;
    void applySoftBodyRuntimeProperties(std::uint32_t index,
                                        const SoftBodyState &normalizedState) noexcept;
    void applyStrandRuntimeProperties(std::uint32_t index,
                                      const StrandState &normalizedState) noexcept;
    void recomputeParticleGridCellSize() noexcept;
    void recomputeSoftBodyBoundsChunkCount() noexcept;
    bool prepareSoftBodyStateForInsert(const SoftBodyState &candidate,
                                       const SoftBodyState *previousState,
                                       SoftBodyDerivedCache &derivedCache) noexcept;
    bool prepareStrandStateForInsert(const StrandState &candidate,
                                     StrandDerivedCache &derivedCache) noexcept;
    bool prepareFluidStateForInsert(const FluidState &candidate,
                                    FluidDerivedCache &derivedCache) noexcept;
    struct TetGenMeshCache
    {
        std::string nodeFile;
        std::string eleFile;
        std::vector<Diligent::float3> objectSpaceRestPositions;
        std::vector<std::uint32_t> tetVertexIndices;
    };

    const TetGenMeshCache *tryGetTetGenMeshCache(common::EntityId entityId) const noexcept;

    RigidBodySoAHost mRigidBodies{};
    mutable ColliderSoAHost mColliders{};
    mutable BodyColliderMappingHost mBodyColliderMapping{};
    mutable RigidJointSceneHost mRigidJointScene{};
    mutable JointCollisionSuppressionHost mJointCollisionSuppression{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToRigidBodyIndex{};
    std::unordered_map<RigidBodyId, std::uint32_t> mRigidBodyIdToIndex{};
    std::unordered_map<ColliderId, std::uint32_t> mColliderIdToIndex{};
    std::unordered_map<common::EntityId, std::vector<ColliderId>> mEntityToColliderIds{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToSoftBodyIndex{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToStrandIndex{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToFluidIndex{};
    std::unordered_map<ParticleSequenceId, std::uint32_t> mParticleSequenceIdToIndex{};
    std::unordered_map<ParticleConstraintId, std::uint32_t> mParticleConstraintIdToIndex{};
    std::unordered_map<RigidParticleAttachmentConstraintId, std::uint32_t>
        mRigidParticleAttachmentConstraintIdToIndex{};
    std::unordered_map<StrandRigidAttachmentConstraintId, std::uint32_t>
        mStrandRigidAttachmentConstraintIdToIndex{};
    std::unordered_map<RigidDistanceConstraintId, std::uint32_t>
        mRigidDistanceConstraintIdToIndex{};
    std::unordered_map<RoutedCableConstraintId, std::uint32_t> mRoutedCableConstraintIdToIndex{};
    std::unordered_map<ParticleCollisionFilterId, std::uint32_t>
        mParticleCollisionFilterIdToIndex{};
    std::unordered_map<SuturingSequenceId, std::uint32_t> mSuturingSequenceIdToIndex{};
    std::unordered_map<common::EntityId, TetGenMeshCache> mTetGenMeshCache{};
    std::vector<RigidBodyState> mRigidBodySnapshot{};
    std::vector<ColliderState> mColliderSnapshot{};
    std::vector<SoftBodyState> mSoftBodySnapshot{};
    std::vector<StrandState> mStrandSnapshot{};
    std::vector<FluidState> mFluidSnapshot{};
    std::vector<AuthoredParticleSequenceState> mParticleSequenceSnapshot{};
    std::vector<AuthoredParticleDistanceConstraintState> mParticleDistanceConstraintSnapshot{};
    std::vector<AuthoredRigidParticleAttachmentConstraintState>
        mRigidParticleAttachmentConstraintSnapshot{};
    std::vector<AuthoredStrandRigidAttachmentConstraintState>
        mStrandRigidAttachmentConstraintSnapshot{};
    std::vector<AuthoredRigidDistanceConstraintState> mRigidDistanceConstraintSnapshot{};
    std::vector<AuthoredRoutedCableConstraintState> mRoutedCableConstraintSnapshot{};
    std::vector<AuthoredParticleCollisionFilterState> mParticleCollisionFilterSnapshot{};
    std::vector<AuthoredSuturingSequenceState> mSuturingSequenceSnapshot{};
    std::vector<BallJointState> mBallJointSnapshot{};
    std::vector<SphericalJointState> mSphericalJointSnapshot{};
    std::vector<HingeJointState> mHingeJointSnapshot{};
    std::vector<SliderJointState> mSliderJointSnapshot{};
    std::vector<SoftBodyDerivedCache> mSoftBodyDerivedCaches{};
    std::vector<StrandDerivedCache> mStrandDerivedCaches{};
    std::vector<FluidDerivedCache> mFluidDerivedCaches{};
    std::vector<StrandSoftSuturingPair> mSuturingPairs{};
    ParticleSoAHost mParticles{};
    std::vector<Diligent::float4> mParticleContactMaterials{};
    std::vector<FluidMaterialGpu> mFluidMaterials{};
    std::vector<DeformableDistanceConstraint> mSoftEdges{};
    std::vector<DeformableBendConstraint> mSoftBends{};
    std::vector<DeformableVolumeConstraint> mSoftTets{};
    std::vector<StrandSegmentConstraint> mStrandSegments{};
    std::vector<StrandJointConstraint> mStrandJoints{};
    std::vector<StrandDistanceConstraint> mStrandDistanceConstraints{};
    std::vector<StrandSegmentState> mStrandSegmentStates{};
    std::vector<RigidParticleAttachmentConstraint> mRigidParticleAttachments{};
    std::vector<StrandRigidAttachmentConstraint> mStrandRigidAttachments{};
    std::vector<RigidDistanceConstraint> mRigidDistanceConstraints{};
    std::vector<RoutedCableConstraint> mRoutedCableConstraints{};
    std::vector<RoutedCableRoutePoint> mRoutedCableRoutePoints{};
    SoftRenderDataHost mSoftRenderData{};
    CurveRenderDataHost mCurveRenderData{};
    std::vector<std::uint32_t> mRigidBodyDirtyIndices{};
    std::vector<std::uint32_t> mColliderDirtyIndices{};
    std::vector<std::uint8_t> mRigidBodyDirtyBits{};
    std::vector<std::uint8_t> mColliderDirtyBits{};
    bool mRigidBodyCountDirty                    = false;
    bool mColliderCountDirty                     = false;
    bool mFullRigidBodyUploadRequired            = false;
    bool mFullColliderUploadRequired             = false;
    mutable bool mBodyColliderMappingDirty       = true;
    mutable bool mRigidJointSceneDirty           = true;
    mutable bool mJointCollisionSuppressionDirty = true;
    PhysicsRebuildFlags mRebuildFlags =
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData |
        PhysicsRebuildFlags::SuturingData | PhysicsRebuildFlags::ResolvedRigidParticleAttachments |
        PhysicsRebuildFlags::ResolvedStrandRigidAttachments |
        PhysicsRebuildFlags::ResolvedRigidDistanceConstraints |
        PhysicsRebuildFlags::ResolvedRoutedCables;
    bool mStaticBroadPhaseDirty                                                  = false;
    std::uint32_t mActiveMovingColliderCount                                     = 0u;
    std::uint32_t mStaticColliderCount                                           = 0u;
    float mParticleGridCellSize                                                  = 0.1f;
    std::uint32_t mSoftBodyBoundsChunkCount                                      = 0u;
    std::uint32_t mMaxSuturingPathsPerPair                                       = 4u;
    std::uint32_t mMaxSuturingNodesPerPath                                       = 128u;
    std::uint32_t mReservedSuturingPathHeaders                                   = 0u;
    std::uint32_t mReservedSuturingPathNodes                                     = 0u;
    std::uint64_t mAuthoredRevision                                              = 0;
    std::uint64_t mSimulationRevision                                            = 0;
    std::uint64_t mRigidBodyTopologyRevision                                     = 0;
    std::uint64_t mRigidJointSceneRevision                                       = 0;
    std::uint64_t mRigidJointModeRevision                                        = 0;
    std::uint64_t mRigidJointTopologyRevision                                    = 0;
    std::uint64_t mSoftBodyTopologyRevision                                      = 0;
    std::uint64_t mSoftParticleRevision                                          = 0;
    std::uint64_t mSoftTopologyRevision                                          = 0;
    std::uint64_t mSoftConstraintAdjacencyRevision                               = 0;
    std::uint64_t mRigidParticleAttachmentDefinitionRevision                     = 0;
    std::uint64_t mRigidParticleAttachmentResolvedRevision                       = 0;
    std::uint64_t mStrandRigidAttachmentDefinitionRevision                       = 0;
    std::uint64_t mStrandRigidAttachmentResolvedRevision                         = 0;
    std::uint64_t mRigidDistanceConstraintDefinitionRevision                     = 0;
    std::uint64_t mRigidDistanceConstraintResolvedRevision                       = 0;
    std::uint64_t mRoutedCableDefinitionRevision                                 = 0;
    std::uint64_t mRoutedCableResolvedRevision                                   = 0;
    std::uint64_t mCurveRenderRevision                                           = 0;
    RigidBodyId mNextRigidBodyId                                                 = 1u;
    ColliderId mNextColliderId                                                   = 1u;
    BallJointId mNextBallJointId                                                 = 1u;
    SphericalJointId mNextSphericalJointId                                       = 1u;
    HingeJointId mNextHingeJointId                                               = 1u;
    SliderJointId mNextSliderJointId                                             = 1u;
    ParticleSequenceId mNextParticleSequenceId                                   = 1u;
    ParticleConstraintId mNextParticleConstraintId                               = 1u;
    RigidParticleAttachmentConstraintId mNextRigidParticleAttachmentConstraintId = 1u;
    StrandRigidAttachmentConstraintId mNextStrandRigidAttachmentConstraintId     = 1u;
    RigidDistanceConstraintId mNextRigidDistanceConstraintId                     = 1u;
    RoutedCableConstraintId mNextRoutedCableConstraintId                         = 1u;
    ParticleCollisionFilterId mNextParticleCollisionFilterId                     = 1u;
    SuturingSequenceId mNextSuturingSequenceId                                   = 1u;
};

PhysicsWorld::PhysicsWorld() : mImpl(std::make_unique<Impl>(*this)) {}

PhysicsWorld::~PhysicsWorld() = default;

PhysicsWorld::PhysicsWorld(const PhysicsWorld &other) : mImpl(std::make_unique<Impl>(*other.mImpl))
{
    mImpl->mOwner = this;
}

PhysicsWorld &PhysicsWorld::operator=(const PhysicsWorld &other)
{
    if (this != &other)
    {
        mImpl         = std::make_unique<Impl>(*other.mImpl);
        mImpl->mOwner = this;
    }
    return *this;
}

PhysicsWorld::PhysicsWorld(PhysicsWorld &&other) noexcept : mImpl(std::move(other.mImpl))
{
    if (mImpl)
    {
        mImpl->mOwner = this;
    }
}

PhysicsWorld &PhysicsWorld::operator=(PhysicsWorld &&other) noexcept
{
    if (this != &other)
    {
        mImpl = std::move(other.mImpl);
        if (mImpl)
        {
            mImpl->mOwner = this;
        }
    }
    return *this;
}

void PhysicsWorld::clear()
{
    mImpl->mRigidBodies.clear();
    mImpl->mColliders.clear();
    mImpl->mBodyColliderMapping.clear();
    mImpl->mRigidJointScene.clear();
    mImpl->mJointCollisionSuppression.clear();
    mImpl->mEntityToRigidBodyIndex.clear();
    mImpl->mRigidBodyIdToIndex.clear();
    mImpl->mColliderIdToIndex.clear();
    mImpl->mEntityToColliderIds.clear();
    mImpl->mEntityToSoftBodyIndex.clear();
    mImpl->mEntityToStrandIndex.clear();
    mImpl->mEntityToFluidIndex.clear();
    mImpl->mParticleSequenceIdToIndex.clear();
    mImpl->mParticleConstraintIdToIndex.clear();
    mImpl->mRigidParticleAttachmentConstraintIdToIndex.clear();
    mImpl->mStrandRigidAttachmentConstraintIdToIndex.clear();
    mImpl->mRigidDistanceConstraintIdToIndex.clear();
    mImpl->mRoutedCableConstraintIdToIndex.clear();
    mImpl->mParticleCollisionFilterIdToIndex.clear();
    mImpl->mSuturingSequenceIdToIndex.clear();
    mImpl->mTetGenMeshCache.clear();
    mImpl->mRigidBodySnapshot.clear();
    mImpl->mColliderSnapshot.clear();
    mImpl->mSoftBodySnapshot.clear();
    mImpl->mStrandSnapshot.clear();
    mImpl->mFluidSnapshot.clear();
    mImpl->mParticleSequenceSnapshot.clear();
    mImpl->mParticleDistanceConstraintSnapshot.clear();
    mImpl->mRigidParticleAttachmentConstraintSnapshot.clear();
    mImpl->mStrandRigidAttachmentConstraintSnapshot.clear();
    mImpl->mRigidDistanceConstraintSnapshot.clear();
    mImpl->mRoutedCableConstraintSnapshot.clear();
    mImpl->mParticleCollisionFilterSnapshot.clear();
    mImpl->mSuturingSequenceSnapshot.clear();
    mImpl->mBallJointSnapshot.clear();
    mImpl->mSphericalJointSnapshot.clear();
    mImpl->mHingeJointSnapshot.clear();
    mImpl->mSliderJointSnapshot.clear();
    mImpl->mSoftBodyDerivedCaches.clear();
    mImpl->mStrandDerivedCaches.clear();
    mImpl->mFluidDerivedCaches.clear();
    mImpl->mSuturingPairs.clear();
    mImpl->mReservedSuturingPathHeaders = 0u;
    mImpl->mReservedSuturingPathNodes   = 0u;
    mImpl->mParticles.clear();
    mImpl->mParticleContactMaterials.clear();
    mImpl->mFluidMaterials.clear();
    mImpl->mSoftEdges.clear();
    mImpl->mSoftBends.clear();
    mImpl->mSoftTets.clear();
    mImpl->mStrandSegments.clear();
    mImpl->mStrandJoints.clear();
    mImpl->mStrandDistanceConstraints.clear();
    mImpl->mStrandSegmentStates.clear();
    mImpl->mRigidParticleAttachments.clear();
    mImpl->mStrandRigidAttachments.clear();
    mImpl->mRigidDistanceConstraints.clear();
    mImpl->mRoutedCableConstraints.clear();
    mImpl->mRoutedCableRoutePoints.clear();
    mImpl->mSoftRenderData.clear();
    mImpl->mCurveRenderData.clear();
    mImpl->mRigidBodyDirtyIndices.clear();
    mImpl->mColliderDirtyIndices.clear();
    mImpl->mRigidBodyDirtyBits.clear();
    mImpl->mColliderDirtyBits.clear();
    mImpl->mRigidBodyCountDirty            = true;
    mImpl->mColliderCountDirty             = true;
    mImpl->mFullRigidBodyUploadRequired    = true;
    mImpl->mFullColliderUploadRequired     = true;
    mImpl->mBodyColliderMappingDirty       = true;
    mImpl->mRigidJointSceneDirty           = true;
    mImpl->mJointCollisionSuppressionDirty = true;
    mImpl->mRebuildFlags =
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData |
        PhysicsRebuildFlags::SuturingData | PhysicsRebuildFlags::ResolvedRigidParticleAttachments |
        PhysicsRebuildFlags::ResolvedStrandRigidAttachments |
        PhysicsRebuildFlags::ResolvedRigidDistanceConstraints |
        PhysicsRebuildFlags::ResolvedRoutedCables;
    mImpl->mStaticBroadPhaseDirty                   = true;
    mImpl->mActiveMovingColliderCount               = 0u;
    mImpl->mStaticColliderCount                     = 0u;
    mImpl->mParticleGridCellSize                    = 0.1f;
    mImpl->mSoftBodyBoundsChunkCount                = 0u;
    mImpl->mNextRigidBodyId                         = 1u;
    mImpl->mNextColliderId                          = 1u;
    mImpl->mNextBallJointId                         = 1u;
    mImpl->mNextSphericalJointId                    = 1u;
    mImpl->mNextHingeJointId                        = 1u;
    mImpl->mNextSliderJointId                       = 1u;
    mImpl->mNextParticleSequenceId                  = 1u;
    mImpl->mNextParticleConstraintId                = 1u;
    mImpl->mNextRigidParticleAttachmentConstraintId = 1u;
    mImpl->mNextStrandRigidAttachmentConstraintId   = 1u;
    mImpl->mNextRigidDistanceConstraintId           = 1u;
    mImpl->mNextRoutedCableConstraintId             = 1u;
    mImpl->mNextParticleCollisionFilterId           = 1u;
    mImpl->mNextSuturingSequenceId                  = 1u;
    ++mImpl->mRigidBodyTopologyRevision;
    ++mImpl->mRigidJointSceneRevision;
    ++mImpl->mRigidJointModeRevision;
    ++mImpl->mRigidJointTopologyRevision;
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mSoftConstraintAdjacencyRevision;
    ++mImpl->mRigidParticleAttachmentDefinitionRevision;
    ++mImpl->mRigidParticleAttachmentResolvedRevision;
    ++mImpl->mStrandRigidAttachmentDefinitionRevision;
    ++mImpl->mStrandRigidAttachmentResolvedRevision;
    ++mImpl->mRigidDistanceConstraintDefinitionRevision;
    ++mImpl->mRigidDistanceConstraintResolvedRevision;
    ++mImpl->mRoutedCableDefinitionRevision;
    ++mImpl->mRoutedCableResolvedRevision;
    ++mImpl->mAuthoredRevision;
    ++mImpl->mSimulationRevision;
}

RigidBodyState &PhysicsWorld::upsertRigidBody(const RigidBodyState &state)
{
    RigidBodyState normalizedState = state;
    mImpl->normalizeRigidBodyState(normalizedState);

    auto it = mImpl->mEntityToRigidBodyIndex.find(normalizedState.entityId);
    if (it == mImpl->mEntityToRigidBodyIndex.end())
    {
        normalizedState.rigidBodyId = mImpl->mNextRigidBodyId++;

        const std::uint32_t index = static_cast<std::uint32_t>(mImpl->mRigidBodies.size());
        mImpl->mEntityToRigidBodyIndex.emplace(normalizedState.entityId, index);
        mImpl->mRigidBodyIdToIndex.emplace(normalizedState.rigidBodyId, index);
        mImpl->mRigidBodySnapshot.push_back(normalizedState);
        mImpl->mRigidBodies.rigidBodyIds.push_back(normalizedState.rigidBodyId);
        mImpl->mRigidBodies.entityIds.push_back(normalizedState.entityId);
        mImpl->mRigidBodies.environmentIndices.push_back(normalizedState.environmentIndex);
        mImpl->mRigidBodies.positionsInvMass.push_back(toPositionInvMass(normalizedState));
        mImpl->mRigidBodies.orientations.push_back(toOrientation(normalizedState));
        mImpl->mRigidBodies.scales.push_back(toScale(normalizedState));
        mImpl->mRigidBodies.linearVelocities.push_back(toLinearVelocity(normalizedState));
        mImpl->mRigidBodies.angularVelocities.push_back(toAngularVelocity(normalizedState));
        mImpl->mRigidBodies.inverseInertiaLocal.push_back(toInverseInertiaLocal(normalizedState));
        mImpl->mRigidBodies.bodyTypes.push_back(
            static_cast<std::uint32_t>(normalizedState.bodyType));
        mImpl->mRigidBodies.kinematicTargetPositions.push_back(
            toKinematicTargetPosition(normalizedState));
        mImpl->mRigidBodies.kinematicTargetOrientations.push_back(
            toKinematicTargetOrientation(normalizedState));
        mImpl->mRigidBodies.kinematicTargetFlags.push_back(
            normalizedState.kinematicTargetEnabled ? 1u : 0u);
        mImpl->mRigidBodies.proxyParticleContactMaterials.push_back(
            normalizedState.proxyParticleContactMaterial);
        mImpl->markRigidBodyDirty(index);
        mImpl->markRigidBodyCountDirty();
        mImpl->mBodyColliderMappingDirty       = true;
        mImpl->mJointCollisionSuppressionDirty = true;
        mImpl->mStaticBroadPhaseDirty =
            mImpl->mStaticBroadPhaseDirty || mImpl->isStaticBody(normalizedState);
        if (!normalizedState.proxyParticleLocalPositions.empty())
        {
            mImpl->invalidateSoftDerivedState();
            ++mImpl->mSoftParticleRevision;
            ++mImpl->mSoftTopologyRevision;
        }
        mImpl->invalidateAllRigidBodyDependentResolvedConstraints();
        ++mImpl->mRigidBodyTopologyRevision;
        ++mImpl->mAuthoredRevision;
        return mImpl->mRigidBodySnapshot.back();
    }

    const std::uint32_t index          = it->second;
    normalizedState.rigidBodyId        = mImpl->mRigidBodySnapshot[index].rigidBodyId;
    const RigidBodyState previousState = mImpl->mRigidBodySnapshot[index];
    if (previousState.bodyType != normalizedState.bodyType)
    {
        const std::uint32_t enabledColliderCount =
            mImpl->enabledColliderCountForEntity(previousState.entityId);
        if (mImpl->isStaticBody(previousState))
        {
            mImpl->mStaticColliderCount -= enabledColliderCount;
        }
        else if (mImpl->isMovingBody(previousState))
        {
            mImpl->mActiveMovingColliderCount -= enabledColliderCount;
        }

        if (mImpl->isStaticBody(normalizedState))
        {
            mImpl->mStaticColliderCount += enabledColliderCount;
        }
        else if (mImpl->isMovingBody(normalizedState))
        {
            mImpl->mActiveMovingColliderCount += enabledColliderCount;
        }
    }
    mImpl->writeRigidBodySoAAt(mImpl->mRigidBodies, index, normalizedState);
    mImpl->mRigidBodySnapshot[index] = normalizedState;
    mImpl->markRigidBodyDirty(index);
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
        mImpl->invalidateSoftDerivedState();
        ++mImpl->mSoftParticleRevision;
        ++mImpl->mSoftTopologyRevision;
    }
    if (previousState.environmentIndex != normalizedState.environmentIndex)
    {
        auto colliderHandlesIt = mImpl->mEntityToColliderIds.find(previousState.entityId);
        if (colliderHandlesIt != mImpl->mEntityToColliderIds.end())
        {
            for (const ColliderId colliderId : colliderHandlesIt->second)
            {
                const auto colliderIt = mImpl->mColliderIdToIndex.find(colliderId);
                if (colliderIt != mImpl->mColliderIdToIndex.end())
                {
                    const std::uint32_t colliderIndex = colliderIt->second;
                    if (colliderIndex < mImpl->mColliders.environmentIndices.size())
                    {
                        mImpl->mColliders.environmentIndices[colliderIndex] =
                            normalizedState.environmentIndex;
                        mImpl->markColliderDirty(colliderIndex);
                    }
                }
            }
        }
        mImpl->invalidateAllRigidBodyDependentResolvedConstraints();
        mImpl->mRigidJointSceneDirty           = true;
        mImpl->mJointCollisionSuppressionDirty = true;
        ++mImpl->mRigidJointSceneRevision;
        ++mImpl->mRigidJointModeRevision;
        ++mImpl->mRigidJointTopologyRevision;
    }
    mImpl->mStaticBroadPhaseDirty = mImpl->mStaticBroadPhaseDirty ||
                                    mImpl->staticBodyPoseChanged(previousState, normalizedState);
    ++mImpl->mAuthoredRevision;
    return mImpl->mRigidBodySnapshot[index];
}

bool PhysicsWorld::removeRigidBody(common::EntityId entityId)
{
    const auto it = mImpl->mEntityToRigidBodyIndex.find(entityId);
    if (it == mImpl->mEntityToRigidBodyIndex.end())
    {
        return false;
    }

    mImpl->removeCollidersForEntity(entityId);

    const std::uint32_t index       = it->second;
    const std::uint32_t last        = static_cast<std::uint32_t>(mImpl->mRigidBodies.size() - 1u);
    const RigidBodyId removedBodyId = mImpl->mRigidBodySnapshot[index].rigidBodyId;
    const bool removedStatic        = mImpl->isStaticBody(mImpl->mRigidBodySnapshot[index]);
    const bool removedProxyParticles =
        !mImpl->mRigidBodySnapshot[index].proxyParticleLocalPositions.empty();
    bool movedProxyParticles = false;

    if (index != last)
    {
        movedProxyParticles = !mImpl->mRigidBodySnapshot[last].proxyParticleLocalPositions.empty();
        mImpl->mRigidBodySnapshot[index]        = mImpl->mRigidBodySnapshot[last];
        mImpl->mRigidBodies.rigidBodyIds[index] = mImpl->mRigidBodies.rigidBodyIds[last];
        mImpl->mRigidBodies.entityIds[index]    = mImpl->mRigidBodies.entityIds[last];
        mImpl->mRigidBodies.environmentIndices[index] =
            mImpl->mRigidBodies.environmentIndices[last];
        mImpl->mRigidBodies.positionsInvMass[index]  = mImpl->mRigidBodies.positionsInvMass[last];
        mImpl->mRigidBodies.orientations[index]      = mImpl->mRigidBodies.orientations[last];
        mImpl->mRigidBodies.scales[index]            = mImpl->mRigidBodies.scales[last];
        mImpl->mRigidBodies.linearVelocities[index]  = mImpl->mRigidBodies.linearVelocities[last];
        mImpl->mRigidBodies.angularVelocities[index] = mImpl->mRigidBodies.angularVelocities[last];
        mImpl->mRigidBodies.inverseInertiaLocal[index] =
            mImpl->mRigidBodies.inverseInertiaLocal[last];
        mImpl->mRigidBodies.bodyTypes[index] = mImpl->mRigidBodies.bodyTypes[last];
        mImpl->mRigidBodies.kinematicTargetPositions[index] =
            mImpl->mRigidBodies.kinematicTargetPositions[last];
        mImpl->mRigidBodies.kinematicTargetOrientations[index] =
            mImpl->mRigidBodies.kinematicTargetOrientations[last];
        mImpl->mRigidBodies.kinematicTargetFlags[index] =
            mImpl->mRigidBodies.kinematicTargetFlags[last];
        mImpl->mEntityToRigidBodyIndex[mImpl->mRigidBodies.entityIds[index]] = index;
        mImpl->mRigidBodyIdToIndex[mImpl->mRigidBodies.rigidBodyIds[index]]  = index;
        mImpl->markRigidBodyDirty(index);
    }

    mImpl->mEntityToRigidBodyIndex.erase(it);
    mImpl->mRigidBodyIdToIndex.erase(removedBodyId);
    mImpl->mRigidBodySnapshot.pop_back();
    mImpl->mRigidBodies.rigidBodyIds.pop_back();
    mImpl->mRigidBodies.entityIds.pop_back();
    mImpl->mRigidBodies.environmentIndices.pop_back();
    mImpl->mRigidBodies.positionsInvMass.pop_back();
    mImpl->mRigidBodies.orientations.pop_back();
    mImpl->mRigidBodies.scales.pop_back();
    mImpl->mRigidBodies.linearVelocities.pop_back();
    mImpl->mRigidBodies.angularVelocities.pop_back();
    mImpl->mRigidBodies.inverseInertiaLocal.pop_back();
    mImpl->mRigidBodies.bodyTypes.pop_back();
    mImpl->mRigidBodies.kinematicTargetPositions.pop_back();
    mImpl->mRigidBodies.kinematicTargetOrientations.pop_back();
    mImpl->mRigidBodies.kinematicTargetFlags.pop_back();
    mImpl->mRigidBodies.proxyParticleContactMaterials.pop_back();

    mImpl->markRigidBodyCountDirty();
    mImpl->markColliderCountDirty(true);
    mImpl->mBodyColliderMappingDirty = true;
    mImpl->mStaticBroadPhaseDirty    = mImpl->mStaticBroadPhaseDirty || removedStatic;
    if (removedProxyParticles || movedProxyParticles)
    {
        mImpl->invalidateSoftDerivedState();
        ++mImpl->mSoftParticleRevision;
        ++mImpl->mSoftTopologyRevision;
    }
    mImpl->invalidateAllRigidBodyDependentResolvedConstraints();
    mImpl->pruneRigidJointsForBody(removedBodyId);
    ++mImpl->mRigidBodyTopologyRevision;
    mImpl->mRigidJointSceneDirty           = true;
    mImpl->mJointCollisionSuppressionDirty = true;
    ++mImpl->mRigidJointSceneRevision;
    ++mImpl->mRigidJointModeRevision;
    ++mImpl->mRigidJointTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

void PhysicsWorld::upsertCollider(const ColliderState &state)
{
    RigidBodyId ownerRigidBodyId = state.ownerRigidBodyId;
    std::uint32_t ownerBodyIndex = 0xffffffffu;

    if (ownerRigidBodyId != kInvalidRigidBodyId)
    {
        const auto bodyIt = mImpl->mRigidBodyIdToIndex.find(ownerRigidBodyId);
        if (bodyIt != mImpl->mRigidBodyIdToIndex.end())
        {
            ownerBodyIndex = bodyIt->second;
        }
    }

    if (ownerBodyIndex == 0xffffffffu)
    {
        const auto bodyIt = mImpl->mEntityToRigidBodyIndex.find(state.entityId);
        if (bodyIt == mImpl->mEntityToRigidBodyIndex.end())
        {
            removeCollider(state.colliderId);
            return;
        }
        ownerBodyIndex   = bodyIt->second;
        ownerRigidBodyId = mImpl->mRigidBodySnapshot[ownerBodyIndex].rigidBodyId;
    }

    ColliderState normalizedState = state;
    mImpl->normalizeColliderState(normalizedState);
    normalizedState.entityId         = mImpl->mRigidBodySnapshot[ownerBodyIndex].entityId;
    normalizedState.ownerRigidBodyId = ownerRigidBodyId;
    if (normalizedState.colliderId == kInvalidColliderId)
    {
        normalizedState.colliderId = mImpl->mNextColliderId++;
    }

    const bool ownerIsStatic = mImpl->isStaticBody(mImpl->mRigidBodySnapshot[ownerBodyIndex]);
    const auto colliderIt    = mImpl->mColliderIdToIndex.find(normalizedState.colliderId);
    if (colliderIt == mImpl->mColliderIdToIndex.end())
    {
        const std::uint32_t colliderIndex = static_cast<std::uint32_t>(mImpl->mColliders.size());
        mImpl->mColliderSnapshot.push_back(normalizedState);
        mImpl->writeColliderSoAAt(mImpl->mColliders, colliderIndex, normalizedState, ownerBodyIndex,
                                  mImpl->mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
        mImpl->mColliderIdToIndex.emplace(normalizedState.colliderId, colliderIndex);
        auto &entityColliderIds = mImpl->mEntityToColliderIds[normalizedState.entityId];
        entityColliderIds.push_back(normalizedState.colliderId);
        const std::uint32_t contribution = mImpl->colliderBroadPhaseContribution(
            normalizedState, &mImpl->mRigidBodySnapshot[ownerBodyIndex]);
        if (contribution == kBroadPhaseContributionStatic)
        {
            ++mImpl->mStaticColliderCount;
        }
        else if (contribution == kBroadPhaseContributionMoving)
        {
            ++mImpl->mActiveMovingColliderCount;
        }
        mImpl->markColliderDirty(colliderIndex);
        mImpl->markColliderCountDirty();
        mImpl->mBodyColliderMappingDirty = true;
        mImpl->mStaticBroadPhaseDirty    = mImpl->mStaticBroadPhaseDirty || ownerIsStatic;
        ++mImpl->mAuthoredRevision;
        return;
    }

    const std::uint32_t colliderIndex = colliderIt->second;
    const ColliderState previousState = mImpl->mColliderSnapshot[colliderIndex];
    const std::uint32_t previousContribution =
        mImpl->broadPhaseContributionForCollider(previousState);
    const std::uint32_t newContribution = mImpl->colliderBroadPhaseContribution(
        normalizedState, &mImpl->mRigidBodySnapshot[ownerBodyIndex]);
    const bool staticContributionChanged = previousContribution == kBroadPhaseContributionStatic ||
                                           newContribution == kBroadPhaseContributionStatic;
    if (previousContribution != newContribution)
    {
        if (previousContribution == kBroadPhaseContributionStatic)
        {
            --mImpl->mStaticColliderCount;
        }
        else if (previousContribution == kBroadPhaseContributionMoving)
        {
            --mImpl->mActiveMovingColliderCount;
        }

        if (newContribution == kBroadPhaseContributionStatic)
        {
            ++mImpl->mStaticColliderCount;
        }
        else if (newContribution == kBroadPhaseContributionMoving)
        {
            ++mImpl->mActiveMovingColliderCount;
        }
    }
    mImpl->writeColliderSoAAt(mImpl->mColliders, colliderIndex, normalizedState, ownerBodyIndex,
                              mImpl->mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
    mImpl->mColliderSnapshot[colliderIndex] = normalizedState;
    mImpl->markColliderDirty(colliderIndex);
    if (previousState.ownerRigidBodyId != normalizedState.ownerRigidBodyId)
    {
        mImpl->mBodyColliderMappingDirty = true;
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
        mImpl->mStaticBroadPhaseDirty = true;
    }
    ++mImpl->mAuthoredRevision;
}

bool PhysicsWorld::removeCollider(ColliderId colliderId)
{
    const auto it = mImpl->mColliderIdToIndex.find(colliderId);
    if (it == mImpl->mColliderIdToIndex.end())
    {
        return false;
    }

    bool removedStaticOwner            = false;
    const std::uint32_t ownerBodyIndex = mImpl->mColliders.ownerRigidBodyIndices[it->second];
    if (ownerBodyIndex != 0xffffffffu && ownerBodyIndex < rigidBodyCount())
    {
        removedStaticOwner = mImpl->isStaticBody(mImpl->mRigidBodySnapshot[ownerBodyIndex]);
    }
    mImpl->removeColliderAtIndex(it->second);
    mImpl->markColliderCountDirty();
    mImpl->mBodyColliderMappingDirty = true;
    mImpl->mStaticBroadPhaseDirty    = mImpl->mStaticBroadPhaseDirty || removedStaticOwner;
    ++mImpl->mAuthoredRevision;
    return true;
}

void PhysicsWorld::replaceColliders(common::EntityId entityId,
                                    const std::vector<ColliderState> &colliders)
{
    mImpl->removeCollidersForEntity(entityId);

    const auto bodyIt = mImpl->mEntityToRigidBodyIndex.find(entityId);
    if (bodyIt == mImpl->mEntityToRigidBodyIndex.end() || colliders.empty())
    {
        mImpl->markColliderCountDirty(true);
        mImpl->mBodyColliderMappingDirty = true;
        ++mImpl->mAuthoredRevision;
        return;
    }

    const std::uint32_t ownerBodyIndex = bodyIt->second;
    const RigidBodyId ownerRigidBodyId = mImpl->mRigidBodySnapshot[ownerBodyIndex].rigidBodyId;

    auto &entityColliderIds = mImpl->mEntityToColliderIds[entityId];
    entityColliderIds.reserve(colliders.size());

    for (ColliderState collider : colliders)
    {
        mImpl->normalizeColliderState(collider);
        if (collider.colliderId == kInvalidColliderId)
        {
            collider.colliderId = mImpl->mNextColliderId++;
        }
        else
        {
            mImpl->mNextColliderId = std::max(mImpl->mNextColliderId, collider.colliderId + 1u);
        }
        collider.entityId         = entityId;
        collider.ownerRigidBodyId = ownerRigidBodyId;

        const std::uint32_t colliderIndex = static_cast<std::uint32_t>(mImpl->mColliders.size());
        mImpl->mColliderSnapshot.push_back(collider);
        mImpl->writeColliderSoAAt(mImpl->mColliders, colliderIndex, collider, ownerBodyIndex,
                                  mImpl->mRigidBodySnapshot[ownerBodyIndex].environmentIndex);
        mImpl->mColliderIdToIndex.emplace(collider.colliderId, colliderIndex);
        entityColliderIds.push_back(collider.colliderId);
        const std::uint32_t contribution = mImpl->colliderBroadPhaseContribution(
            collider, &mImpl->mRigidBodySnapshot[ownerBodyIndex]);
        if (contribution == kBroadPhaseContributionStatic)
        {
            ++mImpl->mStaticColliderCount;
        }
        else if (contribution == kBroadPhaseContributionMoving)
        {
            ++mImpl->mActiveMovingColliderCount;
        }
    }

    mImpl->markColliderCountDirty(true);
    mImpl->mBodyColliderMappingDirty = true;
    mImpl->mStaticBroadPhaseDirty    = true;
    ++mImpl->mAuthoredRevision;
}

bool PhysicsWorld::upsertSoftBody(const SoftBodyState &state)
{
    SoftBodyState normalizedState = state;
    mImpl->normalizeSoftBodyState(normalizedState);

    const auto it = mImpl->mEntityToSoftBodyIndex.find(normalizedState.entityId);
    const SoftBodyState *previousState = nullptr;
    if (it != mImpl->mEntityToSoftBodyIndex.end())
    {
        previousState = &mImpl->mSoftBodySnapshot[it->second];
    }

    if (previousState != nullptr &&
        mImpl->classifySoftBodyChange(*previousState, normalizedState) ==
            Impl::SoftBodyChangeKind::RuntimePropertiesOnly)
    {
        mImpl->applySoftBodyRuntimeProperties(it->second, normalizedState);
        ++mImpl->mSoftParticleRevision;
        ++mImpl->mAuthoredRevision;
        return true;
    }

    Impl::SoftBodyDerivedCache derivedCache;
    if (!mImpl->prepareSoftBodyStateForInsert(normalizedState, previousState, derivedCache))
    {
        return false;
    }

    if (it == mImpl->mEntityToSoftBodyIndex.end())
    {
        mImpl->mEntityToSoftBodyIndex.emplace(
            normalizedState.entityId, static_cast<std::uint32_t>(mImpl->mSoftBodySnapshot.size()));
        mImpl->mSoftBodySnapshot.push_back(normalizedState);
        mImpl->mSoftBodyDerivedCaches.push_back(std::move(derivedCache));
    }
    else
    {
        mImpl->mSoftBodySnapshot[it->second]      = normalizedState;
        mImpl->mSoftBodyDerivedCaches[it->second] = std::move(derivedCache);
    }

    mImpl->invalidateSoftDerivedState();
    mImpl->invalidateResolvedRigidParticleAttachments();
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::upsertStrand(const StrandState &state)
{
    StrandState normalizedState = state;
    mImpl->normalizeStrandState(normalizedState);

    const auto it = mImpl->mEntityToStrandIndex.find(normalizedState.entityId);
    const StrandState *previousState =
        it != mImpl->mEntityToStrandIndex.end() ? &mImpl->mStrandSnapshot[it->second] : nullptr;
    if (previousState != nullptr && mImpl->classifyStrandChange(*previousState, normalizedState) ==
                                        Impl::StrandChangeKind::RuntimePropertiesOnly)
    {
        mImpl->applyStrandRuntimeProperties(it->second, normalizedState);
        ++mImpl->mSoftParticleRevision;
        ++mImpl->mAuthoredRevision;
        return true;
    }

    Impl::StrandDerivedCache derivedCache;
    if (!mImpl->prepareStrandStateForInsert(normalizedState, derivedCache))
    {
        return false;
    }

    if (it == mImpl->mEntityToStrandIndex.end())
    {
        mImpl->mEntityToStrandIndex.emplace(
            normalizedState.entityId, static_cast<std::uint32_t>(mImpl->mStrandSnapshot.size()));
        mImpl->mStrandSnapshot.push_back(normalizedState);
        mImpl->mStrandDerivedCaches.push_back(std::move(derivedCache));
    }
    else
    {
        mImpl->mStrandSnapshot[it->second]      = normalizedState;
        mImpl->mStrandDerivedCaches[it->second] = std::move(derivedCache);
    }

    mImpl->invalidateSoftDerivedState();
    mImpl->invalidateResolvedRigidParticleAttachments();
    mImpl->invalidateResolvedStrandRigidAttachments(true);
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::upsertFluid(const FluidState &state)
{
    FluidState normalizedState = state;
    mImpl->normalizeFluidState(normalizedState);

    const auto it = mImpl->mEntityToFluidIndex.find(normalizedState.entityId);
    const FluidState *previousState =
        it != mImpl->mEntityToFluidIndex.end() ? &mImpl->mFluidSnapshot[it->second] : nullptr;
    if (!mImpl->validateFluidMaterialCompatibility(normalizedState, previousState))
    {
        return false;
    }

    Impl::FluidDerivedCache derivedCache;
    if (!mImpl->prepareFluidStateForInsert(normalizedState, derivedCache))
    {
        return false;
    }

    if (it == mImpl->mEntityToFluidIndex.end())
    {
        mImpl->mEntityToFluidIndex.emplace(
            normalizedState.entityId, static_cast<std::uint32_t>(mImpl->mFluidSnapshot.size()));
        mImpl->mFluidSnapshot.push_back(normalizedState);
        mImpl->mFluidDerivedCaches.push_back(std::move(derivedCache));
    }
    else
    {
        mImpl->mFluidSnapshot[it->second]      = normalizedState;
        mImpl->mFluidDerivedCaches[it->second] = std::move(derivedCache);
    }
    mImpl->invalidateSoftDerivedState();
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

AuthoredParticleDistanceConstraintState &PhysicsWorld::upsertParticleDistanceConstraint(
    const AuthoredParticleDistanceConstraintState &state)
{
    AuthoredParticleDistanceConstraintState normalizedState = state;
    normalizedState.restLength = std::max(normalizedState.restLength, 0.0f);
    normalizedState.compliance = std::max(normalizedState.compliance, 0.0f);

    auto it = mImpl->mParticleConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidParticleConstraintId ||
        it == mImpl->mParticleConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mImpl->mNextParticleConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mImpl->mParticleDistanceConstraintSnapshot.size());
        mImpl->mParticleConstraintIdToIndex.emplace(normalizedState.constraintId, index);
        mImpl->mParticleDistanceConstraintSnapshot.push_back(normalizedState);
        mImpl->markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData);
        ++mImpl->mSoftBodyTopologyRevision;
        ++mImpl->mSoftTopologyRevision;
        ++mImpl->mAuthoredRevision;
        return mImpl->mParticleDistanceConstraintSnapshot.back();
    }

    mImpl->mParticleDistanceConstraintSnapshot[it->second] = normalizedState;
    mImpl->markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData);
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return mImpl->mParticleDistanceConstraintSnapshot[it->second];
}

bool PhysicsWorld::upsertRigidParticleAttachmentConstraint(
    const AuthoredRigidParticleAttachmentConstraintState &state,
    AuthoredRigidParticleAttachmentConstraintState *outAuthored)
{
    AuthoredRigidParticleAttachmentConstraintState normalizedState = state;
    normalizedState.compliance = std::max(normalizedState.compliance, 0.0f);

    const auto bodyIt = mImpl->mEntityToRigidBodyIndex.find(normalizedState.rigidBodyEntityId);
    if (bodyIt == mImpl->mEntityToRigidBodyIndex.end() ||
        bodyIt->second >= mImpl->mRigidBodySnapshot.size())
    {
        return false;
    }

    std::uint32_t particleEnvironmentIndex = 0u;
    switch (normalizedState.particle.type)
    {
    case AuthoredParticleReferenceType::SoftBodyParticle:
    {
        const auto softBodyIt =
            mImpl->mEntityToSoftBodyIndex.find(normalizedState.particle.entityId);
        if (softBodyIt == mImpl->mEntityToSoftBodyIndex.end() ||
            softBodyIt->second >= mImpl->mSoftBodySnapshot.size() ||
            softBodyIt->second >= mImpl->mSoftBodyDerivedCaches.size() ||
            normalizedState.particle.localParticleIndex >=
                mImpl->mSoftBodyDerivedCaches[softBodyIt->second].restPositions.size())
        {
            return false;
        }
        particleEnvironmentIndex = mImpl->mSoftBodySnapshot[softBodyIt->second].environmentIndex;
        break;
    }
    case AuthoredParticleReferenceType::StrandParticle:
    {
        const auto strandIt = mImpl->mEntityToStrandIndex.find(normalizedState.particle.entityId);
        if (strandIt == mImpl->mEntityToStrandIndex.end() ||
            strandIt->second >= mImpl->mStrandSnapshot.size() ||
            strandIt->second >= mImpl->mStrandDerivedCaches.size() ||
            normalizedState.particle.localParticleIndex >=
                mImpl->mStrandDerivedCaches[strandIt->second].restPositions.size())
        {
            return false;
        }
        particleEnvironmentIndex = mImpl->mStrandSnapshot[strandIt->second].environmentIndex;
        break;
    }
    case AuthoredParticleReferenceType::RigidProxyParticle:
    {
        const RigidBodyState *proxyBody = tryGetRigidBody(normalizedState.particle.entityId);
        if (proxyBody == nullptr || normalizedState.particle.localParticleIndex >=
                                        proxyBody->proxyParticleLocalPositions.size())
        {
            return false;
        }
        particleEnvironmentIndex = proxyBody->environmentIndex;
        break;
    }
    }

    if (particleEnvironmentIndex != mImpl->mRigidBodySnapshot[bodyIt->second].environmentIndex)
    {
        return false;
    }

    auto it = mImpl->mRigidParticleAttachmentConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidRigidParticleAttachmentConstraintId ||
        it == mImpl->mRigidParticleAttachmentConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mImpl->mNextRigidParticleAttachmentConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mImpl->mRigidParticleAttachmentConstraintSnapshot.size());
        mImpl->mRigidParticleAttachmentConstraintIdToIndex.emplace(normalizedState.constraintId,
                                                                   index);
        mImpl->mRigidParticleAttachmentConstraintSnapshot.push_back(normalizedState);
        mImpl->invalidateResolvedRigidParticleAttachments();
        ++mImpl->mRigidParticleAttachmentDefinitionRevision;
        ++mImpl->mAuthoredRevision;
        if (outAuthored != nullptr)
        {
            *outAuthored = mImpl->mRigidParticleAttachmentConstraintSnapshot.back();
        }
        return true;
    }

    mImpl->mRigidParticleAttachmentConstraintSnapshot[it->second] = normalizedState;
    mImpl->invalidateResolvedRigidParticleAttachments();
    ++mImpl->mRigidParticleAttachmentDefinitionRevision;
    ++mImpl->mAuthoredRevision;
    if (outAuthored != nullptr)
    {
        *outAuthored = mImpl->mRigidParticleAttachmentConstraintSnapshot[it->second];
    }
    return true;
}

bool PhysicsWorld::upsertStrandRigidAttachmentConstraint(
    const AuthoredStrandRigidAttachmentConstraintState &state,
    AuthoredStrandRigidAttachmentConstraintState *outAuthored)
{
    AuthoredStrandRigidAttachmentConstraintState normalizedState = state;
    normalizedState.segmentT = std::clamp(normalizedState.segmentT, 0.0f, 1.0f);
    normalizedState.localRotation =
        common::runtime_math::normalizeQuaternion(normalizedState.localRotation);
    normalizedState.translationCompliance = std::max(normalizedState.translationCompliance, 0.0f);
    normalizedState.rotationCompliance    = std::max(normalizedState.rotationCompliance, 0.0f);

    const auto strandIt = mImpl->mEntityToStrandIndex.find(normalizedState.strandEntityId);
    const RigidBodyState *rigidBody = tryGetRigidBody(normalizedState.rigidBodyEntityId);
    if (strandIt == mImpl->mEntityToStrandIndex.end() ||
        strandIt->second >= mImpl->mStrandSnapshot.size() ||
        strandIt->second >= mImpl->mStrandDerivedCaches.size() || rigidBody == nullptr ||
        normalizedState.localSegmentIndex >=
            mImpl->mStrandDerivedCaches[strandIt->second].segments.size() ||
        mImpl->mStrandSnapshot[strandIt->second].environmentIndex != rigidBody->environmentIndex)
    {
        return false;
    }

    auto it = mImpl->mStrandRigidAttachmentConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidStrandRigidAttachmentConstraintId ||
        it == mImpl->mStrandRigidAttachmentConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mImpl->mNextStrandRigidAttachmentConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mImpl->mStrandRigidAttachmentConstraintSnapshot.size());
        mImpl->mStrandRigidAttachmentConstraintIdToIndex.emplace(normalizedState.constraintId,
                                                                 index);
        mImpl->mStrandRigidAttachmentConstraintSnapshot.push_back(normalizedState);
        mImpl->invalidateResolvedStrandRigidAttachments(true);
        ++mImpl->mStrandRigidAttachmentDefinitionRevision;
        ++mImpl->mAuthoredRevision;
        if (outAuthored != nullptr)
        {
            *outAuthored = mImpl->mStrandRigidAttachmentConstraintSnapshot.back();
        }
        return true;
    }

    const AuthoredStrandRigidAttachmentConstraintState &previousState =
        mImpl->mStrandRigidAttachmentConstraintSnapshot[it->second];
    const bool affectsSoftAdjacency =
        previousState.enabled != normalizedState.enabled ||
        previousState.strandEntityId != normalizedState.strandEntityId ||
        previousState.localSegmentIndex != normalizedState.localSegmentIndex ||
        previousState.rigidBodyEntityId != normalizedState.rigidBodyEntityId;
    mImpl->mStrandRigidAttachmentConstraintSnapshot[it->second] = normalizedState;
    mImpl->invalidateResolvedStrandRigidAttachments(affectsSoftAdjacency);
    ++mImpl->mStrandRigidAttachmentDefinitionRevision;
    ++mImpl->mAuthoredRevision;
    if (outAuthored != nullptr)
    {
        *outAuthored = mImpl->mStrandRigidAttachmentConstraintSnapshot[it->second];
    }
    return true;
}

bool PhysicsWorld::upsertRigidDistanceConstraint(const AuthoredRigidDistanceConstraintState &state,
                                                 AuthoredRigidDistanceConstraintState *outAuthored)
{
    AuthoredRigidDistanceConstraintState normalizedState = state;
    normalizedState.restDistance = std::max(normalizedState.restDistance, 0.0f);
    normalizedState.compliance   = std::max(normalizedState.compliance, 0.0f);

    const RigidBodyState *bodyA = tryGetRigidBody(normalizedState.entityA);
    const RigidBodyState *bodyB = tryGetRigidBody(normalizedState.entityB);
    if (bodyA == nullptr || bodyB == nullptr ||
        normalizedState.entityA == normalizedState.entityB ||
        bodyA->environmentIndex != bodyB->environmentIndex)
    {
        return false;
    }

    auto it = mImpl->mRigidDistanceConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidRigidDistanceConstraintId ||
        it == mImpl->mRigidDistanceConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mImpl->mNextRigidDistanceConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mImpl->mRigidDistanceConstraintSnapshot.size());
        mImpl->mRigidDistanceConstraintIdToIndex.emplace(normalizedState.constraintId, index);
        mImpl->mRigidDistanceConstraintSnapshot.push_back(normalizedState);
        mImpl->invalidateResolvedRigidDistanceConstraints();
        ++mImpl->mRigidDistanceConstraintDefinitionRevision;
        ++mImpl->mAuthoredRevision;
        if (outAuthored != nullptr)
        {
            *outAuthored = mImpl->mRigidDistanceConstraintSnapshot.back();
        }
        return true;
    }

    mImpl->mRigidDistanceConstraintSnapshot[it->second] = normalizedState;
    mImpl->invalidateResolvedRigidDistanceConstraints();
    ++mImpl->mRigidDistanceConstraintDefinitionRevision;
    ++mImpl->mAuthoredRevision;
    if (outAuthored != nullptr)
    {
        *outAuthored = mImpl->mRigidDistanceConstraintSnapshot[it->second];
    }
    return true;
}

bool PhysicsWorld::upsertRoutedCableConstraint(const AuthoredRoutedCableConstraintState &state,
                                               AuthoredRoutedCableConstraintState *outAuthored)
{
    AuthoredRoutedCableConstraintState normalizedState = state;
    normalizedState.targetLength = std::max(normalizedState.targetLength, 0.0f);
    normalizedState.compliance   = std::max(normalizedState.compliance, 0.0f);

    if (normalizedState.routePoints.size() < 2u)
    {
        return false;
    }

    std::optional<std::uint32_t> environmentIndex;
    std::optional<common::EntityId> previousEntityId;
    for (const AuthoredRoutedCableRoutePoint &routePoint : normalizedState.routePoints)
    {
        const RigidBodyState *body = tryGetRigidBody(routePoint.entityId);
        if (body == nullptr)
        {
            return false;
        }
        if (!environmentIndex.has_value())
        {
            environmentIndex = body->environmentIndex;
        }
        else if (*environmentIndex != body->environmentIndex)
        {
            return false;
        }
        if (previousEntityId.has_value() && *previousEntityId == routePoint.entityId)
        {
            return false;
        }
        previousEntityId = routePoint.entityId;
    }

    auto it = mImpl->mRoutedCableConstraintIdToIndex.find(normalizedState.constraintId);
    if (normalizedState.constraintId == kInvalidRoutedCableConstraintId ||
        it == mImpl->mRoutedCableConstraintIdToIndex.end())
    {
        normalizedState.constraintId = mImpl->mNextRoutedCableConstraintId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mImpl->mRoutedCableConstraintSnapshot.size());
        mImpl->mRoutedCableConstraintIdToIndex.emplace(normalizedState.constraintId, index);
        mImpl->mRoutedCableConstraintSnapshot.push_back(normalizedState);
        mImpl->invalidateResolvedRoutedCables();
        ++mImpl->mRoutedCableDefinitionRevision;
        ++mImpl->mAuthoredRevision;
        if (outAuthored != nullptr)
        {
            *outAuthored = mImpl->mRoutedCableConstraintSnapshot.back();
        }
        return true;
    }

    mImpl->mRoutedCableConstraintSnapshot[it->second] = normalizedState;
    mImpl->invalidateResolvedRoutedCables();
    ++mImpl->mRoutedCableDefinitionRevision;
    ++mImpl->mAuthoredRevision;
    if (outAuthored != nullptr)
    {
        *outAuthored = mImpl->mRoutedCableConstraintSnapshot[it->second];
    }
    return true;
}

AuthoredParticleCollisionFilterState &PhysicsWorld::upsertParticleCollisionFilter(
    const AuthoredParticleCollisionFilterState &state)
{
    AuthoredParticleCollisionFilterState normalizedState = state;
    if (normalizedState.collisionLayer == 0u)
    {
        normalizedState.collisionLayer = 1u;
    }

    auto it = mImpl->mParticleCollisionFilterIdToIndex.find(normalizedState.filterId);
    if (normalizedState.filterId == kInvalidParticleCollisionFilterId ||
        it == mImpl->mParticleCollisionFilterIdToIndex.end())
    {
        normalizedState.filterId = mImpl->mNextParticleCollisionFilterId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mImpl->mParticleCollisionFilterSnapshot.size());
        mImpl->mParticleCollisionFilterIdToIndex.emplace(normalizedState.filterId, index);
        mImpl->mParticleCollisionFilterSnapshot.push_back(normalizedState);
        mImpl->markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData);
        ++mImpl->mSoftParticleRevision;
        ++mImpl->mAuthoredRevision;
        return mImpl->mParticleCollisionFilterSnapshot.back();
    }

    mImpl->mParticleCollisionFilterSnapshot[it->second] = normalizedState;
    mImpl->markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData);
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mAuthoredRevision;
    return mImpl->mParticleCollisionFilterSnapshot[it->second];
}

AuthoredSuturingSequenceState &PhysicsWorld::upsertSuturingSequence(
    const AuthoredSuturingSequenceState &state)
{
    AuthoredSuturingSequenceState normalizedState = state;
    normalizedState.pathNodeSpacing               = std::max(normalizedState.pathNodeSpacing, 0.0f);
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

    auto it = mImpl->mSuturingSequenceIdToIndex.find(normalizedState.sequenceId);
    if (normalizedState.sequenceId == kInvalidSuturingSequenceId ||
        it == mImpl->mSuturingSequenceIdToIndex.end())
    {
        normalizedState.sequenceId = mImpl->mNextSuturingSequenceId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mImpl->mSuturingSequenceSnapshot.size());
        mImpl->mSuturingSequenceIdToIndex.emplace(normalizedState.sequenceId, index);
        mImpl->mSuturingSequenceSnapshot.push_back(normalizedState);
        mImpl->markRebuildDirty(PhysicsRebuildFlags::SuturingData);
        ++mImpl->mSoftBodyTopologyRevision;
        ++mImpl->mSoftTopologyRevision;
        ++mImpl->mAuthoredRevision;
        return mImpl->mSuturingSequenceSnapshot.back();
    }

    mImpl->mSuturingSequenceSnapshot[it->second] = normalizedState;
    mImpl->markRebuildDirty(PhysicsRebuildFlags::SuturingData);
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return mImpl->mSuturingSequenceSnapshot[it->second];
}

AuthoredParticleSequenceState &PhysicsWorld::upsertParticleSequence(
    const AuthoredParticleSequenceState &state)
{
    AuthoredParticleSequenceState normalizedState = state;
    auto it = mImpl->mParticleSequenceIdToIndex.find(normalizedState.sequenceId);
    if (normalizedState.sequenceId == kInvalidParticleSequenceId ||
        it == mImpl->mParticleSequenceIdToIndex.end())
    {
        normalizedState.sequenceId = mImpl->mNextParticleSequenceId++;
        const std::uint32_t index =
            static_cast<std::uint32_t>(mImpl->mParticleSequenceSnapshot.size());
        mImpl->mParticleSequenceIdToIndex.emplace(normalizedState.sequenceId, index);
        mImpl->mParticleSequenceSnapshot.push_back(normalizedState);
        ++mImpl->mAuthoredRevision;
        return mImpl->mParticleSequenceSnapshot.back();
    }

    mImpl->mParticleSequenceSnapshot[it->second] = normalizedState;
    ++mImpl->mAuthoredRevision;
    return mImpl->mParticleSequenceSnapshot[it->second];
}

bool PhysicsWorld::removeSoftBody(common::EntityId entityId)
{
    const auto it = mImpl->mEntityToSoftBodyIndex.find(entityId);
    if (it == mImpl->mEntityToSoftBodyIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mImpl->mSoftBodySnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mSoftBodySnapshot[index]      = mImpl->mSoftBodySnapshot[last];
        mImpl->mSoftBodyDerivedCaches[index] = std::move(mImpl->mSoftBodyDerivedCaches[last]);
        mImpl->mEntityToSoftBodyIndex[mImpl->mSoftBodySnapshot[index].entityId] = index;
    }
    mImpl->mSoftBodySnapshot.pop_back();
    mImpl->mSoftBodyDerivedCaches.pop_back();
    mImpl->mEntityToSoftBodyIndex.erase(it);
    mImpl->mTetGenMeshCache.erase(entityId);
    mImpl->invalidateSoftDerivedState();
    mImpl->invalidateResolvedRigidParticleAttachments();
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeStrand(common::EntityId entityId)
{
    const auto it = mImpl->mEntityToStrandIndex.find(entityId);
    if (it == mImpl->mEntityToStrandIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mImpl->mStrandSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mStrandSnapshot[index]      = mImpl->mStrandSnapshot[last];
        mImpl->mStrandDerivedCaches[index] = std::move(mImpl->mStrandDerivedCaches[last]);
        mImpl->mEntityToStrandIndex[mImpl->mStrandSnapshot[index].entityId] = index;
    }
    mImpl->mStrandSnapshot.pop_back();
    mImpl->mStrandDerivedCaches.pop_back();
    mImpl->mEntityToStrandIndex.erase(it);
    mImpl->invalidateSoftDerivedState();
    mImpl->invalidateResolvedRigidParticleAttachments();
    mImpl->invalidateResolvedStrandRigidAttachments(true);
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeFluid(common::EntityId entityId)
{
    const auto it = mImpl->mEntityToFluidIndex.find(entityId);
    if (it == mImpl->mEntityToFluidIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mImpl->mFluidSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mFluidSnapshot[index]      = mImpl->mFluidSnapshot[last];
        mImpl->mFluidDerivedCaches[index] = std::move(mImpl->mFluidDerivedCaches[last]);
        mImpl->mEntityToFluidIndex[mImpl->mFluidSnapshot[index].entityId] = index;
    }
    mImpl->mFluidSnapshot.pop_back();
    mImpl->mFluidDerivedCaches.pop_back();
    mImpl->mEntityToFluidIndex.erase(it);
    mImpl->invalidateSoftDerivedState();
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeParticleDistanceConstraint(ParticleConstraintId constraintId)
{
    const auto it = mImpl->mParticleConstraintIdToIndex.find(constraintId);
    if (it == mImpl->mParticleConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mImpl->mParticleDistanceConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mParticleDistanceConstraintSnapshot[index] =
            mImpl->mParticleDistanceConstraintSnapshot[last];
        mImpl->mParticleConstraintIdToIndex[mImpl->mParticleDistanceConstraintSnapshot[index]
                                                .constraintId] = index;
    }

    mImpl->mParticleConstraintIdToIndex.erase(it);
    mImpl->mParticleDistanceConstraintSnapshot.pop_back();
    mImpl->markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData);
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeRigidParticleAttachmentConstraint(
    RigidParticleAttachmentConstraintId constraintId)
{
    const auto it = mImpl->mRigidParticleAttachmentConstraintIdToIndex.find(constraintId);
    if (it == mImpl->mRigidParticleAttachmentConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mImpl->mRigidParticleAttachmentConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mRigidParticleAttachmentConstraintSnapshot[index] =
            mImpl->mRigidParticleAttachmentConstraintSnapshot[last];
        mImpl->mRigidParticleAttachmentConstraintIdToIndex
            [mImpl->mRigidParticleAttachmentConstraintSnapshot[index].constraintId] = index;
    }

    mImpl->mRigidParticleAttachmentConstraintIdToIndex.erase(it);
    mImpl->mRigidParticleAttachmentConstraintSnapshot.pop_back();
    mImpl->invalidateResolvedRigidParticleAttachments();
    ++mImpl->mRigidParticleAttachmentDefinitionRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeStrandRigidAttachmentConstraint(
    StrandRigidAttachmentConstraintId constraintId)
{
    const auto it = mImpl->mStrandRigidAttachmentConstraintIdToIndex.find(constraintId);
    if (it == mImpl->mStrandRigidAttachmentConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mImpl->mStrandRigidAttachmentConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mStrandRigidAttachmentConstraintSnapshot[index] =
            mImpl->mStrandRigidAttachmentConstraintSnapshot[last];
        mImpl->mStrandRigidAttachmentConstraintIdToIndex
            [mImpl->mStrandRigidAttachmentConstraintSnapshot[index].constraintId] = index;
    }

    mImpl->mStrandRigidAttachmentConstraintIdToIndex.erase(it);
    mImpl->mStrandRigidAttachmentConstraintSnapshot.pop_back();
    mImpl->invalidateResolvedStrandRigidAttachments(true);
    ++mImpl->mStrandRigidAttachmentDefinitionRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeRigidDistanceConstraint(RigidDistanceConstraintId constraintId)
{
    const auto it = mImpl->mRigidDistanceConstraintIdToIndex.find(constraintId);
    if (it == mImpl->mRigidDistanceConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mImpl->mRigidDistanceConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mRigidDistanceConstraintSnapshot[index] =
            mImpl->mRigidDistanceConstraintSnapshot[last];
        mImpl->mRigidDistanceConstraintIdToIndex[mImpl->mRigidDistanceConstraintSnapshot[index]
                                                     .constraintId] = index;
    }

    mImpl->mRigidDistanceConstraintIdToIndex.erase(it);
    mImpl->mRigidDistanceConstraintSnapshot.pop_back();
    mImpl->invalidateResolvedRigidDistanceConstraints();
    ++mImpl->mRigidDistanceConstraintDefinitionRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeRoutedCableConstraint(RoutedCableConstraintId constraintId)
{
    const auto it = mImpl->mRoutedCableConstraintIdToIndex.find(constraintId);
    if (it == mImpl->mRoutedCableConstraintIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mImpl->mRoutedCableConstraintSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mRoutedCableConstraintSnapshot[index] = mImpl->mRoutedCableConstraintSnapshot[last];
        mImpl->mRoutedCableConstraintIdToIndex[mImpl->mRoutedCableConstraintSnapshot[index]
                                                   .constraintId] = index;
    }

    mImpl->mRoutedCableConstraintIdToIndex.erase(it);
    mImpl->mRoutedCableConstraintSnapshot.pop_back();
    mImpl->invalidateResolvedRoutedCables();
    ++mImpl->mRoutedCableDefinitionRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeParticleSequence(ParticleSequenceId sequenceId)
{
    const auto it = mImpl->mParticleSequenceIdToIndex.find(sequenceId);
    if (it == mImpl->mParticleSequenceIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mImpl->mParticleSequenceSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mParticleSequenceSnapshot[index] = mImpl->mParticleSequenceSnapshot[last];
        mImpl->mParticleSequenceIdToIndex[mImpl->mParticleSequenceSnapshot[index].sequenceId] =
            index;
    }

    mImpl->mParticleSequenceIdToIndex.erase(it);
    mImpl->mParticleSequenceSnapshot.pop_back();
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeParticleCollisionFilter(ParticleCollisionFilterId filterId)
{
    const auto it = mImpl->mParticleCollisionFilterIdToIndex.find(filterId);
    if (it == mImpl->mParticleCollisionFilterIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mImpl->mParticleCollisionFilterSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mParticleCollisionFilterSnapshot[index] =
            mImpl->mParticleCollisionFilterSnapshot[last];
        mImpl->mParticleCollisionFilterIdToIndex[mImpl->mParticleCollisionFilterSnapshot[index]
                                                     .filterId] = index;
    }

    mImpl->mParticleCollisionFilterIdToIndex.erase(it);
    mImpl->mParticleCollisionFilterSnapshot.pop_back();
    mImpl->markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData);
    ++mImpl->mSoftParticleRevision;
    ++mImpl->mAuthoredRevision;
    return true;
}

bool PhysicsWorld::removeSuturingSequence(SuturingSequenceId sequenceId)
{
    const auto it = mImpl->mSuturingSequenceIdToIndex.find(sequenceId);
    if (it == mImpl->mSuturingSequenceIdToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last =
        static_cast<std::uint32_t>(mImpl->mSuturingSequenceSnapshot.size() - 1u);
    if (index != last)
    {
        mImpl->mSuturingSequenceSnapshot[index] = mImpl->mSuturingSequenceSnapshot[last];
        mImpl->mSuturingSequenceIdToIndex[mImpl->mSuturingSequenceSnapshot[index].sequenceId] =
            index;
    }

    mImpl->mSuturingSequenceIdToIndex.erase(it);
    mImpl->mSuturingSequenceSnapshot.pop_back();
    mImpl->markRebuildDirty(PhysicsRebuildFlags::SuturingData);
    ++mImpl->mSoftBodyTopologyRevision;
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
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
        normalized.jointId = mImpl->mNextBallJointId++;
    }
    const auto bodyAIt = mImpl->mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mImpl->mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mImpl->mRigidBodyIdToIndex.end() || bodyBIt == mImpl->mRigidBodyIdToIndex.end())
    {
        return false;
    }
    if (mImpl->mRigidBodySnapshot[bodyAIt->second].environmentIndex !=
        mImpl->mRigidBodySnapshot[bodyBIt->second].environmentIndex)
    {
        return false;
    }

    auto it = std::find_if(mImpl->mBallJointSnapshot.begin(), mImpl->mBallJointSnapshot.end(),
                           [&](const BallJointState &existing)
                           { return existing.jointId == normalized.jointId; });
    const bool inserted = it == mImpl->mBallJointSnapshot.end();
    if (it == mImpl->mBallJointSnapshot.end())
    {
        mImpl->mBallJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    mImpl->applyRigidJointChange(mImpl->classifyBallJointChange(inserted));
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
    if (!normalized.limitEnabled)
    {
        normalized.limitMin = 0.0f;
        normalized.limitMax = 0.0f;
    }
    normalized.constraintCompliance    = std::max(normalized.constraintCompliance, 0.0f);
    normalized.driveCompliance         = std::max(normalized.driveCompliance, 0.0f);
    normalized.driveDamping            = std::max(normalized.driveDamping, 0.0f);
    normalized.driveMaxAngularVelocity = std::max(normalized.driveMaxAngularVelocity, 0.0f);
    if (normalized.jointId == kInvalidHingeJointId)
    {
        normalized.jointId = mImpl->mNextHingeJointId++;
    }
    const auto bodyAIt = mImpl->mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mImpl->mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mImpl->mRigidBodyIdToIndex.end() || bodyBIt == mImpl->mRigidBodyIdToIndex.end())
    {
        return false;
    }
    if (mImpl->mRigidBodySnapshot[bodyAIt->second].environmentIndex !=
        mImpl->mRigidBodySnapshot[bodyBIt->second].environmentIndex)
    {
        return false;
    }

    auto it = std::find_if(mImpl->mHingeJointSnapshot.begin(), mImpl->mHingeJointSnapshot.end(),
                           [&](const HingeJointState &existing)
                           { return existing.jointId == normalized.jointId; });
    const bool inserted                 = it == mImpl->mHingeJointSnapshot.end();
    const HingeJointState previousState = inserted ? HingeJointState{} : *it;
    if (it == mImpl->mHingeJointSnapshot.end())
    {
        mImpl->mHingeJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    mImpl->applyRigidJointChange(
        mImpl->classifyHingeJointChange(previousState, normalized, inserted));
    return true;
}

bool PhysicsWorld::upsertSphericalJoint(const SphericalJointState &state)
{
    SphericalJointState normalized = state;
    if (normalized.bodyA == kInvalidRigidBodyId || normalized.bodyB == kInvalidRigidBodyId ||
        normalized.bodyA == normalized.bodyB)
    {
        return false;
    }

    normalized.localRotationA =
        common::runtime_math::normalizeQuaternion(normalized.localRotationA);
    normalized.localRotationB =
        common::runtime_math::normalizeQuaternion(normalized.localRotationB);
    normalized.driveTargetOrientation =
        common::runtime_math::normalizeQuaternion(normalized.driveTargetOrientation);
    normalized.constraintCompliance = std::max(normalized.constraintCompliance, 0.0f);
    normalized.swingCompliance      = std::max(normalized.swingCompliance, 0.0f);
    normalized.twistCompliance      = std::max(normalized.twistCompliance, 0.0f);
    normalized.driveCompliance      = std::max(normalized.driveCompliance, 0.0f);
    normalized.swingLimitY          = std::max(normalized.swingLimitY, 0.0f);
    normalized.swingLimitZ          = std::max(normalized.swingLimitZ, 0.0f);
    if (normalized.limitEnabled && normalized.twistLimitMin > normalized.twistLimitMax)
    {
        std::swap(normalized.twistLimitMin, normalized.twistLimitMax);
    }
    if (!normalized.limitEnabled)
    {
        normalized.swingLimitY   = 0.0f;
        normalized.swingLimitZ   = 0.0f;
        normalized.twistLimitMin = 0.0f;
        normalized.twistLimitMax = 0.0f;
    }
    if (normalized.driveMode != RigidJointDriveMode::None &&
        normalized.driveMode != RigidJointDriveMode::TargetOrientation)
    {
        return false;
    }
    if (normalized.jointId == kInvalidSphericalJointId)
    {
        normalized.jointId = mImpl->mNextSphericalJointId++;
    }

    const auto bodyAIt = mImpl->mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mImpl->mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mImpl->mRigidBodyIdToIndex.end() || bodyBIt == mImpl->mRigidBodyIdToIndex.end())
    {
        return false;
    }
    if (mImpl->mRigidBodySnapshot[bodyAIt->second].environmentIndex !=
        mImpl->mRigidBodySnapshot[bodyBIt->second].environmentIndex)
    {
        return false;
    }

    auto it =
        std::find_if(mImpl->mSphericalJointSnapshot.begin(), mImpl->mSphericalJointSnapshot.end(),
                     [&](const SphericalJointState &existing)
                     { return existing.jointId == normalized.jointId; });
    const bool inserted                     = it == mImpl->mSphericalJointSnapshot.end();
    const SphericalJointState previousState = inserted ? SphericalJointState{} : *it;
    if (inserted)
    {
        mImpl->mSphericalJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    mImpl->applyRigidJointChange(
        mImpl->classifySphericalJointChange(previousState, normalized, inserted));
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
    normalized.driveDamping         = std::max(normalized.driveDamping, 0.0f);
    normalized.driveMaxVelocity     = std::max(normalized.driveMaxVelocity, 0.0f);

    const auto bodyAIt = mImpl->mRigidBodyIdToIndex.find(normalized.bodyA);
    const auto bodyBIt = mImpl->mRigidBodyIdToIndex.find(normalized.bodyB);
    if (bodyAIt == mImpl->mRigidBodyIdToIndex.end() || bodyBIt == mImpl->mRigidBodyIdToIndex.end())
    {
        return false;
    }
    const RigidBodyState &bodyA = mImpl->mRigidBodySnapshot[bodyAIt->second];
    const RigidBodyState &bodyB = mImpl->mRigidBodySnapshot[bodyBIt->second];
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
        normalized.jointId = mImpl->mNextSliderJointId++;
    }

    auto it = std::find_if(mImpl->mSliderJointSnapshot.begin(), mImpl->mSliderJointSnapshot.end(),
                           [&](const SliderJointState &existing)
                           { return existing.jointId == normalized.jointId; });
    const bool inserted                  = it == mImpl->mSliderJointSnapshot.end();
    const SliderJointState previousState = inserted ? SliderJointState{} : *it;
    if (it == mImpl->mSliderJointSnapshot.end())
    {
        mImpl->mSliderJointSnapshot.push_back(normalized);
    }
    else
    {
        *it = normalized;
    }

    mImpl->applyRigidJointChange(
        mImpl->classifySliderJointChange(previousState, normalized, inserted));
    return true;
}

bool PhysicsWorld::removeBallJoint(BallJointId jointId)
{
    auto it = std::find_if(mImpl->mBallJointSnapshot.begin(), mImpl->mBallJointSnapshot.end(),
                           [&](const BallJointState &state) { return state.jointId == jointId; });
    if (it != mImpl->mBallJointSnapshot.end())
    {
        mImpl->mBallJointSnapshot.erase(it);
        mImpl->applyRigidJointChange(Impl::RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

bool PhysicsWorld::removeHingeJoint(HingeJointId jointId)
{
    auto it = std::find_if(mImpl->mHingeJointSnapshot.begin(), mImpl->mHingeJointSnapshot.end(),
                           [&](const HingeJointState &state) { return state.jointId == jointId; });
    if (it != mImpl->mHingeJointSnapshot.end())
    {
        mImpl->mHingeJointSnapshot.erase(it);
        mImpl->applyRigidJointChange(Impl::RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

bool PhysicsWorld::removeSphericalJoint(SphericalJointId jointId)
{
    auto it =
        std::find_if(mImpl->mSphericalJointSnapshot.begin(), mImpl->mSphericalJointSnapshot.end(),
                     [&](const SphericalJointState &state) { return state.jointId == jointId; });
    if (it != mImpl->mSphericalJointSnapshot.end())
    {
        mImpl->mSphericalJointSnapshot.erase(it);
        mImpl->applyRigidJointChange(Impl::RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

bool PhysicsWorld::removeSliderJoint(SliderJointId jointId)
{
    auto it = std::find_if(mImpl->mSliderJointSnapshot.begin(), mImpl->mSliderJointSnapshot.end(),
                           [&](const SliderJointState &state) { return state.jointId == jointId; });
    if (it != mImpl->mSliderJointSnapshot.end())
    {
        mImpl->mSliderJointSnapshot.erase(it);
        mImpl->applyRigidJointChange(Impl::RigidJointChangeKind::TopologyRebuild);
        return true;
    }
    return false;
}

RigidBodyState *PhysicsWorld::tryGetRigidBody(common::EntityId entityId)
{
    const auto it = mImpl->mEntityToRigidBodyIndex.find(entityId);
    return it == mImpl->mEntityToRigidBodyIndex.end() ? nullptr
                                                      : &mImpl->mRigidBodySnapshot[it->second];
}

const RigidBodyState *PhysicsWorld::tryGetRigidBody(common::EntityId entityId) const
{
    const auto it = mImpl->mEntityToRigidBodyIndex.find(entityId);
    return it == mImpl->mEntityToRigidBodyIndex.end() ? nullptr
                                                      : &mImpl->mRigidBodySnapshot[it->second];
}

const ColliderState *PhysicsWorld::tryGetCollider(ColliderId colliderId) const
{
    const auto it = mImpl->mColliderIdToIndex.find(colliderId);
    return it == mImpl->mColliderIdToIndex.end() ? nullptr : &mImpl->mColliderSnapshot[it->second];
}

const BallJointState *PhysicsWorld::tryGetBallJoint(BallJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mImpl->mBallJointSnapshot.begin(), mImpl->mBallJointSnapshot.end(),
                     [&](const BallJointState &state) { return state.jointId == jointId; });
    return it == mImpl->mBallJointSnapshot.end() ? nullptr : &(*it);
}

const SphericalJointState *PhysicsWorld::tryGetSphericalJoint(
    SphericalJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mImpl->mSphericalJointSnapshot.begin(), mImpl->mSphericalJointSnapshot.end(),
                     [&](const SphericalJointState &state) { return state.jointId == jointId; });
    return it == mImpl->mSphericalJointSnapshot.end() ? nullptr : &(*it);
}

const HingeJointState *PhysicsWorld::tryGetHingeJoint(HingeJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mImpl->mHingeJointSnapshot.begin(), mImpl->mHingeJointSnapshot.end(),
                     [&](const HingeJointState &state) { return state.jointId == jointId; });
    return it == mImpl->mHingeJointSnapshot.end() ? nullptr : &(*it);
}

const SliderJointState *PhysicsWorld::tryGetSliderJoint(SliderJointId jointId) const noexcept
{
    const auto it =
        std::find_if(mImpl->mSliderJointSnapshot.begin(), mImpl->mSliderJointSnapshot.end(),
                     [&](const SliderJointState &state) { return state.jointId == jointId; });
    return it == mImpl->mSliderJointSnapshot.end() ? nullptr : &(*it);
}

SoftBodyState *PhysicsWorld::tryGetSoftBody(common::EntityId entityId)
{
    const auto it = mImpl->mEntityToSoftBodyIndex.find(entityId);
    return it == mImpl->mEntityToSoftBodyIndex.end() ? nullptr
                                                     : &mImpl->mSoftBodySnapshot[it->second];
}

const SoftBodyState *PhysicsWorld::tryGetSoftBody(common::EntityId entityId) const
{
    const auto it = mImpl->mEntityToSoftBodyIndex.find(entityId);
    return it == mImpl->mEntityToSoftBodyIndex.end() ? nullptr
                                                     : &mImpl->mSoftBodySnapshot[it->second];
}

StrandState *PhysicsWorld::tryGetStrand(common::EntityId entityId)
{
    const auto it = mImpl->mEntityToStrandIndex.find(entityId);
    return it == mImpl->mEntityToStrandIndex.end() ? nullptr : &mImpl->mStrandSnapshot[it->second];
}

const StrandState *PhysicsWorld::tryGetStrand(common::EntityId entityId) const
{
    const auto it = mImpl->mEntityToStrandIndex.find(entityId);
    return it == mImpl->mEntityToStrandIndex.end() ? nullptr : &mImpl->mStrandSnapshot[it->second];
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
    const Impl::TetGenMeshCache *tetGenCache = mImpl->tryGetTetGenMeshCache(entityId);
    const TetMeshData *cacheView             = nullptr;
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
    const auto it = mImpl->mEntityToFluidIndex.find(entityId);
    return it == mImpl->mEntityToFluidIndex.end() ? nullptr : &mImpl->mFluidSnapshot[it->second];
}

const FluidState *PhysicsWorld::tryGetFluid(common::EntityId entityId) const
{
    const auto it = mImpl->mEntityToFluidIndex.find(entityId);
    return it == mImpl->mEntityToFluidIndex.end() ? nullptr : &mImpl->mFluidSnapshot[it->second];
}

AuthoredParticleSequenceState *PhysicsWorld::tryGetParticleSequence(ParticleSequenceId sequenceId)
{
    const auto it = mImpl->mParticleSequenceIdToIndex.find(sequenceId);
    return it == mImpl->mParticleSequenceIdToIndex.end()
               ? nullptr
               : &mImpl->mParticleSequenceSnapshot[it->second];
}

const AuthoredParticleSequenceState *PhysicsWorld::tryGetParticleSequence(
    ParticleSequenceId sequenceId) const
{
    const auto it = mImpl->mParticleSequenceIdToIndex.find(sequenceId);
    return it == mImpl->mParticleSequenceIdToIndex.end()
               ? nullptr
               : &mImpl->mParticleSequenceSnapshot[it->second];
}

AuthoredParticleDistanceConstraintState *PhysicsWorld::tryGetParticleDistanceConstraint(
    ParticleConstraintId constraintId)
{
    const auto it = mImpl->mParticleConstraintIdToIndex.find(constraintId);
    return it == mImpl->mParticleConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mParticleDistanceConstraintSnapshot[it->second];
}

const AuthoredParticleDistanceConstraintState *PhysicsWorld::tryGetParticleDistanceConstraint(
    ParticleConstraintId constraintId) const
{
    const auto it = mImpl->mParticleConstraintIdToIndex.find(constraintId);
    return it == mImpl->mParticleConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mParticleDistanceConstraintSnapshot[it->second];
}

AuthoredRigidParticleAttachmentConstraintState *PhysicsWorld::
    tryGetRigidParticleAttachmentConstraint(RigidParticleAttachmentConstraintId constraintId)
{
    const auto it = mImpl->mRigidParticleAttachmentConstraintIdToIndex.find(constraintId);
    return it == mImpl->mRigidParticleAttachmentConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mRigidParticleAttachmentConstraintSnapshot[it->second];
}

const AuthoredRigidParticleAttachmentConstraintState *PhysicsWorld::
    tryGetRigidParticleAttachmentConstraint(RigidParticleAttachmentConstraintId constraintId) const
{
    const auto it = mImpl->mRigidParticleAttachmentConstraintIdToIndex.find(constraintId);
    return it == mImpl->mRigidParticleAttachmentConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mRigidParticleAttachmentConstraintSnapshot[it->second];
}

AuthoredStrandRigidAttachmentConstraintState *PhysicsWorld::tryGetStrandRigidAttachmentConstraint(
    StrandRigidAttachmentConstraintId constraintId)
{
    const auto it = mImpl->mStrandRigidAttachmentConstraintIdToIndex.find(constraintId);
    return it == mImpl->mStrandRigidAttachmentConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mStrandRigidAttachmentConstraintSnapshot[it->second];
}

const AuthoredStrandRigidAttachmentConstraintState *PhysicsWorld::
    tryGetStrandRigidAttachmentConstraint(StrandRigidAttachmentConstraintId constraintId) const
{
    const auto it = mImpl->mStrandRigidAttachmentConstraintIdToIndex.find(constraintId);
    return it == mImpl->mStrandRigidAttachmentConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mStrandRigidAttachmentConstraintSnapshot[it->second];
}

AuthoredRigidDistanceConstraintState *PhysicsWorld::tryGetRigidDistanceConstraint(
    RigidDistanceConstraintId constraintId)
{
    const auto it = mImpl->mRigidDistanceConstraintIdToIndex.find(constraintId);
    return it == mImpl->mRigidDistanceConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mRigidDistanceConstraintSnapshot[it->second];
}

const AuthoredRigidDistanceConstraintState *PhysicsWorld::tryGetRigidDistanceConstraint(
    RigidDistanceConstraintId constraintId) const
{
    const auto it = mImpl->mRigidDistanceConstraintIdToIndex.find(constraintId);
    return it == mImpl->mRigidDistanceConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mRigidDistanceConstraintSnapshot[it->second];
}

AuthoredRoutedCableConstraintState *PhysicsWorld::tryGetRoutedCableConstraint(
    RoutedCableConstraintId constraintId)
{
    const auto it = mImpl->mRoutedCableConstraintIdToIndex.find(constraintId);
    return it == mImpl->mRoutedCableConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mRoutedCableConstraintSnapshot[it->second];
}

const AuthoredRoutedCableConstraintState *PhysicsWorld::tryGetRoutedCableConstraint(
    RoutedCableConstraintId constraintId) const
{
    const auto it = mImpl->mRoutedCableConstraintIdToIndex.find(constraintId);
    return it == mImpl->mRoutedCableConstraintIdToIndex.end()
               ? nullptr
               : &mImpl->mRoutedCableConstraintSnapshot[it->second];
}

AuthoredParticleCollisionFilterState *PhysicsWorld::tryGetParticleCollisionFilter(
    ParticleCollisionFilterId filterId)
{
    const auto it = mImpl->mParticleCollisionFilterIdToIndex.find(filterId);
    return it == mImpl->mParticleCollisionFilterIdToIndex.end()
               ? nullptr
               : &mImpl->mParticleCollisionFilterSnapshot[it->second];
}

const AuthoredParticleCollisionFilterState *PhysicsWorld::tryGetParticleCollisionFilter(
    ParticleCollisionFilterId filterId) const
{
    const auto it = mImpl->mParticleCollisionFilterIdToIndex.find(filterId);
    return it == mImpl->mParticleCollisionFilterIdToIndex.end()
               ? nullptr
               : &mImpl->mParticleCollisionFilterSnapshot[it->second];
}

AuthoredSuturingSequenceState *PhysicsWorld::tryGetSuturingSequence(SuturingSequenceId sequenceId)
{
    const auto it = mImpl->mSuturingSequenceIdToIndex.find(sequenceId);
    return it == mImpl->mSuturingSequenceIdToIndex.end()
               ? nullptr
               : &mImpl->mSuturingSequenceSnapshot[it->second];
}

const AuthoredSuturingSequenceState *PhysicsWorld::tryGetSuturingSequence(
    SuturingSequenceId sequenceId) const
{
    const auto it = mImpl->mSuturingSequenceIdToIndex.find(sequenceId);
    return it == mImpl->mSuturingSequenceIdToIndex.end()
               ? nullptr
               : &mImpl->mSuturingSequenceSnapshot[it->second];
}

const std::vector<RigidBodyState> &PhysicsWorld::rigidBodySnapshot() const noexcept
{
    return mImpl->mRigidBodySnapshot;
}

const std::vector<ColliderState> &PhysicsWorld::colliderSnapshot() const noexcept
{
    return mImpl->mColliderSnapshot;
}

const std::vector<SoftBodyState> &PhysicsWorld::softBodySnapshot() const noexcept
{
    return mImpl->mSoftBodySnapshot;
}

const std::vector<StrandState> &PhysicsWorld::strandSnapshot() const noexcept
{
    return mImpl->mStrandSnapshot;
}

const std::vector<FluidState> &PhysicsWorld::fluidSnapshot() const noexcept
{
    return mImpl->mFluidSnapshot;
}

const std::vector<AuthoredParticleSequenceState> &PhysicsWorld::particleSequenceSnapshot()
    const noexcept
{
    return mImpl->mParticleSequenceSnapshot;
}

const std::vector<AuthoredParticleDistanceConstraintState> &PhysicsWorld::
    particleDistanceConstraintSnapshot() const noexcept
{
    return mImpl->mParticleDistanceConstraintSnapshot;
}

const std::vector<AuthoredRigidParticleAttachmentConstraintState> &PhysicsWorld::
    rigidParticleAttachmentConstraintSnapshot() const noexcept
{
    return mImpl->mRigidParticleAttachmentConstraintSnapshot;
}

const std::vector<AuthoredStrandRigidAttachmentConstraintState> &PhysicsWorld::
    strandRigidAttachmentConstraintSnapshot() const noexcept
{
    return mImpl->mStrandRigidAttachmentConstraintSnapshot;
}

const std::vector<AuthoredRigidDistanceConstraintState> &PhysicsWorld::
    rigidDistanceConstraintSnapshot() const noexcept
{
    return mImpl->mRigidDistanceConstraintSnapshot;
}

const std::vector<AuthoredRoutedCableConstraintState> &PhysicsWorld::routedCableConstraintSnapshot()
    const noexcept
{
    return mImpl->mRoutedCableConstraintSnapshot;
}

const std::vector<AuthoredParticleCollisionFilterState> &PhysicsWorld::
    particleCollisionFilterSnapshot() const noexcept
{
    return mImpl->mParticleCollisionFilterSnapshot;
}

const std::vector<AuthoredSuturingSequenceState> &PhysicsWorld::suturingSequenceSnapshot()
    const noexcept
{
    return mImpl->mSuturingSequenceSnapshot;
}

const std::vector<BallJointState> &PhysicsWorld::ballJointSnapshot() const noexcept
{
    return mImpl->mBallJointSnapshot;
}

const std::vector<SphericalJointState> &PhysicsWorld::sphericalJointSnapshot() const noexcept
{
    return mImpl->mSphericalJointSnapshot;
}

const std::vector<HingeJointState> &PhysicsWorld::hingeJointSnapshot() const noexcept
{
    return mImpl->mHingeJointSnapshot;
}

const std::vector<SliderJointState> &PhysicsWorld::sliderJointSnapshot() const noexcept
{
    return mImpl->mSliderJointSnapshot;
}

const RigidBodySoAHost &PhysicsWorld::rigidBodySoA() const noexcept
{
    return mImpl->mRigidBodies;
}

const ColliderSoAHost &PhysicsWorld::colliderSoA() const noexcept
{
    ensureDerivedStateUpToDate();
    return mImpl->mColliders;
}

const BodyColliderMappingHost &PhysicsWorld::bodyColliderMapping() const noexcept
{
    ensureDerivedStateUpToDate();
    return mImpl->mBodyColliderMapping;
}

const RigidJointSceneHost &PhysicsWorld::rigidJointScene() const noexcept
{
    ensureDerivedStateUpToDate();
    return mImpl->mRigidJointScene;
}

const JointCollisionSuppressionHost &PhysicsWorld::jointCollisionSuppression() const noexcept
{
    ensureDerivedStateUpToDate();
    return mImpl->mJointCollisionSuppression;
}

const ParticleSoAHost &PhysicsWorld::particles() const noexcept
{
    const_cast<PhysicsWorld *>(this)->ensureSoftBodyDerivedStateUpToDate();
    return mImpl->mParticles;
}

const std::vector<Diligent::float4> &PhysicsWorld::particleContactMaterials() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout);
    return mImpl->mParticleContactMaterials;
}

const std::vector<FluidMaterialGpu> &PhysicsWorld::fluidMaterials() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout);
    return mImpl->mFluidMaterials;
}

const std::vector<DeformableDistanceConstraint> &PhysicsWorld::distanceConstraints() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData);
    return mImpl->mSoftEdges;
}

const std::vector<DeformableBendConstraint> &PhysicsWorld::bendConstraints() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData);
    return mImpl->mSoftBends;
}

const std::vector<DeformableVolumeConstraint> &PhysicsWorld::volumeConstraints() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData);
    return mImpl->mSoftTets;
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
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData);
    return mImpl->mStrandSegments;
}

const std::vector<StrandJointConstraint> &PhysicsWorld::strandJoints() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData);
    return mImpl->mStrandJoints;
}

const std::vector<StrandDistanceConstraint> &PhysicsWorld::strandDistanceConstraints()
    const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData);
    return mImpl->mStrandDistanceConstraints;
}

const std::vector<StrandSegmentState> &PhysicsWorld::strandSegmentStates() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData);
    return mImpl->mStrandSegmentStates;
}

const std::vector<RigidParticleAttachmentConstraint> &PhysicsWorld::rigidParticleAttachments()
    const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureResolvedConstraintStateUpToDate();
    return mImpl->mRigidParticleAttachments;
}

const std::vector<StrandRigidAttachmentConstraint> &PhysicsWorld::strandRigidAttachments()
    const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureResolvedConstraintStateUpToDate();
    return mImpl->mStrandRigidAttachments;
}

const std::vector<RigidDistanceConstraint> &PhysicsWorld::rigidDistanceConstraints() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureResolvedConstraintStateUpToDate();
    return mImpl->mRigidDistanceConstraints;
}

const std::vector<RoutedCableConstraint> &PhysicsWorld::routedCableConstraints() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureResolvedConstraintStateUpToDate();
    return mImpl->mRoutedCableConstraints;
}

const std::vector<RoutedCableRoutePoint> &PhysicsWorld::routedCableRoutePoints() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureResolvedConstraintStateUpToDate();
    return mImpl->mRoutedCableRoutePoints;
}

const std::vector<StrandSoftSuturingPair> &PhysicsWorld::suturingPairs() const noexcept
{
    const_cast<PhysicsWorld *>(this)->ensureSoftBodyDerivedStateUpToDate();
    return mImpl->mSuturingPairs;
}

const std::vector<std::uint32_t> &PhysicsWorld::suturingParticleIndices() const noexcept
{
    const_cast<PhysicsWorld *>(this)->ensureSoftBodyDerivedStateUpToDate();
    return mImpl->mParticles.suturingParticleIndices;
}

const SoftRenderDataHost &PhysicsWorld::softRenderData() const noexcept
{
    return mImpl->mSoftRenderData;
}

void PhysicsWorld::setSoftRenderData(const SoftRenderDataHost &data)
{
    mImpl->mSoftRenderData = data;
    mImpl->recomputeSoftBodyBoundsChunkCount();
    ++mImpl->mSoftTopologyRevision;
    ++mImpl->mAuthoredRevision;
}

const CurveRenderDataHost &PhysicsWorld::curveRenderData() const noexcept
{
    return mImpl->mCurveRenderData;
}

void PhysicsWorld::setCurveRenderData(const CurveRenderDataHost &data)
{
    mImpl->mCurveRenderData = data;
    ++mImpl->mCurveRenderRevision;
}

void PhysicsWorld::ensureDerivedStateUpToDate() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureResolvedConstraintStateUpToDate();
    if (mImpl->mBodyColliderMappingDirty)
    {
        mImpl->rebuildBodyColliderMapping();
        mImpl->mBodyColliderMappingDirty = false;
    }
    if (mImpl->mRigidJointSceneDirty)
    {
        mImpl->rebuildRigidJointScene();
        mImpl->mRigidJointSceneDirty = false;
    }
    if (mImpl->mJointCollisionSuppressionDirty)
    {
        mImpl->rebuildJointCollisionSuppression();
        mImpl->mJointCollisionSuppressionDirty = false;
    }
}

void PhysicsWorld::ensureSoftBodyDerivedStateUpToDate() noexcept
{
    mImpl->ensureRebuildDomainsUpToDate(PhysicsRebuildFlags::SoftParticleLayout |
                                        PhysicsRebuildFlags::SoftConstraintData |
                                        PhysicsRebuildFlags::SuturingData);
}

void PhysicsWorld::Impl::ensureResolvedConstraintStateUpToDate() noexcept
{
    ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout | PhysicsRebuildFlags::SoftConstraintData |
        PhysicsRebuildFlags::SuturingData | PhysicsRebuildFlags::ResolvedRigidParticleAttachments |
        PhysicsRebuildFlags::ResolvedStrandRigidAttachments |
        PhysicsRebuildFlags::ResolvedRigidDistanceConstraints |
        PhysicsRebuildFlags::ResolvedRoutedCables);
}

void PhysicsWorld::Impl::invalidateSoftDerivedState() noexcept
{
    markRebuildDirty(PhysicsRebuildFlags::SoftParticleLayout |
                     PhysicsRebuildFlags::SoftConstraintData | PhysicsRebuildFlags::SuturingData);
}

void PhysicsWorld::Impl::invalidateResolvedRigidParticleAttachments() noexcept
{
    markRebuildDirty(PhysicsRebuildFlags::ResolvedRigidParticleAttachments);
}

void PhysicsWorld::Impl::invalidateResolvedStrandRigidAttachments(bool alsoSoftAdjacency) noexcept
{
    markRebuildDirty(PhysicsRebuildFlags::ResolvedStrandRigidAttachments);
    if (alsoSoftAdjacency)
    {
        markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData);
    }
}

void PhysicsWorld::Impl::invalidateResolvedRigidDistanceConstraints() noexcept
{
    markRebuildDirty(PhysicsRebuildFlags::ResolvedRigidDistanceConstraints);
}

void PhysicsWorld::Impl::invalidateResolvedRoutedCables() noexcept
{
    markRebuildDirty(PhysicsRebuildFlags::ResolvedRoutedCables);
}

void PhysicsWorld::Impl::invalidateAllRigidBodyDependentResolvedConstraints() noexcept
{
    invalidateResolvedRigidParticleAttachments();
    invalidateResolvedStrandRigidAttachments(true);
    invalidateResolvedRigidDistanceConstraints();
    invalidateResolvedRoutedCables();
}

void PhysicsWorld::Impl::markRebuildDirty(PhysicsRebuildFlags flags) noexcept
{
    if ((flags & PhysicsRebuildFlags::SoftParticleLayout) != PhysicsRebuildFlags::None)
    {
        flags |= PhysicsRebuildFlags::SoftConstraintData | PhysicsRebuildFlags::SuturingData |
                 PhysicsRebuildFlags::ResolvedRigidParticleAttachments |
                 PhysicsRebuildFlags::ResolvedStrandRigidAttachments |
                 PhysicsRebuildFlags::ResolvedRigidDistanceConstraints |
                 PhysicsRebuildFlags::ResolvedRoutedCables;
    }
    if ((flags & PhysicsRebuildFlags::SoftConstraintData) != PhysicsRebuildFlags::None)
    {
        flags |= PhysicsRebuildFlags::ResolvedRigidParticleAttachments |
                 PhysicsRebuildFlags::ResolvedStrandRigidAttachments;
    }
    if ((flags & PhysicsRebuildFlags::SuturingData) != PhysicsRebuildFlags::None)
    {
        flags |= PhysicsRebuildFlags::ResolvedStrandRigidAttachments;
    }
    mRebuildFlags |= flags;
}

void PhysicsWorld::Impl::clearRebuildDirty(PhysicsRebuildFlags flags) noexcept
{
    mRebuildFlags = static_cast<PhysicsRebuildFlags>(static_cast<std::uint32_t>(mRebuildFlags) &
                                                     ~static_cast<std::uint32_t>(flags));
}

bool PhysicsWorld::Impl::isRebuildDirty(PhysicsRebuildFlags flags) const noexcept
{
    return (mRebuildFlags & flags) != PhysicsRebuildFlags::None;
}

void PhysicsWorld::Impl::ensureRebuildDomainsUpToDate(PhysicsRebuildFlags flags) noexcept
{
    if ((flags & PhysicsRebuildFlags::SoftParticleLayout) != PhysicsRebuildFlags::None &&
        isRebuildDirty(PhysicsRebuildFlags::SoftParticleLayout))
    {
        rebuildSoftParticleLayout();
    }
    if ((flags & PhysicsRebuildFlags::SoftConstraintData) != PhysicsRebuildFlags::None &&
        isRebuildDirty(PhysicsRebuildFlags::SoftConstraintData))
    {
        rebuildSoftConstraintData();
    }
    if ((flags & PhysicsRebuildFlags::SuturingData) != PhysicsRebuildFlags::None &&
        isRebuildDirty(PhysicsRebuildFlags::SuturingData))
    {
        rebuildSuturingData();
    }
    if ((flags & PhysicsRebuildFlags::ResolvedRigidParticleAttachments) !=
            PhysicsRebuildFlags::None &&
        isRebuildDirty(PhysicsRebuildFlags::ResolvedRigidParticleAttachments))
    {
        rebuildResolvedRigidParticleAttachments();
    }
    if ((flags & PhysicsRebuildFlags::ResolvedStrandRigidAttachments) !=
            PhysicsRebuildFlags::None &&
        isRebuildDirty(PhysicsRebuildFlags::ResolvedStrandRigidAttachments))
    {
        rebuildResolvedStrandRigidAttachments();
    }
    if ((flags & PhysicsRebuildFlags::ResolvedRigidDistanceConstraints) !=
            PhysicsRebuildFlags::None &&
        isRebuildDirty(PhysicsRebuildFlags::ResolvedRigidDistanceConstraints))
    {
        rebuildResolvedRigidDistanceConstraints();
    }
    if ((flags & PhysicsRebuildFlags::ResolvedRoutedCables) != PhysicsRebuildFlags::None &&
        isRebuildDirty(PhysicsRebuildFlags::ResolvedRoutedCables))
    {
        rebuildResolvedRoutedCables();
    }
}

const std::vector<std::uint32_t> &PhysicsWorld::rigidBodyDirtyIndices() const noexcept
{
    return mImpl->mRigidBodyDirtyIndices;
}

const std::vector<std::uint32_t> &PhysicsWorld::colliderDirtyIndices() const noexcept
{
    return mImpl->mColliderDirtyIndices;
}

std::uint32_t PhysicsWorld::rigidBodyCount() const noexcept
{
    return static_cast<std::uint32_t>(mImpl->mRigidBodies.size());
}

std::uint32_t PhysicsWorld::colliderCount() const noexcept
{
    return static_cast<std::uint32_t>(mImpl->mColliders.size());
}

std::uint32_t PhysicsWorld::softBodyCount() const noexcept
{
    return static_cast<std::uint32_t>(mImpl->mSoftBodySnapshot.size());
}

std::uint32_t PhysicsWorld::strandCount() const noexcept
{
    return static_cast<std::uint32_t>(mImpl->mStrandSnapshot.size());
}

std::uint32_t PhysicsWorld::fluidCount() const noexcept
{
    return static_cast<std::uint32_t>(mImpl->mFluidSnapshot.size());
}

bool PhysicsWorld::rigidBodyCountDirty() const noexcept
{
    return mImpl->mRigidBodyCountDirty;
}

bool PhysicsWorld::colliderCountDirty() const noexcept
{
    return mImpl->mColliderCountDirty;
}

bool PhysicsWorld::fullRigidBodyUploadRequired() const noexcept
{
    return mImpl->mFullRigidBodyUploadRequired;
}

bool PhysicsWorld::fullColliderUploadRequired() const noexcept
{
    return mImpl->mFullColliderUploadRequired;
}

void PhysicsWorld::clearRigidBodyUploadState() noexcept
{
    clearDirtyIndices(mImpl->mRigidBodyDirtyIndices, mImpl->mRigidBodyDirtyBits);
    mImpl->mRigidBodyCountDirty         = false;
    mImpl->mFullRigidBodyUploadRequired = false;
}

void PhysicsWorld::clearColliderUploadState() noexcept
{
    clearDirtyIndices(mImpl->mColliderDirtyIndices, mImpl->mColliderDirtyBits);
    mImpl->mColliderCountDirty         = false;
    mImpl->mFullColliderUploadRequired = false;
}

bool PhysicsWorld::staticBroadPhaseDirty() const noexcept
{
    return mImpl->mStaticBroadPhaseDirty;
}

void PhysicsWorld::clearStaticBroadPhaseDirty() noexcept
{
    mImpl->mStaticBroadPhaseDirty = false;
}

std::uint32_t PhysicsWorld::activeMovingColliderCount() const noexcept
{
    return mImpl->mActiveMovingColliderCount;
}

std::uint32_t PhysicsWorld::staticColliderCount() const noexcept
{
    return mImpl->mStaticColliderCount;
}

float PhysicsWorld::particleGridCellSize() const noexcept
{
    const_cast<PhysicsWorld *>(this)->mImpl->ensureRebuildDomainsUpToDate(
        PhysicsRebuildFlags::SoftParticleLayout);
    return mImpl->mParticleGridCellSize;
}

std::uint32_t PhysicsWorld::softBodyBoundsChunkCount() const noexcept
{
    return mImpl->mSoftBodyBoundsChunkCount;
}

std::uint32_t PhysicsWorld::maxSuturingPathsPerPair() const noexcept
{
    return mImpl->mMaxSuturingPathsPerPair;
}

std::uint32_t PhysicsWorld::maxSuturingNodesPerPath() const noexcept
{
    return mImpl->mMaxSuturingNodesPerPath;
}

std::uint32_t PhysicsWorld::suturingParticleCount() const noexcept
{
    const_cast<PhysicsWorld *>(this)->ensureSoftBodyDerivedStateUpToDate();
    return static_cast<std::uint32_t>(mImpl->mParticles.suturingParticleIndices.size());
}

std::uint32_t PhysicsWorld::reservedSuturingPathHeaderCount() const noexcept
{
    const_cast<PhysicsWorld *>(this)->ensureSoftBodyDerivedStateUpToDate();
    return mImpl->mReservedSuturingPathHeaders;
}

std::uint32_t PhysicsWorld::reservedSuturingPathNodeCount() const noexcept
{
    const_cast<PhysicsWorld *>(this)->ensureSoftBodyDerivedStateUpToDate();
    return mImpl->mReservedSuturingPathNodes;
}

void PhysicsWorld::integrateRigidBodiesCpu(float dt) noexcept
{
    if (mImpl->mRigidBodies.empty())
    {
        return;
    }

    for (std::uint32_t i = 0; i < rigidBodyCount(); ++i)
    {
        Diligent::float4 &positionInvMass     = mImpl->mRigidBodies.positionsInvMass[i];
        const Diligent::float4 linearVelocity = mImpl->mRigidBodies.linearVelocities[i];
        positionInvMass.x += linearVelocity.x * dt;
        positionInvMass.y += linearVelocity.y * dt;
        positionInvMass.z += linearVelocity.z * dt;

        RigidBodyState &state = mImpl->mRigidBodySnapshot[i];
        state.position.x      = positionInvMass.x;
        state.position.y      = positionInvMass.y;
        state.position.z      = positionInvMass.z;
    }

    mImpl->markAllRigidBodiesDirty();
    ++mImpl->mAuthoredRevision;
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

    mImpl->mRigidBodies.positionsInvMass[index]  = positionInvMass;
    mImpl->mRigidBodies.orientations[index]      = orientation;
    mImpl->mRigidBodies.linearVelocities[index]  = linearVelocity;
    mImpl->mRigidBodies.angularVelocities[index] = angularVelocity;
    RigidBodyState &state                        = mImpl->mRigidBodySnapshot[index];
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
    if (mImpl->mRigidBodies.empty())
    {
        return;
    }
    ++mImpl->mSimulationRevision;
}

bool PhysicsWorld::syncParticleStateFromSimulation(std::uint32_t index,
                                                   const Diligent::float4 &positionInvMass,
                                                   const Diligent::float4 &previousPosition,
                                                   const Diligent::float4 &velocity) noexcept
{
    if (index >= mImpl->mParticles.size())
    {
        return false;
    }

    mImpl->mParticles.positionsInvMass[index]  = positionInvMass;
    mImpl->mParticles.previousPositions[index] = previousPosition;
    mImpl->mParticles.velocities[index]        = velocity;
    return true;
}

void PhysicsWorld::finalizeParticleWriteback() noexcept
{
    if (mImpl->mParticles.empty())
    {
        return;
    }
    ++mImpl->mSimulationRevision;
}

std::uint64_t PhysicsWorld::authoredRevision() const noexcept
{
    return mImpl->mAuthoredRevision;
}

std::uint64_t PhysicsWorld::simulationRevision() const noexcept
{
    return mImpl->mSimulationRevision;
}

std::uint64_t PhysicsWorld::rigidBodyTopologyRevision() const noexcept
{
    return mImpl->mRigidBodyTopologyRevision;
}

std::uint64_t PhysicsWorld::rigidJointTopologyRevision() const noexcept
{
    return mImpl->mRigidJointTopologyRevision;
}

std::uint64_t PhysicsWorld::rigidJointSceneRevision() const noexcept
{
    return mImpl->mRigidJointSceneRevision;
}

std::uint64_t PhysicsWorld::rigidJointModeRevision() const noexcept
{
    return mImpl->mRigidJointModeRevision;
}

std::uint64_t PhysicsWorld::softBodyTopologyRevision() const noexcept
{
    return mImpl->mSoftBodyTopologyRevision;
}

std::uint64_t PhysicsWorld::softParticleRevision() const noexcept
{
    return mImpl->mSoftParticleRevision;
}

std::uint64_t PhysicsWorld::softTopologyRevision() const noexcept
{
    return mImpl->mSoftTopologyRevision;
}

std::uint64_t PhysicsWorld::softConstraintAdjacencyRevision() const noexcept
{
    return mImpl->mSoftConstraintAdjacencyRevision;
}

std::uint64_t PhysicsWorld::rigidParticleAttachmentDefinitionRevision() const noexcept
{
    return mImpl->mRigidParticleAttachmentDefinitionRevision;
}

std::uint64_t PhysicsWorld::rigidParticleAttachmentResolvedRevision() const noexcept
{
    return mImpl->mRigidParticleAttachmentResolvedRevision;
}

std::uint64_t PhysicsWorld::strandRigidAttachmentDefinitionRevision() const noexcept
{
    return mImpl->mStrandRigidAttachmentDefinitionRevision;
}

std::uint64_t PhysicsWorld::strandRigidAttachmentResolvedRevision() const noexcept
{
    return mImpl->mStrandRigidAttachmentResolvedRevision;
}

std::uint64_t PhysicsWorld::rigidDistanceConstraintDefinitionRevision() const noexcept
{
    return mImpl->mRigidDistanceConstraintDefinitionRevision;
}

std::uint64_t PhysicsWorld::rigidDistanceConstraintResolvedRevision() const noexcept
{
    return mImpl->mRigidDistanceConstraintResolvedRevision;
}

std::uint64_t PhysicsWorld::routedCableDefinitionRevision() const noexcept
{
    return mImpl->mRoutedCableDefinitionRevision;
}

std::uint64_t PhysicsWorld::routedCableResolvedRevision() const noexcept
{
    return mImpl->mRoutedCableResolvedRevision;
}

std::uint64_t PhysicsWorld::curveRenderRevision() const noexcept
{
    return mImpl->mCurveRenderRevision;
}

void PhysicsWorld::Impl::writeRigidBodySoAAt(RigidBodySoAHost &soa, std::uint32_t index,
                                             const RigidBodyState &state)
{
    soa.rigidBodyIds[index]                  = state.rigidBodyId;
    soa.entityIds[index]                     = state.entityId;
    soa.environmentIndices[index]            = state.environmentIndex;
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

void PhysicsWorld::Impl::writeColliderSoAAt(ColliderSoAHost &soa, std::uint32_t index,
                                            const ColliderState &state,
                                            std::uint32_t ownerBodyIndex,
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

bool PhysicsWorld::Impl::isStaticBody(const RigidBodyState &state) noexcept
{
    return state.bodyType == RigidBodyType::Static;
}

bool PhysicsWorld::Impl::isMovingBody(const RigidBodyState &state) noexcept
{
    return state.bodyType == RigidBodyType::Kinematic || state.bodyType == RigidBodyType::Dynamic;
}

std::uint32_t PhysicsWorld::Impl::colliderBroadPhaseContribution(
    const ColliderState &collider, const RigidBodyState *owner) noexcept
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

bool PhysicsWorld::Impl::staticBodyPoseChanged(const RigidBodyState &before,
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

void PhysicsWorld::Impl::normalizeRigidBodyState(RigidBodyState &state) noexcept
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

void PhysicsWorld::Impl::normalizeColliderState(ColliderState &state) noexcept
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

void PhysicsWorld::Impl::normalizeSoftBodyState(SoftBodyState &state) noexcept
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

void PhysicsWorld::Impl::normalizeStrandState(StrandState &state) noexcept
{
    normalizeParticleContactMaterial(state.material.contact);
    state.particleMass           = std::max(state.particleMass, 1.0e-4f);
    state.particleRadius         = std::max(state.particleRadius, 1.0e-4f);
    state.stretchShearCompliance = std::max(state.stretchShearCompliance, 0.0f);
    state.bendCompliance         = std::max(state.bendCompliance, 0.0f);
    state.twistCompliance        = std::max(state.twistCompliance, 0.0f);
    if (state.distanceCompliance < 0.0f)
    {
        state.distanceCompliance = state.stretchShearCompliance;
    }
    state.distanceCompliance = std::max(state.distanceCompliance, 0.0f);
    state.pathNodeSpacing    = std::max(state.pathNodeSpacing, 1.0e-4f);
    if (state.collisionLayer == 0u)
    {
        state.collisionLayer = 1u;
    }
}

void PhysicsWorld::Impl::normalizeFluidState(FluidState &state) noexcept
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

bool PhysicsWorld::Impl::validateFluidMaterialCompatibility(
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

PhysicsWorld::Impl::SoftBodyChangeKind PhysicsWorld::Impl::classifySoftBodyChange(
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

PhysicsWorld::Impl::StrandChangeKind PhysicsWorld::Impl::classifyStrandChange(
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

PhysicsWorld::Impl::RigidJointChangeKind PhysicsWorld::Impl::classifyBallJointChange(
    bool inserted) noexcept
{
    return inserted ? RigidJointChangeKind::TopologyRebuild : RigidJointChangeKind::PayloadOnly;
}

PhysicsWorld::Impl::RigidJointChangeKind PhysicsWorld::Impl::classifyHingeJointChange(
    const HingeJointState &previousState, const HingeJointState &candidate, bool inserted) noexcept
{
    if (inserted)
    {
        return RigidJointChangeKind::TopologyRebuild;
    }

    if (previousState.bodyA != candidate.bodyA || previousState.bodyB != candidate.bodyB ||
        previousState.localRotationA != candidate.localRotationA ||
        previousState.localRotationB != candidate.localRotationB)
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

PhysicsWorld::Impl::RigidJointChangeKind PhysicsWorld::Impl::classifySphericalJointChange(
    const SphericalJointState &previousState, const SphericalJointState &candidate,
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

PhysicsWorld::Impl::RigidJointChangeKind PhysicsWorld::Impl::classifySliderJointChange(
    const SliderJointState &previousState, const SliderJointState &candidate,
    bool inserted) noexcept
{
    if (inserted)
    {
        return RigidJointChangeKind::TopologyRebuild;
    }

    if (previousState.bodyA != candidate.bodyA || previousState.bodyB != candidate.bodyB ||
        previousState.localAnchorA != candidate.localAnchorA ||
        previousState.localAnchorB != candidate.localAnchorB ||
        previousState.localRotationA != candidate.localRotationA ||
        previousState.localRotationB != candidate.localRotationB)
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

void PhysicsWorld::Impl::applySoftBodyRuntimeProperties(
    std::uint32_t index, const SoftBodyState &normalizedState) noexcept
{
    if (index >= mSoftBodySnapshot.size())
    {
        return;
    }

    ensureRebuildDomainsUpToDate(PhysicsRebuildFlags::SoftParticleLayout |
                                 PhysicsRebuildFlags::SoftConstraintData);

    if (index >= mSoftBodySnapshot.size() || index >= mSoftBodyDerivedCaches.size())
    {
        return;
    }

    SoftBodyState &softBody                      = mSoftBodySnapshot[index];
    const std::uint32_t previousEnvironmentIndex = softBody.environmentIndex;
    softBody.environmentIndex                    = normalizedState.environmentIndex;
    softBody.collisionLayer                      = normalizedState.collisionLayer;
    softBody.collisionMask                       = normalizedState.collisionMask;
    softBody.material                            = normalizedState.material;
    softBody.particleMass                        = normalizedState.particleMass;
    softBody.particleRadius                      = normalizedState.particleRadius;
    softBody.edgeCompliance                      = normalizedState.edgeCompliance;
    softBody.volumeCompliance                    = normalizedState.volumeCompliance;
    softBody.simulated                           = normalizedState.simulated;
    softBody.selfCollisionEnabled                = normalizedState.selfCollisionEnabled;
    softBody.supportsSuturing                    = normalizedState.supportsSuturing;

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
    if (previousEnvironmentIndex != softBody.environmentIndex)
    {
        markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData |
                         PhysicsRebuildFlags::SuturingData);
        invalidateResolvedRigidParticleAttachments();
        ++mSoftTopologyRevision;
    }
}

void PhysicsWorld::Impl::applyStrandRuntimeProperties(std::uint32_t index,
                                                      const StrandState &normalizedState) noexcept
{
    if (index >= mStrandSnapshot.size())
    {
        return;
    }

    ensureRebuildDomainsUpToDate(PhysicsRebuildFlags::SoftParticleLayout |
                                 PhysicsRebuildFlags::SoftConstraintData |
                                 PhysicsRebuildFlags::SuturingData);

    if (index >= mStrandSnapshot.size() || index >= mStrandDerivedCaches.size())
    {
        return;
    }

    StrandState &strand                          = mStrandSnapshot[index];
    const std::uint32_t previousEnvironmentIndex = strand.environmentIndex;
    strand.environmentIndex                      = normalizedState.environmentIndex;
    strand.collisionLayer                        = normalizedState.collisionLayer;
    strand.collisionMask                         = normalizedState.collisionMask;
    strand.material                              = normalizedState.material;
    strand.particleMass                          = normalizedState.particleMass;
    strand.particleRadius                        = normalizedState.particleRadius;
    strand.stretchShearCompliance                = normalizedState.stretchShearCompliance;
    strand.bendCompliance                        = normalizedState.bendCompliance;
    strand.twistCompliance                       = normalizedState.twistCompliance;
    strand.distanceCompliance                    = normalizedState.distanceCompliance;
    strand.rootMaterialNormal                    = normalizedState.rootMaterialNormal;
    strand.simulated                             = normalizedState.simulated;
    strand.selfCollisionEnabled                  = normalizedState.selfCollisionEnabled;
    strand.suturingEnabled                       = normalizedState.suturingEnabled;
    strand.pathNodeSpacing                       = normalizedState.pathNodeSpacing;
    strand.staticParticleIndices                 = normalizedState.staticParticleIndices;
    strand.contactMaterialIndex                  = findOrAppendParticleContactMaterial(
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
    const std::uint32_t segmentEnd   = std::min(segmentBegin + strand.segmentCount,
                                                static_cast<std::uint32_t>(mStrandSegments.size()));
    for (std::uint32_t segmentIndex = segmentBegin; segmentIndex < segmentEnd; ++segmentIndex)
    {
        mStrandSegments[segmentIndex].stretchShearCompliance = strand.stretchShearCompliance;
    }

    const std::uint32_t jointBegin = strand.jointOffset;
    const std::uint32_t jointEnd =
        std::min(jointBegin + strand.jointCount, static_cast<std::uint32_t>(mStrandJoints.size()));
    for (std::uint32_t jointIndex = jointBegin; jointIndex < jointEnd; ++jointIndex)
    {
        mStrandJoints[jointIndex].bendCompliance  = strand.bendCompliance;
        mStrandJoints[jointIndex].twistCompliance = strand.twistCompliance;
    }

    const std::uint32_t distanceBegin = strand.segmentOffset;
    const std::uint32_t distanceEnd =
        std::min(distanceBegin + strand.segmentCount,
                 static_cast<std::uint32_t>(mStrandDistanceConstraints.size()));
    for (std::uint32_t distanceIndex = distanceBegin; distanceIndex < distanceEnd; ++distanceIndex)
    {
        mStrandDistanceConstraints[distanceIndex].distanceCompliance = strand.distanceCompliance;
    }

    recomputeParticleGridCellSize();
    if (previousEnvironmentIndex != strand.environmentIndex)
    {
        markRebuildDirty(PhysicsRebuildFlags::SoftConstraintData |
                         PhysicsRebuildFlags::SuturingData);
        invalidateResolvedRigidParticleAttachments();
        invalidateResolvedStrandRigidAttachments(true);
        ++mSoftTopologyRevision;
    }
}

void PhysicsWorld::Impl::removeCollidersForEntity(common::EntityId entityId) noexcept
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

void PhysicsWorld::Impl::removeColliderAtIndex(std::uint32_t index) noexcept
{
    if (index >= static_cast<std::uint32_t>(mColliders.size()))
    {
        return;
    }

    const std::uint32_t last                = static_cast<std::uint32_t>(mColliders.size()) - 1u;
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

std::uint32_t PhysicsWorld::Impl::broadPhaseContributionForCollider(
    const ColliderState &collider) const noexcept
{
    const auto bodyIt = mRigidBodyIdToIndex.find(collider.ownerRigidBodyId);
    const RigidBodyState *owner =
        bodyIt != mRigidBodyIdToIndex.end() ? &mRigidBodySnapshot[bodyIt->second] : nullptr;
    return colliderBroadPhaseContribution(collider, owner);
}

std::uint32_t PhysicsWorld::Impl::enabledColliderCountForEntity(
    common::EntityId entityId) const noexcept
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

bool PhysicsWorld::Impl::pruneRigidJointsForBody(RigidBodyId rigidBodyId) noexcept
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

    const bool removedBall      = removeByBody(mBallJointSnapshot);
    const bool removedSpherical = removeByBody(mSphericalJointSnapshot);
    const bool removedHinge     = removeByBody(mHingeJointSnapshot);
    const bool removedSlider    = removeByBody(mSliderJointSnapshot);
    return removedBall || removedSpherical || removedHinge || removedSlider;
}

void PhysicsWorld::Impl::rebuildBodyColliderMapping() const noexcept
{
    mBodyColliderMapping.colliderOffsets.assign(static_cast<std::uint32_t>(mRigidBodies.size()),
                                                0u);
    mBodyColliderMapping.colliderCounts.assign(static_cast<std::uint32_t>(mRigidBodies.size()), 0u);
    mBodyColliderMapping.colliderIndices.assign(static_cast<std::uint32_t>(mColliders.size()), 0u);

    for (std::uint32_t colliderIndex = 0;
         colliderIndex < static_cast<std::uint32_t>(mColliders.size()); ++colliderIndex)
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
    for (std::uint32_t bodyIndex = 0; bodyIndex < static_cast<std::uint32_t>(mRigidBodies.size());
         ++bodyIndex)
    {
        mBodyColliderMapping.colliderOffsets[bodyIndex] = runningOffset;
        runningOffset += mBodyColliderMapping.colliderCounts[bodyIndex];
    }

    std::vector<std::uint32_t> cursor = mBodyColliderMapping.colliderOffsets;
    for (std::uint32_t colliderIndex = 0;
         colliderIndex < static_cast<std::uint32_t>(mColliders.size()); ++colliderIndex)
    {
        const std::uint32_t bodyIndex = mColliders.ownerRigidBodyIndices[colliderIndex];
        if (bodyIndex == 0xffffffffu ||
            bodyIndex >= static_cast<std::uint32_t>(mRigidBodies.size()))
        {
            continue;
        }
        mBodyColliderMapping.colliderIndices[cursor[bodyIndex]++] = colliderIndex;
    }
}

void PhysicsWorld::Impl::rebuildRigidJointScene() const noexcept
{
    auto *self = const_cast<Impl *>(this);
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
        const Diligent::float3 axisA1 = safeNormalize(
            quaternionRotate(joint.localRotationA, Diligent::float3{0.0f, 1.0f, 0.0f}),
            Diligent::float3{0.0f, 1.0f, 0.0f});
        const Diligent::float3 axisB1 = safeNormalize(
            quaternionRotate(joint.localRotationB, Diligent::float3{0.0f, 1.0f, 0.0f}),
            Diligent::float3{0.0f, 1.0f, 0.0f});

        self->mRigidJointScene.hinge.bodyIndicesA.push_back(bodyIndexA);
        self->mRigidJointScene.hinge.bodyIndicesB.push_back(bodyIndexB);
        self->mRigidJointScene.hinge.enabledFlags.push_back(joint.enabled ? 1u : 0u);
        self->mRigidJointScene.hinge.driveModes.push_back(
            static_cast<std::uint32_t>(joint.driveMode));
        self->mRigidJointScene.hinge.localAnchorsA.push_back(toFloat4(joint.localAnchorA, 0.0f));
        self->mRigidJointScene.hinge.localAnchorsB.push_back(toFloat4(joint.localAnchorB, 0.0f));
        self->mRigidJointScene.hinge.localAxesA0.push_back(toFloat4(axisA0, 0.0f));
        self->mRigidJointScene.hinge.localAxesA1.push_back(toFloat4(axisA1, 0.0f));
        self->mRigidJointScene.hinge.localAxesB1.push_back(toFloat4(axisB1, 0.0f));
        self->mRigidJointScene.hinge.limitEnabledFlags.push_back(joint.limitEnabled ? 1u : 0u);
        self->mRigidJointScene.hinge.limitMins.push_back(joint.limitMin);
        self->mRigidJointScene.hinge.limitMaxs.push_back(joint.limitMax);
        self->mRigidJointScene.hinge.constraintCompliances.push_back(joint.constraintCompliance);
        self->mRigidJointScene.hinge.driveCompliances.push_back(joint.driveCompliance);
        self->mRigidJointScene.hinge.driveTargetAngles.push_back(joint.driveTargetAngle);
        self->mRigidJointScene.hinge.driveDampings.push_back(joint.driveDamping);
        self->mRigidJointScene.hinge.driveMaxAngularVelocities.push_back(
            joint.driveMaxAngularVelocity);
        self->mRigidJointScene.hinge.driveTargetAngularVelocities.push_back(
            joint.driveTargetAngularVelocity);
        self->mRigidJointScene.hinge.projectionRow0.push_back(projectionRows[0]);
        self->mRigidJointScene.hinge.projectionRow1.push_back(projectionRows[1]);
        self->mRigidJointScene.hinge.projectionRow2.push_back(projectionRows[2]);
    };

    auto appendSpherical = [&](const SphericalJointState &joint)
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

        self->mRigidJointScene.spherical.bodyIndicesA.push_back(bodyIndexA);
        self->mRigidJointScene.spherical.bodyIndicesB.push_back(bodyIndexB);
        self->mRigidJointScene.spherical.enabledFlags.push_back(joint.enabled ? 1u : 0u);
        self->mRigidJointScene.spherical.driveModes.push_back(
            static_cast<std::uint32_t>(joint.driveMode));
        self->mRigidJointScene.spherical.localAnchorsA.push_back(
            toFloat4(joint.localAnchorA, 0.0f));
        self->mRigidJointScene.spherical.localAnchorsB.push_back(
            toFloat4(joint.localAnchorB, 0.0f));
        self->mRigidJointScene.spherical.localRotationsA.push_back(
            Diligent::float4{joint.localRotationA.q.x, joint.localRotationA.q.y,
                             joint.localRotationA.q.z, joint.localRotationA.q.w});
        self->mRigidJointScene.spherical.localRotationsB.push_back(
            Diligent::float4{joint.localRotationB.q.x, joint.localRotationB.q.y,
                             joint.localRotationB.q.z, joint.localRotationB.q.w});
        self->mRigidJointScene.spherical.limitEnabledFlags.push_back(joint.limitEnabled ? 1u : 0u);
        self->mRigidJointScene.spherical.swingLimitYs.push_back(joint.swingLimitY);
        self->mRigidJointScene.spherical.swingLimitZs.push_back(joint.swingLimitZ);
        self->mRigidJointScene.spherical.twistLimitMins.push_back(joint.twistLimitMin);
        self->mRigidJointScene.spherical.twistLimitMaxs.push_back(joint.twistLimitMax);
        self->mRigidJointScene.spherical.constraintCompliances.push_back(
            joint.constraintCompliance);
        self->mRigidJointScene.spherical.swingCompliances.push_back(joint.swingCompliance);
        self->mRigidJointScene.spherical.twistCompliances.push_back(joint.twistCompliance);
        self->mRigidJointScene.spherical.driveCompliances.push_back(joint.driveCompliance);
        self->mRigidJointScene.spherical.driveTargetOrientations.push_back(
            Diligent::float4{joint.driveTargetOrientation.q.x, joint.driveTargetOrientation.q.y,
                             joint.driveTargetOrientation.q.z, joint.driveTargetOrientation.q.w});
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
        self->mRigidJointScene.slider.driveDampings.push_back(joint.driveDamping);
        self->mRigidJointScene.slider.driveMaxVelocities.push_back(joint.driveMaxVelocity);
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
    for (const SphericalJointState &joint : self->mSphericalJointSnapshot)
    {
        appendSpherical(joint);
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

void PhysicsWorld::Impl::rebuildJointCollisionSuppression() const noexcept
{
    auto *self = const_cast<Impl *>(this);
    self->mJointCollisionSuppression.clear();
    self->mJointCollisionSuppression.neighborOffsets.assign(
        static_cast<std::uint32_t>(mRigidBodies.size()) + 1u, 0u);
    if (static_cast<std::uint32_t>(mRigidBodies.size()) == 0u)
    {
        return;
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> directedPairs;
    directedPairs.reserve((mBallJointSnapshot.size() + mSphericalJointSnapshot.size() +
                           mHingeJointSnapshot.size() + mSliderJointSnapshot.size()) *
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
    for (const SphericalJointState &joint : mSphericalJointSnapshot)
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

    for (std::uint32_t bodyIndex = 0u; bodyIndex < static_cast<std::uint32_t>(mRigidBodies.size());
         ++bodyIndex)
    {
        self->mJointCollisionSuppression.neighborOffsets[bodyIndex + 1u] +=
            self->mJointCollisionSuppression.neighborOffsets[bodyIndex];
    }
}

void PhysicsWorld::Impl::rebuildSoftParticleLayout() noexcept
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
            distanceConstraint.particleA          = segment.particleA;
            distanceConstraint.particleB          = segment.particleB;
            distanceConstraint.restLength         = segment.restLength;
            distanceConstraint.distanceCompliance = strand.distanceCompliance;
            mStrandDistanceConstraints.push_back(distanceConstraint);
        }

        for (std::uint32_t jointIndex = 0u;
             jointIndex < static_cast<std::uint32_t>(topology.joints.size()); ++jointIndex)
        {
            const auto &jointDesc = topology.joints[jointIndex];
            StrandJointConstraint joint{};
            joint.segmentA                = strand.segmentOffset + jointDesc[0];
            joint.segmentB                = strand.segmentOffset + jointDesc[1];
            joint.bendCompliance          = strand.bendCompliance;
            joint.twistCompliance         = strand.twistCompliance;
            const Diligent::QuaternionF q = topology.restJointRelativeOrientations[jointIndex];
            joint.restRelativeOrientation = Diligent::float4{q.q.x, q.q.y, q.q.z, q.q.w};
            mStrandJoints.push_back(joint);
        }

        strand.segmentCount =
            static_cast<std::uint32_t>(mStrandSegments.size()) - strand.segmentOffset;
        strand.jointCount = static_cast<std::uint32_t>(mStrandJoints.size()) - strand.jointOffset;
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

        resolved.environmentIndex = *environmentIndex;
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
                const RigidBodyState *rigidBody = mOwner->tryGetRigidBody(tipReference.entityId);
                resolved.pathNodeSpacing =
                    rigidBody != nullptr ? std::max(rigidBody->proxyParticleRadius * 1.5f, 1.0e-4f)
                                         : 0.2f;
            }
            else
            {
                const StrandState *strand = mOwner->tryGetStrand(tipReference.entityId);
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
        resolved.enabled    = constraint.enabled ? 1u : 0u;
        mSoftEdges.push_back(resolved);
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
            pair.suturingGroupId     = strandIndex;
            pair.softBodyIndex       = softBodyIndex;
            pair.strandParticleStart = strand.particleOffset;
            pair.strandParticleCount = strand.particleCount;
            pair.tipParticleIndex    = strand.particleOffset;
            pair.softTetStart        = softBody.tetOffset;
            pair.softTetCount        = softBody.tetCount;
            pair.pathStart           = mReservedSuturingPathHeaders;
            pair.pathCount           = mMaxSuturingPathsPerPair;
            pair.nodeStart           = mReservedSuturingPathNodes;
            pair.nodeCount           = mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
            pair.activePathIndex     = kInvalidSuturingIndex;
            pair.environmentIndex    = strand.environmentIndex;
            pair.pathNodeSpacing     = strand.pathNodeSpacing;
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
            pair.suturingGroupId     = sequence.groupId;
            pair.softBodyIndex       = softBodyIndex;
            pair.strandParticleStart = particleStart;
            pair.strandParticleCount = particleEnd - particleStart + 1u;
            pair.tipParticleIndex    = sequence.particleIndices[sequence.tipEntryIndex];
            pair.softTetStart        = softBody.tetOffset;
            pair.softTetCount        = softBody.tetCount;
            pair.pathStart           = mReservedSuturingPathHeaders;
            pair.pathCount           = mMaxSuturingPathsPerPair;
            pair.nodeStart           = mReservedSuturingPathNodes;
            pair.nodeCount           = mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
            pair.activePathIndex     = kInvalidSuturingIndex;
            pair.environmentIndex    = sequence.environmentIndex;
            pair.pathNodeSpacing     = sequence.pathNodeSpacing;
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
    clearRebuildDirty(PhysicsRebuildFlags::SoftParticleLayout |
                      PhysicsRebuildFlags::SoftConstraintData | PhysicsRebuildFlags::SuturingData);
}

void PhysicsWorld::Impl::rebuildSoftConstraintData() noexcept
{
    const std::vector<std::uint32_t> previousAdjacencyOffsets = mParticles.adjacencyOffsets;
    const std::vector<std::uint32_t> previousAdjacencyCounts  = mParticles.adjacencyCounts;
    const std::vector<std::uint32_t> previousAdjacencyIndices = mParticles.adjacencyIndices;

    mSoftEdges.clear();
    mSoftBends.clear();
    mSoftTets.clear();
    mStrandSegments.clear();
    mStrandJoints.clear();
    mStrandDistanceConstraints.clear();
    mStrandSegmentStates.clear();

    std::vector<std::vector<std::uint32_t>> adjacencyLists(mParticles.size());

    for (const RigidBodyState &rigidBody : mRigidBodySnapshot)
    {
        const std::uint32_t end =
            std::min(rigidBody.proxyParticleOffset + rigidBody.proxyParticleCount,
                     static_cast<std::uint32_t>(std::min(mParticles.collisionLayers.size(),
                                                         mParticles.collisionMasks.size())));
        for (std::uint32_t particleIndex = rigidBody.proxyParticleOffset; particleIndex < end;
             ++particleIndex)
        {
            mParticles.collisionLayers[particleIndex] = rigidBody.proxyCollisionLayer;
            mParticles.collisionMasks[particleIndex]  = rigidBody.proxyCollisionMask;
        }
    }

    for (std::uint32_t softBodyIndex = 0u; softBodyIndex < mSoftBodySnapshot.size();
         ++softBodyIndex)
    {
        SoftBodyState &softBody = mSoftBodySnapshot[softBodyIndex];
        normalizeSoftBodyState(softBody);
        softBody.edgeOffset = static_cast<std::uint32_t>(mSoftEdges.size());
        softBody.tetOffset  = static_cast<std::uint32_t>(mSoftTets.size());

        if (softBodyIndex >= mSoftBodyDerivedCaches.size())
        {
            softBody.edgeCount = 0u;
            softBody.tetCount  = 0u;
            continue;
        }

        const SoftBodyDerivedCache &topology = mSoftBodyDerivedCaches[softBodyIndex];
        const std::uint32_t particleEnd =
            std::min(softBody.particleOffset + softBody.particleCount,
                     static_cast<std::uint32_t>(std::min(mParticles.collisionLayers.size(),
                                                         mParticles.collisionMasks.size())));
        for (std::uint32_t particleIndex = softBody.particleOffset; particleIndex < particleEnd;
             ++particleIndex)
        {
            mParticles.collisionLayers[particleIndex] = softBody.collisionLayer;
            mParticles.collisionMasks[particleIndex]  = softBody.collisionMask;
        }

        for (std::uint32_t localParticleIndex = 0u;
             localParticleIndex < static_cast<std::uint32_t>(topology.adjacencyLists.size()) &&
             softBody.particleOffset + localParticleIndex < adjacencyLists.size();
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
        strand.segmentOffset = static_cast<std::uint32_t>(mStrandSegments.size());
        strand.jointOffset   = static_cast<std::uint32_t>(mStrandJoints.size());

        if (strandIndex >= mStrandDerivedCaches.size())
        {
            strand.segmentCount = 0u;
            strand.jointCount   = 0u;
            continue;
        }

        const StrandDerivedCache &topology = mStrandDerivedCaches[strandIndex];
        const std::uint32_t particleEnd =
            std::min(strand.particleOffset + strand.particleCount,
                     static_cast<std::uint32_t>(std::min(mParticles.collisionLayers.size(),
                                                         mParticles.collisionMasks.size())));
        for (std::uint32_t particleIndex = strand.particleOffset; particleIndex < particleEnd;
             ++particleIndex)
        {
            mParticles.collisionLayers[particleIndex] = strand.collisionLayer;
            mParticles.collisionMasks[particleIndex]  = strand.collisionMask;
        }

        for (std::uint32_t localParticleIndex = 0u;
             localParticleIndex < static_cast<std::uint32_t>(topology.adjacencyLists.size()) &&
             strand.particleOffset + localParticleIndex < adjacencyLists.size();
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
            distanceConstraint.particleA          = segment.particleA;
            distanceConstraint.particleB          = segment.particleB;
            distanceConstraint.restLength         = segment.restLength;
            distanceConstraint.distanceCompliance = strand.distanceCompliance;
            mStrandDistanceConstraints.push_back(distanceConstraint);
        }

        for (std::uint32_t jointIndex = 0u;
             jointIndex < static_cast<std::uint32_t>(topology.joints.size()); ++jointIndex)
        {
            const auto &jointDesc = topology.joints[jointIndex];
            StrandJointConstraint joint{};
            joint.segmentA                = strand.segmentOffset + jointDesc[0];
            joint.segmentB                = strand.segmentOffset + jointDesc[1];
            joint.bendCompliance          = strand.bendCompliance;
            joint.twistCompliance         = strand.twistCompliance;
            const Diligent::QuaternionF q = topology.restJointRelativeOrientations[jointIndex];
            joint.restRelativeOrientation = Diligent::float4{q.q.x, q.q.y, q.q.z, q.q.w};
            mStrandJoints.push_back(joint);
        }

        strand.segmentCount =
            static_cast<std::uint32_t>(mStrandSegments.size()) - strand.segmentOffset;
        strand.jointCount = static_cast<std::uint32_t>(mStrandJoints.size()) - strand.jointOffset;
    }

    for (const FluidState &fluid : mFluidSnapshot)
    {
        const std::uint32_t end =
            std::min(fluid.particleOffset + fluid.particleCount,
                     static_cast<std::uint32_t>(std::min(mParticles.collisionLayers.size(),
                                                         mParticles.collisionMasks.size())));
        for (std::uint32_t particleIndex = fluid.particleOffset; particleIndex < end;
             ++particleIndex)
        {
            mParticles.collisionLayers[particleIndex] = fluid.collisionLayer;
            mParticles.collisionMasks[particleIndex]  = fluid.collisionMask;
        }
    }

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

    for (const AuthoredParticleDistanceConstraintState &constraint :
         mParticleDistanceConstraintSnapshot)
    {
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
        resolved.enabled    = constraint.enabled ? 1u : 0u;
        mSoftEdges.push_back(resolved);
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

    if (triviallyComparableVectorsDiffer(previousAdjacencyOffsets, mParticles.adjacencyOffsets) ||
        triviallyComparableVectorsDiffer(previousAdjacencyCounts, mParticles.adjacencyCounts) ||
        triviallyComparableVectorsDiffer(previousAdjacencyIndices, mParticles.adjacencyIndices))
    {
        ++mSoftConstraintAdjacencyRevision;
    }
    clearRebuildDirty(PhysicsRebuildFlags::SoftConstraintData);
}

void PhysicsWorld::Impl::rebuildSuturingData() noexcept
{
    for (std::uint32_t particleIndex = 0u;
         particleIndex < static_cast<std::uint32_t>(mParticles.strandIds.size()); ++particleIndex)
    {
        mParticles.strandIds[particleIndex]    = 0xffffffffu;
        mParticles.strandOrders[particleIndex] = 0xffffffffu;
        mParticles.strandRoles[particleIndex] =
            static_cast<std::uint32_t>(ParticleStrandRole::None);
        mParticles.suturingNeighborLinks[particleIndex] = Diligent::uint4{
            kInvalidSuturingIndex, kInvalidSuturingIndex, kInvalidSuturingIndex, 0u};
    }

    const std::uint32_t rigidProxySuturingGroupBase =
        static_cast<std::uint32_t>(mStrandSnapshot.size());

    for (std::uint32_t rigidBodyIndex = 0u; rigidBodyIndex < mRigidBodySnapshot.size();
         ++rigidBodyIndex)
    {
        const RigidBodyState &rigidBody = mRigidBodySnapshot[rigidBodyIndex];
        if (!rigidBody.suturingEnabled || rigidBody.proxyParticleCount == 0u)
        {
            continue;
        }

        for (std::uint32_t proxyLocalIndex = 0u; proxyLocalIndex < rigidBody.proxyParticleCount;
             ++proxyLocalIndex)
        {
            const std::uint32_t particleIndex = rigidBody.proxyParticleOffset + proxyLocalIndex;
            if (particleIndex >= mParticles.strandIds.size())
            {
                break;
            }

            mParticles.strandIds[particleIndex]    = rigidProxySuturingGroupBase + rigidBodyIndex;
            mParticles.strandOrders[particleIndex] = proxyLocalIndex;
            mParticles.strandRoles[particleIndex]  = static_cast<std::uint32_t>(
                proxyLocalIndex == rigidBody.needleTipProxyIndex ? ParticleStrandRole::NeedleTip
                                                                 : ParticleStrandRole::NeedleBody);
            mParticles.suturingNeighborLinks[particleIndex].x =
                proxyLocalIndex > 0u ? particleIndex - 1u : kInvalidSuturingIndex;
            mParticles.suturingNeighborLinks[particleIndex].y =
                proxyLocalIndex + 1u < rigidBody.proxyParticleCount ? particleIndex + 1u
                                                                    : kInvalidSuturingIndex;
        }
    }

    for (std::uint32_t strandIndex = 0u; strandIndex < mStrandSnapshot.size(); ++strandIndex)
    {
        const StrandState &strand = mStrandSnapshot[strandIndex];
        if (!strand.suturingEnabled || strand.particleCount == 0u)
        {
            continue;
        }

        for (std::uint32_t localParticleIndex = 0u; localParticleIndex < strand.particleCount;
             ++localParticleIndex)
        {
            const std::uint32_t particleIndex = strand.particleOffset + localParticleIndex;
            if (particleIndex >= mParticles.strandIds.size())
            {
                break;
            }

            mParticles.strandIds[particleIndex]    = strandIndex;
            mParticles.strandOrders[particleIndex] = localParticleIndex;
            mParticles.strandRoles[particleIndex]  = static_cast<std::uint32_t>(
                localParticleIndex == 0u ? ParticleStrandRole::NeedleTip
                                         : ParticleStrandRole::NeedleBody);
            mParticles.suturingNeighborLinks[particleIndex].x =
                localParticleIndex > 0u ? particleIndex - 1u : kInvalidSuturingIndex;
            mParticles.suturingNeighborLinks[particleIndex].y =
                localParticleIndex + 1u < strand.particleCount ? particleIndex + 1u
                                                               : kInvalidSuturingIndex;
        }
    }

    struct ResolvedSuturingSequence
    {
        const AuthoredSuturingSequenceState *authored = nullptr;
        std::vector<std::uint32_t> particleIndices{};
        std::uint32_t environmentIndex = 0u;
        std::uint32_t tipEntryIndex    = 0u;
        float pathNodeSpacing          = 0.0f;
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

        resolved.environmentIndex = *environmentIndex;
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
                const RigidBodyState *rigidBody = mOwner->tryGetRigidBody(tipReference.entityId);
                resolved.pathNodeSpacing =
                    rigidBody != nullptr ? std::max(rigidBody->proxyParticleRadius * 1.5f, 1.0e-4f)
                                         : 0.2f;
            }
            else
            {
                const StrandState *strand = mOwner->tryGetStrand(tipReference.entityId);
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
            mParticles.suturingNeighborLinks[particleIndex].x = kInvalidSuturingIndex;
            mParticles.suturingNeighborLinks[particleIndex].y = kInvalidSuturingIndex;
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
            mParticles.suturingNeighborLinks[particleIndex].x = kInvalidSuturingIndex;
            mParticles.suturingNeighborLinks[particleIndex].y = kInvalidSuturingIndex;
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
                kInvalidSuturingIndex, 0u};
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
            pair.suturingGroupId     = strandIndex;
            pair.softBodyIndex       = softBodyIndex;
            pair.strandParticleStart = strand.particleOffset;
            pair.strandParticleCount = strand.particleCount;
            pair.tipParticleIndex    = strand.particleOffset;
            pair.softTetStart        = softBody.tetOffset;
            pair.softTetCount        = softBody.tetCount;
            pair.pathStart           = mReservedSuturingPathHeaders;
            pair.pathCount           = mMaxSuturingPathsPerPair;
            pair.nodeStart           = mReservedSuturingPathNodes;
            pair.nodeCount           = mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
            pair.activePathIndex     = kInvalidSuturingIndex;
            pair.environmentIndex    = strand.environmentIndex;
            pair.pathNodeSpacing     = strand.pathNodeSpacing;
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
            pair.suturingGroupId     = sequence.groupId;
            pair.softBodyIndex       = softBodyIndex;
            pair.strandParticleStart = particleStart;
            pair.strandParticleCount = particleEnd - particleStart + 1u;
            pair.tipParticleIndex    = sequence.particleIndices[sequence.tipEntryIndex];
            pair.softTetStart        = softBody.tetOffset;
            pair.softTetCount        = softBody.tetCount;
            pair.pathStart           = mReservedSuturingPathHeaders;
            pair.pathCount           = mMaxSuturingPathsPerPair;
            pair.nodeStart           = mReservedSuturingPathNodes;
            pair.nodeCount           = mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
            pair.activePathIndex     = kInvalidSuturingIndex;
            pair.environmentIndex    = sequence.environmentIndex;
            pair.pathNodeSpacing     = sequence.pathNodeSpacing;
            mSuturingPairs.push_back(pair);

            mReservedSuturingPathHeaders += mMaxSuturingPathsPerPair;
            mReservedSuturingPathNodes += mMaxSuturingPathsPerPair * mMaxSuturingNodesPerPath;
        }
    }

    clearRebuildDirty(PhysicsRebuildFlags::SuturingData);
}

std::optional<std::uint32_t> PhysicsWorld::Impl::resolveParticleReference(
    const AuthoredParticleReference &reference) const noexcept
{
    switch (reference.type)
    {
    case AuthoredParticleReferenceType::SoftBodyParticle:
    {
        const SoftBodyState *softBody = mOwner->tryGetSoftBody(reference.entityId);
        if (softBody == nullptr || reference.localParticleIndex >= softBody->particleCount)
        {
            return std::nullopt;
        }
        return softBody->particleOffset + reference.localParticleIndex;
    }
    case AuthoredParticleReferenceType::StrandParticle:
    {
        const StrandState *strand = mOwner->tryGetStrand(reference.entityId);
        if (strand == nullptr || reference.localParticleIndex >= strand->particleCount)
        {
            return std::nullopt;
        }
        return strand->particleOffset + reference.localParticleIndex;
    }
    case AuthoredParticleReferenceType::RigidProxyParticle:
    {
        const RigidBodyState *rigidBody = mOwner->tryGetRigidBody(reference.entityId);
        if (rigidBody == nullptr || reference.localParticleIndex >= rigidBody->proxyParticleCount)
        {
            return std::nullopt;
        }
        return rigidBody->proxyParticleOffset + reference.localParticleIndex;
    }
    }

    return std::nullopt;
}

void PhysicsWorld::Impl::rebuildResolvedRigidParticleAttachments() noexcept
{
    std::vector<RigidParticleAttachmentConstraint> rebuilt;

    for (const AuthoredRigidParticleAttachmentConstraintState &constraint :
         mRigidParticleAttachmentConstraintSnapshot)
    {
        const auto particleIndex = resolveParticleReference(constraint.particle);
        const auto rigidBodyIt   = mEntityToRigidBodyIndex.find(constraint.rigidBodyEntityId);
        if (!particleIndex.has_value() || rigidBodyIt == mEntityToRigidBodyIndex.end() ||
            *particleIndex >= mParticles.environmentIndices.size() ||
            rigidBodyIt->second >= mRigidBodySnapshot.size())
        {
            continue;
        }

        if (mParticles.environmentIndices[*particleIndex] !=
            mRigidBodySnapshot[rigidBodyIt->second].environmentIndex)
        {
            continue;
        }

        rebuilt.push_back(RigidParticleAttachmentConstraint{
            *particleIndex,
            rigidBodyIt->second,
            constraint.compliance,
            constraint.enabled ? 1u : 0u,
            Diligent::float4{constraint.localAnchor.x, constraint.localAnchor.y,
                             constraint.localAnchor.z, 0.0f},
        });
    }

    if (triviallyComparableVectorsDiffer(mRigidParticleAttachments, rebuilt))
    {
        mRigidParticleAttachments = std::move(rebuilt);
        ++mRigidParticleAttachmentResolvedRevision;
    }
    else
    {
        mRigidParticleAttachments = std::move(rebuilt);
    }
    clearRebuildDirty(PhysicsRebuildFlags::ResolvedRigidParticleAttachments);
}

void PhysicsWorld::Impl::rebuildResolvedStrandRigidAttachments() noexcept
{
    std::vector<StrandRigidAttachmentConstraint> rebuilt;
    std::vector<std::uint32_t> adjacencySignature;

    for (const AuthoredStrandRigidAttachmentConstraintState &constraint :
         mStrandRigidAttachmentConstraintSnapshot)
    {
        const StrandState *strand = mOwner->tryGetStrand(constraint.strandEntityId);
        const auto rigidBodyIt    = mEntityToRigidBodyIndex.find(constraint.rigidBodyEntityId);
        if (strand == nullptr || rigidBodyIt == mEntityToRigidBodyIndex.end() ||
            constraint.localSegmentIndex >= strand->segmentCount ||
            rigidBodyIt->second >= mRigidBodySnapshot.size())
        {
            continue;
        }

        const std::uint32_t segmentIndex = strand->segmentOffset + constraint.localSegmentIndex;
        if (segmentIndex >= mStrandSegments.size())
        {
            continue;
        }

        const StrandSegmentConstraint &segment = mStrandSegments[segmentIndex];
        if (segment.particleA >= mParticles.environmentIndices.size() ||
            mParticles.environmentIndices[segment.particleA] !=
                mRigidBodySnapshot[rigidBodyIt->second].environmentIndex)
        {
            continue;
        }

        const Diligent::QuaternionF localRotation =
            common::runtime_math::normalizeQuaternion(constraint.localRotation);
        rebuilt.push_back(StrandRigidAttachmentConstraint{
            segmentIndex,
            rigidBodyIt->second,
            std::clamp(constraint.segmentT, 0.0f, 1.0f),
            constraint.translationCompliance,
            constraint.rotationCompliance,
            constraint.enabled ? 1u : 0u,
            0u,
            0u,
            Diligent::float4{constraint.localAnchor.x, constraint.localAnchor.y,
                             constraint.localAnchor.z, 0.0f},
            Diligent::float4{localRotation.q.x, localRotation.q.y, localRotation.q.z,
                             localRotation.q.w},
        });
        adjacencySignature.push_back(segmentIndex);
    }

    std::vector<std::uint32_t> previousAdjacencySignature;
    previousAdjacencySignature.reserve(mStrandRigidAttachments.size());
    for (const StrandRigidAttachmentConstraint &attachment : mStrandRigidAttachments)
    {
        previousAdjacencySignature.push_back(attachment.segmentIndex);
    }

    if (triviallyComparableVectorsDiffer(mStrandRigidAttachments, rebuilt))
    {
        mStrandRigidAttachments = std::move(rebuilt);
        ++mStrandRigidAttachmentResolvedRevision;
    }
    else
    {
        mStrandRigidAttachments = std::move(rebuilt);
    }

    if (triviallyComparableVectorsDiffer(previousAdjacencySignature, adjacencySignature))
    {
        ++mSoftConstraintAdjacencyRevision;
    }
    clearRebuildDirty(PhysicsRebuildFlags::ResolvedStrandRigidAttachments);
}

void PhysicsWorld::Impl::rebuildResolvedRigidDistanceConstraints() noexcept
{
    std::vector<RigidDistanceConstraint> rebuilt;

    for (const AuthoredRigidDistanceConstraintState &constraint : mRigidDistanceConstraintSnapshot)
    {
        const auto rigidBodyIndexA = mEntityToRigidBodyIndex.find(constraint.entityA);
        const auto rigidBodyIndexB = mEntityToRigidBodyIndex.find(constraint.entityB);
        if (rigidBodyIndexA == mEntityToRigidBodyIndex.end() ||
            rigidBodyIndexB == mEntityToRigidBodyIndex.end() ||
            rigidBodyIndexA->second == rigidBodyIndexB->second ||
            rigidBodyIndexA->second >= mRigidBodySnapshot.size() ||
            rigidBodyIndexB->second >= mRigidBodySnapshot.size())
        {
            continue;
        }

        const RigidBodyState &bodyA = mRigidBodySnapshot[rigidBodyIndexA->second];
        const RigidBodyState &bodyB = mRigidBodySnapshot[rigidBodyIndexB->second];
        if (bodyA.environmentIndex != bodyB.environmentIndex)
        {
            continue;
        }

        rebuilt.push_back(RigidDistanceConstraint{
            rigidBodyIndexA->second,
            rigidBodyIndexB->second,
            constraint.restDistance,
            constraint.compliance,
            constraint.enabled ? 1u : 0u,
            0u,
            Diligent::float4{constraint.localAnchorA.x, constraint.localAnchorA.y,
                             constraint.localAnchorA.z, 0.0f},
            Diligent::float4{constraint.localAnchorB.x, constraint.localAnchorB.y,
                             constraint.localAnchorB.z, 0.0f},
        });
    }

    if (triviallyComparableVectorsDiffer(mRigidDistanceConstraints, rebuilt))
    {
        mRigidDistanceConstraints = std::move(rebuilt);
        ++mRigidDistanceConstraintResolvedRevision;
    }
    else
    {
        mRigidDistanceConstraints = std::move(rebuilt);
    }
    clearRebuildDirty(PhysicsRebuildFlags::ResolvedRigidDistanceConstraints);
}

void PhysicsWorld::Impl::rebuildResolvedRoutedCables() noexcept
{
    std::vector<RoutedCableConstraint> rebuiltConstraints;
    std::vector<RoutedCableRoutePoint> rebuiltRoutePoints;

    for (const AuthoredRoutedCableConstraintState &constraint : mRoutedCableConstraintSnapshot)
    {
        if (constraint.routePoints.size() < 2u)
        {
            continue;
        }

        std::optional<std::uint32_t> environmentIndex;
        const std::uint32_t routePointStart = static_cast<std::uint32_t>(rebuiltRoutePoints.size());
        bool validRoute                     = true;
        std::optional<std::uint32_t> previousRigidBodyIndex;
        for (const AuthoredRoutedCableRoutePoint &routePoint : constraint.routePoints)
        {
            const auto rigidBodyIt = mEntityToRigidBodyIndex.find(routePoint.entityId);
            if (rigidBodyIt == mEntityToRigidBodyIndex.end() ||
                rigidBodyIt->second >= mRigidBodySnapshot.size())
            {
                validRoute = false;
                break;
            }

            const RigidBodyState &rigidBody = mRigidBodySnapshot[rigidBodyIt->second];
            if (!environmentIndex.has_value())
            {
                environmentIndex = rigidBody.environmentIndex;
            }
            else if (*environmentIndex != rigidBody.environmentIndex)
            {
                validRoute = false;
                break;
            }

            if (previousRigidBodyIndex.has_value() &&
                *previousRigidBodyIndex == rigidBodyIt->second)
            {
                validRoute = false;
                break;
            }

            previousRigidBodyIndex = rigidBodyIt->second;
            rebuiltRoutePoints.push_back(RoutedCableRoutePoint{
                rigidBodyIt->second, 0u, 0u, 0u,
                Diligent::float4{routePoint.localGuideOffset.x, routePoint.localGuideOffset.y,
                                 routePoint.localGuideOffset.z, 0.0f}});
        }

        if (!validRoute)
        {
            rebuiltRoutePoints.resize(routePointStart);
            continue;
        }

        rebuiltConstraints.push_back(RoutedCableConstraint{
            routePointStart, static_cast<std::uint32_t>(constraint.routePoints.size()),
            constraint.targetLength, constraint.compliance, constraint.tensionOnly ? 1u : 0u,
            constraint.enabled ? 1u : 0u, 0u, 0u});
    }

    const bool constraintsChanged =
        triviallyComparableVectorsDiffer(mRoutedCableConstraints, rebuiltConstraints);
    const bool routePointsChanged =
        triviallyComparableVectorsDiffer(mRoutedCableRoutePoints, rebuiltRoutePoints);
    mRoutedCableConstraints = std::move(rebuiltConstraints);
    mRoutedCableRoutePoints = std::move(rebuiltRoutePoints);
    if (constraintsChanged || routePointsChanged)
    {
        ++mRoutedCableResolvedRevision;
    }
    clearRebuildDirty(PhysicsRebuildFlags::ResolvedRoutedCables);
}

void PhysicsWorld::Impl::recomputeParticleGridCellSize() noexcept
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

void PhysicsWorld::Impl::recomputeSoftBodyBoundsChunkCount() noexcept
{
    mSoftBodyBoundsChunkCount = 0u;
    for (const Diligent::uint2 &range : mSoftRenderData.softBodyParticleRanges)
    {
        mSoftBodyBoundsChunkCount += (range.y + 64u - 1u) / 64u;
    }
}

bool PhysicsWorld::Impl::prepareFluidStateForInsert(const FluidState &candidate,
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

bool PhysicsWorld::Impl::prepareStrandStateForInsert(const StrandState &candidate,
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

    const std::uint32_t particleCount = static_cast<std::uint32_t>(candidate.restPositions.size());
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
        const Diligent::float3 tangent = safeNormalize(delta, Diligent::float3{1.0f, 0.0f, 0.0f});
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
        derivedCache.restJointRelativeOrientations.push_back(
            quaternionMultiply(quaternionConjugate(derivedCache.restSegmentOrientations[i]),
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

bool PhysicsWorld::Impl::prepareSoftBodyStateForInsert(const SoftBodyState &candidate,
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

const PhysicsWorld::Impl::TetGenMeshCache *PhysicsWorld::Impl::tryGetTetGenMeshCache(
    common::EntityId entityId) const noexcept
{
    const auto it = mTetGenMeshCache.find(entityId);
    return it == mTetGenMeshCache.end() ? nullptr : &it->second;
}

void PhysicsWorld::Impl::markAllRigidBodiesDirty() noexcept
{
    mFullRigidBodyUploadRequired = true;
    mRigidBodyCountDirty         = true;
    mRigidBodyDirtyBits.assign(static_cast<std::uint32_t>(mRigidBodies.size()), 1u);
    mRigidBodyDirtyIndices.resize(static_cast<std::uint32_t>(mRigidBodies.size()));
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(mRigidBodies.size()); ++i)
    {
        mRigidBodyDirtyIndices[i] = i;
    }
}

void PhysicsWorld::Impl::markAllCollidersDirty() noexcept
{
    mFullColliderUploadRequired = true;
    mColliderCountDirty         = true;
    mColliderDirtyBits.assign(static_cast<std::uint32_t>(mColliders.size()), 1u);
    mColliderDirtyIndices.resize(static_cast<std::uint32_t>(mColliders.size()));
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(mColliders.size()); ++i)
    {
        mColliderDirtyIndices[i] = i;
    }
}

void PhysicsWorld::Impl::markRigidBodyDirty(std::uint32_t index) noexcept
{
    enqueueDirtyIndex(index, mRigidBodyDirtyIndices, mRigidBodyDirtyBits);
}

void PhysicsWorld::Impl::markColliderDirty(std::uint32_t index) noexcept
{
    enqueueDirtyIndex(index, mColliderDirtyIndices, mColliderDirtyBits);
}

void PhysicsWorld::Impl::markRigidBodyCountDirty(bool fullUploadRequired) noexcept
{
    mRigidBodyCountDirty         = true;
    mFullRigidBodyUploadRequired = mFullRigidBodyUploadRequired || fullUploadRequired;
}

void PhysicsWorld::Impl::markColliderCountDirty(bool fullUploadRequired) noexcept
{
    mColliderCountDirty         = true;
    mFullColliderUploadRequired = mFullColliderUploadRequired || fullUploadRequired;
}

void PhysicsWorld::Impl::applyRigidJointChange(RigidJointChangeKind changeKind) noexcept
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

void PhysicsWorld::Impl::markJointSceneDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mAuthoredRevision;
}

void PhysicsWorld::Impl::markJointModeDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mAuthoredRevision;
}

void PhysicsWorld::Impl::markJointTopologyDirty() noexcept
{
    mRigidJointSceneDirty           = true;
    mJointCollisionSuppressionDirty = true;
    ++mRigidJointSceneRevision;
    ++mRigidJointModeRevision;
    ++mRigidJointTopologyRevision;
    ++mAuthoredRevision;
}

} // namespace cressim::neo::physics
