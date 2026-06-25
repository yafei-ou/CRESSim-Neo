#include "common/logger.h"
#include "engine/runtime.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace
{

struct AuthoredCartpoleEnv
{
    cressim::neo::common::EntityId base = cressim::neo::common::kInvalidEntityId;
    cressim::neo::common::EntityId cart = cressim::neo::common::kInvalidEntityId;
    cressim::neo::common::EntityId pole = cressim::neo::common::kInvalidEntityId;
    cressim::neo::physics::SliderJointId slider = cressim::neo::physics::kInvalidSliderJointId;
    cressim::neo::physics::HingeJointId hinge = cressim::neo::physics::kInvalidHingeJointId;
};

constexpr float kCartHalfX = 0.18f;
constexpr float kCartHalfY = 0.10f;
constexpr float kCartHalfZ = 0.10f;
constexpr float kPoleHalfY = 0.50f;
constexpr float kPoleHalfX = 0.05f;
constexpr float kPoleHalfZ = 0.05f;
constexpr float kEnvSpacingZ = 2.5f;
constexpr float kEpsilon = 1.0e-6f;

Diligent::QuaternionF quaternionFromBasis(const Diligent::float3 &x, const Diligent::float3 &y,
                                          const Diligent::float3 &z)
{
    const float m00 = x.x, m01 = y.x, m02 = z.x;
    const float m10 = x.y, m11 = y.y, m12 = z.y;
    const float m20 = x.z, m21 = y.z, m22 = z.z;
    const float trace = m00 + m11 + m22;

    Diligent::QuaternionF q{};
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.q.w         = 0.25f * s;
        q.q.x         = (m21 - m12) / s;
        q.q.y         = (m02 - m20) / s;
        q.q.z         = (m10 - m01) / s;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.q.w         = (m21 - m12) / s;
        q.q.x         = 0.25f * s;
        q.q.y         = (m01 + m10) / s;
        q.q.z         = (m02 + m20) / s;
    }
    else if (m11 > m22)
    {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.q.w         = (m02 - m20) / s;
        q.q.x         = (m01 + m10) / s;
        q.q.y         = 0.25f * s;
        q.q.z         = (m12 + m21) / s;
    }
    else
    {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.q.w         = (m10 - m01) / s;
        q.q.x         = (m02 + m20) / s;
        q.q.y         = (m12 + m21) / s;
        q.q.z         = 0.25f * s;
    }

    const float lengthSq = Diligent::dot(q.q, q.q);
    if (lengthSq <= kEpsilon)
    {
        return Diligent::QuaternionF{0.0f, 0.0f, 0.0f, 1.0f};
    }
    return Diligent::normalize(q);
}

Diligent::QuaternionF makeJointFrameRotation(const Diligent::float3 &axisX)
{
    const float lengthSq = Diligent::dot(axisX, axisX);
    const Diligent::float3 x = lengthSq <= kEpsilon ? Diligent::float3{1.0f, 0.0f, 0.0f}
                                                    : axisX * (1.0f / std::sqrt(lengthSq));
    Diligent::float3 reference{1.0f, 0.0f, 0.0f};
    if (std::abs(Diligent::dot(reference, x)) > 0.99f)
    {
        reference = {0.0f, 1.0f, 0.0f};
    }
    Diligent::float3 y = Diligent::cross(x, reference);
    const float yLengthSq = Diligent::dot(y, y);
    y = yLengthSq <= kEpsilon ? Diligent::float3{0.0f, 1.0f, 0.0f}
                              : y * (1.0f / std::sqrt(yLengthSq));
    Diligent::float3 z = Diligent::cross(x, y);
    const float zLengthSq = Diligent::dot(z, z);
    z = zLengthSq <= kEpsilon ? Diligent::float3{0.0f, 0.0f, 1.0f}
                              : z * (1.0f / std::sqrt(zLengthSq));
    return quaternionFromBasis(x, y, z);
}

AuthoredCartpoleEnv authorEnv(cressim::neo::engine::World &world, const std::uint32_t envIndex)
{
    using namespace cressim::neo;

    const float zOffset = static_cast<float>(envIndex) * kEnvSpacingZ;
    AuthoredCartpoleEnv authored{};

    authored.base = world.createEntity(envIndex);
    engine::TransformComponent baseTransform{};
    baseTransform.worldTransform.position = {0.0f, 0.5f, zOffset};
    world.setTransform(authored.base, baseTransform);

    engine::RigidBodyComponent baseBody{};
    baseBody.bodyType    = physics::RigidBodyType::Static;
    baseBody.inverseMass = 0.0f;
    baseBody.simulated   = true;
    world.setRigidBody(authored.base, baseBody);

    engine::ColliderComponent baseCollider{};
    baseCollider.shapeType   = physics::ColliderShapeType::Box;
    baseCollider.shapeParams = {0.15f, 0.15f, 0.15f, 0.0f};
    world.addCollider(authored.base, baseCollider);

    authored.cart = world.createEntity(envIndex);
    engine::TransformComponent cartTransform{};
    cartTransform.worldTransform.position = {0.0f, 0.5f, zOffset};
    world.setTransform(authored.cart, cartTransform);

    engine::RigidBodyComponent cartBody{};
    cartBody.bodyType          = physics::RigidBodyType::Dynamic;
    cartBody.inverseMass       = 1.0f;
    cartBody.inverseInertiaLocal = {2.0f, 2.0f, 2.0f};
    cartBody.simulated         = true;
    world.setRigidBody(authored.cart, cartBody);

    engine::ColliderComponent cartCollider{};
    cartCollider.shapeType   = physics::ColliderShapeType::Box;
    cartCollider.shapeParams = {kCartHalfX, kCartHalfY, kCartHalfZ, 0.0f};
    world.addCollider(authored.cart, cartCollider);

    authored.pole = world.createEntity(envIndex);
    engine::TransformComponent poleTransform{};
    poleTransform.worldTransform.position = {0.0f, 0.5f + kCartHalfY + kPoleHalfY, zOffset};
    world.setTransform(authored.pole, poleTransform);

    engine::RigidBodyComponent poleBody{};
    poleBody.bodyType            = physics::RigidBodyType::Dynamic;
    poleBody.inverseMass         = 0.5f;
    poleBody.inverseInertiaLocal = {1.5f, 1.5f, 1.5f};
    poleBody.simulated           = true;
    world.setRigidBody(authored.pole, poleBody);

    engine::ColliderComponent poleCollider{};
    poleCollider.shapeType   = physics::ColliderShapeType::Box;
    poleCollider.shapeParams = {kPoleHalfX, kPoleHalfY, kPoleHalfZ, 0.0f};
    world.addCollider(authored.pole, poleCollider);

    physics::SliderJointState slider{};
    slider.jointId                        = 1000u + envIndex;
    slider.bodyA                          = authored.base;
    slider.bodyB                          = authored.cart;
    slider.suppressConnectedBodyCollisions = true;
    slider.driveMode                      = physics::RigidJointDriveMode::TargetVelocity;
    slider.localRotationA                 = makeJointFrameRotation({1.0f, 0.0f, 0.0f});
    slider.localRotationB                 = makeJointFrameRotation({1.0f, 0.0f, 0.0f});
    slider.limitEnabled                   = true;
    slider.limitMin                       = -2.4f;
    slider.limitMax                       = 2.4f;
    slider.driveTargetVelocity            = 0.0f;
    if (!world.upsertSliderJoint(slider))
    {
        authored.slider = physics::kInvalidSliderJointId;
        return authored;
    }
    authored.slider = slider.jointId;

    physics::HingeJointState hinge{};
    hinge.jointId                         = 2000u + envIndex;
    hinge.bodyA                           = authored.cart;
    hinge.bodyB                           = authored.pole;
    hinge.suppressConnectedBodyCollisions = true;
    hinge.localAnchorA                    = {0.0f, kCartHalfY, 0.0f};
    hinge.localAnchorB                    = {0.0f, -kPoleHalfY, 0.0f};
    hinge.localRotationA                  = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    hinge.localRotationB                  = makeJointFrameRotation({0.0f, 0.0f, 1.0f});
    hinge.limitEnabled                    = false;
    if (!world.upsertHingeJoint(hinge))
    {
        authored.hinge = physics::kInvalidHingeJointId;
        return authored;
    }
    authored.hinge = hinge.jointId;

    return authored;
}

bool verifyMapping(const cressim::neo::engine::JointLayoutMapping &mapping,
                   const std::vector<AuthoredCartpoleEnv> &authored)
{
    if (mapping.bindingGeneration == 0u)
    {
        CRESSIM_LOG_ERROR("Prepared joint mapping generation should be non-zero after prepare.");
        return false;
    }
    if (mapping.hingeJointCount != authored.size() || mapping.sliderJointCount != authored.size())
    {
        CRESSIM_LOG_ERROR("Unexpected hinge/slider counts in prepared joint mapping.");
        return false;
    }

    for (std::size_t envIndex = 0; envIndex < authored.size(); ++envIndex)
    {
        if (mapping.sliderJointIds[envIndex] != authored[envIndex].slider ||
            mapping.hingeJointIds[envIndex] != authored[envIndex].hinge)
        {
            CRESSIM_LOG_ERROR("Joint mapping did not preserve authored joint ids.");
            return false;
        }
        if (mapping.sliderEnvironmentIndices[envIndex] != envIndex ||
            mapping.hingeEnvironmentIndices[envIndex] != envIndex)
        {
            CRESSIM_LOG_ERROR("Joint mapping environment indices are incorrect.");
            return false;
        }
        if (mapping.sliderBodyIdsA[envIndex] != authored[envIndex].base ||
            mapping.sliderBodyIdsB[envIndex] != authored[envIndex].cart ||
            mapping.hingeBodyIdsA[envIndex] != authored[envIndex].cart ||
            mapping.hingeBodyIdsB[envIndex] != authored[envIndex].pole)
        {
            CRESSIM_LOG_ERROR("Joint mapping body ids are incorrect.");
            return false;
        }
    }

    return true;
}

bool hasResource(const std::vector<cressim::neo::engine::CustomComputeResourceDesc> &resources,
                 const char *key)
{
    return std::any_of(resources.begin(), resources.end(),
                       [key](const cressim::neo::engine::CustomComputeResourceDesc &resource)
                       { return resource.key == key; });
}

} // namespace

int main()
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.sceneLayout.envCount           = 3u;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping runtime joint layout mapping test because runtime "
                            "initialization failed.");
        return 0;
    }

    auto &world = runtime.getWorld();
    std::vector<AuthoredCartpoleEnv> authored;
    for (std::uint32_t envIndex = 0; envIndex < config.sceneLayout.envCount; ++envIndex)
    {
        authored.push_back(authorEnv(world, envIndex));
        if (authored.back().slider == physics::kInvalidSliderJointId ||
            authored.back().hinge == physics::kInvalidHingeJointId)
        {
            CRESSIM_LOG_ERROR("Failed to author cartpole joints for prepared mapping test.");
            runtime.shutdown();
            return 1;
        }
    }

    runtime.prepare();

    engine::JointLayoutMapping mapping{};
    if (!runtime.tryGetPreparedJointLayoutMapping(mapping))
    {
        CRESSIM_LOG_ERROR("Runtime prepared joint layout mapping query failed.");
        runtime.shutdown();
        return 1;
    }
    if (!verifyMapping(mapping, authored))
    {
        runtime.shutdown();
        return 1;
    }

    const std::uint64_t previousGeneration = mapping.bindingGeneration;

    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload world before querying custom compute resources.");
        runtime.shutdown();
        return 1;
    }

    const std::vector<engine::CustomComputeResourceDesc> resources =
        runtime.listCustomComputeResources();
    if (!hasResource(resources, "joint.hinge") || !hasResource(resources, "joint.slider") ||
        !hasResource(resources, "rigid.linear_velocities") ||
        !hasResource(resources, "rigid.angular_velocities"))
    {
        CRESSIM_LOG_ERROR("Expected joint/velocity custom compute resources are missing.");
        runtime.shutdown();
        return 1;
    }

    const AuthoredCartpoleEnv added = authorEnv(world, 2u);
    if (added.slider == physics::kInvalidSliderJointId || added.hinge == physics::kInvalidHingeJointId)
    {
        CRESSIM_LOG_ERROR("Failed to author additional cartpole env for generation test.");
        runtime.shutdown();
        return 1;
    }

    runtime.prepare();
    engine::JointLayoutMapping updatedMapping{};
    if (!runtime.tryGetPreparedJointLayoutMapping(updatedMapping))
    {
        CRESSIM_LOG_ERROR("Updated prepared joint layout mapping query failed.");
        runtime.shutdown();
        return 1;
    }
    if (updatedMapping.hingeJointCount != mapping.hingeJointCount + 1u ||
        updatedMapping.sliderJointCount != mapping.sliderJointCount + 1u)
    {
        CRESSIM_LOG_ERROR("Joint layout mapping counts did not update after structural authoring.");
        runtime.shutdown();
        return 1;
    }
    if (updatedMapping.bindingGeneration == previousGeneration)
    {
        CRESSIM_LOG_ERROR("Joint layout binding generation did not change after structural authoring.");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    return 0;
}
