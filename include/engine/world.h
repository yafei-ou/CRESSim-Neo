#ifndef CRESSIM_NEO_ENGINE_WORLD_H
#define CRESSIM_NEO_ENGINE_WORLD_H

#include "common/id.h"
#include "engine/components.h"
#include "engine/export.h"
#include "engine/render_scene_types.h"
#include "graphics/host_scene.h"
#include "physics/physics_world.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace cressim::neo::engine
{

/// @brief Primary ECS scene graph container managing entities, components, graphics views, and physics bindings.
class CRESSIM_NEO_ENGINE_API World
{
public:
    using ColliderHandle = engine::ColliderHandle;

    World();
    ~World();

    World(const World &other);
    World &operator=(const World &other);
    World(World &&other) noexcept;
    World &operator=(World &&other) noexcept;

    /// @brief Creates a new entity within the world scene graph.
    /// @param envIndex Environment index for parallel RL environments (default: 0).
    /// @return Unique EntityId for the created entity.
    common::EntityId createEntity(std::uint32_t envIndex = 0u);

    /// @brief Destroys an existing entity and removes all associated components.
    /// @param entityId Entity ID to destroy.
    /// @return True if entity was destroyed; false if invalid or not found.
    bool destroyEntity(common::EntityId entityId);

    /// @brief Sets the scene layout capacities.
    /// @param layout Scene layout descriptor.
    void setSceneLayout(const common::SceneLayoutDesc &layout);

    /// @brief Gets the current scene layout capacity descriptor.
    /// @return Reference to current SceneLayoutDesc.
    const common::SceneLayoutDesc &sceneLayout() const noexcept;

    /// @brief Assigns an entity to a specific environment index.
    /// @param entityId Entity ID.
    /// @param envIndex Environment index.
    /// @return True if assignment succeeded.
    bool setEntityEnvironment(common::EntityId entityId, std::uint32_t envIndex);

    /// @brief Gets the environment index assigned to an entity.
    /// @param entityId Entity ID.
    /// @return Environment index.
    std::uint32_t entityEnvironment(common::EntityId entityId) const noexcept;

    /// @brief Configures Image-Based Lighting (IBL) environment maps for an environment index.
    /// @param envIndex Environment index.
    /// @param desc Environment IBL descriptor.
    /// @return True if configured successfully.
    bool setEnvironmentIbl(std::uint32_t envIndex, const graphics::EnvironmentIblDesc &desc);

    /// @brief Tries to get the Environment IBL descriptor for an environment index.
    /// @param envIndex Environment index.
    /// @return Pointer to EnvironmentIblDesc if set; nullptr otherwise.
    const graphics::EnvironmentIblDesc *tryGetEnvironmentIbl(std::uint32_t envIndex) const noexcept;

    /// @brief Configures environment fluid properties for an environment index.
    /// @param envIndex Environment index.
    /// @param desc Environment fluid descriptor.
    /// @return True if configured successfully.
    bool setEnvironmentFluid(std::uint32_t envIndex, const graphics::EnvironmentFluidDesc &desc);

    /// @brief Tries to get the Environment fluid descriptor for an environment index.
    /// @param envIndex Environment index.
    /// @return Pointer to EnvironmentFluidDesc if set; nullptr otherwise.
    const graphics::EnvironmentFluidDesc *tryGetEnvironmentFluid(
        std::uint32_t envIndex) const noexcept;

    /// @brief Checks if an entity is alive and active in the world.
    /// @param entityId Entity ID to query.
    /// @return True if alive, false otherwise.
    bool isAlive(common::EntityId entityId) const;

    /// @brief Returns the list of all active entity IDs in the world.
    /// @return Const reference to vector of active EntityIds.
    const std::vector<common::EntityId> &entities() const noexcept;

    /// @brief Assigns or updates the TransformComponent for an entity.
    /// @param entityId Target entity ID.
    /// @param component Transform component data.
    void setTransform(common::EntityId entityId, const TransformComponent &component);

    /// @brief Assigns or updates the MeshRendererComponent for an entity.
    /// @param entityId Target entity ID.
    /// @param component Mesh renderer component data.
    void setMeshRenderer(common::EntityId entityId, const MeshRendererComponent &component);

    /// @brief Assigns or updates the CameraComponent for an entity.
    /// @param entityId Target entity ID.
    /// @param component Camera component data.
    void setCamera(common::EntityId entityId, const CameraComponent &component);

    /// @brief Assigns or updates a DirectionalLightComponent for an entity.
    /// @param entityId Target entity ID.
    /// @param component Directional light component data.
    void setDirectionalLight(common::EntityId entityId, const DirectionalLightComponent &component);

    /// @brief Assigns or updates a PointLightComponent for an entity.
    /// @param entityId Target entity ID.
    /// @param component Point light component data.
    void setPointLight(common::EntityId entityId, const PointLightComponent &component);

    /// @brief Assigns or updates a SpotLightComponent for an entity.
    /// @param entityId Target entity ID.
    /// @param component Spot light component data.
    void setSpotLight(common::EntityId entityId, const SpotLightComponent &component);

    /// @brief Assigns or updates a RigidBodyComponent for an entity.
    /// @param entityId Target entity ID.
    /// @param component Rigid body component data.
    void setRigidBody(common::EntityId entityId, const RigidBodyComponent &component);

    /// @brief Removes the RigidBodyComponent from an entity.
    /// @param entityId Target entity ID.
    /// @return True if removed; false otherwise.
    bool removeRigidBody(common::EntityId entityId);

    /// @brief Assigns a SoftBodyComponent to an entity.
    /// @param entityId Target entity ID.
    /// @param component Soft body component data.
    /// @return True if set successfully.
    bool setSoftBody(common::EntityId entityId, const SoftBodyComponent &component);

    /// @brief Assigns a MeshfreeSoftBodyComponent to an entity.
    /// @param entityId Target entity ID.
    /// @param component Meshfree soft body component data.
    /// @return True if set successfully.
    bool setMeshfreeSoftBody(common::EntityId entityId, const MeshfreeSoftBodyComponent &component);

    /// @brief Removes the SoftBodyComponent from an entity.
    /// @param entityId Target entity ID.
    /// @return True if removed; false otherwise.
    bool removeSoftBody(common::EntityId entityId);

    /// @brief Assigns a StrandComponent to an entity.
    /// @param entityId Target entity ID.
    /// @param component Strand component data.
    /// @return True if set successfully.
    bool setStrand(common::EntityId entityId, const StrandComponent &component);

    /// @brief Removes the StrandComponent from an entity.
    /// @param entityId Target entity ID.
    /// @return True if removed; false otherwise.
    bool removeStrand(common::EntityId entityId);

    /// @brief Assigns a ProceduralDeformableCurveRenderComponent to an entity.
    /// @param entityId Target entity ID.
    /// @param component Procedural curve render component data.
    void setProceduralDeformableCurveRender(
        common::EntityId entityId, const ProceduralDeformableCurveRenderComponent &component);

    /// @brief Removes ProceduralDeformableCurveRenderComponent from an entity.
    /// @param entityId Target entity ID.
    /// @return True if removed; false otherwise.
    bool removeProceduralDeformableCurveRender(common::EntityId entityId);

    /// @brief Assigns a FluidComponent to an entity.
    /// @param entityId Target entity ID.
    /// @param component Fluid component data.
    /// @return True if set successfully.
    bool setFluid(common::EntityId entityId, const FluidComponent &component);

    /// @brief Removes the FluidComponent from an entity.
    /// @param entityId Target entity ID.
    /// @return True if removed; false otherwise.
    bool removeFluid(common::EntityId entityId);

    physics::AuthoredParticleSequenceState &upsertParticleSequence(
        const physics::AuthoredParticleSequenceState &state);
    physics::AuthoredParticleDistanceConstraintState &upsertParticleDistanceConstraint(
        const physics::AuthoredParticleDistanceConstraintState &state);
    bool upsertRigidParticleAttachmentConstraint(
        const physics::AuthoredRigidParticleAttachmentConstraintState &state,
        physics::AuthoredRigidParticleAttachmentConstraintState *outAuthored = nullptr);
    bool upsertStrandRigidAttachmentConstraint(
        const physics::AuthoredStrandRigidAttachmentConstraintState &state,
        physics::AuthoredStrandRigidAttachmentConstraintState *outAuthored = nullptr);
    bool upsertRigidDistanceConstraint(
        const physics::AuthoredRigidDistanceConstraintState &state,
        physics::AuthoredRigidDistanceConstraintState *outAuthored = nullptr);
    bool upsertRoutedCableConstraint(
        const physics::AuthoredRoutedCableConstraintState &state,
        physics::AuthoredRoutedCableConstraintState *outAuthored = nullptr);
    bool upsertBallJoint(const physics::BallJointState &state);
    bool upsertHingeJoint(const physics::HingeJointState &state);
    bool upsertSphericalJoint(const physics::SphericalJointState &state);
    bool upsertSliderJoint(const physics::SliderJointState &state);
    physics::AuthoredParticleCollisionFilterState &upsertParticleCollisionFilter(
        const physics::AuthoredParticleCollisionFilterState &state);
    physics::AuthoredSuturingSequenceState &upsertSuturingSequence(
        const physics::AuthoredSuturingSequenceState &state);
    bool removeParticleDistanceConstraint(physics::ParticleConstraintId constraintId);
    bool removeRigidParticleAttachmentConstraint(
        physics::RigidParticleAttachmentConstraintId constraintId);
    bool removeStrandRigidAttachmentConstraint(
        physics::StrandRigidAttachmentConstraintId constraintId);
    bool removeRigidDistanceConstraint(physics::RigidDistanceConstraintId constraintId);
    bool removeRoutedCableConstraint(physics::RoutedCableConstraintId constraintId);
    bool removeBallJoint(physics::BallJointId jointId);
    bool removeHingeJoint(physics::HingeJointId jointId);
    bool removeSphericalJoint(physics::SphericalJointId jointId);
    bool removeSliderJoint(physics::SliderJointId jointId);
    bool removeParticleCollisionFilter(physics::ParticleCollisionFilterId filterId);
    bool removeSuturingSequence(physics::SuturingSequenceId sequenceId);
    void setUltrasoundProbe(common::EntityId entityId, const UltrasoundProbeComponent &component);
    bool removeUltrasoundProbe(common::EntityId entityId);
    void setUltrasoundRenderer(common::EntityId entityId,
                               const UltrasoundRendererComponent &component);
    bool removeUltrasoundRenderer(common::EntityId entityId);
    void setUltrasoundScattererSource(common::EntityId entityId,
                                      const UltrasoundScattererSourceComponent &component);
    bool removeUltrasoundScattererSource(common::EntityId entityId);
    void setUltrasoundScattererAmplitudeRanges(common::EntityId entityId,
                                               const std::vector<UltrasoundAmplitudeRange> &ranges);
    bool clearUltrasoundScattererAmplitudeRanges(common::EntityId entityId);

    ColliderHandle addCollider(common::EntityId entityId, const ColliderComponent &component);
    void updateCollider(ColliderHandle handle, const ColliderComponent &component);
    bool removeCollider(ColliderHandle handle);
    bool replaceColliders(common::EntityId entityId,
                          const std::vector<ColliderComponent> &components);

    bool removeTransform(common::EntityId entityId);
    bool removeMeshRenderer(common::EntityId entityId);
    bool removeCamera(common::EntityId entityId);
    bool removeDirectionalLight(common::EntityId entityId);
    bool removePointLight(common::EntityId entityId);
    bool removeSpotLight(common::EntityId entityId);

    std::optional<TransformComponent> tryGetTransform(common::EntityId entityId) const;
    std::optional<MeshRendererComponent> tryGetMeshRenderer(common::EntityId entityId) const;
    std::optional<CameraComponent> tryGetCamera(common::EntityId entityId) const;
    std::optional<DirectionalLightComponent> tryGetDirectionalLight(
        common::EntityId entityId) const;
    std::optional<PointLightComponent> tryGetPointLight(common::EntityId entityId) const;
    std::optional<SpotLightComponent> tryGetSpotLight(common::EntityId entityId) const;

    std::optional<RigidBodyComponent> tryGetRigidBody(common::EntityId entityId) const;
    std::optional<SoftBodyComponent> tryGetSoftBody(common::EntityId entityId) const;
    std::optional<StrandComponent> tryGetStrand(common::EntityId entityId) const;
    std::optional<ProceduralDeformableCurveRenderComponent> tryGetProceduralDeformableCurveRender(
        common::EntityId entityId) const;
    std::optional<FluidComponent> tryGetFluid(common::EntityId entityId) const;
    const physics::AuthoredParticleSequenceState *tryGetParticleSequence(
        physics::ParticleSequenceId sequenceId) const noexcept;
    const physics::AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        physics::ParticleConstraintId constraintId) const noexcept;
    const physics::AuthoredRigidParticleAttachmentConstraintState *
    tryGetRigidParticleAttachmentConstraint(
        physics::RigidParticleAttachmentConstraintId constraintId) const noexcept;
    const physics::AuthoredStrandRigidAttachmentConstraintState *
    tryGetStrandRigidAttachmentConstraint(
        physics::StrandRigidAttachmentConstraintId constraintId) const noexcept;
    const physics::AuthoredRigidDistanceConstraintState *tryGetRigidDistanceConstraint(
        physics::RigidDistanceConstraintId constraintId) const noexcept;
    const physics::AuthoredRoutedCableConstraintState *tryGetRoutedCableConstraint(
        physics::RoutedCableConstraintId constraintId) const noexcept;
    const physics::BallJointState *tryGetBallJoint(physics::BallJointId jointId) const noexcept;
    const physics::HingeJointState *tryGetHingeJoint(physics::HingeJointId jointId) const noexcept;
    const physics::SphericalJointState *tryGetSphericalJoint(
        physics::SphericalJointId jointId) const noexcept;
    const physics::SliderJointState *tryGetSliderJoint(
        physics::SliderJointId jointId) const noexcept;
    const physics::AuthoredParticleCollisionFilterState *tryGetParticleCollisionFilter(
        physics::ParticleCollisionFilterId filterId) const noexcept;
    const physics::AuthoredSuturingSequenceState *tryGetSuturingSequence(
        physics::SuturingSequenceId sequenceId) const noexcept;
    std::optional<UltrasoundProbeComponent> tryGetUltrasoundProbe(common::EntityId entityId) const;
    std::optional<UltrasoundRendererComponent> tryGetUltrasoundRenderer(
        common::EntityId entityId) const;
    std::optional<UltrasoundScattererSourceComponent> tryGetUltrasoundScattererSource(
        common::EntityId entityId) const;
    std::optional<SoftBodyAuthoringParticles> tryGetSoftBodyAuthoringParticles(
        common::EntityId entityId) const;
    const std::vector<UltrasoundAmplitudeRange> *tryGetUltrasoundScattererAmplitudeRanges(
        common::EntityId entityId) const noexcept;
    const UltrasoundProbeResult *tryGetUltrasoundProbeResult(
        common::EntityId entityId) const noexcept;
    std::optional<ColliderComponent> tryGetCollider(ColliderHandle handle) const;
    const std::vector<ColliderHandle> &colliderHandles(common::EntityId entityId) const;

    /// @brief Gets a reference to the internal PhysicsWorld instance.
    /// @return Reference to PhysicsWorld.
    physics::PhysicsWorld &physicsWorld() noexcept;

    /// @brief Gets a const reference to the internal PhysicsWorld instance.
    /// @return Const reference to PhysicsWorld.
    const physics::PhysicsWorld &physicsWorld() const noexcept;

    void setGpuEntityScene(const graphics::GpuEntitySceneView &sceneView) noexcept;

    const std::vector<graphics::RenderableInstance> &renderables() const noexcept;
    const std::vector<graphics::CameraData> &cameras() const noexcept;
    const std::vector<graphics::LightData> &lights() const noexcept;
    std::uint32_t entityPoseSlot(common::EntityId entityId) const noexcept;
    const std::vector<Diligent::float4> &entityPosePositions() const noexcept;
    const std::vector<Diligent::float4> &entityPoseOrientations() const noexcept;
    const std::vector<Diligent::float4> &entityPoseScales() const noexcept;
    const std::vector<Diligent::float4> &renderObjectPositions() const noexcept;
    const std::vector<Diligent::float4> &renderObjectOrientations() const noexcept;
    const std::vector<Diligent::float4> &renderObjectScales() const noexcept;
    const std::vector<graphics::GpuRenderableMetadata> &renderableMetadata() const noexcept;
    const std::vector<graphics::GpuRenderableQueueInfo> &renderableQueueInfo() const noexcept;
    const std::vector<graphics::GpuCameraInput> &cameraInputs() const noexcept;
    const std::vector<graphics::GpuLightInput> &lightInputs() const noexcept;
    const std::vector<graphics::GpuLocalLightSelection> &localLightSelections() const noexcept;
    const std::vector<graphics::GpuSoftBodyVertexBinding> &softBodyVertexBindings() const noexcept;
    const std::vector<graphics::IndirectCommandRegistryEntry> &opaqueDrawRegistry() const noexcept;
    const std::vector<graphics::TransparentDrawEntry> &transparentDrawRegistry() const noexcept;
    const std::vector<graphics::IndirectCommandRegistryEntry> &shadowDrawRegistry() const noexcept;
    const std::vector<graphics::IndirectCommandRegistryEntry> &localShadowDrawRegistry()
        const noexcept;
    const std::vector<EntityPoseMappingEntry> &physicsRenderableMappings();
    std::uint64_t entityPoseRevision() const noexcept;
    std::uint64_t renderableMetadataRevision() const noexcept;
    std::uint64_t renderableQueueInfoRevision() const noexcept;
    std::uint64_t softBodyVertexBindingRevision() const noexcept;
    std::uint64_t cameraInputRevision() const noexcept;
    std::uint64_t lightInputRevision() const noexcept;
    std::uint64_t localLightSelectionRevision() const noexcept;
    const graphics::GpuEntitySceneView &gpuEntityScene() const noexcept;
    graphics::HostSceneView hostSceneView() const noexcept;
    void ensureRenderStateUpToDate(const graphics::RenderResourceManager &resources);
    const std::unordered_map<common::EntityId, UltrasoundProbeComponent> &
    ultrasoundProbeComponents() const noexcept;
    const std::unordered_map<common::EntityId, UltrasoundRendererComponent> &
    ultrasoundRendererComponents() const noexcept;
    const std::unordered_map<common::EntityId, UltrasoundScattererSourceComponent> &
    ultrasoundScattererSourceComponents() const noexcept;
    std::uint64_t ultrasoundScattererAmplitudeRevision() const noexcept;
    void setUltrasoundProbeResult(common::EntityId entityId, const UltrasoundProbeResult &result);
    void clearUltrasoundProbeResult(common::EntityId entityId);
    bool removeParticleSequence(physics::ParticleSequenceId sequenceId);

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_WORLD_H
