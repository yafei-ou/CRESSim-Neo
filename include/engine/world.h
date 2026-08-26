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

/// @brief Primary ECS scene graph container managing entities, components, graphics views, and
/// physics bindings.
class CRESSIM_NEO_ENGINE_API World
{
public:
    using ColliderHandle = engine::ColliderHandle;

    /// @brief Constructs an empty world with the default scene layout.
    World();

    /// @brief Releases world state.
    ~World();

    /// @brief Deep-copies world state from @p other.
    World(const World &other);

    /// @brief Replaces this world with a deep copy of @p other.
    World &operator=(const World &other);

    /// @brief Transfers world state from @p other; the moved-from object is valid only for
    /// destruction or reassignment.
    World(World &&other) noexcept;

    /// @brief Transfers world state from @p other; the moved-from object is valid only for
    /// destruction or reassignment.
    World &operator=(World &&other) noexcept;

    /// @brief Creates a new entity within the world scene graph.
    /// @param envIndex Environment index for parallel RL environments (default: 0).
    /// @return Unique EntityId for the created entity, or kInvalidEntityId if envIndex is outside
    /// the configured environment count.
    common::EntityId createEntity(std::uint32_t envIndex = 0u);

    /// @brief Destroys an existing entity and removes all associated components.
    /// @param entityId Entity ID to destroy.
    /// @return True if entity was destroyed; false if invalid or not found.
    bool destroyEntity(common::EntityId entityId);

    /// @brief Sets the scene layout capacities before any scene authoring occurs.
    /// @param layout Scene layout descriptor.
    /// @note Calls made after authoring has begun leave the existing layout unchanged.
    void setSceneLayout(const common::SceneLayoutDesc &layout);

    /// @brief Gets the current scene layout capacity descriptor.
    /// @return Reference to current SceneLayoutDesc.
    const common::SceneLayoutDesc &sceneLayout() const noexcept;

    /// @brief Assigns an entity to a specific environment index, migrating its associated state.
    /// @param entityId Entity ID.
    /// @param envIndex Environment index.
    /// @return True if assignment succeeded; false for an invalid entity or environment index, or
    /// if associated state cannot be migrated.
    bool setEntityEnvironment(common::EntityId entityId, std::uint32_t envIndex);

    /// @brief Gets the environment index assigned to an entity.
    /// @param entityId Entity ID.
    /// @return Environment index, or zero when @p entityId is not alive.
    std::uint32_t entityEnvironment(common::EntityId entityId) const noexcept;

    /// @brief Configures Image-Based Lighting (IBL) environment maps for an environment index.
    /// @param envIndex Environment index.
    /// @param desc Environment IBL descriptor.
    /// @return True if configured successfully.
    bool setEnvironmentIbl(std::uint32_t envIndex, const graphics::EnvironmentIblDesc &desc);

    /// @brief Gets the Environment IBL descriptor stored for an initialized environment index.
    /// @param envIndex Environment index.
    /// @return Pointer to EnvironmentIblDesc (including its default value when not explicitly
    /// configured), or nullptr if host scene storage has not been initialized for envIndex.
    const graphics::EnvironmentIblDesc *tryGetEnvironmentIbl(std::uint32_t envIndex) const noexcept;

    /// @brief Configures environment fluid properties for an environment index.
    /// @param envIndex Environment index.
    /// @param desc Environment fluid descriptor.
    /// @return True if configured successfully.
    bool setEnvironmentFluid(std::uint32_t envIndex, const graphics::EnvironmentFluidDesc &desc);

    /// @brief Gets the Environment fluid descriptor stored for an initialized environment index.
    /// @param envIndex Environment index.
    /// @return Pointer to EnvironmentFluidDesc (including its default value when not explicitly
    /// configured), or nullptr if host scene storage has not been initialized for envIndex.
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
    ///
    /// On an entity with a rigid body, this immediately overwrites the body's pose (a teleport);

    /// it does not create a kinematic target.
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

    /// @brief Removes the RigidBodyComponent from an entity and removes its colliders.
    /// @param entityId Target entity ID.
    /// @return True if removed; false otherwise.
    bool removeRigidBody(common::EntityId entityId);

    /// @brief Assigns a SoftBodyComponent to an entity.
    ///
    /// A successful replacement clears any authored ultrasound scatterer amplitude ranges for the
    /// entity because their required count depends on the soft body's rest-particle set.
    /// @param entityId Target entity ID.
    /// @param component Soft body component data.
    /// @return True if set successfully.
    bool setSoftBody(common::EntityId entityId, const SoftBodyComponent &component);

    /// @brief Assigns a MeshfreeSoftBodyComponent to an entity.
    /// @param entityId Target entity ID.
    /// @param component Meshfree soft body component data.
    /// @return True if set successfully.
    bool setMeshfreeSoftBody(common::EntityId entityId, const MeshfreeSoftBodyComponent &component);

    /// @brief Removes the SoftBodyComponent and any authored ultrasound amplitude ranges.
    /// @param entityId Target entity ID.
    /// @return True if either the soft body or its amplitude ranges existed and was removed.
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

    /// @brief Creates or updates an authored particle sequence.
    /// @param state Sequence state to store.
    /// @return Reference to the stored sequence state.
    physics::AuthoredParticleSequenceState &upsertParticleSequence(
        const physics::AuthoredParticleSequenceState &state);

    /// @brief Creates or updates an authored particle distance constraint.
    ///
    /// An invalid or unknown ID creates a new constraint. Negative rest length and compliance are
    /// clamped to zero; particle references remain authored until derived state is rebuilt.
    /// @param state Constraint state to store.
    /// @return Reference to the stored constraint state.
    physics::AuthoredParticleDistanceConstraintState &upsertParticleDistanceConstraint(
        const physics::AuthoredParticleDistanceConstraintState &state);

    /// @brief Creates or updates a rigid-particle attachment constraint.
    ///
    /// Its referenced particle and rigid body must exist in the same environment. Negative
    /// compliance is clamped to zero.
    /// @param state Constraint state to store.
    /// @param outAuthored Optional output receiving the stored state.
    /// @return False for invalid/out-of-range references or different environments.
    bool upsertRigidParticleAttachmentConstraint(
        const physics::AuthoredRigidParticleAttachmentConstraintState &state,
        physics::AuthoredRigidParticleAttachmentConstraintState *outAuthored = nullptr);

    /// @brief Creates or updates a strand-rigid attachment constraint.
    ///
    /// The strand segment and rigid body must exist in the same environment. `segmentT` is clamped
    /// to [0, 1], the local rotation is normalized, and negative compliance is clamped to zero.
    /// @return False for invalid references or different environments.
    bool upsertStrandRigidAttachmentConstraint(
        const physics::AuthoredStrandRigidAttachmentConstraintState &state,
        physics::AuthoredStrandRigidAttachmentConstraintState *outAuthored = nullptr);

    /// @brief Creates or updates a rigid distance constraint.
    ///
    /// Bodies must be distinct and in the same environment. Negative rest distance and compliance
    /// are clamped to zero.
    /// @return False for invalid/same-body references or different environments.
    bool upsertRigidDistanceConstraint(
        const physics::AuthoredRigidDistanceConstraintState &state,
        physics::AuthoredRigidDistanceConstraintState *outAuthored = nullptr);

    /// @brief Creates or updates a routed cable constraint.
    ///
    /// The route needs at least two rigid-body guide points in one environment, with no consecutive
    /// duplicate bodies. Negative target length and compliance are clamped to zero.
    /// @return False for an invalid route or guide-body reference.
    bool upsertRoutedCableConstraint(
        const physics::AuthoredRoutedCableConstraintState &state,
        physics::AuthoredRoutedCableConstraintState *outAuthored = nullptr);

    /// @brief Creates or updates a ball joint between two rigid-body entities.
    /// @return False unless both entities are alive, have rigid bodies, and share an environment.
    bool upsertBallJoint(const physics::BallJointState &state);

    /// @brief Creates or updates a hinge joint between two rigid-body entities.
    /// @return False unless both entities are alive, have rigid bodies, and share an environment.
    bool upsertHingeJoint(const physics::HingeJointState &state);

    /// @brief Creates or updates a spherical joint between two rigid-body entities.
    /// @return False unless both entities are alive, have rigid bodies, and share an environment.
    bool upsertSphericalJoint(const physics::SphericalJointState &state);

    /// @brief Creates or updates a slider joint between two rigid-body entities.
    /// @return False unless both entities are alive, have rigid bodies, and share an environment.
    bool upsertSliderJoint(const physics::SliderJointState &state);

    /// @brief Creates or updates an authored particle collision filter.
    ///
    /// An invalid or unknown filter ID creates a new filter; a zero collision layer becomes 1.
    physics::AuthoredParticleCollisionFilterState &upsertParticleCollisionFilter(
        const physics::AuthoredParticleCollisionFilterState &state);

    /// @brief Creates or updates an authored suturing sequence.
    ///
    /// Negative path-node spacing is clamped to zero and the tip index is clamped to the entries.
    physics::AuthoredSuturingSequenceState &upsertSuturingSequence(
        const physics::AuthoredSuturingSequenceState &state);

    /// @brief Removes an authored particle distance constraint.
    bool removeParticleDistanceConstraint(physics::ParticleConstraintId constraintId);

    /// @brief Removes a rigid-particle attachment constraint.
    bool removeRigidParticleAttachmentConstraint(
        physics::RigidParticleAttachmentConstraintId constraintId);

    /// @brief Removes a strand-rigid attachment constraint.
    bool removeStrandRigidAttachmentConstraint(
        physics::StrandRigidAttachmentConstraintId constraintId);

    /// @brief Removes a rigid distance constraint.
    bool removeRigidDistanceConstraint(physics::RigidDistanceConstraintId constraintId);

    /// @brief Removes a routed cable constraint.
    bool removeRoutedCableConstraint(physics::RoutedCableConstraintId constraintId);

    /// @brief Removes a ball joint.
    bool removeBallJoint(physics::BallJointId jointId);

    /// @brief Removes a hinge joint.
    bool removeHingeJoint(physics::HingeJointId jointId);

    /// @brief Removes a spherical joint.
    bool removeSphericalJoint(physics::SphericalJointId jointId);

    /// @brief Removes a slider joint.
    bool removeSliderJoint(physics::SliderJointId jointId);

    /// @brief Removes a particle collision filter.
    bool removeParticleCollisionFilter(physics::ParticleCollisionFilterId filterId);

    /// @brief Removes a suturing sequence.
    bool removeSuturingSequence(physics::SuturingSequenceId sequenceId);

    /// @brief Assigns or updates an enabled ultrasound probe component for an entity.
    /// @note Passing a disabled component removes the probe and its published result.
    void setUltrasoundProbe(common::EntityId entityId, const UltrasoundProbeComponent &component);

    /// @brief Removes an ultrasound probe component and its result from an entity.
    bool removeUltrasoundProbe(common::EntityId entityId);

    /// @brief Assigns or updates an ultrasound renderer component for an entity.
    void setUltrasoundRenderer(common::EntityId entityId,
                               const UltrasoundRendererComponent &component);

    /// @brief Removes an ultrasound renderer component and clears any published probe result.
    bool removeUltrasoundRenderer(common::EntityId entityId);

    /// @brief Assigns or updates an enabled ultrasound scatterer source component for an entity.
    /// @note Passing a disabled component removes the source and its amplitude ranges.
    void setUltrasoundScattererSource(common::EntityId entityId,
                                      const UltrasoundScattererSourceComponent &component);

    /// @brief Removes an ultrasound scatterer source and its amplitude ranges from an entity.
    bool removeUltrasoundScattererSource(common::EntityId entityId);

    /// @brief Sets ultrasound scatterer amplitude ranges for an entity.
    /// @note The entity must have a soft body and provide exactly one range per authored particle.
    void setUltrasoundScattererAmplitudeRanges(common::EntityId entityId,
                                               const std::vector<UltrasoundAmplitudeRange> &ranges);

    /// @brief Clears ultrasound scatterer amplitude ranges for an entity.
    bool clearUltrasoundScattererAmplitudeRanges(common::EntityId entityId);

    /// @brief Adds a collider to an entity with a rigid body and returns its handle.
    /// @return An invalid handle if the entity is invalid, not alive, or has no rigid body.
    ColliderHandle addCollider(common::EntityId entityId, const ColliderComponent &component);

    /// @brief Updates the component of a registered collider handle.
    void updateCollider(ColliderHandle handle, const ColliderComponent &component);

    /// @brief Removes a registered collider handle.
    bool removeCollider(ColliderHandle handle);

    /// @brief Replaces all colliders on an entity with a rigid body.
    /// @return False if the entity is invalid, not alive, or has no rigid body.
    bool replaceColliders(common::EntityId entityId,
                          const std::vector<ColliderComponent> &components);

    /// @brief Removes the transform component from an entity.
    bool removeTransform(common::EntityId entityId);

    /// @brief Removes the mesh renderer component from an entity.
    bool removeMeshRenderer(common::EntityId entityId);

    /// @brief Removes the camera component from an entity.
    bool removeCamera(common::EntityId entityId);

    /// @brief Removes the directional-light component from an entity.
    bool removeDirectionalLight(common::EntityId entityId);

    /// @brief Removes the point-light component from an entity.
    bool removePointLight(common::EntityId entityId);

    /// @brief Removes the spot-light component from an entity.
    bool removeSpotLight(common::EntityId entityId);

    /// @brief Returns the transform component for an entity, or std::nullopt.
    std::optional<TransformComponent> tryGetTransform(common::EntityId entityId) const;

    /// @brief Returns the mesh renderer component for an entity, or std::nullopt.
    std::optional<MeshRendererComponent> tryGetMeshRenderer(common::EntityId entityId) const;

    /// @brief Returns the camera component for an entity, or std::nullopt.
    std::optional<CameraComponent> tryGetCamera(common::EntityId entityId) const;

    /// @brief Returns the directional-light component for an entity, or std::nullopt.
    std::optional<DirectionalLightComponent> tryGetDirectionalLight(
        common::EntityId entityId) const;

    /// @brief Returns the point-light component for an entity, or std::nullopt.
    std::optional<PointLightComponent> tryGetPointLight(common::EntityId entityId) const;

    /// @brief Returns the spot-light component for an entity, or std::nullopt.
    std::optional<SpotLightComponent> tryGetSpotLight(common::EntityId entityId) const;

    /// @brief Returns the rigid-body component for an entity, or std::nullopt.
    std::optional<RigidBodyComponent> tryGetRigidBody(common::EntityId entityId) const;

    /// @brief Returns the soft-body component for an entity, or std::nullopt.
    std::optional<SoftBodyComponent> tryGetSoftBody(common::EntityId entityId) const;

    /// @brief Returns the strand component for an entity, or std::nullopt.
    std::optional<StrandComponent> tryGetStrand(common::EntityId entityId) const;

    /// @brief Returns the procedural deformable-curve renderer for an entity, or std::nullopt.
    std::optional<ProceduralDeformableCurveRenderComponent> tryGetProceduralDeformableCurveRender(
        common::EntityId entityId) const;

    /// @brief Returns the fluid component for an entity, or std::nullopt.
    std::optional<FluidComponent> tryGetFluid(common::EntityId entityId) const;

    /// @brief Returns an authored particle sequence by ID, or nullptr.
    const physics::AuthoredParticleSequenceState *tryGetParticleSequence(
        physics::ParticleSequenceId sequenceId) const noexcept;

    /// @brief Returns an authored particle distance constraint by ID, or nullptr.
    const physics::AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        physics::ParticleConstraintId constraintId) const noexcept;

    /// @brief Returns a rigid-particle attachment constraint by ID, or nullptr.
    const physics::AuthoredRigidParticleAttachmentConstraintState *
    tryGetRigidParticleAttachmentConstraint(
        physics::RigidParticleAttachmentConstraintId constraintId) const noexcept;

    /// @brief Returns a strand-rigid attachment constraint by ID, or nullptr.
    const physics::AuthoredStrandRigidAttachmentConstraintState *
    tryGetStrandRigidAttachmentConstraint(
        physics::StrandRigidAttachmentConstraintId constraintId) const noexcept;

    /// @brief Returns a rigid distance constraint by ID, or nullptr.
    const physics::AuthoredRigidDistanceConstraintState *tryGetRigidDistanceConstraint(
        physics::RigidDistanceConstraintId constraintId) const noexcept;

    /// @brief Returns a routed cable constraint by ID, or nullptr.
    const physics::AuthoredRoutedCableConstraintState *tryGetRoutedCableConstraint(
        physics::RoutedCableConstraintId constraintId) const noexcept;

    /// @brief Returns a ball joint by ID, or nullptr.
    const physics::BallJointState *tryGetBallJoint(physics::BallJointId jointId) const noexcept;

    /// @brief Returns a hinge joint by ID, or nullptr.
    const physics::HingeJointState *tryGetHingeJoint(physics::HingeJointId jointId) const noexcept;

    /// @brief Returns a spherical joint by ID, or nullptr.
    const physics::SphericalJointState *tryGetSphericalJoint(
        physics::SphericalJointId jointId) const noexcept;

    /// @brief Returns a slider joint by ID, or nullptr.
    const physics::SliderJointState *tryGetSliderJoint(
        physics::SliderJointId jointId) const noexcept;

    /// @brief Returns an authored particle collision filter by ID, or nullptr.
    const physics::AuthoredParticleCollisionFilterState *tryGetParticleCollisionFilter(
        physics::ParticleCollisionFilterId filterId) const noexcept;

    /// @brief Returns an authored suturing sequence by ID, or nullptr.
    const physics::AuthoredSuturingSequenceState *tryGetSuturingSequence(
        physics::SuturingSequenceId sequenceId) const noexcept;

    /// @brief Returns the ultrasound probe component for an entity, or std::nullopt.
    std::optional<UltrasoundProbeComponent> tryGetUltrasoundProbe(common::EntityId entityId) const;

    /// @brief Returns the ultrasound renderer component for an entity, or std::nullopt.
    std::optional<UltrasoundRendererComponent> tryGetUltrasoundRenderer(
        common::EntityId entityId) const;

    /// @brief Returns the ultrasound scatterer source component for an entity, or std::nullopt.
    std::optional<UltrasoundScattererSourceComponent> tryGetUltrasoundScattererSource(
        common::EntityId entityId) const;

    /// @brief Returns authored rest positions for a soft body, or std::nullopt.
    std::optional<SoftBodyAuthoringParticles> tryGetSoftBodyAuthoringParticles(
        common::EntityId entityId) const;

    /// @brief Returns ultrasound scatterer amplitude ranges for an entity, or nullptr.
    const std::vector<UltrasoundAmplitudeRange> *tryGetUltrasoundScattererAmplitudeRanges(
        common::EntityId entityId) const noexcept;

    /// @brief Returns the most recently published ultrasound probe result, or nullptr.
    const UltrasoundProbeResult *tryGetUltrasoundProbeResult(
        common::EntityId entityId) const noexcept;

    /// @brief Returns the component for a registered collider handle, or std::nullopt.
    std::optional<ColliderComponent> tryGetCollider(ColliderHandle handle) const;

    /// @brief Returns collider handles belonging to an entity.
    /// @return An empty shared list when the entity has no collider links.
    const std::vector<ColliderHandle> &colliderHandles(common::EntityId entityId) const;

    /// @brief Gets a reference to the internal PhysicsWorld instance.
    /// @return Reference to PhysicsWorld.
    physics::PhysicsWorld &physicsWorld() noexcept;

    /// @brief Gets a const reference to the internal PhysicsWorld instance.
    /// @return Const reference to PhysicsWorld.
    const physics::PhysicsWorld &physicsWorld() const noexcept;

    /// @brief Sets the GPU scene view used by render-state accessors.
    void setGpuEntityScene(const graphics::GpuEntitySceneView &sceneView) noexcept;

    /// @brief Returns authored renderable instances.
    const std::vector<graphics::RenderableInstance> &renderables() const noexcept;

    /// @brief Returns authored camera data.
    const std::vector<graphics::CameraData> &cameras() const noexcept;

    /// @brief Returns authored light data.
    const std::vector<graphics::LightData> &lights() const noexcept;

    /// @brief Returns an entity's pose slot, or an invalid slot when none is assigned.
    std::uint32_t entityPoseSlot(common::EntityId entityId) const noexcept;

    /// @brief Returns entity-pose positions prepared for GPU upload.
    const std::vector<Diligent::float4> &entityPosePositions() const noexcept;

    /// @brief Returns entity-pose orientations prepared for GPU upload.
    const std::vector<Diligent::float4> &entityPoseOrientations() const noexcept;

    /// @brief Returns entity-pose scales prepared for GPU upload.
    const std::vector<Diligent::float4> &entityPoseScales() const noexcept;

    /// @brief Returns render-object positions prepared for GPU upload.
    const std::vector<Diligent::float4> &renderObjectPositions() const noexcept;

    /// @brief Returns render-object orientations prepared for GPU upload.
    const std::vector<Diligent::float4> &renderObjectOrientations() const noexcept;

    /// @brief Returns render-object scales prepared for GPU upload.
    const std::vector<Diligent::float4> &renderObjectScales() const noexcept;

    /// @brief Returns renderable metadata prepared for GPU upload.
    const std::vector<graphics::GpuRenderableMetadata> &renderableMetadata() const noexcept;

    /// @brief Returns renderable queue information prepared for GPU upload.
    const std::vector<graphics::GpuRenderableQueueInfo> &renderableQueueInfo() const noexcept;

    /// @brief Returns camera inputs prepared for GPU upload.
    const std::vector<graphics::GpuCameraInput> &cameraInputs() const noexcept;

    /// @brief Returns light inputs prepared for GPU upload.
    const std::vector<graphics::GpuLightInput> &lightInputs() const noexcept;

    /// @brief Returns per-environment local-light selections prepared for GPU upload.
    const std::vector<graphics::GpuLocalLightSelection> &localLightSelections() const noexcept;

    /// @brief Returns soft-body vertex bindings prepared for GPU upload.
    const std::vector<graphics::GpuSoftBodyVertexBinding> &softBodyVertexBindings() const noexcept;

    /// @brief Returns the opaque indirect-draw registry.
    const std::vector<graphics::IndirectCommandRegistryEntry> &opaqueDrawRegistry() const noexcept;

    /// @brief Returns the transparent draw registry.
    const std::vector<graphics::TransparentDrawEntry> &transparentDrawRegistry() const noexcept;

    /// @brief Returns the directional-shadow indirect-draw registry.
    const std::vector<graphics::IndirectCommandRegistryEntry> &shadowDrawRegistry() const noexcept;

    /// @brief Returns the local-light-shadow indirect-draw registry.
    const std::vector<graphics::IndirectCommandRegistryEntry> &localShadowDrawRegistry()
        const noexcept;

    /// @brief Returns mappings between physics bodies and renderables, rebuilding them if needed.
    const std::vector<EntityPoseMappingEntry> &physicsRenderableMappings();

    /// @brief Returns the entity-pose data revision.
    std::uint64_t entityPoseRevision() const noexcept;

    /// @brief Returns the renderable-metadata revision.
    std::uint64_t renderableMetadataRevision() const noexcept;

    /// @brief Returns the renderable-queue revision.
    std::uint64_t renderableQueueInfoRevision() const noexcept;

    /// @brief Returns the soft-body vertex-binding revision.
    std::uint64_t softBodyVertexBindingRevision() const noexcept;

    /// @brief Returns the camera-input revision.
    std::uint64_t cameraInputRevision() const noexcept;

    /// @brief Returns the light-input revision.
    std::uint64_t lightInputRevision() const noexcept;

    /// @brief Returns the local-light-selection revision.
    std::uint64_t localLightSelectionRevision() const noexcept;

    /// @brief Returns the GPU entity-scene view.
    const graphics::GpuEntitySceneView &gpuEntityScene() const noexcept;

    /// @brief Returns an immutable bundle of host scene data for rendering.
    graphics::HostSceneView hostSceneView() const noexcept;

    /// @brief Updates render state derived from authored world and resource state.
    void ensureRenderStateUpToDate(const graphics::RenderResourceManager &resources);

    /// @brief Returns ultrasound probe components keyed by entity ID.
    const std::unordered_map<common::EntityId, UltrasoundProbeComponent> &
    ultrasoundProbeComponents() const noexcept;

    /// @brief Returns ultrasound renderer components keyed by entity ID.
    const std::unordered_map<common::EntityId, UltrasoundRendererComponent> &
    ultrasoundRendererComponents() const noexcept;

    /// @brief Returns ultrasound scatterer source components keyed by entity ID.
    const std::unordered_map<common::EntityId, UltrasoundScattererSourceComponent> &
    ultrasoundScattererSourceComponents() const noexcept;

    /// @brief Returns the ultrasound scatterer-amplitude revision.
    std::uint64_t ultrasoundScattererAmplitudeRevision() const noexcept;

    /// @brief Publishes an ultrasound probe result under an entity ID without validating it.
    void setUltrasoundProbeResult(common::EntityId entityId, const UltrasoundProbeResult &result);

    /// @brief Clears a published ultrasound probe result for an entity ID, if present.
    void clearUltrasoundProbeResult(common::EntityId entityId);

    /// @brief Removes an authored particle sequence.
    bool removeParticleSequence(physics::ParticleSequenceId sequenceId);

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_WORLD_H
