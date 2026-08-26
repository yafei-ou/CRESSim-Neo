#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H

#include "common/flags.h"
#include "physics/export.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <memory>
#include <vector>

/// @file physics_world.h
/// @brief Host physics world container managing authored simulation bodies, colliders, joints,
/// constraints, and SoA scene snapshot staging.

namespace cressim::neo::physics
{

/// @brief Bitmask flags indicating internal data structures that need rebuilding after authoring
/// updates.
enum class PhysicsRebuildFlags : std::uint32_t
{
    None               = 0u,       ///< No rebuild required.
    SoftParticleLayout = 1u << 0u, ///< Particle memory layout changed.
    SoftConstraintData = 1u << 1u, ///< Edge, tetrahedral, or bending constraints changed.
    SuturingData       = 1u << 2u, ///< Needle suturing pairing metadata changed.
    ResolvedRigidParticleAttachments = 1u << 3u, ///< Rigid-particle attachments need re-resolution.
    ResolvedStrandRigidAttachments   = 1u << 4u, ///< Strand-rigid attachments need re-resolution.
    ResolvedRigidDistanceConstraints =
        1u << 5u,                    ///< Rigid-to-rigid distance constraints need re-resolution.
    ResolvedRoutedCables = 1u << 6u, ///< Routed cable paths need re-resolution.
};

CRESSIM_NEO_DEFINE_ENUM_FLAGS(PhysicsRebuildFlags)

/// @brief Host container managing authored physics scene state, rigid bodies, deformable meshes,
/// strands, fluids, joints, and constraints.
///
/// `PhysicsWorld` acts as the authoring database and host synchronization bridge for CRESSim-Neo
/// physics. It maintains authored component states (rigid bodies, colliders, soft bodies, strands,
/// fluids, joints, constraints), transforms them into compact Structure-of-Arrays (SoA)
/// representations for GPU uploads, tracks revision counters, and receives simulation writeback
/// updates from GPU solvers.
class CRESSIM_NEO_PHYSICS_API PhysicsWorld
{
public:
    /// @brief Default constructor initializing empty physics world.
    PhysicsWorld();
    /// @brief Destructor releasing internal implementation resources.
    ~PhysicsWorld();

    /// @brief Copy constructor performing deep clone of world state.
    /// @param other Source physics world to copy.
    PhysicsWorld(const PhysicsWorld &other);
    /// @brief Copy assignment operator.
    /// @param other Source physics world to copy.
    /// @return Reference to this world.
    PhysicsWorld &operator=(const PhysicsWorld &other);
    /// @brief Move constructor transferring ownership of world state.
    /// @param other Source physics world to move from.
    PhysicsWorld(PhysicsWorld &&other) noexcept;
    /// @brief Move assignment operator.
    /// @param other Source physics world to move from.
    /// @return Reference to this world.
    PhysicsWorld &operator=(PhysicsWorld &&other) noexcept;

    /// @brief Clears all bodies, colliders, joints, particles, and constraints from the world.
    void clear();

    /// @brief Inserts or updates an authored rigid body.
    /// @param state Rigid body configuration state to insert or modify.
    /// @return Reference to the stored RigidBodyState.
    RigidBodyState &upsertRigidBody(const RigidBodyState &state);
    /// @brief Removes a rigid body by its associated scene entity ID.
    /// @param entityId Entity ID of the rigid body to remove.
    /// @return True if a rigid body was found and removed, false otherwise.
    bool removeRigidBody(common::EntityId entityId);
    /// @brief Inserts or updates a single collider attachment.
    /// @param collider Collider state to insert or update.
    void upsertCollider(const ColliderState &collider);
    /// @brief Removes a collider by its ColliderId.
    /// @param colliderId Unique identifier of the collider to remove.
    /// @return True if found and removed, false otherwise.
    bool removeCollider(ColliderId colliderId);
    /// @brief Replaces all colliders belonging to a specific entity.
    /// @param entityId Owning entity ID.
    /// @param colliders New list of colliders to assign.
    void replaceColliders(common::EntityId entityId, const std::vector<ColliderState> &colliders);
    /// @brief Inserts or updates an authored 3D deformable soft body.
    /// @param state Soft body state.
    /// @return True if successfully inserted or updated.
    bool upsertSoftBody(const SoftBodyState &state);
    /// @brief Inserts or updates an authored 1D elastic strand.
    /// @param state Strand state.
    /// @return True if successfully inserted or updated.
    bool upsertStrand(const StrandState &state);
    /// @brief Inserts or updates an authored fluid body.
    /// @param state Fluid state.
    /// @return True if successfully inserted or updated.
    bool upsertFluid(const FluidState &state);
    /// @brief Inserts or updates an authored particle sequence.
    /// @param state Sequence state.
    /// @return Reference to stored sequence.
    AuthoredParticleSequenceState &upsertParticleSequence(
        const AuthoredParticleSequenceState &state);
    /// @brief Inserts or updates an authored particle-to-particle distance constraint.
    ///
    /// An invalid or unknown ID creates a new constraint. Negative rest length and compliance are
    /// clamped to zero. Particle references are retained as authored here and resolved when the
    /// derived state is rebuilt.
    /// @param state Constraint state.
    /// @return Reference to the normalized stored constraint.
    AuthoredParticleDistanceConstraintState &upsertParticleDistanceConstraint(
        const AuthoredParticleDistanceConstraintState &state);
    /// @brief Inserts or updates an authored rigid-to-particle attachment constraint.
    ///
    /// The referenced particle and rigid body must exist and belong to the same environment.
    /// Negative compliance is clamped to zero. An invalid or unknown constraint ID creates a new
    /// constraint; on success, @p outAuthored receives its normalized state when non-null.
    /// @param state Constraint state to insert or update.
    /// @param outAuthored Optional output pointer receiving the normalized stored constraint.
    /// @return False if either reference is invalid, out of range, or in a different environment.
    bool upsertRigidParticleAttachmentConstraint(
        const AuthoredRigidParticleAttachmentConstraintState &state,
        AuthoredRigidParticleAttachmentConstraintState *outAuthored = nullptr);
    /// @brief Inserts or updates an authored strand-to-rigid attachment constraint.
    ///
    /// The strand segment and rigid body must exist and belong to the same environment. `segmentT`
    /// is clamped to [0, 1], the local rotation is normalized, and negative compliance values are
    /// clamped to zero. An invalid or unknown constraint ID creates a new constraint.
    /// @param state Constraint state.
    /// @param outAuthored Optional output pointer receiving the normalized stored state.
    /// @return False if the strand segment or rigid body is invalid, or their environments differ.
    bool upsertStrandRigidAttachmentConstraint(
        const AuthoredStrandRigidAttachmentConstraintState &state,
        AuthoredStrandRigidAttachmentConstraintState *outAuthored = nullptr);
    /// @brief Inserts or updates an authored rigid-to-rigid distance constraint.
    ///
    /// The bodies must be distinct existing rigid bodies in the same environment. Negative rest
    /// distance and compliance are clamped to zero. An invalid or unknown constraint ID creates a
    /// new constraint.
    /// @param state Constraint state.
    /// @param outAuthored Optional output pointer receiving the normalized stored state.
    /// @return False if either body is invalid, both names refer to the same body, or environments
    /// differ.
    bool upsertRigidDistanceConstraint(const AuthoredRigidDistanceConstraintState &state,
                                       AuthoredRigidDistanceConstraintState *outAuthored = nullptr);
    /// @brief Inserts or updates an authored routed cable constraint.
    ///
    /// The route must contain at least two guide points on existing rigid bodies in one
    /// environment; adjacent guide points may not use the same body. Negative target length and
    /// compliance are clamped to zero. An invalid or unknown constraint ID creates a new
    /// constraint.
    /// @param state Constraint state.
    /// @param outAuthored Optional output pointer receiving the normalized stored state.
    /// @return False if the route is too short, has an invalid body, crosses environments, or
    /// repeats a body at adjacent points.
    bool upsertRoutedCableConstraint(const AuthoredRoutedCableConstraintState &state,
                                     AuthoredRoutedCableConstraintState *outAuthored = nullptr);
    /// @brief Inserts or updates an authored particle collision filter.
    ///
    /// An invalid or unknown filter ID creates a new filter. A zero collision layer is normalized
    /// to layer 1; the particle reference is retained as authored and resolved on rebuild.
    /// @param state Filter state.
    /// @return Reference to the normalized stored filter state.
    AuthoredParticleCollisionFilterState &upsertParticleCollisionFilter(
        const AuthoredParticleCollisionFilterState &state);
    /// @brief Inserts or updates an authored suturing sequence.
    ///
    /// An invalid or unknown ID creates a new sequence. Negative path-node spacing is clamped to
    /// zero, and the tip index is clamped to the entries (or set to zero for an empty sequence).
    /// @param state Sequence state.
    /// @return Reference to the normalized stored suturing sequence state.
    AuthoredSuturingSequenceState &upsertSuturingSequence(
        const AuthoredSuturingSequenceState &state);
    /// @brief Removes a soft body by its entity ID.
    /// @param entityId Entity ID of the soft body to remove.
    /// @return True if removed, false otherwise.
    bool removeSoftBody(common::EntityId entityId);
    /// @brief Removes a strand by its entity ID.
    /// @param entityId Entity ID of the strand to remove.
    /// @return True if removed, false otherwise.
    bool removeStrand(common::EntityId entityId);
    /// @brief Removes a fluid body by its entity ID.
    /// @param entityId Entity ID of the fluid to remove.
    /// @return True if removed, false otherwise.
    bool removeFluid(common::EntityId entityId);
    /// @brief Removes a particle sequence by its ID.
    /// @param sequenceId Sequence identifier to remove.
    /// @return True if removed, false otherwise.
    bool removeParticleSequence(ParticleSequenceId sequenceId);
    /// @brief Removes a particle distance constraint.
    /// @param constraintId Constraint ID to remove.
    /// @return True if removed, false otherwise.
    bool removeParticleDistanceConstraint(ParticleConstraintId constraintId);
    /// @brief Removes a rigid-particle attachment constraint.
    /// @param constraintId Constraint ID to remove.
    /// @return True if removed, false otherwise.
    bool removeRigidParticleAttachmentConstraint(RigidParticleAttachmentConstraintId constraintId);
    /// @brief Removes a strand-rigid attachment constraint.
    /// @param constraintId Constraint ID to remove.
    /// @return True if removed, false otherwise.
    bool removeStrandRigidAttachmentConstraint(StrandRigidAttachmentConstraintId constraintId);
    /// @brief Removes a rigid distance constraint.
    /// @param constraintId Constraint ID to remove.
    /// @return True if removed, false otherwise.
    bool removeRigidDistanceConstraint(RigidDistanceConstraintId constraintId);
    /// @brief Removes a routed cable constraint.
    /// @param constraintId Constraint ID to remove.
    /// @return True if removed, false otherwise.
    bool removeRoutedCableConstraint(RoutedCableConstraintId constraintId);
    /// @brief Removes a particle collision filter.
    /// @param filterId Filter ID to remove.
    /// @return True if removed, false otherwise.
    bool removeParticleCollisionFilter(ParticleCollisionFilterId filterId);
    /// @brief Removes a suturing sequence.
    /// @param sequenceId Sequence ID to remove.
    /// @return True if removed, false otherwise.
    bool removeSuturingSequence(SuturingSequenceId sequenceId);
    /// @brief Inserts or updates an articulated ball joint.
    ///
    /// The two body IDs must name distinct registered rigid bodies in the same environment. An
    /// invalid joint ID creates a new joint.
    /// @param state Ball joint state.
    /// @return False if either body is invalid, both body IDs are equal, or environments differ.
    bool upsertBallJoint(const BallJointState &state);
    /// @brief Inserts or updates an articulated spherical joint.
    ///
    /// The bodies must be distinct registered rigid bodies in the same environment. Rotations are
    /// normalized, negative compliance and swing limits are clamped to zero, reversed enabled
    /// twist limits are reordered, and disabled limits are reset. Only `None` and
    /// `TargetOrientation` drive modes are accepted.
    /// @param state Spherical joint state.
    /// @return False if the body references are invalid, environments differ, or the drive mode is
    /// unsupported.
    bool upsertSphericalJoint(const SphericalJointState &state);
    /// @brief Inserts or updates an articulated hinge joint.
    ///
    /// The bodies must be distinct registered rigid bodies in the same environment. Rotations are
    /// normalized, negative compliance, damping, and maximum angular velocity are clamped to zero,
    /// and disabled limits are reset.
    /// @param state Hinge joint state.
    /// @return False if either body is invalid, both body IDs are equal, or environments differ.
    bool upsertHingeJoint(const HingeJointState &state);
    /// @brief Inserts or updates an articulated slider joint.
    ///
    /// The bodies must be distinct registered rigid bodies in the same environment. Rotations are
    /// normalized, reversed enabled limits are reordered, disabled limits are reset, and negative
    /// compliance, damping, and maximum velocity are clamped to zero. When both local anchors are
    /// near zero, anchors are derived from body B's current world position.
    /// @param state Slider joint state.
    /// @return False if either body is invalid, both body IDs are equal, or environments differ.
    bool upsertSliderJoint(const SliderJointState &state);
    /// @brief Removes a ball joint.
    /// @param jointId Joint identifier to remove.
    /// @return True if removed.
    bool removeBallJoint(BallJointId jointId);
    /// @brief Removes a spherical joint.
    /// @param jointId Joint identifier to remove.
    /// @return True if removed.
    bool removeSphericalJoint(SphericalJointId jointId);
    /// @brief Removes a hinge joint.
    /// @param jointId Joint identifier to remove.
    /// @return True if removed.
    bool removeHingeJoint(HingeJointId jointId);
    /// @brief Removes a slider joint.
    /// @param jointId Joint identifier to remove.
    /// @return True if removed.
    bool removeSliderJoint(SliderJointId jointId);

    /// @brief Attempts to retrieve mutable pointer to a rigid body by entity ID.
    /// @param entityId Entity ID of the rigid body.
    /// @return Pointer to RigidBodyState or nullptr if not found.
    RigidBodyState *tryGetRigidBody(common::EntityId entityId);
    /// @brief Attempts to retrieve const pointer to a rigid body by entity ID.
    /// @param entityId Entity ID of the rigid body.
    /// @return Const pointer to RigidBodyState or nullptr if not found.
    const RigidBodyState *tryGetRigidBody(common::EntityId entityId) const;
    /// @brief Attempts to retrieve const pointer to a collider by ColliderId.
    /// @param colliderId Unique collider ID.
    /// @return Const pointer to ColliderState or nullptr if not found.
    const ColliderState *tryGetCollider(ColliderId colliderId) const;
    /// @brief Attempts to retrieve mutable pointer to a soft body by entity ID.
    /// @param entityId Entity ID of the soft body.
    /// @return Pointer to SoftBodyState or nullptr if not found.
    SoftBodyState *tryGetSoftBody(common::EntityId entityId);
    /// @brief Attempts to retrieve const pointer to a soft body by entity ID.
    /// @param entityId Entity ID of the soft body.
    /// @return Const pointer to SoftBodyState or nullptr if not found.
    const SoftBodyState *tryGetSoftBody(common::EntityId entityId) const;
    /// @brief Attempts to retrieve mutable pointer to a strand by entity ID.
    /// @param entityId Entity ID of the strand.
    /// @return Pointer to StrandState or nullptr if not found.
    StrandState *tryGetStrand(common::EntityId entityId);
    /// @brief Attempts to retrieve const pointer to a strand by entity ID.
    /// @param entityId Entity ID of the strand.
    /// @return Const pointer to StrandState or nullptr if not found.
    const StrandState *tryGetStrand(common::EntityId entityId) const;
    /// @brief Retrieves authoring rest positions for a soft body.
    /// @param entityId Entity ID.
    /// @param[out] outRestPositions Output vector receiving rest positions.
    /// @return True if found and rest positions populated.
    bool tryGetSoftBodyAuthoringRestPositions(
        common::EntityId entityId, std::vector<Diligent::float3> &outRestPositions) const;
    /// @brief Attempts to retrieve mutable pointer to a fluid body by entity ID.
    /// @param entityId Entity ID of the fluid.
    /// @return Pointer to FluidState or nullptr if not found.
    FluidState *tryGetFluid(common::EntityId entityId);
    /// @brief Attempts to retrieve const pointer to a fluid body by entity ID.
    /// @param entityId Entity ID of the fluid.
    /// @return Const pointer to FluidState or nullptr if not found.
    const FluidState *tryGetFluid(common::EntityId entityId) const;
    /// @brief Attempts to retrieve mutable pointer to a particle sequence.
    /// @param sequenceId Sequence identifier.
    /// @return Pointer to AuthoredParticleSequenceState or nullptr.
    AuthoredParticleSequenceState *tryGetParticleSequence(ParticleSequenceId sequenceId);
    /// @brief Attempts to retrieve const pointer to a particle sequence.
    /// @param sequenceId Sequence identifier.
    /// @return Const pointer to AuthoredParticleSequenceState or nullptr.
    const AuthoredParticleSequenceState *tryGetParticleSequence(
        ParticleSequenceId sequenceId) const;
    /// @brief Attempts to retrieve mutable pointer to a particle distance constraint.
    /// @param constraintId Constraint identifier.
    /// @return Pointer to AuthoredParticleDistanceConstraintState or nullptr.
    AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        ParticleConstraintId constraintId);
    /// @brief Attempts to retrieve const pointer to a particle distance constraint.
    /// @param constraintId Constraint identifier.
    /// @return Const pointer to AuthoredParticleDistanceConstraintState or nullptr.
    const AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        ParticleConstraintId constraintId) const;
    /// @brief Attempts to retrieve mutable pointer to a rigid-particle attachment constraint.
    /// @param constraintId Constraint identifier.
    /// @return Pointer to AuthoredRigidParticleAttachmentConstraintState or nullptr.
    AuthoredRigidParticleAttachmentConstraintState *tryGetRigidParticleAttachmentConstraint(
        RigidParticleAttachmentConstraintId constraintId);
    /// @brief Attempts to retrieve const pointer to a rigid-particle attachment constraint.
    /// @param constraintId Constraint identifier.
    /// @return Const pointer to AuthoredRigidParticleAttachmentConstraintState or nullptr.
    const AuthoredRigidParticleAttachmentConstraintState *tryGetRigidParticleAttachmentConstraint(
        RigidParticleAttachmentConstraintId constraintId) const;
    /// @brief Attempts to retrieve mutable pointer to a strand-rigid attachment constraint.
    /// @param constraintId Constraint identifier.
    /// @return Pointer to AuthoredStrandRigidAttachmentConstraintState or nullptr.
    AuthoredStrandRigidAttachmentConstraintState *tryGetStrandRigidAttachmentConstraint(
        StrandRigidAttachmentConstraintId constraintId);
    /// @brief Attempts to retrieve const pointer to a strand-rigid attachment constraint.
    /// @param constraintId Constraint identifier.
    /// @return Const pointer to AuthoredStrandRigidAttachmentConstraintState or nullptr.
    const AuthoredStrandRigidAttachmentConstraintState *tryGetStrandRigidAttachmentConstraint(
        StrandRigidAttachmentConstraintId constraintId) const;
    /// @brief Attempts to retrieve mutable pointer to a rigid distance constraint.
    /// @param constraintId Constraint identifier.
    /// @return Pointer to AuthoredRigidDistanceConstraintState or nullptr.
    AuthoredRigidDistanceConstraintState *tryGetRigidDistanceConstraint(
        RigidDistanceConstraintId constraintId);
    /// @brief Attempts to retrieve const pointer to a rigid distance constraint.
    /// @param constraintId Constraint identifier.
    /// @return Const pointer to AuthoredRigidDistanceConstraintState or nullptr.
    const AuthoredRigidDistanceConstraintState *tryGetRigidDistanceConstraint(
        RigidDistanceConstraintId constraintId) const;
    /// @brief Attempts to retrieve mutable pointer to a routed cable constraint.
    /// @param constraintId Constraint identifier.
    /// @return Pointer to AuthoredRoutedCableConstraintState or nullptr.
    AuthoredRoutedCableConstraintState *tryGetRoutedCableConstraint(
        RoutedCableConstraintId constraintId);
    /// @brief Attempts to retrieve const pointer to a routed cable constraint.
    /// @param constraintId Constraint identifier.
    /// @return Const pointer to AuthoredRoutedCableConstraintState or nullptr.
    const AuthoredRoutedCableConstraintState *tryGetRoutedCableConstraint(
        RoutedCableConstraintId constraintId) const;
    /// @brief Attempts to retrieve mutable pointer to a particle collision filter.
    /// @param filterId Filter identifier.
    /// @return Pointer to AuthoredParticleCollisionFilterState or nullptr.
    AuthoredParticleCollisionFilterState *tryGetParticleCollisionFilter(
        ParticleCollisionFilterId filterId);
    /// @brief Attempts to retrieve const pointer to a particle collision filter.
    /// @param filterId Filter identifier.
    /// @return Const pointer to AuthoredParticleCollisionFilterState or nullptr.
    const AuthoredParticleCollisionFilterState *tryGetParticleCollisionFilter(
        ParticleCollisionFilterId filterId) const;
    /// @brief Attempts to retrieve mutable pointer to a suturing sequence.
    /// @param sequenceId Sequence identifier.
    /// @return Pointer to AuthoredSuturingSequenceState or nullptr.
    AuthoredSuturingSequenceState *tryGetSuturingSequence(SuturingSequenceId sequenceId);
    /// @brief Attempts to retrieve const pointer to a suturing sequence.
    /// @param sequenceId Sequence identifier.
    /// @return Const pointer to AuthoredSuturingSequenceState or nullptr.
    const AuthoredSuturingSequenceState *tryGetSuturingSequence(
        SuturingSequenceId sequenceId) const;
    /// @brief Attempts to retrieve const pointer to a ball joint.
    /// @param jointId Joint identifier.
    /// @return Const pointer to BallJointState or nullptr.
    const BallJointState *tryGetBallJoint(BallJointId jointId) const noexcept;
    /// @brief Attempts to retrieve const pointer to a spherical joint.
    /// @param jointId Joint identifier.
    /// @return Const pointer to SphericalJointState or nullptr.
    const SphericalJointState *tryGetSphericalJoint(SphericalJointId jointId) const noexcept;
    /// @brief Attempts to retrieve const pointer to a hinge joint.
    /// @param jointId Joint identifier.
    /// @return Const pointer to HingeJointState or nullptr.
    const HingeJointState *tryGetHingeJoint(HingeJointId jointId) const noexcept;
    /// @brief Attempts to retrieve const pointer to a slider joint.
    /// @param jointId Joint identifier.
    /// @return Const pointer to SliderJointState or nullptr.
    const SliderJointState *tryGetSliderJoint(SliderJointId jointId) const noexcept;

    /// @brief Gets snapshot vector of all authored rigid bodies.
    /// @return Const reference to rigid body state vector.
    const std::vector<RigidBodyState> &rigidBodySnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored colliders.
    /// @return Const reference to collider state vector.
    const std::vector<ColliderState> &colliderSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored soft bodies.
    /// @return Const reference to soft body state vector.
    const std::vector<SoftBodyState> &softBodySnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored strands.
    /// @return Const reference to strand state vector.
    const std::vector<StrandState> &strandSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored fluids.
    /// @return Const reference to fluid state vector.
    const std::vector<FluidState> &fluidSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored particle sequences.
    /// @return Const reference to particle sequence state vector.
    const std::vector<AuthoredParticleSequenceState> &particleSequenceSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored particle distance constraints.
    /// @return Const reference to particle distance constraint vector.
    const std::vector<AuthoredParticleDistanceConstraintState> &particleDistanceConstraintSnapshot()
        const noexcept;
    /// @brief Gets snapshot vector of all authored rigid-particle attachment constraints.
    /// @return Const reference to rigid-particle attachment vector.
    const std::vector<AuthoredRigidParticleAttachmentConstraintState> &
    rigidParticleAttachmentConstraintSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored strand-rigid attachment constraints.
    /// @return Const reference to strand-rigid attachment vector.
    const std::vector<AuthoredStrandRigidAttachmentConstraintState> &
    strandRigidAttachmentConstraintSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored rigid-to-rigid distance constraints.
    /// @return Const reference to rigid distance constraint vector.
    const std::vector<AuthoredRigidDistanceConstraintState> &rigidDistanceConstraintSnapshot()
        const noexcept;
    /// @brief Gets snapshot vector of all authored routed cable constraints.
    /// @return Const reference to routed cable constraint vector.
    const std::vector<AuthoredRoutedCableConstraintState> &routedCableConstraintSnapshot()
        const noexcept;
    /// @brief Gets snapshot vector of all authored particle collision filters.
    /// @return Const reference to particle collision filter vector.
    const std::vector<AuthoredParticleCollisionFilterState> &particleCollisionFilterSnapshot()
        const noexcept;
    /// @brief Gets snapshot vector of all authored suturing sequences.
    /// @return Const reference to suturing sequence vector.
    const std::vector<AuthoredSuturingSequenceState> &suturingSequenceSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored ball joints.
    /// @return Const reference to ball joint state vector.
    const std::vector<BallJointState> &ballJointSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored spherical joints.
    /// @return Const reference to spherical joint state vector.
    const std::vector<SphericalJointState> &sphericalJointSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored hinge joints.
    /// @return Const reference to hinge joint state vector.
    const std::vector<HingeJointState> &hingeJointSnapshot() const noexcept;
    /// @brief Gets snapshot vector of all authored slider joints.
    /// @return Const reference to slider joint state vector.
    const std::vector<SliderJointState> &sliderJointSnapshot() const noexcept;
    /// @brief Gets the rigid body Structure-of-Arrays (SoA) host container.
    /// @return Const reference to RigidBodySoAHost.
    const RigidBodySoAHost &rigidBodySoA() const noexcept;
    /// @brief Gets the collider Structure-of-Arrays (SoA) host container.
    /// @return Const reference to ColliderSoAHost.
    const ColliderSoAHost &colliderSoA() const noexcept;
    /// @brief Gets the rigid-to-collider index mapping table.
    /// @return Const reference to BodyColliderMappingHost.
    const BodyColliderMappingHost &bodyColliderMapping() const noexcept;
    /// @brief Gets the combined rigid joint SoA scene container.
    /// @return Const reference to RigidJointSceneHost.
    const RigidJointSceneHost &rigidJointScene() const noexcept;
    /// @brief Gets the collision suppression pairs table for connected joints.
    /// @return Const reference to JointCollisionSuppressionHost.
    const JointCollisionSuppressionHost &jointCollisionSuppression() const noexcept;
    /// @brief Gets the unified particle Structure-of-Arrays (SoA) host container.
    /// @return Const reference to ParticleSoAHost.
    const ParticleSoAHost &particles() const noexcept;
    /// @brief Gets the global particle contact material property table.
    /// @return Const reference to float4 packed material vector.
    const std::vector<Diligent::float4> &particleContactMaterials() const noexcept;
    /// @brief Gets the global GPU fluid material properties table.
    /// @return Const reference to FluidMaterialGpu vector.
    const std::vector<FluidMaterialGpu> &fluidMaterials() const noexcept;
    /// @brief Gets resolved deformable distance constraints.
    /// @return Const reference to DeformableDistanceConstraint vector.
    const std::vector<DeformableDistanceConstraint> &distanceConstraints() const noexcept;
    /// @brief Gets resolved deformable bending constraints.
    /// @return Const reference to DeformableBendConstraint vector.
    const std::vector<DeformableBendConstraint> &bendConstraints() const noexcept;
    /// @brief Gets resolved deformable volume constraints.
    /// @return Const reference to DeformableVolumeConstraint vector.
    const std::vector<DeformableVolumeConstraint> &volumeConstraints() const noexcept;
    /// @brief Gets alias view of deformable edge constraints.
    /// @return Const reference to SoftEdge vector.
    const std::vector<SoftEdge> &softEdges() const noexcept;
    /// @brief Gets alias view of deformable bending constraints.
    /// @return Const reference to SoftBend vector.
    const std::vector<SoftBend> &softBends() const noexcept;
    /// @brief Gets alias view of tetrahedral volume constraints.
    /// @return Const reference to SoftTet vector.
    const std::vector<SoftTet> &softTets() const noexcept;
    /// @brief Gets resolved strand segment constraints.
    /// @return Const reference to StrandSegmentConstraint vector.
    const std::vector<StrandSegmentConstraint> &strandSegments() const noexcept;
    /// @brief Gets resolved strand joint constraints.
    /// @return Const reference to StrandJointConstraint vector.
    const std::vector<StrandJointConstraint> &strandJoints() const noexcept;
    /// @brief Gets resolved strand distance constraints.
    /// @return Const reference to StrandDistanceConstraint vector.
    const std::vector<StrandDistanceConstraint> &strandDistanceConstraints() const noexcept;
    /// @brief Gets strand segment runtime orientation states.
    /// @return Const reference to StrandSegmentState vector.
    const std::vector<StrandSegmentState> &strandSegmentStates() const noexcept;
    /// @brief Gets resolved rigid-to-particle attachment constraints.
    /// @return Const reference to RigidParticleAttachmentConstraint vector.
    const std::vector<RigidParticleAttachmentConstraint> &rigidParticleAttachments() const noexcept;
    /// @brief Gets resolved strand-to-rigid attachment constraints.
    /// @return Const reference to StrandRigidAttachmentConstraint vector.
    const std::vector<StrandRigidAttachmentConstraint> &strandRigidAttachments() const noexcept;
    /// @brief Gets resolved rigid distance constraints.
    /// @return Const reference to RigidDistanceConstraint vector.
    const std::vector<RigidDistanceConstraint> &rigidDistanceConstraints() const noexcept;
    /// @brief Gets resolved routed cable constraints.
    /// @return Const reference to RoutedCableConstraint vector.
    const std::vector<RoutedCableConstraint> &routedCableConstraints() const noexcept;
    /// @brief Gets resolved routed cable route points.
    /// @return Const reference to RoutedCableRoutePoint vector.
    const std::vector<RoutedCableRoutePoint> &routedCableRoutePoints() const noexcept;
    /// @brief Gets surgical strand-soft tissue puncture pairs.
    /// @return Const reference to StrandSoftSuturingPair vector.
    const std::vector<StrandSoftSuturingPair> &suturingPairs() const noexcept;
    /// @brief Gets global particle indices participating in suturing tracking.
    /// @return Const reference to suturing particle indices vector.
    const std::vector<std::uint32_t> &suturingParticleIndices() const noexcept;
    /// @brief Gets host soft-body skinning and rendering mesh bindings.
    /// @return Const reference to SoftRenderDataHost.
    const SoftRenderDataHost &softRenderData() const noexcept;
    /// @brief Sets host soft-body skinning and rendering mesh bindings.
    /// @param data Skinning and surface vertex binding data.
    void setSoftRenderData(const SoftRenderDataHost &data);
    /// @brief Gets host curve rendering descriptors.
    /// @return Const reference to CurveRenderDataHost.
    const CurveRenderDataHost &curveRenderData() const noexcept;
    /// @brief Sets host curve rendering descriptors.
    /// @param data Curve rendering descriptor data.
    void setCurveRenderData(const CurveRenderDataHost &data);
    /// @brief Ensures all derived GPU-ready SoA tables and constraints are up to date.
    ///
    /// Although const, this call may rebuild internal cached mappings, resolved constraints, and
    /// joint/collider derived tables.
    void ensureDerivedStateUpToDate() const noexcept;
    /// @brief Ensures soft-body derived topology and particle layout tables are up to date.
    ///
    /// This may rebuild particle, constraint, and suturing caches.
    void ensureSoftBodyDerivedStateUpToDate() noexcept;
    /// @brief Gets indices of rigid bodies modified since last GPU upload.
    /// @return Vector of dirty rigid body indices.
    const std::vector<std::uint32_t> &rigidBodyDirtyIndices() const noexcept;
    /// @brief Gets indices of colliders modified since last GPU upload.
    /// @return Vector of dirty collider indices.
    const std::vector<std::uint32_t> &colliderDirtyIndices() const noexcept;
    /// @brief Gets total number of active rigid bodies.
    /// @return Number of rigid bodies.
    std::uint32_t rigidBodyCount() const noexcept;
    /// @brief Gets total number of active colliders.
    /// @return Number of colliders.
    std::uint32_t colliderCount() const noexcept;
    /// @brief Gets total number of active soft bodies.
    /// @return Number of soft bodies.
    std::uint32_t softBodyCount() const noexcept;
    /// @brief Gets total number of active strands.
    /// @return Number of strands.
    std::uint32_t strandCount() const noexcept;
    /// @brief Gets total number of active fluids.
    /// @return Number of fluid bodies.
    std::uint32_t fluidCount() const noexcept;
    /// @brief Checks if rigid body count changed requiring GPU reallocation.
    /// @return True if count is dirty.
    bool rigidBodyCountDirty() const noexcept;
    /// @brief Checks if collider count changed requiring GPU reallocation.
    /// @return True if count is dirty.
    bool colliderCountDirty() const noexcept;
    /// @brief Checks if a full upload of all rigid bodies is required.
    /// @return True if full upload required.
    bool fullRigidBodyUploadRequired() const noexcept;
    /// @brief Checks if a full upload of all colliders is required.
    /// @return True if full upload required.
    bool fullColliderUploadRequired() const noexcept;
    /// @brief Clears dirty flags and indices for rigid body GPU upload tracking.
    void clearRigidBodyUploadState() noexcept;
    /// @brief Clears dirty flags and indices for collider GPU upload tracking.
    void clearColliderUploadState() noexcept;
    /// @brief Checks if static collider broadphase acceleration structures need rebuilding.
    /// @return True if static broadphase is dirty.
    bool staticBroadPhaseDirty() const noexcept;
    /// @brief Clears static broadphase dirty flag.
    void clearStaticBroadPhaseDirty() noexcept;
    /// @brief Gets count of dynamic or kinematic (moving) colliders.
    /// @return Number of active moving colliders.
    std::uint32_t activeMovingColliderCount() const noexcept;
    /// @brief Gets count of static colliders.
    /// @return Number of static colliders.
    std::uint32_t staticColliderCount() const noexcept;
    /// @brief Gets spatial hashing cell dimension for particle neighbor search.
    /// @return Grid cell dimension.
    float particleGridCellSize() const noexcept;
    /// @brief Gets number of bounding box chunks for soft body broadphase.
    /// @return Chunk count.
    std::uint32_t softBodyBoundsChunkCount() const noexcept;
    /// @brief Gets maximum suturing paths allocated per needle-soft pair.
    /// @return Path capacity.
    std::uint32_t maxSuturingPathsPerPair() const noexcept;
    /// @brief Gets maximum nodes allocated per suturing path.
    /// @return Node capacity.
    std::uint32_t maxSuturingNodesPerPath() const noexcept;
    /// @brief Gets number of particles participating in suturing tracking.
    /// @return Particle count.
    std::uint32_t suturingParticleCount() const noexcept;
    /// @brief Gets total reserved suturing path header count in GPU buffer.
    /// @return Header buffer size.
    std::uint32_t reservedSuturingPathHeaderCount() const noexcept;
    /// @brief Gets total reserved suturing path node count in GPU buffer.
    /// @return Node buffer size.
    std::uint32_t reservedSuturingPathNodeCount() const noexcept;

    /// @brief Advances rigid body motion on CPU for simple kinematic/dynamic steps.
    /// @param dt Time step size in seconds.
    void integrateRigidBodiesCpu(float dt) noexcept;
    /// @brief Synchronizes a rigid body's pose and velocities back from GPU simulation.
    /// @param index Rigid body index in SoA.
    /// @param positionInvMass New world position and inverse mass.
    /// @param orientation New orientation quaternion.
    /// @param linearVelocity New linear velocity vector.
    /// @param angularVelocity New angular velocity vector.
    /// @return True if synchronization succeeded.
    bool syncRigidBodyStateFromSimulation(std::uint32_t index,
                                          const Diligent::float4 &positionInvMass,
                                          const Diligent::float4 &orientation,
                                          const Diligent::float4 &linearVelocity,
                                          const Diligent::float4 &angularVelocity) noexcept;
    /// @brief Finalizes rigid body state writeback into snapshot structures.
    void finalizeRigidBodyWriteback() noexcept;
    /// @brief Synchronizes a particle's position and velocity back from GPU simulation.
    /// @param index Particle index in SoA.
    /// @param positionInvMass New world position and inverse mass.
    /// @param previousPosition Previous sub-step position.
    /// @param velocity New velocity vector.
    /// @return True if synchronization succeeded.
    bool syncParticleStateFromSimulation(std::uint32_t index,
                                         const Diligent::float4 &positionInvMass,
                                         const Diligent::float4 &previousPosition,
                                         const Diligent::float4 &velocity) noexcept;
    /// @brief Finalizes particle state writeback into snapshot structures.
    void finalizeParticleWriteback() noexcept;

    /// @brief Monotonically increasing revision counter incremented on any authored scene change.
    /// @return Authoring revision number.
    std::uint64_t authoredRevision() const noexcept;
    /// @brief Monotonically increasing revision counter incremented on simulation step writeback.
    /// @return Simulation revision number.
    std::uint64_t simulationRevision() const noexcept;
    /// @brief Revision counter tracking rigid body count and topology changes.
    /// @return Revision number.
    std::uint64_t rigidBodyTopologyRevision() const noexcept;
    /// @brief Revision counter tracking rigid joint count and topology changes.
    /// @return Revision number.
    std::uint64_t rigidJointTopologyRevision() const noexcept;
    /// @brief Revision counter tracking joint parameter changes.
    /// @return Revision number.
    std::uint64_t rigidJointSceneRevision() const noexcept;
    /// @brief Revision counter tracking joint drive mode changes.
    /// @return Revision number.
    std::uint64_t rigidJointModeRevision() const noexcept;
    /// @brief Revision counter tracking soft body entity topology changes.
    /// @return Revision number.
    std::uint64_t softBodyTopologyRevision() const noexcept;
    /// @brief Revision counter tracking particle count and memory layout changes.
    /// @return Revision number.
    std::uint64_t softParticleRevision() const noexcept;
    /// @brief Revision counter tracking internal soft body constraint topologies.
    /// @return Revision number.
    std::uint64_t softTopologyRevision() const noexcept;
    /// @brief Revision counter tracking constraint adjacency and neighbor graph updates.
    /// @return Revision number.
    std::uint64_t softConstraintAdjacencyRevision() const noexcept;
    /// @brief Revision counter tracking rigid-particle attachment authoring changes.
    /// @return Revision number.
    std::uint64_t rigidParticleAttachmentDefinitionRevision() const noexcept;
    /// @brief Revision counter tracking resolved GPU rigid-particle attachments.
    /// @return Revision number.
    std::uint64_t rigidParticleAttachmentResolvedRevision() const noexcept;
    /// @brief Revision counter tracking strand-rigid attachment authoring changes.
    /// @return Revision number.
    std::uint64_t strandRigidAttachmentDefinitionRevision() const noexcept;
    /// @brief Revision counter tracking resolved GPU strand-rigid attachments.
    /// @return Revision number.
    std::uint64_t strandRigidAttachmentResolvedRevision() const noexcept;
    /// @brief Revision counter tracking rigid distance constraint authoring changes.
    /// @return Revision number.
    std::uint64_t rigidDistanceConstraintDefinitionRevision() const noexcept;
    /// @brief Revision counter tracking resolved GPU rigid distance constraints.
    /// @return Revision number.
    std::uint64_t rigidDistanceConstraintResolvedRevision() const noexcept;
    /// @brief Revision counter tracking routed cable constraint authoring changes.
    /// @return Revision number.
    std::uint64_t routedCableDefinitionRevision() const noexcept;
    /// @brief Revision counter tracking resolved GPU routed cable constraints.
    /// @return Revision number.
    std::uint64_t routedCableResolvedRevision() const noexcept;
    /// @brief Revision counter tracking curve rendering descriptors.
    /// @return Revision number.
    std::uint64_t curveRenderRevision() const noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
