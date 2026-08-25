#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H

#include "common/scene_primitives.h"

#include <cstdint>

/// @file physics_gpu_scene_view.h
/// @brief Non-owning GPU buffer views and descriptor structures published by the physics solver for rendering and compute pipelines.

namespace cressim::neo::physics
{

/// @brief GPU buffer bindings and counts representing rigid bodies, colliders, and attachments.
struct PhysicsGpuRigidSceneView
{
    common::PoseBufferView poses{};                                 ///< Instanced pose buffer view for render transforms.
    Diligent::IBuffer *statePositionsBuffer                = nullptr; ///< GPU buffer of rigid-body center-of-mass positions and inverse masses.
    Diligent::IBuffer *stateOrientationsBuffer             = nullptr; ///< GPU buffer of rigid-body orientation quaternions.
    Diligent::IBuffer *stateLinearVelocitiesBuffer         = nullptr; ///< GPU buffer of rigid-body linear velocity vectors.
    Diligent::IBuffer *stateAngularVelocitiesBuffer        = nullptr; ///< GPU buffer of rigid-body angular velocity vectors.
    Diligent::IBuffer *inverseInertiaLocalBuffer           = nullptr; ///< GPU buffer of body-local diagonal inverse inertia tensors.
    Diligent::IBuffer *bodyTypesBuffer                     = nullptr; ///< GPU buffer of RigidBodyType values.
    Diligent::IBuffer *proxyParticleContactMaterialsBuffer = nullptr; ///< GPU buffer of proxy contact material parameters.
    Diligent::IBuffer *kinematicTargetPositionsBuffer      = nullptr; ///< GPU buffer of kinematic target positions.
    Diligent::IBuffer *kinematicTargetOrientationsBuffer   = nullptr; ///< GPU buffer of kinematic target orientations.
    Diligent::IBuffer *kinematicTargetFlagsBuffer          = nullptr; ///< GPU buffer of kinematic drive target active flags.
    Diligent::IBuffer *colliderOwnerBodyIndicesBuffer      = nullptr; ///< GPU buffer mapping colliders to owning rigid-body indices.
    Diligent::IBuffer *colliderBroadPhaseBuffer            = nullptr; ///< GPU buffer of collider axis-aligned bounding boxes (AABBs).
    Diligent::IBuffer *colliderGeometryBuffer              = nullptr; ///< GPU buffer of collider geometric dimensions (radius, half-extents).
    Diligent::IBuffer *colliderMaterialsBuffer             = nullptr; ///< GPU buffer of collider friction and restitution parameters.
    Diligent::IBuffer *colliderShapeTypesBuffer            = nullptr; ///< GPU buffer of ColliderShapeType values.
    Diligent::IBuffer *colliderEnabledFlagsBuffer          = nullptr; ///< GPU buffer of collider enabled bitflags.
    Diligent::IBuffer *bodyColliderOffsetsBuffer           = nullptr; ///< GPU buffer of start offsets into the collider index list per body.
    Diligent::IBuffer *bodyColliderCountsBuffer            = nullptr; ///< GPU buffer of collider counts per rigid body.
    Diligent::IBuffer *bodyColliderRangesBuffer            = nullptr; ///< GPU buffer of collider contiguous ranges.
    Diligent::IBuffer *bodyColliderIndicesBuffer           = nullptr; ///< GPU buffer of flattened collider indices.
    // Constraint item buffers preserve prepare()/uploadWorld()-resolved slot ownership but allow
    // runtime mutation of supported per-slot fields through custom compute. For rigid-particle
    // attachments this includes the live descriptor payload read directly by the solver
    // (`particleIndex`, `rigidBodyIndex`, `compliance`, `enabled`) as long as the caller does not
    // assume prepared host-side layout mappings will reflect those GPU-only edits.
    Diligent::IBuffer *rigidParticleAttachmentsBuffer      = nullptr; ///< GPU buffer of RigidParticleAttachmentConstraint descriptors.
    Diligent::IBuffer *strandRigidAttachmentsBuffer        = nullptr; ///< GPU buffer of StrandRigidAttachmentConstraint descriptors.
    Diligent::IBuffer *rigidDistanceConstraintsBuffer      = nullptr; ///< GPU buffer of RigidDistanceConstraint descriptors.
    Diligent::IBuffer *routedCableDescriptorsBuffer        = nullptr; ///< GPU buffer of RoutedCableConstraint descriptors.
    Diligent::IBuffer *routedCableRoutePointsBuffer        = nullptr; ///< GPU buffer of RoutedCableRoutePoint descriptors.
    Diligent::IBuffer *routedCableDebugSegmentsBuffer      = nullptr; ///< GPU buffer for routed cable debug line visualizer segments.
    std::uint32_t bodyCount                                = 0;       ///< Total active rigid bodies in the scene.
    std::uint32_t rigidParticleAttachmentCount             = 0;       ///< Number of rigid-particle attachment constraints.
    std::uint32_t strandRigidAttachmentCount               = 0;       ///< Number of strand-rigid attachment constraints.
    std::uint32_t rigidDistanceConstraintCount             = 0;       ///< Number of rigid-to-rigid distance constraints.
    std::uint32_t routedCableCount                         = 0;       ///< Number of routed cable constraints.
    std::uint32_t routedCableRoutePointCount               = 0;       ///< Total guide points across all routed cables.
    std::uint32_t routedCableDebugSegmentCount             = 0;       ///< Number of debug line segments for cable rendering.
    std::uint32_t colliderCount                            = 0;       ///< Total colliders registered in the scene.
    std::uint64_t bindingGeneration                        = 0;       ///< Revision counter for rigid scene buffer allocations.
    std::uint64_t constraintBindingGeneration              = 0;       ///< Revision counter for constraint buffer allocations.
};

/// @brief GPU buffer bindings and counts for articulated rigid joints.
struct PhysicsGpuJointSceneView
{
    Diligent::IBuffer *ballJointsBuffer               = nullptr; ///< GPU buffer of BallJointState descriptors.
    Diligent::IBuffer *sphericalJointsBuffer          = nullptr; ///< GPU buffer of SphericalJointState descriptors.
    Diligent::IBuffer *hingeJointsBuffer              = nullptr; ///< GPU buffer of HingeJointState descriptors.
    Diligent::IBuffer *hingeJointRuntimeStatesBuffer  = nullptr; ///< GPU buffer of dynamic runtime hinge joint state.
    Diligent::IBuffer *sliderJointsBuffer             = nullptr; ///< GPU buffer of SliderJointState descriptors.
    Diligent::IBuffer *sliderJointRuntimeStatesBuffer = nullptr; ///< GPU buffer of dynamic runtime slider joint state.
    std::uint32_t ballJointCount                      = 0u;      ///< Number of active ball joints.
    std::uint32_t hingeJointCount                     = 0u;      ///< Number of active hinge joints.
    std::uint32_t sphericalJointCount                 = 0u;      ///< Number of active spherical joints.
    std::uint32_t sliderJointCount                    = 0u;      ///< Number of active slider joints.
    std::uint64_t bindingGeneration                   = 0u;      ///< Revision counter for joint buffer allocations.
};

/// @brief GPU buffer bindings for unified particle systems (soft bodies, strands, fluids, and proxy particles).
struct PhysicsGpuParticleBufferView
{
    Diligent::IBuffer *positionsInvMassBuffer              = nullptr; ///< GPU buffer of float4 particle positions (xyz) and inverse mass (w).
    Diligent::IBuffer *previousPositionsBuffer             = nullptr; ///< GPU buffer of particle positions from previous substep.
    Diligent::IBuffer *velocitiesBuffer                    = nullptr; ///< GPU buffer of float4 particle velocity vectors.
    Diligent::IBuffer *radiiBuffer                         = nullptr; ///< GPU buffer of particle collision radii.
    Diligent::IBuffer *environmentIndicesBuffer            = nullptr; ///< GPU buffer of environment assignments.
    Diligent::IBuffer *particleKindsBuffer                 = nullptr; ///< GPU buffer of ParticleKind values.
    Diligent::IBuffer *ownerTypesBuffer                    = nullptr; ///< GPU buffer of ParticleOwnerType values.
    Diligent::IBuffer *ownerIndicesBuffer                  = nullptr; ///< GPU buffer of owning body/strand/fluid index.
    Diligent::IBuffer *strandIdsBuffer                     = nullptr; ///< GPU buffer of strand IDs for strand particles.
    Diligent::IBuffer *strandRolesBuffer                   = nullptr; ///< GPU buffer of ParticleStrandRole suturing values.
    Diligent::IBuffer *suturingNeighborLinksBuffer         = nullptr; ///< GPU buffer of suturing neighbor topological links.
    Diligent::IBuffer *owningSoftBodyIndicesBuffer         = nullptr; ///< GPU buffer mapping particles to owning soft-body index.
    Diligent::IBuffer *particleMaterialIndicesBuffer       = nullptr; ///< GPU buffer of contact material indices.
    Diligent::IBuffer *fluidMaterialIndicesBuffer          = nullptr; ///< GPU buffer of fluid material indices.
    Diligent::IBuffer *fluidVisualsBuffer                  = nullptr; ///< GPU buffer of fluid particle render colors.
    Diligent::IBuffer *particleContactMaterialsBuffer      = nullptr; ///< GPU buffer of contact material parameters.
    Diligent::IBuffer *fluidMaterialsBuffer                = nullptr; ///< GPU buffer of FluidMaterialGpu parameters.
    Diligent::IBuffer *phasesBuffer                        = nullptr; ///< GPU buffer of packed particle collision phase bitfields.
    Diligent::IBuffer *collisionLayersBuffer               = nullptr; ///< GPU buffer of collision layer bitmasks.
    Diligent::IBuffer *collisionMasksBuffer                = nullptr; ///< GPU buffer of collision filter bitmasks.
    Diligent::IBuffer *adjacencyOffsetsBuffer              = nullptr; ///< GPU buffer of neighbor adjacency list start offsets.
    Diligent::IBuffer *adjacencyCountsBuffer               = nullptr; ///< GPU buffer of neighbor adjacency counts.
    Diligent::IBuffer *adjacencyIndicesBuffer              = nullptr; ///< GPU buffer of flattened neighbor particle indices.
    Diligent::IBuffer *fluidSurfaceNormalConstraintsBuffer = nullptr; ///< GPU buffer of fluid surface normal constraints.
    Diligent::IBuffer *fluidAnisotropy1Buffer              = nullptr; ///< GPU buffer of fluid surface anisotropy tensor column 1.
    Diligent::IBuffer *fluidAnisotropy2Buffer              = nullptr; ///< GPU buffer of fluid surface anisotropy tensor column 2.
    Diligent::IBuffer *fluidAnisotropy3Buffer              = nullptr; ///< GPU buffer of fluid surface anisotropy tensor column 3.
    std::uint32_t count                                    = 0;       ///< Total number of particles in the buffer.
    std::uint32_t fluidVisualCount                         = 0;       ///< Number of fluid particles rendered.
    std::uint32_t contactMaterialCount                     = 0;       ///< Number of contact material entries.
    std::uint32_t fluidMaterialCount                       = 0;       ///< Number of fluid material entries.
};

/// @brief GPU buffer bindings and counts for deformable soft bodies, strands, and suturing paths.
struct PhysicsGpuSoftSceneView
{
    PhysicsGpuParticleBufferView particles{};                          ///< Unified particle buffer view.
    Diligent::IBuffer *edgesBuffer                       = nullptr;    ///< GPU buffer of DeformableDistanceConstraint edge constraints.
    Diligent::IBuffer *bendsBuffer                       = nullptr;    ///< GPU buffer of DeformableBendConstraint bending constraints.
    Diligent::IBuffer *tetsBuffer                        = nullptr;    ///< GPU buffer of DeformableVolumeConstraint tetrahedral volume constraints.
    Diligent::IBuffer *strandSegmentsBuffer              = nullptr;    ///< GPU buffer of StrandSegmentConstraint descriptors.
    Diligent::IBuffer *strandJointsBuffer                = nullptr;    ///< GPU buffer of StrandJointConstraint descriptors.
    Diligent::IBuffer *strandDistanceConstraintsBuffer   = nullptr;    ///< GPU buffer of StrandDistanceConstraint descriptors.
    Diligent::IBuffer *strandSegmentStatesBuffer         = nullptr;    ///< GPU buffer of StrandSegmentState dynamic orientations.
    Diligent::IBuffer *segmentStrandJointRangesBuffer    = nullptr;    ///< GPU buffer of joint ranges incident to strand segments.
    Diligent::IBuffer *segmentIncidentStrandJointsBuffer = nullptr;    ///< GPU buffer of incident strand joint indices.
    Diligent::IBuffer *suturingPairsBuffer               = nullptr;    ///< GPU buffer of StrandSoftSuturingPair descriptors.
    Diligent::IBuffer *suturingParticleRefsBuffer        = nullptr;    ///< GPU buffer of suturing particle references.
    Diligent::IBuffer *suturingInsertionStatesBuffer     = nullptr;    ///< GPU buffer of GpuSuturingInsertionState tracking.
    Diligent::IBuffer *suturingPathHeadersBuffer         = nullptr;    ///< GPU buffer of SuturingPathHeader descriptors.
    Diligent::IBuffer *suturingPathNodesBuffer           = nullptr;    ///< GPU buffer of SuturingPathNode interpolation points.
    Diligent::IBuffer *renderPositionsBuffer             = nullptr;    ///< GPU buffer of interpolated surface render vertex positions.
    Diligent::IBuffer *renderNormalsBuffer               = nullptr;    ///< GPU buffer of interpolated surface render vertex normals.
    Diligent::IBuffer *worldAabbsBuffer                  = nullptr;    ///< GPU buffer of world-space AABBs for soft bodies.
    std::uint32_t softBodyCount                          = 0;          ///< Total soft bodies in the scene.
    std::uint32_t edgeCount                              = 0;          ///< Number of deformable edge constraints.
    std::uint32_t bendCount                              = 0;          ///< Number of deformable bending constraints.
    std::uint32_t tetCount                               = 0;          ///< Number of tetrahedral volume constraints.
    std::uint32_t strandSegmentCount                     = 0;          ///< Number of strand segments.
    std::uint32_t strandJointCount                       = 0;          ///< Number of strand joints.
    std::uint32_t strandDistanceCount                    = 0;          ///< Number of strand distance constraints.
    std::uint32_t suturingPairCount                      = 0;          ///< Number of active suturing interaction pairs.
    std::uint32_t suturingPathHeaderCount                = 0;          ///< Number of active suturing path headers.
    std::uint32_t suturingPathNodeCount                  = 0;          ///< Number of generated suturing path nodes.
    std::uint64_t bindingGeneration                      = 0;          ///< Revision counter for soft scene buffer allocations.
};

/// @brief GPU buffer bindings for strand curve rendering and extrusion geometry.
struct PhysicsGpuCurveSceneView
{
    Diligent::IBuffer *descriptorsBuffer     = nullptr; ///< GPU buffer of CurveRenderDescriptorHost descriptors.
    Diligent::IBuffer *particleIndicesBuffer = nullptr; ///< GPU buffer of particle indices forming each curve.
    Diligent::IBuffer *positionsBuffer       = nullptr; ///< GPU buffer of generated curve vertex positions.
    Diligent::IBuffer *normalsBuffer         = nullptr; ///< GPU buffer of generated curve vertex normal vectors.
    Diligent::IBuffer *worldAabbsBuffer      = nullptr; ///< GPU buffer of world-space AABBs for curves.
    std::uint32_t curveCount                 = 0;       ///< Number of curve render descriptors.
    std::uint64_t bindingGeneration          = 0;       ///< Revision counter for curve scene buffer allocations.
};

/// @brief Aggregate GPU scene view grouping rigid, joint, soft, and curve sub-views.
struct PhysicsGpuSceneView
{
    PhysicsGpuRigidSceneView rigid{};   ///< GPU view for rigid bodies, colliders, and rigid constraints.
    PhysicsGpuJointSceneView joints{};  ///< GPU view for articulated rigid joints.
    PhysicsGpuSoftSceneView soft{};     ///< GPU view for soft bodies, strands, and fluids.
    PhysicsGpuCurveSceneView curve{};   ///< GPU view for strand curve rendering.
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
