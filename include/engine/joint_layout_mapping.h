#ifndef CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H

#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

/// @brief Prepared parallel arrays mapping rigid-joint definitions to rigid-body slots.
struct JointLayoutMapping
{
    std::uint32_t ballJointCount      = 0u; ///< Number of ball-joint slots.
    std::uint32_t hingeJointCount     = 0u; ///< Number of hinge-joint slots.
    std::uint32_t sphericalJointCount = 0u; ///< Number of spherical-joint slots.
    std::uint32_t sliderJointCount    = 0u; ///< Number of slider-joint slots.
    /// Prepared host-side rigid-joint slot-layout invalidation key produced by prepare().
    /// This describes when authored slot interpretation changes and is not the same as the live
    /// GPU custom-compute resource bindingGeneration exposed after uploadWorld().
    std::uint64_t layoutRevision      = 0u;

    /// @brief Ball-joint IDs, indexed by ball-joint slot.
    std::vector<physics::BallJointId> ballJointIds;
    std::vector<std::uint32_t> ballEnvironmentIndices; ///< Environment of body A.
    std::vector<physics::RigidBodyId> ballBodyIdsA;    ///< First body IDs.
    std::vector<physics::RigidBodyId> ballBodyIdsB;    ///< Second body IDs.
    std::vector<std::uint32_t> ballBodyIndicesA;       ///< First body slots.
    std::vector<std::uint32_t> ballBodyIndicesB;       ///< Second body slots.

    /// @brief Hinge-joint IDs, indexed by hinge-joint slot.
    std::vector<physics::HingeJointId> hingeJointIds;
    std::vector<std::uint32_t> hingeEnvironmentIndices; ///< Environment of body A.
    std::vector<physics::RigidBodyId> hingeBodyIdsA;    ///< First body IDs.
    std::vector<physics::RigidBodyId> hingeBodyIdsB;    ///< Second body IDs.
    std::vector<std::uint32_t> hingeBodyIndicesA;       ///< First body slots.
    std::vector<std::uint32_t> hingeBodyIndicesB;       ///< Second body slots.

    /// @brief Spherical-joint IDs, indexed by spherical-joint slot.
    std::vector<physics::SphericalJointId> sphericalJointIds;
    std::vector<std::uint32_t> sphericalEnvironmentIndices; ///< Environment of body A.
    std::vector<physics::RigidBodyId> sphericalBodyIdsA;    ///< First body IDs.
    std::vector<physics::RigidBodyId> sphericalBodyIdsB;    ///< Second body IDs.
    std::vector<std::uint32_t> sphericalBodyIndicesA;       ///< First body slots.
    std::vector<std::uint32_t> sphericalBodyIndicesB;       ///< Second body slots.

    /// @brief Slider-joint IDs, indexed by slider-joint slot.
    std::vector<physics::SliderJointId> sliderJointIds;
    std::vector<std::uint32_t> sliderEnvironmentIndices; ///< Environment of body A.
    std::vector<physics::RigidBodyId> sliderBodyIdsA;    ///< First body IDs.
    std::vector<physics::RigidBodyId> sliderBodyIdsB;    ///< Second body IDs.
    std::vector<std::uint32_t> sliderBodyIndicesA;       ///< First body slots.
    std::vector<std::uint32_t> sliderBodyIndicesB;       ///< Second body slots.
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H
