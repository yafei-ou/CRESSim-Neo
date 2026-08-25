#ifndef CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H

#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

struct JointLayoutMapping
{
    std::uint32_t ballJointCount      = 0u;
    std::uint32_t hingeJointCount     = 0u;
    std::uint32_t sphericalJointCount = 0u;
    std::uint32_t sliderJointCount    = 0u;
    /// Prepared host-side rigid-joint slot-layout invalidation key produced by prepare().
    /// This describes when authored slot interpretation changes and is not the same as the live
    /// GPU custom-compute resource bindingGeneration exposed after uploadWorld().
    std::uint64_t layoutRevision      = 0u;

    std::vector<physics::BallJointId> ballJointIds;
    std::vector<std::uint32_t> ballEnvironmentIndices;
    std::vector<physics::RigidBodyId> ballBodyIdsA;
    std::vector<physics::RigidBodyId> ballBodyIdsB;
    std::vector<std::uint32_t> ballBodyIndicesA;
    std::vector<std::uint32_t> ballBodyIndicesB;

    std::vector<physics::HingeJointId> hingeJointIds;
    std::vector<std::uint32_t> hingeEnvironmentIndices;
    std::vector<physics::RigidBodyId> hingeBodyIdsA;
    std::vector<physics::RigidBodyId> hingeBodyIdsB;
    std::vector<std::uint32_t> hingeBodyIndicesA;
    std::vector<std::uint32_t> hingeBodyIndicesB;

    std::vector<physics::SphericalJointId> sphericalJointIds;
    std::vector<std::uint32_t> sphericalEnvironmentIndices;
    std::vector<physics::RigidBodyId> sphericalBodyIdsA;
    std::vector<physics::RigidBodyId> sphericalBodyIdsB;
    std::vector<std::uint32_t> sphericalBodyIndicesA;
    std::vector<std::uint32_t> sphericalBodyIndicesB;

    std::vector<physics::SliderJointId> sliderJointIds;
    std::vector<std::uint32_t> sliderEnvironmentIndices;
    std::vector<physics::RigidBodyId> sliderBodyIdsA;
    std::vector<physics::RigidBodyId> sliderBodyIdsB;
    std::vector<std::uint32_t> sliderBodyIndicesA;
    std::vector<std::uint32_t> sliderBodyIndicesB;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H
