#ifndef CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H

#include "engine/export.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

struct CRESSIM_NEO_ENGINE_API JointLayoutMapping
{
    std::uint32_t hingeJointCount   = 0u;
    std::uint32_t sliderJointCount  = 0u;
    std::uint64_t bindingGeneration = 0u;

    std::vector<physics::HingeJointId> hingeJointIds;
    std::vector<std::uint32_t> hingeEnvironmentIndices;
    std::vector<physics::RigidBodyId> hingeBodyIdsA;
    std::vector<physics::RigidBodyId> hingeBodyIdsB;
    std::vector<std::uint32_t> hingeBodyIndicesA;
    std::vector<std::uint32_t> hingeBodyIndicesB;

    std::vector<physics::SliderJointId> sliderJointIds;
    std::vector<std::uint32_t> sliderEnvironmentIndices;
    std::vector<physics::RigidBodyId> sliderBodyIdsA;
    std::vector<physics::RigidBodyId> sliderBodyIdsB;
    std::vector<std::uint32_t> sliderBodyIndicesA;
    std::vector<std::uint32_t> sliderBodyIndicesB;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_JOINT_LAYOUT_MAPPING_H
