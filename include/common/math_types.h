#ifndef CRESSIM_NEO_COMMON_MATH_TYPES_H
#define CRESSIM_NEO_COMMON_MATH_TYPES_H

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

namespace cressim::neo::common
{

struct Transform
{
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float3 scale{1.0f, 1.0f, 1.0f};
};

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_MATH_TYPES_H
