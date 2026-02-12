#ifndef CRESSIM_NEO_COMMON_MATH_TYPES_H
#define CRESSIM_NEO_COMMON_MATH_TYPES_H

namespace cressim::neo::common
{

struct Vec3f
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Quatf
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct Transform
{
    Vec3f position{};
    Quatf rotation{};
    Vec3f scale{1.0f, 1.0f, 1.0f};
};

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_MATH_TYPES_H
