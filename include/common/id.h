#ifndef CRESSIM_NEO_COMMON_ID_H
#define CRESSIM_NEO_COMMON_ID_H

#include <cstdint>

namespace cressim::neo::common
{

using EntityId                      = std::uint32_t;
constexpr EntityId kInvalidEntityId = 0;

using ResourceId                        = std::uint32_t;
constexpr ResourceId kInvalidResourceId = 0;

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_ID_H
