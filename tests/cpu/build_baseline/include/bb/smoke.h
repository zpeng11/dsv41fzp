#pragma once

#include <cstddef>
#include <cstdint>

namespace bb {

// Sum with overflow detection; returns false if the result does not fit.
bool checked_sum(const std::int32_t* values, std::size_t count, std::int32_t* out);

}  // namespace bb
