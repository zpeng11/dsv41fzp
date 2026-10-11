#include "bb/smoke.h"

namespace bb {

bool checked_sum(const std::int32_t* values, std::size_t count, std::int32_t* out) {
  std::int64_t acc = 0;
  for (std::size_t i = 0; i < count; ++i) {
    acc += values[i];
    if (acc > INT32_MAX || acc < INT32_MIN) return false;
  }
  *out = static_cast<std::int32_t>(acc);
  return true;
}

}  // namespace bb
