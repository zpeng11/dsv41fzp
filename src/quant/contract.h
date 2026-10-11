#pragma once

#include <cstdint>

namespace df::quant {

// Quantization format contract version, independent of the engine VERSION. It must be
// bumped whenever any bit pattern, zero sign, packing order or failure contract changes;
// the current value is recorded in docs/QUANT.md.
inline constexpr std::uint32_t kFormatContractVersion = 1;

}  // namespace df::quant
