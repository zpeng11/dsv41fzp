#pragma once

#include <cstdint>
#include <span>

// Single-row BF16 raw bits to E4M3FN codes and one E8M0 scale per 32 elements
// under docs/ACTQUANT.md §3 (P0017, D0013). All stages use integer bit operations
// with RNE and signed gradual underflow, without reading or changing the FP environment.
// K is the input length, positive and divisible by 32. Output capacities must be at
// least K and K/32; only those prefixes are written. Input and written output ranges
// must be pairwise disjoint. Invalid metadata, overlap or nonfinite BF16 input throws
// std::invalid_argument. Metadata is checked before input reads, all contents before
// any output write; both outputs remain unchanged on failure.
namespace df::quant {

void act_quant(std::span<const std::uint16_t> input, std::span<std::uint8_t> a_codes,
               std::span<std::uint8_t> scale_a);

}  // namespace df::quant
