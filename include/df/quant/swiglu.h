#pragma once

#include <cstdint>
#include <span>

// Single-token SwiGLU bridge under docs/EXPERT.md (P0021, D0016). Gate and up
// contain equal-length BF16 raw bits; empty inputs are accepted. Each element is
// widened exactly, clamped, passed through correctly rounded FP32 SiLU, multiplied
// by up and optionally by a finite FP32 route weight, then rounded to BF16. The
// overload without a route weight omits that multiplication entirely.
//
// Out holds at least gate.size() elements; only that prefix is written. Read-only
// inputs may overlap each other, but the written prefix must be disjoint from both.
// Invalid lengths, capacities, byte counts, overlap or route weight throw
// std::invalid_argument before reading input elements or writing output. On
// rejection out remains entirely unchanged. Nonfinite gate/up values follow the
// bridge numerical contract. No dynamic storage is allocated and the complete
// floating-point environment is unchanged.
namespace df::quant {

void swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
            std::span<std::uint16_t> out);
void swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
            std::uint32_t route_weight_bits, std::span<std::uint16_t> out);

}  // namespace df::quant
