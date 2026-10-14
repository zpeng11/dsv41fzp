#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// Single-token, bias-free expert linear layer (P0019). K is input.size(); N and K
// must be positive and K divisible by 32. Weights are N contiguous packed E2M1
// rows (K/2 bytes each, earlier element in the low nibble); scale_b holds N rows
// of K/32 E8M0 codes. Both matrix views must have exactly those lengths.
// Caller-owned workspaces hold at least K E4M3FN codes and K/32 E8M0 scales;
// out holds at least N FP32 or BF16 raw-bit results. Only these prefixes are
// written. The input is quantized once by act_quant, then each row is evaluated
// by expert_grouped_dot, retaining their numerical and floating-point environment
// contracts. No dynamic storage is allocated.
//
// Written prefixes must be mutually disjoint and disjoint from every input read
// range. Read-only ranges may overlap; unused output/workspace tails do not
// participate. Invalid dimensions, overflowing sizes, mismatched matrix lengths,
// insufficient capacities, forbidden overlap, nonfinite BF16 input or scale_b
// code 0xFF throw std::invalid_argument. All metadata checks precede content reads;
// all content checks precede writes. On rejection, out and both workspaces remain
// entirely unchanged. Inputs are never modified; legal Inf/NaN results are allowed.
namespace df::quant {

void expert_linear(std::span<const std::uint16_t> input, std::size_t n,
                   std::span<const std::uint8_t> w_packed, std::span<const std::uint8_t> scale_b,
                   std::span<std::uint8_t> a_codes, std::span<std::uint8_t> scale_a,
                   std::span<std::uint32_t> out);
void expert_linear(std::span<const std::uint16_t> input, std::size_t n,
                   std::span<const std::uint8_t> w_packed, std::span<const std::uint8_t> scale_b,
                   std::span<std::uint8_t> a_codes, std::span<std::uint8_t> scale_a,
                   std::span<std::uint16_t> out);

}  // namespace df::quant
