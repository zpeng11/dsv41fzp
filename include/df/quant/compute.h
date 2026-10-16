#pragma once

#include <cstdint>
#include <span>

// Single-output expert grouped dot under docs/COMPUTE.md (P0013, D0009): E4M3FN
// activations times packed E2M1 weights, both grouped by 32 along K with one E8M0
// scale per group on each side. K is the activation view length; the packed weights
// hold K/2 bytes with the earlier element in the low nibble, each scale view K/32.
// Every stage rounds to binary32 as specified in COMPUTE §3 and is built with integer
// bit operations, so the result does not depend on the caller's floating-point
// environment, which is never read or changed. Inf/NaN produced by legal inputs are
// results, not failures.
//
// The output is caller-owned FP32 or BF16 raw bits; exactly one element is written.
// Failures throw std::invalid_argument: K zero or not divisible by 32, mismatched
// weight or scale lengths, empty output, NaN activation (0x7F/0xFF) or scale (0xFF)
// codes, and any input overlapping the output element. Lengths and overlap are
// validated before any input is read, contents before the output is written; on
// failure the output is untouched.
namespace df::quant {

void expert_grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_packed,
                        std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                        std::span<std::uint32_t> out);
void expert_grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_packed,
                        std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                        std::span<std::uint16_t> out);

// Single-output FP8 grouped dot under docs/COMPUTE.md §8 (P0028, D0021).
// Both code views contain K E4M3FN bytes; each E8M0 scale view contains K/32.
// Outputs, environment and validation order follow the expert entry above, with
// NaN weight codes also rejected. No packed weights or matrix layout is accepted.
void fp8_grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_codes,
                     std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                     std::span<std::uint32_t> out);
void fp8_grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_codes,
                     std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                     std::span<std::uint16_t> out);

}  // namespace df::quant
