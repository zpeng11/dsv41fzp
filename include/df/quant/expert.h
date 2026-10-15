#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// Single-token, single-selected-expert FP4 MLP under docs/EXPERT.md (P0022, D0017).
// W1/W3 are H-by-D and W2 is D-by-H; every dimension is a positive multiple of 32.
// Matrix payloads have exact row-major packed-weight and scale lengths. Caller-owned
// workspace capacities cover both activation-quantization boundaries and all H-element
// BF16 bridge tensors. Out holds at least D BF16 elements and only that prefix is written.
// No dynamic storage is allocated and descriptors are snapshotted by value.
//
// Metadata, capacities, byte counts and storage separation are checked before content;
// finite input, finite optional route weight and all matrix scales are checked before the
// first write. Effective workspace writes are pairwise disjoint and disjoint from all
// reads. The complete out span, including its tail, is disjoint from workspace writes.
// Parameter rejection leaves all storage unchanged. A nonfinite hidden BF16 value rejects
// at the second quantization boundary after workspace prefixes may have changed, while the
// complete out span remains unchanged. Final W2 Inf/NaN results are accepted.
namespace df::quant {

struct ExpertMatrix {
  std::size_t n;
  std::size_t k;
  std::span<const std::uint8_t> w_packed;
  std::span<const std::uint8_t> scale_b;
};

struct ExpertWorkspace {
  std::span<std::uint8_t> input_codes;
  std::span<std::uint8_t> input_scales;
  std::span<std::uint16_t> gate;
  std::span<std::uint16_t> up;
  std::span<std::uint16_t> hidden;
  std::span<std::uint8_t> hidden_codes;
  std::span<std::uint8_t> hidden_scales;
};

void expert(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
            ExpertMatrix w3, ExpertWorkspace work, std::span<std::uint16_t> out);
void expert(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
            ExpertMatrix w3, std::uint32_t route_weight_bits, ExpertWorkspace work,
            std::span<std::uint16_t> out);

}  // namespace df::quant
