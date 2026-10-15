#pragma once

#include "df/quant/codec.h"
#include "df/quant/expert.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Private building blocks shared by the production codec, dequantization (P0011),
// expert grouped dot (P0013), activation quantization (P0017) and SwiGLU (P0021).
namespace df::quant::detail {

std::size_t element_count(Shape shape);
std::uint32_t e2m1_bits(std::uint8_t code);
std::uint32_t e8m0_bits(std::uint8_t code);
std::uint32_t e4m3fn_bits(std::uint8_t code);
std::uint16_t bf16_rne(std::uint32_t bits);

// RNE of significand * 2^-shift; left shifts must fit in uint64_t.
std::uint64_t round_shift(std::uint64_t significand, int shift) noexcept;

// E4M3FN RNE for finite binary32 bits already clamped to [-448, 448]; preserves zero sign.
std::uint8_t encode_e4m3fn(std::uint32_t bits) noexcept;

// Both operands must be finite binary32 bit patterns. Results use RNE, gradual
// underflow and signed infinity on overflow, without touching the FP environment.
std::uint32_t multiply_finite(std::uint32_t left, std::uint32_t right) noexcept;

// One binary32 RNE multiplication of arbitrary bit patterns, including specials.
std::uint32_t multiply_fp32(std::uint32_t left, std::uint32_t right) noexcept;

// Rounds sign * significand * 2^exponent to binary32 with RNE, gradual underflow and
// signed infinity on overflow; significand == 0 yields the signed zero.
std::uint32_t binary_to_fp32(std::uint32_t sign, std::uint64_t significand, int exponent) noexcept;

// One binary32 RNE addition of arbitrary bit patterns, specials included, under the
// zero-sign and NaN rules of docs/COMPUTE.md §3.
std::uint32_t add_fp32(std::uint32_t left, std::uint32_t right) noexcept;

// Per-stage FP32 bit patterns of one expert grouped dot: products and partial_sums
// hold K entries, the block-level spans K/32.
struct DotTrace {
  std::span<std::uint32_t> products;
  std::span<std::uint32_t> partial_sums;
  std::span<std::uint32_t> dot;
  std::span<std::uint32_t> after_a;
  std::span<std::uint32_t> block;
  std::span<std::uint32_t> acc;
};

// Validates exactly like the public expert_grouped_dot entries, additionally requires
// trace spans of the sizes above, and runs the same computation core while recording
// every stage. Returns the FP32 result bits.
std::uint32_t trace_expert_grouped_dot(std::span<const std::uint8_t> a_codes,
                                       std::span<const std::uint8_t> w_packed,
                                       std::span<const std::uint8_t> scale_a,
                                       std::span<const std::uint8_t> scale_b,
                                       const DotTrace& trace);

// Activation quantization stages: the first four spans hold K/32 entries, the last
// two K entries. Codes and scales are supplied through the normal output spans.
struct ActQuantTrace {
  std::span<std::uint32_t> amax_bits;
  std::span<std::uint32_t> bounded_amax_bits;
  std::span<std::uint32_t> product_bits;
  std::span<std::int32_t> exponent;
  std::span<std::uint32_t> quotient_bits;
  std::span<std::uint32_t> clamped_bits;
};

// Uses the public validation and computation core, with exact trace lengths and
// trace writes disjoint from each other, the input and written output prefixes.
void trace_act_quant(std::span<const std::uint16_t> input, std::span<std::uint8_t> a_codes,
                     std::span<std::uint8_t> scale_a, const ActQuantTrace& trace);

struct SwigluTrace {
  std::span<std::uint32_t> gate_bits;
  std::span<std::uint32_t> up_bits;
  std::span<std::uint32_t> silu_bits;
  std::span<std::uint32_t> gated_bits;
  std::span<std::uint32_t> weighted_bits;
};

struct ExpertCounts {
  std::size_t input_groups;
  std::size_t hidden_groups;
  std::size_t w1_weight_bytes;
  std::size_t w1_scale_bytes;
  std::size_t w2_weight_bytes;
  std::size_t w2_scale_bytes;
  std::size_t w3_weight_bytes;
  std::size_t w3_scale_bytes;
};

ExpertCounts check_expert_counts(std::size_t input_count, std::size_t w1_n, std::size_t w1_k,
                                 std::size_t w2_n, std::size_t w2_k, std::size_t w3_n,
                                 std::size_t w3_k, std::size_t out_count, bool traced);
void check_expert_storage(std::span<const std::span<const std::byte>> reads,
                          std::span<const std::span<std::byte>> scratch_writes,
                          std::span<std::byte> out, std::size_t out_write_bytes);

void trace_expert(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
                  ExpertMatrix w3, ExpertWorkspace work, std::span<std::uint16_t> out,
                  const SwigluTrace& trace);
void trace_expert(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
                  ExpertMatrix w3, std::uint32_t route_weight_bits, ExpertWorkspace work,
                  std::span<std::uint16_t> out, const SwigluTrace& trace);

void trace_swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
                  std::span<std::uint16_t> out, const SwigluTrace& trace);
void trace_swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
                  std::uint32_t route_weight_bits, std::span<std::uint16_t> out,
                  const SwigluTrace& trace);

// Shared byte-range check behind public and trace SwiGLU entries. Read ranges may
// overlap each other; every write range must be disjoint from both reads and all
// earlier writes. Byte spans let the executor test mixed element types legally.
void check_swiglu_storage(std::span<const std::byte> gate, std::span<const std::byte> up,
                          std::span<const std::span<std::byte>> writes);

// Count checks shared by every SwiGLU entry: input bytes and, when traced, stage
// bytes must not overflow size_t. Scalar arguments keep huge counts testable.
void check_swiglu_counts(std::size_t count, bool traced);

// Only the actually consumed prefixes participate; empty ranges never overlap.
template <class In, class Out>
void require_disjoint(std::span<In> in, std::size_t in_count, std::span<Out> out, std::size_t out_count) {
  if (in_count == 0 || out_count == 0) {
    return;
  }
  const auto in_begin = reinterpret_cast<std::uintptr_t>(in.data());
  const auto out_begin = reinterpret_cast<std::uintptr_t>(out.data());
  const auto in_bytes = in_count * sizeof(In);
  const auto out_bytes = out_count * sizeof(Out);
  if (in_begin <= out_begin ? out_begin - in_begin < in_bytes : in_begin - out_begin < out_bytes) {
    throw std::invalid_argument("input and output buffers overlap");
  }
}

}  // namespace df::quant::detail
