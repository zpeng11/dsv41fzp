#pragma once

#include "df/quant/codec.h"

#include <cstdint>
#include <stdexcept>

// Private building blocks shared by the production codec and dequantization (P0011).
namespace df::quant::detail {

std::size_t element_count(Shape shape);
std::uint32_t e2m1_bits(std::uint8_t code);
std::uint32_t e8m0_bits(std::uint8_t code);
std::uint32_t e4m3fn_bits(std::uint8_t code);
std::uint16_t bf16_rne(std::uint32_t bits);

// Both operands must be finite binary32 bit patterns. Results use RNE, gradual
// underflow and signed infinity on overflow, without touching the FP environment.
std::uint32_t multiply_finite(std::uint32_t left, std::uint32_t right) noexcept;

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
