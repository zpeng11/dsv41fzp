#include "df/quant/dequant.h"

#include "detail.h"

#include <bit>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace df::quant {
namespace detail {

// Shared RNE for significands up to 64 bits; callers bound any left shift.
std::uint64_t round_shift(std::uint64_t significand, int shift) noexcept {
  if (shift <= 0) {
    return significand << -shift;
  }
  if (shift > 64) {
    return 0;
  }
  if (shift == 64) {
    return significand > (std::uint64_t{1} << 63) ? 1 : 0;
  }
  const std::uint64_t whole = significand >> shift;
  const std::uint64_t remainder = significand & ((std::uint64_t{1} << shift) - 1);
  const std::uint64_t midpoint = std::uint64_t{1} << (shift - 1);
  return whole + (remainder > midpoint || (remainder == midpoint && (whole & 1)));
}

std::uint32_t binary_to_fp32(std::uint32_t sign, std::uint64_t significand, int exponent) noexcept {
  if (significand == 0) {
    return sign;
  }
  const int width = std::bit_width(significand);
  int top_exponent = width - 1 + exponent;
  if (top_exponent < -126) {
    return sign | static_cast<std::uint32_t>(round_shift(significand, -149 - exponent));
  }
  std::uint64_t rounded = round_shift(significand, width - 24);
  if (rounded == (std::uint64_t{1} << 24)) {
    rounded >>= 1;
    ++top_exponent;
  }
  if (top_exponent > 127) {
    return sign | 0x7F800000u;
  }
  return sign | (static_cast<std::uint32_t>(top_exponent + 127) << 23) |
         (static_cast<std::uint32_t>(rounded) & 0x7FFFFFu);
}

}  // namespace detail

namespace {

std::size_t checked_product(std::size_t left, std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error("dequant metadata product overflows size_t");
  }
  return left * right;
}

// Output byte counts are checked before any buffer is read or written.
template <class Out>
void check_buffers(std::span<const std::uint8_t> data, std::size_t data_count,
                   std::span<const std::uint8_t> scales, std::size_t scale_count,
                   std::span<Out> out, std::size_t out_count) {
  checked_product(out_count, sizeof(Out));
  if (data.size() != data_count || scales.size() != scale_count) {
    throw std::invalid_argument("data or scale length does not match shape");
  }
  if (out.size() < out_count) {
    throw std::invalid_argument("output capacity is smaller than the result");
  }
  detail::require_disjoint(data, data_count, out, out_count);
  detail::require_disjoint(scales, scale_count, out, out_count);
}

void check_contents(std::span<const std::uint8_t> data, std::span<const std::uint8_t> scales,
                    bool packed, bool fp8_scale) {
  if (!packed) {
    for (std::uint8_t code : data) {
      if ((code & 0x7Fu) == 0x7Fu) {
        throw std::invalid_argument("model dequantization does not accept NaN data");
      }
    }
  }
  for (std::uint8_t code : scales) {
    if (fp8_scale ? (code == 0 || code >= 0x7Fu) : code == 0xFFu) {
      throw std::invalid_argument("scales must be finite and strictly positive");
    }
  }
}

template <class Out>
Out output_bits(std::uint32_t bits) {
  if constexpr (std::is_same_v<Out, std::uint16_t>) {
    return detail::bf16_rne(bits);
  } else {
    return bits;
  }
}

template <class Out>
void rowwise(std::span<const std::uint8_t> data, Shape data_shape,
             std::span<const std::uint8_t> scales, Shape scale_shape,
             std::span<Out> out, std::size_t group, bool packed, bool fp8_scale, bool matrix) {
  if (data_shape.empty() || (matrix && data_shape.size() != 2)) {
    throw std::invalid_argument("invalid dequant data rank");
  }
  const std::size_t cols = checked_product(data_shape.back(), packed ? 2 : 1);
  if (cols % group != 0) {
    throw std::invalid_argument("last logical dimension must be divisible by the group size");
  }
  if (scale_shape.size() != data_shape.size() || scale_shape.back() != cols / group) {
    throw std::invalid_argument("scale shape must exactly match row groups");
  }
  for (std::size_t axis = 0; axis + 1 < data_shape.size(); ++axis) {
    if (scale_shape[axis] != data_shape[axis]) {
      throw std::invalid_argument("scale shape must exactly match row groups");
    }
  }
  const std::size_t data_count = detail::element_count(data_shape);
  const std::size_t out_count = checked_product(data_count, packed ? 2 : 1);
  const std::size_t scale_count = detail::element_count(scale_shape);
  check_buffers(data, data_count, scales, scale_count, out, out_count);
  check_contents(data, scales, packed, fp8_scale);
  for (std::size_t i = 0; i < out_count; ++i) {
    const std::uint8_t code = packed ? (i % 2 == 0 ? data[i / 2] & 0xFu : data[i / 2] >> 4) : data[i];
    const std::uint32_t value = packed ? detail::e2m1_bits(code) : detail::e4m3fn_bits(code);
    const std::uint8_t scale_code = scales[i / group];
    const std::uint32_t scale = fp8_scale ? detail::e4m3fn_bits(scale_code) : detail::e8m0_bits(scale_code);
    out[i] = output_bits<Out>(detail::multiply_finite(value, scale));
  }
}

template <class Out>
void blocks(std::span<const std::uint8_t> data, Shape data_shape,
            std::span<const std::uint8_t> scales, Shape scale_shape, std::span<Out> out) {
  if (data_shape.size() != 2 || scale_shape.size() != 2) {
    throw std::invalid_argument("FP8 weights and block scales must be matrices");
  }
  const std::size_t rows = data_shape[0];
  const std::size_t cols = data_shape[1];
  const std::size_t block_rows = rows / 32 + (rows % 32 != 0);
  const std::size_t block_cols = cols / 32 + (cols % 32 != 0);
  if (scale_shape[0] != block_rows || scale_shape[1] != block_cols) {
    throw std::invalid_argument("scale shape must cover the 32x32 blocks, including tails");
  }
  const std::size_t count = detail::element_count(data_shape);
  const std::size_t scale_count = detail::element_count(scale_shape);
  check_buffers(data, count, scales, scale_count, out, count);
  check_contents(data, scales, false, false);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t scale_index = ((i / cols) / 32) * block_cols + (i % cols) / 32;
    out[i] = output_bits<Out>(detail::multiply_finite(detail::e4m3fn_bits(data[i]),
                                                   detail::e8m0_bits(scales[scale_index])));
  }
}

}  // namespace

namespace detail {

std::uint32_t multiply_finite(std::uint32_t left, std::uint32_t right) noexcept {
  const std::uint32_t sign = (left ^ right) & 0x80000000u;
  const unsigned left_exp = (left >> 23) & 0xFFu;
  const unsigned right_exp = (right >> 23) & 0xFFu;
  const std::uint32_t left_sig = (left & 0x7FFFFFu) | (left_exp == 0 ? 0 : 0x800000u);
  const std::uint32_t right_sig = (right & 0x7FFFFFu) | (right_exp == 0 ? 0 : 0x800000u);
  const int exponent = (left_exp == 0 ? -149 : static_cast<int>(left_exp) - 150) +
                       (right_exp == 0 ? -149 : static_cast<int>(right_exp) - 150);
  return detail::binary_to_fp32(sign, static_cast<std::uint64_t>(left_sig) * right_sig, exponent);
}

std::uint32_t multiply_fp32(std::uint32_t left, std::uint32_t right) noexcept {
  constexpr std::uint32_t kSign = 0x80000000u;
  constexpr std::uint32_t kMagnitude = 0x7FFFFFFFu;
  constexpr std::uint32_t kInf = 0x7F800000u;
  constexpr std::uint32_t kQuietNan = 0x7FC00000u;
  const std::uint32_t left_magnitude = left & kMagnitude;
  const std::uint32_t right_magnitude = right & kMagnitude;
  if (left_magnitude > kInf || right_magnitude > kInf) {
    return kQuietNan;
  }
  const std::uint32_t sign = (left ^ right) & kSign;
  if (left_magnitude == kInf || right_magnitude == kInf) {
    return left_magnitude == 0 || right_magnitude == 0 ? kQuietNan : sign | kInf;
  }
  return multiply_finite(left, right);
}

}  // namespace detail

void dequantize_expert(std::span<const std::uint8_t> data, Shape data_shape,
                       std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint32_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 32, true, false, true);
}

void dequantize_expert(std::span<const std::uint8_t> data, Shape data_shape,
                       std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint16_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 32, true, false, true);
}

void dequantize_fp8_weights(std::span<const std::uint8_t> data, Shape data_shape,
                            std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint32_t> out) {
  blocks(data, data_shape, scales, scale_shape, out);
}

void dequantize_fp8_weights(std::span<const std::uint8_t> data, Shape data_shape,
                            std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint16_t> out) {
  blocks(data, data_shape, scales, scale_shape, out);
}

void dequantize_engram(std::span<const std::uint8_t> data, Shape data_shape,
                       std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint32_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 32, false, false, true);
}

void dequantize_engram(std::span<const std::uint8_t> data, Shape data_shape,
                       std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint16_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 32, false, false, true);
}

void dequantize_swa(std::span<const std::uint8_t> data, Shape data_shape,
                    std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint32_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 32, false, false, false);
}

void dequantize_swa(std::span<const std::uint8_t> data, Shape data_shape,
                    std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint16_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 32, false, false, false);
}

void dequantize_main_kv(std::span<const std::uint8_t> data, Shape data_shape,
                        std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint32_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 16, true, true, false);
}

void dequantize_main_kv(std::span<const std::uint8_t> data, Shape data_shape,
                        std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint16_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 16, true, true, false);
}

void dequantize_index_k(std::span<const std::uint8_t> data, Shape data_shape,
                        std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint32_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 32, true, false, false);
}

void dequantize_index_k(std::span<const std::uint8_t> data, Shape data_shape,
                        std::span<const std::uint8_t> scales, Shape scale_shape, std::span<std::uint16_t> out) {
  rowwise(data, data_shape, scales, scale_shape, out, 32, true, false, false);
}

}  // namespace df::quant
