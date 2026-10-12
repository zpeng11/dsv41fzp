#include "df/quant/compute.h"

#include "detail.h"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace df::quant {
namespace {

constexpr std::size_t kGroup = 32;
constexpr std::uint32_t kSign = 0x80000000u;
constexpr std::uint32_t kMagnitude = 0x7FFFFFFFu;
constexpr std::uint32_t kInf = 0x7F800000u;
constexpr std::uint32_t kQuietNan = 0x7FC00000u;
constexpr int kAlign = 3;

// Unbiased view of a finite binary32: value = significand * 2^exponent.
struct Unpacked {
  std::uint64_t significand;
  int exponent;
};

Unpacked unpack(std::uint32_t bits) noexcept {
  const unsigned field = (bits >> 23) & 0xFFu;
  const std::uint64_t mantissa = bits & 0x7FFFFFu;
  if (field == 0) {
    return {mantissa, -149};
  }
  return {mantissa | 0x800000u, static_cast<int>(field) - 150};
}

// Scale multiplication with the special-value split of COMPUTE §3: scales are finite
// and strictly positive, so Inf keeps its sign and NaN propagates unchanged.
std::uint32_t scale_by(std::uint32_t value, std::uint32_t scale) noexcept {
  if ((value & kInf) == kInf) {
    return value;
  }
  return detail::multiply_finite(value, scale);
}

std::size_t check_metadata(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_packed,
                           std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b) {
  const std::size_t k = a_codes.size();
  if (k == 0 || k % kGroup != 0) {
    throw std::invalid_argument("K must be positive and divisible by 32");
  }
  if (w_packed.size() != k / 2) {
    throw std::invalid_argument("packed weight length must be K/2");
  }
  if (scale_a.size() != k / kGroup || scale_b.size() != k / kGroup) {
    throw std::invalid_argument("each scale length must be K/32");
  }
  return k;
}

void check_contents(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> scale_a,
                    std::span<const std::uint8_t> scale_b) {
  for (std::uint8_t code : a_codes) {
    if ((code & 0x7Fu) == 0x7Fu) {
      throw std::invalid_argument("NaN activation codes are not accepted");
    }
  }
  for (std::span<const std::uint8_t> scales : {scale_a, scale_b}) {
    for (std::uint8_t code : scales) {
      if (code == 0xFFu) {
        throw std::invalid_argument("NaN scale codes are not accepted");
      }
    }
  }
}

// The single computation core behind the public entries (trace == nullptr) and the
// private trace connection; recording never feeds back into the arithmetic.
std::uint32_t grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_packed,
                          std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                          const detail::DotTrace* trace) noexcept {
  std::uint32_t acc = 0;
  for (std::size_t j = 0; j < a_codes.size() / kGroup; ++j) {
    std::uint32_t dot = 0;
    for (std::size_t r = 0; r < kGroup; ++r) {
      const std::size_t index = kGroup * j + r;
      const std::uint8_t byte = w_packed[index / 2];
      const std::uint8_t nibble = index % 2 == 0 ? byte & 0xFu : byte >> 4;
      const std::uint32_t product =
          detail::multiply_finite(detail::e4m3fn_bits(a_codes[index]), detail::e2m1_bits(nibble));
      dot = detail::add_fp32(dot, product);
      if (trace != nullptr) {
        trace->products[index] = product;
        trace->partial_sums[index] = dot;
      }
    }
    const std::uint32_t after_a = scale_by(dot, detail::e8m0_bits(scale_a[j]));
    const std::uint32_t block = scale_by(after_a, detail::e8m0_bits(scale_b[j]));
    acc = detail::add_fp32(acc, block);
    if (trace != nullptr) {
      trace->dot[j] = dot;
      trace->after_a[j] = after_a;
      trace->block[j] = block;
      trace->acc[j] = acc;
    }
  }
  return acc;
}

template <class Out>
void evaluate(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_packed,
              std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
              std::span<Out> out) {
  const std::size_t k = check_metadata(a_codes, w_packed, scale_a, scale_b);
  if (out.empty()) {
    throw std::invalid_argument("output capacity is smaller than the result");
  }
  detail::require_disjoint(a_codes, k, out, 1);
  detail::require_disjoint(w_packed, k / 2, out, 1);
  detail::require_disjoint(scale_a, k / kGroup, out, 1);
  detail::require_disjoint(scale_b, k / kGroup, out, 1);
  check_contents(a_codes, scale_a, scale_b);
  const std::uint32_t bits = grouped_dot(a_codes, w_packed, scale_a, scale_b, nullptr);
  if constexpr (std::is_same_v<Out, std::uint16_t>) {
    out[0] = detail::bf16_rne(bits);
  } else {
    out[0] = bits;
  }
}

}  // namespace

namespace detail {

// Finite operands are ordered so that |big| >= |small|; both 24-bit significands gain
// kAlign low bits (guard, round, sticky), so aligned operands stay below 2^27 and their
// sum below 2^28. small is shifted right by the exponent difference d (0..253); shifts
// of 64 or more are taken as zero, never executed. Bits shifted out are jammed into
// the LSB (round to odd). That is exact whenever d <= kAlign, which covers every near
// cancellation; otherwise big is normal, the sum exceeds 2^25, and binary_to_fp32
// discards at least two bits above the jammed LSB, so RNE of the jammed sum equals RNE
// of the exact sum. Exact cancellation yields +0; equal-sign zeros keep their sign.
std::uint32_t add_fp32(std::uint32_t left, std::uint32_t right) noexcept {
  const std::uint32_t left_mag = left & kMagnitude;
  const std::uint32_t right_mag = right & kMagnitude;
  if (left_mag > kInf || right_mag > kInf) {
    return kQuietNan;
  }
  if (left_mag == kInf || right_mag == kInf) {
    if (left_mag == right_mag && ((left ^ right) & kSign) != 0) {
      return kQuietNan;
    }
    return left_mag == kInf ? left : right;
  }
  if (left_mag == 0 && right_mag == 0) {
    return left & right & kSign;
  }
  if (right_mag == 0) {
    return left;
  }
  if (left_mag == 0) {
    return right;
  }
  const bool left_bigger = left_mag >= right_mag;
  const std::uint32_t big_bits = left_bigger ? left : right;
  const std::uint32_t small_bits = left_bigger ? right : left;
  const Unpacked big = unpack(big_bits);
  const Unpacked small = unpack(small_bits);
  const int shift = big.exponent - small.exponent;
  const std::uint64_t big_aligned = big.significand << kAlign;
  const std::uint64_t small_wide = small.significand << kAlign;
  std::uint64_t small_aligned = 0;
  bool lost = small_wide != 0;
  if (shift < 64) {
    small_aligned = small_wide >> shift;
    lost = shift != 0 && (small_wide & ((std::uint64_t{1} << shift) - 1)) != 0;
  }
  small_aligned |= lost ? 1u : 0u;
  const bool subtract = ((big_bits ^ small_bits) & kSign) != 0;
  const std::uint64_t total = subtract ? big_aligned - small_aligned : big_aligned + small_aligned;
  if (total == 0) {
    return 0;
  }
  return binary_to_fp32(big_bits & kSign, total, big.exponent - kAlign);
}

std::uint32_t trace_expert_grouped_dot(std::span<const std::uint8_t> a_codes,
                                       std::span<const std::uint8_t> w_packed,
                                       std::span<const std::uint8_t> scale_a,
                                       std::span<const std::uint8_t> scale_b,
                                       const DotTrace& trace) {
  const std::size_t k = check_metadata(a_codes, w_packed, scale_a, scale_b);
  const std::size_t blocks = k / kGroup;
  if (trace.products.size() != k || trace.partial_sums.size() != k || trace.dot.size() != blocks ||
      trace.after_a.size() != blocks || trace.block.size() != blocks || trace.acc.size() != blocks) {
    throw std::invalid_argument("trace spans do not match K");
  }
  check_contents(a_codes, scale_a, scale_b);
  return grouped_dot(a_codes, w_packed, scale_a, scale_b, &trace);
}

}  // namespace detail

void expert_grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_packed,
                        std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                        std::span<std::uint32_t> out) {
  evaluate(a_codes, w_packed, scale_a, scale_b, out);
}

void expert_grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_packed,
                        std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                        std::span<std::uint16_t> out) {
  evaluate(a_codes, w_packed, scale_a, scale_b, out);
}

}  // namespace df::quant
