#include "df/quant/act_quant.h"

#include "detail.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace df::quant {
namespace {

constexpr std::size_t kGroup = 32;
constexpr std::uint32_t kSign = 0x80000000u;
constexpr std::uint32_t kMagnitude = 0x7FFFFFFFu;
constexpr std::uint32_t kAmaxFloor = 0x38D1B717u;
constexpr std::uint32_t kFp8MaxInverse = 0x3B124925u;
constexpr std::uint32_t kFp8Max = 0x43E00000u;

// All byte counts, capacities and write-range intersections precede any input read.
void check_metadata(std::span<const std::uint16_t> input, std::span<std::uint8_t> a_codes,
                    std::span<std::uint8_t> scale_a, const detail::ActQuantTrace* trace) {
  const std::size_t k = input.size();
  if (k == 0 || k % kGroup != 0) {
    throw std::invalid_argument("K must be positive and divisible by 32");
  }
  if (k > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
    throw std::invalid_argument("input byte count overflows size_t");
  }
  const std::size_t groups = k / kGroup;
  if (a_codes.size() < k || scale_a.size() < groups) {
    throw std::invalid_argument("output capacity is smaller than the result");
  }
  if (trace != nullptr) {
    if (trace->amax_bits.size() != groups || trace->bounded_amax_bits.size() != groups ||
        trace->product_bits.size() != groups || trace->exponent.size() != groups ||
        trace->quotient_bits.size() != k || trace->clamped_bits.size() != k) {
      throw std::invalid_argument("trace spans do not match K");
    }
    if (k > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      throw std::invalid_argument("trace byte count overflows size_t");
    }
  }
  std::array<std::span<std::byte>, 8> writes{};
  writes[0] = std::as_writable_bytes(a_codes.first(k));
  writes[1] = std::as_writable_bytes(scale_a.first(groups));
  if (trace != nullptr) {
    writes[2] = std::as_writable_bytes(trace->amax_bits);
    writes[3] = std::as_writable_bytes(trace->bounded_amax_bits);
    writes[4] = std::as_writable_bytes(trace->product_bits);
    writes[5] = std::as_writable_bytes(trace->exponent);
    writes[6] = std::as_writable_bytes(trace->quotient_bits);
    writes[7] = std::as_writable_bytes(trace->clamped_bits);
  }
  const auto input_bytes = std::as_bytes(input);
  for (std::size_t i = 0; i < writes.size(); ++i) {
    detail::require_disjoint(input_bytes, input_bytes.size(), writes[i], writes[i].size());
    for (std::size_t j = 0; j < i; ++j) {
      detail::require_disjoint(writes[j], writes[j].size(), writes[i], writes[i].size());
    }
  }
}

void check_contents(std::span<const std::uint16_t> input) {
  for (std::uint16_t code : input) {
    if ((code & 0x7F80u) == 0x7F80u) {
      throw std::invalid_argument("nonfinite BF16 activations are not accepted");
    }
  }
}

// Exact multiplication by 2^-exponent, rounded once to binary32 with signed underflow.
std::uint32_t quotient_bits(std::uint16_t code, int exponent) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(code) << 16;
  const unsigned field = (bits >> 23) & 0xFFu;
  const std::uint32_t significand = (bits & 0x7FFFFFu) | (field == 0 ? 0 : 0x800000u);
  const int binary_exponent = field == 0 ? -149 : static_cast<int>(field) - 150;
  return detail::binary_to_fp32(bits & kSign, significand, binary_exponent - exponent);
}

// The public and trace entries share one core; observation never feeds the arithmetic.
void quantize(std::span<const std::uint16_t> input, std::span<std::uint8_t> a_codes,
              std::span<std::uint8_t> scale_a, const detail::ActQuantTrace* trace) noexcept {
  for (std::size_t j = 0; j < input.size() / kGroup; ++j) {
    const std::size_t offset = j * kGroup;
    std::uint32_t amax = 0;
    for (std::size_t r = 0; r < kGroup; ++r) {
      amax = std::max(amax, static_cast<std::uint32_t>(input[offset + r] & 0x7FFFu) << 16);
    }
    const std::uint32_t bounded = std::max(amax, kAmaxFloor);
    const std::uint32_t product = detail::multiply_finite(bounded, kFp8MaxInverse);
    const int exponent = static_cast<int>((product >> 23) & 0xFFu) - 127 +
                         ((product & 0x7FFFFFu) != 0);
    scale_a[j] = static_cast<std::uint8_t>(exponent + 127);
    if (trace != nullptr) {
      trace->amax_bits[j] = amax;
      trace->bounded_amax_bits[j] = bounded;
      trace->product_bits[j] = product;
      trace->exponent[j] = exponent;
    }
    for (std::size_t r = 0; r < kGroup; ++r) {
      const std::size_t index = offset + r;
      const std::uint32_t quotient = quotient_bits(input[index], exponent);
      const std::uint32_t clamped = (quotient & kSign) | std::min(quotient & kMagnitude, kFp8Max);
      a_codes[index] = detail::encode_e4m3fn(clamped);
      if (trace != nullptr) {
        trace->quotient_bits[index] = quotient;
        trace->clamped_bits[index] = clamped;
      }
    }
  }
}

void evaluate(std::span<const std::uint16_t> input, std::span<std::uint8_t> a_codes,
              std::span<std::uint8_t> scale_a, const detail::ActQuantTrace* trace) {
  check_metadata(input, a_codes, scale_a, trace);
  check_contents(input);
  quantize(input, a_codes, scale_a, trace);
}

}  // namespace

namespace detail {

void trace_act_quant(std::span<const std::uint16_t> input, std::span<std::uint8_t> a_codes,
                     std::span<std::uint8_t> scale_a, const ActQuantTrace& trace) {
  const ActQuantTrace snapshot = trace;
  evaluate(input, a_codes, scale_a, &snapshot);
}

}  // namespace detail

void act_quant(std::span<const std::uint16_t> input, std::span<std::uint8_t> a_codes,
               std::span<std::uint8_t> scale_a) {
  evaluate(input, a_codes, scale_a, nullptr);
}

}  // namespace df::quant
