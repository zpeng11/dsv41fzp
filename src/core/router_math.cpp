#include "router_detail.h"

#include "router_fixed.h"
#include "../quant/detail.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace df::core::detail {
namespace {

constexpr std::uint32_t kSign = 0x80000000u;
constexpr std::uint32_t kMagnitude = 0x7FFFFFFFu;
constexpr std::uint32_t kInf = 0x7F800000u;
constexpr std::uint32_t kTwenty = 0x41A00000u;
constexpr std::uint32_t kOneTwentyEight = 0x43000000u;

struct Binary {
  std::uint64_t significand;
  int exponent;
};

Binary unpack(std::uint32_t bits) noexcept {
  const unsigned field = (bits >> 23) & 0xFFu;
  const std::uint64_t fraction = bits & 0x7FFFFFu;
  if (field == 0) {
    return {fraction, -149};
  }
  return {fraction | 0x800000u, static_cast<int>(field) - 150};
}

void require_finite(std::uint32_t bits, const char* message) {
  if ((bits & kMagnitude) >= kInf) {
    throw std::invalid_argument(message);
  }
}

int ratio_top_exponent(std::uint64_t numerator, std::uint64_t denominator) noexcept {
  int exponent = static_cast<int>(std::bit_width(numerator)) -
                 static_cast<int>(std::bit_width(denominator));
  const bool reaches_power =
      exponent >= 0 ? numerator >= denominator * (std::uint64_t{1} << exponent)
                    : numerator * (std::uint64_t{1} << -exponent) >= denominator;
  return reaches_power ? exponent : exponent - 1;
}

struct SmallQuotient {
  std::uint64_t value;
  fixed::Big remainder;
  bool overflow;
};

SmallQuotient divide_to_u64(const fixed::Big& numerator, const fixed::Big& denominator) {
  fixed::Big remainder;
  std::uint64_t quotient = 0;
  bool overflow = false;
  for (std::size_t position = numerator.bit_width(); position != 0; --position) {
    fixed::shift_left_one(remainder);
    if (numerator.bit(position - 1)) {
      remainder.limb[0] |= 1u;
    }
    if (fixed::compare(remainder, denominator) >= 0) {
      fixed::subtract_assign(remainder, denominator);
      const std::size_t quotient_bit = position - 1;
      if (quotient_bit < 64) {
        quotient |= std::uint64_t{1} << quotient_bit;
      } else {
        overflow = true;
      }
    }
  }
  return {quotient, remainder, overflow};
}

std::uint64_t rounded_scaled_ratio(std::uint64_t numerator, std::uint64_t denominator,
                                   int shift) {
  fixed::Big wide_numerator = fixed::Big::from_u64(numerator);
  fixed::Big wide_denominator = fixed::Big::from_u64(denominator);
  if (shift >= 0) {
    wide_numerator = fixed::shift_left(wide_numerator, static_cast<unsigned>(shift));
  } else {
    wide_denominator = fixed::shift_left(wide_denominator, static_cast<unsigned>(-shift));
  }
  SmallQuotient result = divide_to_u64(wide_numerator, wide_denominator);
  if (result.overflow) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  fixed::shift_left_one(result.remainder);
  const int midpoint = fixed::compare(result.remainder, wide_denominator);
  if (midpoint > 0 || (midpoint == 0 && (result.value & 1u) != 0)) {
    ++result.value;
  }
  return result.value;
}

int compare_scaled(std::uint64_t left, int left_exponent, std::uint64_t right,
                   int right_exponent) {
  const int common = std::min(left_exponent, right_exponent);
  fixed::Big wide_left = fixed::shift_left(fixed::Big::from_u64(left),
                                           static_cast<unsigned>(left_exponent - common));
  fixed::Big wide_right = fixed::shift_left(fixed::Big::from_u64(right),
                                            static_cast<unsigned>(right_exponent - common));
  return fixed::compare(wide_left, wide_right);
}

fixed::Big exact_fixed(std::uint32_t magnitude, unsigned precision) {
  const Binary value = unpack(magnitude);
  return fixed::shift_left(fixed::Big::from_u64(value.significand),
                           static_cast<unsigned>(value.exponent + static_cast<int>(precision)));
}

std::uint32_t fixed_to_fp32(const fixed::Big& value, unsigned precision) {
  if (value.empty()) {
    return 0;
  }
  const int top = static_cast<int>(value.bit_width()) - 1 - static_cast<int>(precision);
  const int unit = top >= -126 ? top - 23 : -149;
  const int shift = static_cast<int>(precision) + unit;
  if (shift < 0) {
    throw std::overflow_error("router fixed-point result lacks conversion capacity");
  }
  const std::uint64_t rounded = fixed::rounded_right(value, static_cast<unsigned>(shift));
  return df::quant::detail::binary_to_fp32(0, rounded, unit);
}

struct Interval {
  fixed::Big lower;
  fixed::Big upper;
};

Interval exp_small(const fixed::Big& argument, unsigned precision) {
  Interval term{fixed::one(precision), fixed::one(precision)};
  Interval sum = term;
  for (std::uint32_t denominator = 1;; ++denominator) {
    Interval next{
        fixed::divide_small(fixed::multiply_fixed(term.lower, argument, precision, false),
                            denominator, false),
        fixed::divide_small(fixed::multiply_fixed(term.upper, argument, precision, true),
                            denominator, true)};
    if (next.upper.bit_width() <= 1) {
      fixed::add_assign(sum.upper, next.upper);
      fixed::add_assign(sum.upper, next.upper);
      return sum;
    }
    fixed::add_assign(sum.lower, next.lower);
    fixed::add_assign(sum.upper, next.upper);
    term = next;
    if (denominator == std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("router exponential series did not converge");
    }
  }
}

// FP32 inputs have at most 149 binary fractional places, while every supported P is
// at least 192. Their P-bit fixed encoding therefore has eight trailing zero bits, so
// division by 256 is this exact shift rather than an interval operation.
Interval exp_negative_abs(const fixed::Big& absolute, unsigned precision) {
  const fixed::Big reduced = fixed::shift_right(absolute, 8);
  const Interval positive = exp_small(reduced, precision);
  const fixed::Big unit = fixed::one(precision);
  Interval result{fixed::divide_fixed(unit, positive.upper, precision, false),
                  fixed::divide_fixed(unit, positive.lower, precision, true)};
  for (unsigned i = 0; i < 8; ++i) {
    result = {fixed::multiply_fixed(result.lower, result.lower, precision, false),
              fixed::multiply_fixed(result.upper, result.upper, precision, true)};
  }
  return result;
}

Interval log1p(const Interval& value, unsigned precision) {
  fixed::Big two = fixed::one(precision);
  fixed::shift_left_one(two);
  fixed::Big lower_denominator = two;
  fixed::Big upper_denominator = two;
  fixed::add_assign(lower_denominator, value.lower);
  fixed::add_assign(upper_denominator, value.upper);
  Interval q{fixed::divide_fixed(value.lower, lower_denominator, precision, false),
             fixed::divide_fixed(value.upper, upper_denominator, precision, true)};

  fixed::Big half = fixed::one(precision - 1);
  if (fixed::compare(q.upper, half) >= 0) {
    throw std::overflow_error("router logarithm reduction escaped its proven range");
  }

  Interval q_squared{fixed::multiply_fixed(q.lower, q.lower, precision, false),
                     fixed::multiply_fixed(q.upper, q.upper, precision, true)};
  Interval power = q;
  Interval sum = q;
  for (std::uint32_t denominator = 3;; denominator += 2) {
    Interval next_power{
        fixed::multiply_fixed(power.lower, q_squared.lower, precision, false),
        fixed::multiply_fixed(power.upper, q_squared.upper, precision, true)};
    Interval next_term{fixed::divide_small(next_power.lower, denominator, false),
                       fixed::divide_small(next_power.upper, denominator, true)};
    if (next_term.upper.bit_width() <= 1) {
      fixed::add_assign(sum.upper, next_term.upper);
      fixed::add_assign(sum.upper, next_term.upper);
      fixed::shift_left_one(sum.lower);
      fixed::shift_left_one(sum.upper);
      return sum;
    }
    fixed::add_assign(sum.lower, next_term.lower);
    fixed::add_assign(sum.upper, next_term.upper);
    power = next_power;
    if (denominator > std::numeric_limits<std::uint32_t>::max() - 2) {
      throw std::runtime_error("router logarithm series did not converge");
    }
  }
}

SoftplusBounds certify_softplus(std::uint32_t bits, unsigned precision) {
  const std::uint32_t magnitude = bits & kMagnitude;
  const bool negative = (bits & kSign) != 0;
  const fixed::Big absolute = exact_fixed(magnitude, precision);
  Interval result = log1p(exp_negative_abs(absolute, precision), precision);
  if (!negative) {
    fixed::add_assign(result.lower, absolute);
    fixed::add_assign(result.upper, absolute);
  }
  return {fixed_to_fp32(result.lower, precision), fixed_to_fp32(result.upper, precision)};
}

}  // namespace

std::uint32_t divide_fp32(std::uint32_t numerator, std::uint32_t denominator) {
  require_finite(numerator, "division numerator must be finite FP32");
  require_finite(denominator, "division denominator must be finite FP32");
  if ((denominator & kMagnitude) == 0) {
    throw std::invalid_argument("division denominator must be nonzero");
  }
  const std::uint32_t sign = (numerator ^ denominator) & kSign;
  if ((numerator & kMagnitude) == 0) {
    return sign;
  }

  const Binary left = unpack(numerator & kMagnitude);
  const Binary right = unpack(denominator & kMagnitude);
  const int exact_exponent = left.exponent - right.exponent;
  const int top = ratio_top_exponent(left.significand, right.significand) + exact_exponent;
  const int unit = top >= -126 ? top - 23 : -149;
  const std::uint64_t rounded = rounded_scaled_ratio(
      left.significand, right.significand, exact_exponent - unit);
  return df::quant::detail::binary_to_fp32(sign, rounded, unit);
}

std::uint32_t sqrt_fp32(std::uint32_t bits) {
  require_finite(bits, "sqrt input must be finite FP32");
  if ((bits & kMagnitude) == 0) {
    return bits;
  }
  if ((bits & kSign) != 0) {
    throw std::invalid_argument("sqrt input must be nonnegative");
  }

  const Binary value = unpack(bits);
  const int value_top = static_cast<int>(std::bit_width(value.significand)) - 1 + value.exponent;
  const int root_top = value_top >= 0 ? value_top / 2 : -((-value_top + 1) / 2);
  const int unit = root_top - 23;

  std::uint64_t lower = std::uint64_t{1} << 23;
  std::uint64_t upper = std::uint64_t{1} << 24;
  while (lower + 1 < upper) {
    const std::uint64_t middle = (lower + upper) / 2;
    const int comparison = compare_scaled(middle * middle, 2 * unit,
                                          value.significand, value.exponent);
    if (comparison <= 0) {
      lower = middle;
    } else {
      upper = middle;
    }
  }

  const std::uint64_t doubled_midpoint = 2 * lower + 1;
  const int midpoint_comparison = compare_scaled(
      doubled_midpoint * doubled_midpoint, 2 * unit - 2,
      value.significand, value.exponent);
  if (midpoint_comparison < 0 || (midpoint_comparison == 0 && (lower & 1u) != 0)) {
    ++lower;
  }
  return df::quant::detail::binary_to_fp32(0, lower, unit);
}

SoftplusBounds softplus_bounds(std::uint32_t bits, unsigned precision) {
  require_finite(bits, "softplus input must be finite FP32");
  if (!fixed::valid_precision(precision)) {
    throw std::invalid_argument("softplus precision must be 192 doubled through 12288");
  }
  const std::uint32_t magnitude = bits & kMagnitude;
  if ((bits & kSign) == 0 && magnitude > kTwenty) {
    return {bits, bits};
  }
  if ((bits & kSign) != 0 && magnitude >= kOneTwentyEight) {
    return {0, 0};
  }
  return certify_softplus(bits, precision);
}

std::uint32_t softplus_fp32(std::uint32_t bits, unsigned max_precision) {
  require_finite(bits, "softplus input must be finite FP32");
  const std::uint32_t magnitude = bits & kMagnitude;
  if ((bits & kSign) == 0 && magnitude > kTwenty) {
    return bits;
  }
  if ((bits & kSign) != 0 && magnitude >= kOneTwentyEight) {
    return 0;
  }
  if (max_precision == 0) {
    throw std::runtime_error("softplus binary32 rounding interval did not converge");
  }
  if (!fixed::valid_precision(max_precision)) {
    throw std::invalid_argument("softplus precision must be 192 doubled through 12288");
  }
  for (unsigned precision = fixed::kInitialPrecision;; precision *= 2) {
    const SoftplusBounds bounds = certify_softplus(bits, precision);
    if (bounds.lower_bits == bounds.upper_bits) {
      return bounds.lower_bits;
    }
    if (precision == max_precision) {
      break;
    }
  }
  throw std::runtime_error("softplus binary32 rounding interval did not converge");
}

}  // namespace df::core::detail
