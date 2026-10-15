#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace df::core::detail::fixed {

inline constexpr unsigned kInitialPrecision = 192;
inline constexpr unsigned kMaximumPrecision = 12288;
inline constexpr unsigned kValueGuard = 64;
inline constexpr unsigned kProductGuard = 2 * kValueGuard;
inline constexpr std::size_t kValueWords = (kMaximumPrecision + kValueGuard) / 32;
inline constexpr std::size_t kProductWords = (2 * kMaximumPrecision + kProductGuard) / 32;

struct Big {
  std::array<std::uint32_t, kValueWords> limb{};

  static Big from_u64(std::uint64_t value) noexcept {
    Big out;
    out.limb[0] = static_cast<std::uint32_t>(value);
    out.limb[1] = static_cast<std::uint32_t>(value >> 32);
    return out;
  }

  bool empty() const noexcept {
    for (std::uint32_t word : limb) {
      if (word != 0) {
        return false;
      }
    }
    return true;
  }

  std::size_t used() const noexcept {
    for (std::size_t i = limb.size(); i != 0; --i) {
      if (limb[i - 1] != 0) {
        return i;
      }
    }
    return 0;
  }

  std::size_t bit_width() const noexcept {
    const std::size_t count = used();
    return count == 0 ? 0 : 32 * (count - 1) + std::bit_width(limb[count - 1]);
  }

  bool bit(std::size_t index) const noexcept {
    const std::size_t word = index / 32;
    return word < limb.size() && ((limb[word] >> (index % 32)) & 1u) != 0;
  }

  bool any_below(std::size_t bits) const noexcept {
    const std::size_t whole = bits / 32;
    for (std::size_t i = 0; i < whole && i < limb.size(); ++i) {
      if (limb[i] != 0) {
        return true;
      }
    }
    const unsigned partial = bits % 32;
    return partial != 0 && whole < limb.size() &&
           (limb[whole] & ((std::uint32_t{1} << partial) - 1)) != 0;
  }
};

inline int compare(const Big& left, const Big& right) noexcept {
  for (std::size_t i = kValueWords; i != 0; --i) {
    if (left.limb[i - 1] != right.limb[i - 1]) {
      return left.limb[i - 1] < right.limb[i - 1] ? -1 : 1;
    }
  }
  return 0;
}

inline void add_assign(Big& left, const Big& right) {
  std::uint64_t carry = 0;
  for (std::size_t i = 0; i < kValueWords; ++i) {
    const std::uint64_t sum = static_cast<std::uint64_t>(left.limb[i]) + right.limb[i] + carry;
    left.limb[i] = static_cast<std::uint32_t>(sum);
    carry = sum >> 32;
  }
  if (carry != 0) {
    throw std::overflow_error("router fixed-point addition exceeds private capacity");
  }
}

inline void add_small(Big& value, std::uint32_t addend) {
  std::uint64_t carry = addend;
  for (std::size_t i = 0; carry != 0 && i < kValueWords; ++i) {
    const std::uint64_t sum = static_cast<std::uint64_t>(value.limb[i]) + carry;
    value.limb[i] = static_cast<std::uint32_t>(sum);
    carry = sum >> 32;
  }
  if (carry != 0) {
    throw std::overflow_error("router fixed-point increment exceeds private capacity");
  }
}

inline void subtract_assign(Big& left, const Big& right) noexcept {
  std::uint64_t borrow = 0;
  for (std::size_t i = 0; i < kValueWords; ++i) {
    const std::uint64_t subtrahend = static_cast<std::uint64_t>(right.limb[i]) + borrow;
    const std::uint64_t original = left.limb[i];
    left.limb[i] = static_cast<std::uint32_t>(original - subtrahend);
    borrow = original < subtrahend;
  }
}

inline void shift_left_one(Big& value) {
  std::uint32_t carry = 0;
  for (std::uint32_t& word : value.limb) {
    const std::uint32_t next = word >> 31;
    word = (word << 1) | carry;
    carry = next;
  }
  if (carry != 0) {
    throw std::overflow_error("router fixed-point shift exceeds private capacity");
  }
}

inline Big shift_left(Big value, unsigned shift) {
  const unsigned words = shift / 32;
  const unsigned bits = shift % 32;
  if (value.empty()) {
    return value;
  }
  if (words >= kValueWords || value.used() + words > kValueWords) {
    throw std::overflow_error("router fixed-point shift exceeds private capacity");
  }
  Big out;
  for (std::size_t i = 0; i < value.used(); ++i) {
    const std::size_t target = i + words;
    out.limb[target] |= value.limb[i] << bits;
    if (bits != 0) {
      const std::uint32_t high = value.limb[i] >> (32 - bits);
      if (target + 1 == kValueWords) {
        if (high != 0) {
          throw std::overflow_error("router fixed-point shift exceeds private capacity");
        }
      } else {
        out.limb[target + 1] |= high;
      }
    }
  }
  return out;
}

inline Big shift_right(const Big& value, unsigned shift) noexcept {
  const unsigned words = shift / 32;
  const unsigned bits = shift % 32;
  Big out;
  if (words >= kValueWords) {
    return out;
  }
  for (std::size_t target = 0; target + words < kValueWords; ++target) {
    const std::size_t source = target + words;
    out.limb[target] = value.limb[source] >> bits;
    if (bits != 0 && source + 1 < kValueWords) {
      out.limb[target] |= value.limb[source + 1] << (32 - bits);
    }
  }
  return out;
}

inline Big one(unsigned precision) {
  Big out;
  out.limb[precision / 32] = std::uint32_t{1} << (precision % 32);
  return out;
}

inline Big multiply_fixed(const Big& left, const Big& right, unsigned precision, bool upward) {
  std::array<std::uint32_t, kProductWords> product{};
  const std::size_t left_used = left.used();
  const std::size_t right_used = right.used();
  if (left_used + right_used > kProductWords) {
    throw std::overflow_error("router fixed-point product exceeds private capacity");
  }
  for (std::size_t i = 0; i < left_used; ++i) {
    std::uint64_t carry = 0;
    for (std::size_t j = 0; j < right_used; ++j) {
      const std::size_t index = i + j;
      const std::uint64_t current = static_cast<std::uint64_t>(left.limb[i]) * right.limb[j] +
                                    product[index] + carry;
      product[index] = static_cast<std::uint32_t>(current);
      carry = current >> 32;
    }
    product[i + right_used] = static_cast<std::uint32_t>(carry);
  }

  const std::size_t discarded_words = precision / 32;
  bool discarded = false;
  for (std::size_t i = 0; i < discarded_words; ++i) {
    discarded |= product[i] != 0;
  }
  Big out;
  for (std::size_t i = 0; i < kValueWords; ++i) {
    out.limb[i] = product[i + discarded_words];
  }
  for (std::size_t i = discarded_words + kValueWords; i < kProductWords; ++i) {
    if (product[i] != 0) {
      throw std::overflow_error("router fixed-point product quotient exceeds private capacity");
    }
  }
  if (upward && discarded) {
    add_small(out, 1);
  }
  return out;
}

inline Big divide_small(const Big& numerator, std::uint32_t denominator, bool upward) {
  Big quotient;
  std::uint64_t remainder = 0;
  for (std::size_t i = kValueWords; i != 0; --i) {
    const std::uint64_t current = (remainder << 32) | numerator.limb[i - 1];
    quotient.limb[i - 1] = static_cast<std::uint32_t>(current / denominator);
    remainder = current % denominator;
  }
  if (upward && remainder != 0) {
    add_small(quotient, 1);
  }
  return quotient;
}

inline Big divide_fixed(const Big& numerator, const Big& denominator, unsigned precision,
                        bool upward) {
  if (denominator.empty()) {
    throw std::invalid_argument("router fixed-point denominator is zero");
  }
  Big quotient;
  Big remainder;
  const std::size_t total_bits = numerator.bit_width() + precision;
  for (std::size_t position = total_bits; position != 0; --position) {
    shift_left_one(remainder);
    const std::size_t source = position - 1;
    if (source >= precision && numerator.bit(source - precision)) {
      remainder.limb[0] |= 1u;
    }
    if (compare(remainder, denominator) >= 0) {
      subtract_assign(remainder, denominator);
      if (source >= kValueWords * 32) {
        throw std::overflow_error("router fixed-point quotient exceeds private capacity");
      }
      quotient.limb[source / 32] |= std::uint32_t{1} << (source % 32);
    }
  }
  if (upward && !remainder.empty()) {
    add_small(quotient, 1);
  }
  return quotient;
}

inline std::uint64_t rounded_right(const Big& value, unsigned shift) {
  std::uint64_t whole = 0;
  for (unsigned i = 0; i < 64; ++i) {
    if (value.bit(static_cast<std::size_t>(shift) + i)) {
      whole |= std::uint64_t{1} << i;
    }
  }
  if (shift == 0) {
    return whole;
  }
  const bool half = value.bit(shift - 1);
  const bool below = value.any_below(shift - 1);
  return whole + (half && (below || (whole & 1u) != 0));
}

inline bool valid_precision(unsigned precision) noexcept {
  if (precision < kInitialPrecision || precision > kMaximumPrecision) {
    return false;
  }
  while (precision > kInitialPrecision && (precision % 2) == 0) {
    precision /= 2;
  }
  return precision == kInitialPrecision;
}

}  // namespace df::core::detail::fixed
