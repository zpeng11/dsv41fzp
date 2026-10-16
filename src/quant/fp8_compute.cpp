#include "df/quant/compute.h"

#include "detail.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace df::quant {
namespace {

constexpr std::size_t kGroup = 32;

// All sizes and write-range intersections are checked before any input is read.
template <class Out>
void check_metadata(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_codes,
                    std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                    std::span<Out> out, const detail::DotTrace* trace) {
  const std::size_t k = a_codes.size();
  if (k == 0 || k % kGroup != 0) {
    throw std::invalid_argument("K must be positive and divisible by 32");
  }
  const std::size_t groups = k / kGroup;
  if (w_codes.size() != k || scale_a.size() != groups || scale_b.size() != groups) {
    throw std::invalid_argument("weight or scale length does not match K");
  }
  if (out.empty()) {
    throw std::invalid_argument("output capacity is smaller than the result");
  }
  if (trace != nullptr) {
    if (trace->products.size() != k || trace->partial_sums.size() != k ||
        trace->dot.size() != groups || trace->after_a.size() != groups ||
        trace->block.size() != groups || trace->acc.size() != groups) {
      throw std::invalid_argument("trace spans do not match K");
    }
    if (k > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      throw std::invalid_argument("trace byte count overflows size_t");
    }
  }
  const std::array reads{std::as_bytes(a_codes), std::as_bytes(w_codes),
                         std::as_bytes(scale_a), std::as_bytes(scale_b)};
  std::array<std::span<std::byte>, 7> writes{};
  writes[0] = std::as_writable_bytes(out.first(1));
  if (trace != nullptr) {
    writes[1] = std::as_writable_bytes(trace->products);
    writes[2] = std::as_writable_bytes(trace->partial_sums);
    writes[3] = std::as_writable_bytes(trace->dot);
    writes[4] = std::as_writable_bytes(trace->after_a);
    writes[5] = std::as_writable_bytes(trace->block);
    writes[6] = std::as_writable_bytes(trace->acc);
  }
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) {
      detail::require_disjoint(read, read.size(), writes[i], writes[i].size());
    }
    for (std::size_t j = 0; j < i; ++j) {
      detail::require_disjoint(writes[j], writes[j].size(), writes[i], writes[i].size());
    }
  }
}

void check_contents(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_codes,
                    std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b) {
  for (const auto codes : {a_codes, w_codes}) {
    for (const std::uint8_t code : codes) {
      if ((code & 0x7Fu) == 0x7Fu) {
        throw std::invalid_argument("NaN FP8 codes are not accepted");
      }
    }
  }
  for (const auto scales : {scale_a, scale_b}) {
    for (const std::uint8_t code : scales) {
      if (code == 0xFFu) {
        throw std::invalid_argument("NaN scale codes are not accepted");
      }
    }
  }
}

// The null observer and trace entries execute the same ordered integer arithmetic.
std::uint32_t grouped_dot(std::span<const std::uint8_t> a_codes,
                          std::span<const std::uint8_t> w_codes,
                          std::span<const std::uint8_t> scale_a,
                          std::span<const std::uint8_t> scale_b,
                          const detail::DotTrace* trace) noexcept {
  std::uint32_t acc = 0;
  for (std::size_t j = 0; j < a_codes.size() / kGroup; ++j) {
    std::uint32_t dot = 0;
    for (std::size_t r = 0; r < kGroup; ++r) {
      const std::size_t index = kGroup * j + r;
      const std::uint32_t product = detail::multiply_finite(
          detail::e4m3fn_bits(a_codes[index]), detail::e4m3fn_bits(w_codes[index]));
      dot = detail::add_fp32(dot, product);
      if (trace != nullptr) {
        trace->products[index] = product;
        trace->partial_sums[index] = dot;
      }
    }
    const std::uint32_t after_a = detail::multiply_fp32(dot, detail::e8m0_bits(scale_a[j]));
    const std::uint32_t block = detail::multiply_fp32(after_a, detail::e8m0_bits(scale_b[j]));
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
void evaluate(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_codes,
              std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
              std::span<Out> out, const detail::DotTrace* trace) {
  check_metadata(a_codes, w_codes, scale_a, scale_b, out, trace);
  check_contents(a_codes, w_codes, scale_a, scale_b);
  const std::uint32_t bits = grouped_dot(a_codes, w_codes, scale_a, scale_b, trace);
  if constexpr (std::is_same_v<Out, std::uint16_t>) {
    out[0] = detail::bf16_rne(bits);
  } else {
    out[0] = bits;
  }
}

}  // namespace

namespace detail {

void trace_fp8_grouped_dot(std::span<const std::uint8_t> a_codes,
                           std::span<const std::uint8_t> w_codes,
                           std::span<const std::uint8_t> scale_a,
                           std::span<const std::uint8_t> scale_b,
                           std::span<std::uint32_t> out, const DotTrace& trace) {
  const DotTrace snapshot = trace;
  evaluate(a_codes, w_codes, scale_a, scale_b, out, &snapshot);
}

void trace_fp8_grouped_dot(std::span<const std::uint8_t> a_codes,
                           std::span<const std::uint8_t> w_codes,
                           std::span<const std::uint8_t> scale_a,
                           std::span<const std::uint8_t> scale_b,
                           std::span<std::uint16_t> out, const DotTrace& trace) {
  const DotTrace snapshot = trace;
  evaluate(a_codes, w_codes, scale_a, scale_b, out, &snapshot);
}

}  // namespace detail

void fp8_grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_codes,
                     std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                     std::span<std::uint32_t> out) {
  evaluate(a_codes, w_codes, scale_a, scale_b, out, nullptr);
}

void fp8_grouped_dot(std::span<const std::uint8_t> a_codes, std::span<const std::uint8_t> w_codes,
                     std::span<const std::uint8_t> scale_a, std::span<const std::uint8_t> scale_b,
                     std::span<std::uint16_t> out) {
  evaluate(a_codes, w_codes, scale_a, scale_b, out, nullptr);
}

}  // namespace df::quant
