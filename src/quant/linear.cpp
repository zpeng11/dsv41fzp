#include "df/quant/linear.h"

#include "df/quant/act_quant.h"
#include "df/quant/compute.h"
#include "detail.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace df::quant {
namespace {

constexpr std::size_t kGroup = 32;

std::size_t checked_product(std::size_t count, std::size_t stride) {
  if (count > std::numeric_limits<std::size_t>::max() / stride) {
    throw std::invalid_argument("linear element or byte count overflows size_t");
  }
  return count * stride;
}

// Validate the complete matrix and all effective byte ranges before reading content.
template <class Out>
void check_metadata(std::span<const std::uint16_t> input, std::size_t n,
                    std::span<const std::uint8_t> w_packed,
                    std::span<const std::uint8_t> scale_b, std::span<std::uint8_t> a_codes,
                    std::span<std::uint8_t> scale_a, std::span<Out> out) {
  const std::size_t k = input.size();
  if (n == 0 || k == 0 || k % kGroup != 0) {
    throw std::invalid_argument("N and K must be positive and K divisible by 32");
  }
  const std::size_t groups = k / kGroup;
  checked_product(k, sizeof(std::uint16_t));
  checked_product(n, sizeof(Out));
  const std::size_t weight_bytes = checked_product(n, k / 2);
  const std::size_t scale_bytes = checked_product(n, groups);
  if (w_packed.size() != weight_bytes || scale_b.size() != scale_bytes) {
    throw std::invalid_argument("linear weight and scale lengths must match N and K");
  }
  if (a_codes.size() < k || scale_a.size() < groups || out.size() < n) {
    throw std::invalid_argument("linear workspace or output capacity is smaller than the result");
  }
  const std::array reads{std::as_bytes(input), std::as_bytes(w_packed), std::as_bytes(scale_b)};
  const std::array writes{std::as_writable_bytes(a_codes.first(k)),
                          std::as_writable_bytes(scale_a.first(groups)),
                          std::as_writable_bytes(out.first(n))};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) {
      detail::require_disjoint(read, read.size(), writes[i], writes[i].size());
    }
    for (std::size_t j = 0; j < i; ++j) {
      detail::require_disjoint(writes[j], writes[j].size(), writes[i], writes[i].size());
    }
  }
}

// Whole-matrix scale rejection precedes act_quant's write-atomic finite-input check.
// Its validated results satisfy every row's content contract, so no row can reject
// after the first write. Numerical stages remain exclusively in the public kernels.
template <class Out>
void evaluate(std::span<const std::uint16_t> input, std::size_t n,
              std::span<const std::uint8_t> w_packed, std::span<const std::uint8_t> scale_b,
              std::span<std::uint8_t> a_codes, std::span<std::uint8_t> scale_a,
              std::span<Out> out) {
  check_metadata(input, n, w_packed, scale_b, a_codes, scale_a, out);
  for (const std::uint8_t code : scale_b) {
    if (code == 0xFFu) {
      throw std::invalid_argument("NaN weight scale codes are not accepted");
    }
  }
  const std::size_t k = input.size();
  const std::size_t groups = k / kGroup;
  const auto codes = a_codes.first(k);
  const auto scales = scale_a.first(groups);
  act_quant(input, codes, scales);
  for (std::size_t row = 0; row < n; ++row) {
    expert_grouped_dot(codes, w_packed.subspan(row * (k / 2), k / 2), scales,
                       scale_b.subspan(row * groups, groups), out.subspan(row, 1));
  }
}

}  // namespace

void expert_linear(std::span<const std::uint16_t> input, std::size_t n,
                   std::span<const std::uint8_t> w_packed, std::span<const std::uint8_t> scale_b,
                   std::span<std::uint8_t> a_codes, std::span<std::uint8_t> scale_a,
                   std::span<std::uint32_t> out) {
  evaluate(input, n, w_packed, scale_b, a_codes, scale_a, out);
}

void expert_linear(std::span<const std::uint16_t> input, std::size_t n,
                   std::span<const std::uint8_t> w_packed, std::span<const std::uint8_t> scale_b,
                   std::span<std::uint8_t> a_codes, std::span<std::uint8_t> scale_a,
                   std::span<std::uint16_t> out) {
  evaluate(input, n, w_packed, scale_b, a_codes, scale_a, out);
}

}  // namespace df::quant
