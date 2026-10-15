#include "df/core/router.h"

#include "../quant/detail.h"
#include "router_detail.h"
#include "router_fixed.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace df::core {
namespace {

constexpr std::uint32_t kFp32Magnitude = 0x7FFFFFFFu;
constexpr std::uint32_t kFp32Inf = 0x7F800000u;
constexpr std::uint32_t kFp32One = 0x3F800000u;
constexpr std::uint32_t kRoutingEpsilon = 0x1E3CE508u;
constexpr std::uint32_t kRouteScale = 0x3FC00000u;
constexpr unsigned kCertificationBits = 12288;

std::size_t checked_bytes(std::size_t count, std::size_t width) {
  if (count > std::numeric_limits<std::size_t>::max() / width) {
    throw std::invalid_argument("router element or byte count overflows size_t");
  }
  return count * width;
}

std::size_t checked_product(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument("router element or byte count overflows size_t");
  }
  return left * right;
}

template <class T, std::size_t Extent>
void check_span_bytes(std::span<T, Extent> values) {
  checked_bytes(values.size(), sizeof(T));
}

bool is_finite_fp32(std::uint32_t bits) noexcept {
  return (bits & kFp32Magnitude) < kFp32Inf;
}

int compare_finite_fp32(std::uint32_t left, std::uint32_t right) noexcept {
  const std::uint32_t left_magnitude = left & kFp32Magnitude;
  const std::uint32_t right_magnitude = right & kFp32Magnitude;
  if (left_magnitude == 0 && right_magnitude == 0) {
    return 0;
  }
  const bool left_negative = (left >> 31) != 0;
  const bool right_negative = (right >> 31) != 0;
  if (left_negative != right_negative) {
    return left_negative ? -1 : 1;
  }
  if (left_magnitude == right_magnitude) {
    return 0;
  }
  const int magnitude_order = left_magnitude < right_magnitude ? -1 : 1;
  return left_negative ? -magnitude_order : magnitude_order;
}

bool ranks_before(std::size_t candidate, std::size_t incumbent,
                  std::span<const std::uint32_t> selection) noexcept {
  const int order = compare_finite_fp32(selection[candidate], selection[incumbent]);
  return order > 0 || (order == 0 && candidate < incumbent);
}

void check_capacity(std::span<const std::uint16_t> input, GateMatrix gate,
                    std::span<const std::uint32_t> bias, RouterWorkspace work,
                    const std::optional<std::span<const std::uint32_t>>& bias_vl,
                    std::size_t weight_count) {
  check_span_bytes(input);
  check_span_bytes(gate.weight_bf16);
  check_span_bytes(bias);
  check_span_bytes(work.linear_bits);
  check_span_bytes(work.temp_bits);
  check_span_bytes(work.softplus_bits);
  check_span_bytes(work.score_bits);
  check_span_bytes(work.selection_bits);
  if (bias_vl.has_value()) {
    check_span_bytes(*bias_vl);
  }

  if (input.size() != gate.k) {
    throw std::invalid_argument("router input length must equal K");
  }
  if (gate.weight_bf16.size() != weight_count) {
    throw std::invalid_argument("router gate weight length must equal N*K");
  }
  if (bias.size() != gate.n) {
    throw std::invalid_argument("router bias length must equal N");
  }
  if (bias_vl.has_value() && bias_vl->size() != gate.n) {
    throw std::invalid_argument("router vision bias length must equal N");
  }
  if (work.linear_bits.size() < gate.n || work.temp_bits.size() < gate.n ||
      work.softplus_bits.size() < gate.n || work.score_bits.size() < gate.n ||
      work.selection_bits.size() < gate.n) {
    throw std::invalid_argument("router workspace capacity is smaller than N");
  }
}

void check_storage(std::span<const std::uint16_t> input, GateMatrix gate,
                   std::span<const std::uint32_t> bias, RouterWorkspace work,
                   const std::optional<std::span<const std::uint32_t>>& bias_vl,
                   detail::RouterTrace* trace) {
  std::array<std::span<const std::byte>, 4> reads{};
  reads[0] = std::as_bytes(input);
  reads[1] = std::as_bytes(gate.weight_bf16);
  reads[2] = std::as_bytes(bias);
  if (bias_vl.has_value()) {
    reads[3] = std::as_bytes(*bias_vl);
  }

  std::array<std::span<std::byte>, 6> writes{};
  writes[0] = std::as_writable_bytes(work.linear_bits.first(gate.n));
  writes[1] = std::as_writable_bytes(work.temp_bits.first(gate.n));
  writes[2] = std::as_writable_bytes(work.softplus_bits.first(gate.n));
  writes[3] = std::as_writable_bytes(work.score_bits.first(gate.n));
  writes[4] = std::as_writable_bytes(work.selection_bits.first(gate.n));
  if (trace != nullptr) {
    writes[5] = std::as_writable_bytes(std::span<detail::RouterTrace>(trace, 1));
  }
  detail::check_router_storage(reads, writes);
}

void check_contents(std::span<const std::uint16_t> input, GateMatrix gate,
                    std::span<const std::uint32_t> bias,
                    const std::optional<std::span<const std::uint32_t>>& bias_vl) {
  for (const std::uint16_t bits : input) {
    if ((bits & 0x7F80u) == 0x7F80u) {
      throw std::invalid_argument("router input contains nonfinite BF16 values");
    }
  }
  for (const std::uint16_t bits : gate.weight_bf16) {
    if ((bits & 0x7F80u) == 0x7F80u) {
      throw std::invalid_argument("router gate contains nonfinite BF16 values");
    }
  }
  for (const std::uint32_t bits : bias) {
    if (!is_finite_fp32(bits)) {
      throw std::invalid_argument("router bias contains nonfinite FP32 values");
    }
  }
  if (bias_vl.has_value()) {
    for (const std::uint32_t bits : *bias_vl) {
      if (!is_finite_fp32(bits)) {
        throw std::invalid_argument("router vision bias contains nonfinite FP32 values");
      }
    }
  }
}

std::array<std::size_t, kRouterTopK> select_topk(
    std::span<const std::uint32_t> selection) noexcept {
  std::array<std::size_t, kRouterTopK> indices{};
  std::size_t selected = 0;
  for (std::size_t index = 0; index < selection.size(); ++index) {
    std::size_t position = selected;
    while (position > 0 && ranks_before(index, indices[position - 1], selection)) {
      --position;
    }
    if (position >= kRouterTopK) {
      continue;
    }
    const std::size_t new_size = selected < kRouterTopK ? selected + 1 : selected;
    for (std::size_t slot = new_size; slot > position + 1; --slot) {
      indices[slot - 1] = indices[slot - 2];
    }
    indices[position] = index;
    selected = new_size;
  }
  return indices;
}

// Runs the fixed router stages after completing every parameter and storage check.
RouteResult evaluate(std::span<const std::uint16_t> input, GateMatrix gate,
                     std::span<const std::uint32_t> bias, RouterWorkspace work,
                     detail::RouterTrace* trace, bool image_token,
                     std::optional<std::span<const std::uint32_t>> bias_vl,
                     unsigned certification_bits) {
  const GateMatrix gate_snapshot = gate;
  const RouterWorkspace work_snapshot = work;
  const auto bias_vl_snapshot = bias_vl;

  if (certification_bits != 0 && !detail::fixed::valid_precision(certification_bits)) {
    throw std::invalid_argument("softplus precision must be 192 doubled through 12288");
  }
  detail::check_router_counts(gate_snapshot.n, gate_snapshot.k);
  const std::size_t weight_count = checked_product(gate_snapshot.n, gate_snapshot.k);
  check_capacity(input, gate_snapshot, bias, work_snapshot, bias_vl_snapshot, weight_count);
  check_storage(input, gate_snapshot, bias, work_snapshot, bias_vl_snapshot, trace);
  check_contents(input, gate_snapshot, bias, bias_vl_snapshot);

  const auto linear = work_snapshot.linear_bits.first(gate_snapshot.n);
  const auto temp = work_snapshot.temp_bits.first(gate_snapshot.n);
  const auto softplus = work_snapshot.softplus_bits.first(gate_snapshot.n);
  const auto score = work_snapshot.score_bits.first(gate_snapshot.n);
  const auto selection = work_snapshot.selection_bits.first(gate_snapshot.n);
  const std::span<const std::uint32_t> chosen_bias =
      image_token && bias_vl_snapshot.has_value() ? *bias_vl_snapshot : bias;

  for (std::size_t row = 0; row < gate_snapshot.n; ++row) {
    std::uint32_t accumulator = 0;
    const std::size_t offset = row * gate_snapshot.k;
    for (std::size_t column = 0; column < gate_snapshot.k; ++column) {
      const std::uint32_t input_bits = static_cast<std::uint32_t>(input[column]) << 16;
      const std::uint32_t weight_bits =
          static_cast<std::uint32_t>(gate_snapshot.weight_bf16[offset + column]) << 16;
      const std::uint32_t product =
          df::quant::detail::multiply_finite(input_bits, weight_bits);
      if (!is_finite_fp32(product)) {
        throw std::invalid_argument("nonfinite router projection product");
      }
      accumulator = df::quant::detail::add_fp32(accumulator, product);
      if (!is_finite_fp32(accumulator)) {
        throw std::invalid_argument("nonfinite router projection accumulator");
      }
    }
    linear[row] = accumulator;
  }
  for (std::size_t row = 0; row < gate_snapshot.n; ++row) {
    temp[row] = detail::divide_fp32(linear[row], kFp32One);
  }
  for (std::size_t row = 0; row < gate_snapshot.n; ++row) {
    softplus[row] = detail::softplus_fp32(temp[row], certification_bits);
  }
  for (std::size_t row = 0; row < gate_snapshot.n; ++row) {
    score[row] = detail::sqrt_fp32(softplus[row]);
  }
  for (std::size_t row = 0; row < gate_snapshot.n; ++row) {
    selection[row] = df::quant::detail::add_fp32(score[row], chosen_bias[row]);
    if (!is_finite_fp32(selection[row])) {
      throw std::invalid_argument("nonfinite router selection score");
    }
  }

  RouteResult result{};
  detail::RouterTrace pending_trace{};
  result.indices = select_topk(selection);
  std::uint32_t sum_bits = 0;
  for (std::size_t slot = 0; slot < kRouterTopK; ++slot) {
    pending_trace.selected_bits[slot] = score[result.indices[slot]];
    sum_bits = df::quant::detail::add_fp32(sum_bits, pending_trace.selected_bits[slot]);
  }
  pending_trace.sum_bits = sum_bits;
  pending_trace.denominator_bits =
      df::quant::detail::add_fp32(pending_trace.sum_bits, kRoutingEpsilon);
  for (std::size_t slot = 0; slot < kRouterTopK; ++slot) {
    pending_trace.normalized_bits[slot] =
        detail::divide_fp32(pending_trace.selected_bits[slot], pending_trace.denominator_bits);
    result.weights_bits[slot] =
        df::quant::detail::multiply_finite(pending_trace.normalized_bits[slot], kRouteScale);
  }
  if (trace != nullptr) {
    *trace = pending_trace;
  }
  return result;
}

}  // namespace

namespace detail {

void check_router_counts(std::size_t n, std::size_t k) {
  if (n < kRouterTopK || k == 0) {
    throw std::invalid_argument("router dimensions require N >= 6 and K >= 1");
  }
  const std::size_t weight_count = checked_product(n, k);
  checked_bytes(k, sizeof(std::uint16_t));
  checked_bytes(weight_count, sizeof(std::uint16_t));
  checked_bytes(n, sizeof(std::uint32_t));
}

void check_router_storage(std::span<const std::span<const std::byte>> reads,
                          std::span<const std::span<std::byte>> writes) {
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) {
      df::quant::detail::require_disjoint(read, read.size(), writes[i], writes[i].size());
    }
    for (std::size_t j = 0; j < i; ++j) {
      df::quant::detail::require_disjoint(writes[j], writes[j].size(), writes[i],
                                         writes[i].size());
    }
  }
}

RouteResult trace_route(std::span<const std::uint16_t> input, GateMatrix gate,
                        std::span<const std::uint32_t> bias, RouterWorkspace work,
                        RouterTrace& trace, bool image_token,
                        std::optional<std::span<const std::uint32_t>> bias_vl,
                        unsigned certification_bits) {
  return evaluate(input, gate, bias, work, &trace, image_token, bias_vl, certification_bits);
}

}  // namespace detail

RouteResult route(std::span<const std::uint16_t> input, GateMatrix gate,
                  std::span<const std::uint32_t> bias, RouterWorkspace work, bool image_token,
                  std::optional<std::span<const std::uint32_t>> bias_vl) {
  return evaluate(input, gate, bias, work, nullptr, image_token, bias_vl, kCertificationBits);
}

}  // namespace df::core
