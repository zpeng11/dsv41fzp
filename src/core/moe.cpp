#include "df/core/moe.h"

#include "../quant/detail.h"
#include "moe_detail.h"
#include "router_detail.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace df::core {
namespace {

constexpr std::size_t kGroup = 32;

std::size_t checked_bytes(std::size_t count, std::size_t width) {
  if (count > std::numeric_limits<std::size_t>::max() / width) {
    throw std::invalid_argument("routed element or byte count overflows size_t");
  }
  return count * width;
}

std::size_t checked_product(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument("routed element or byte count overflows size_t");
  }
  return left * right;
}

template <class T, std::size_t Extent>
void check_range(std::span<T, Extent> values) {
  const std::size_t bytes = checked_bytes(values.size(), sizeof(T));
  if (bytes == 0) {
    return;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(values.data());
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    throw std::invalid_argument("routed address range overflows uintptr_t");
  }
}

void check_matrix_range(df::quant::ExpertMatrix matrix) {
  check_range(matrix.w_packed);
  check_range(matrix.scale_b);
}

void require_disjoint(std::span<const std::byte> read, std::span<std::byte> write) {
  df::quant::detail::require_disjoint(read, read.size(), write, write.size());
}

void require_disjoint(std::span<std::byte> left, std::span<std::byte> right) {
  df::quant::detail::require_disjoint(left, left.size(), right, right.size());
}

struct Validation {
  std::size_t d;
  std::size_t h;
  std::size_t input_groups;
  std::size_t hidden_groups;
  std::size_t trace_count;
};

Validation validate_metadata(std::span<const std::uint16_t> input, GateMatrix gate,
                             std::span<const std::uint32_t> bias,
                             std::span<const RoutedExpert> experts, RoutedWorkspace work,
                             std::span<std::uint32_t> out, const detail::RoutedTrace* trace,
                             std::span<std::uint32_t> trace_work,
                             const std::optional<std::span<const std::uint32_t>>& bias_vl) {
  check_range(input);
  check_range(gate.weight_bf16);
  check_range(bias);
  check_range(experts);
  check_range(out);
  if (bias_vl.has_value()) {
    check_range(*bias_vl);
  }

  detail::validate_router_parameters(input, gate, bias, work.router, bias_vl);
  if (experts.size() != gate.n) {
    throw std::invalid_argument("routed expert count must equal gate N");
  }
  if (out.size() < gate.k || work.expert_output.size() < gate.k ||
      work.accumulator_bits.size() < gate.k) {
    throw std::invalid_argument("routed output or workspace capacity is smaller than D");
  }

  const std::size_t trace_count = checked_product(kRouterTopK, gate.k);
  if (trace == nullptr) {
    if (!trace_work.empty()) {
      throw std::invalid_argument("ordinary routed entry does not accept trace workspace");
    }
  } else {
    check_range(std::span<const detail::RoutedTrace>(trace, 1));
    check_range(trace->partial_sums_bits);
    check_range(trace_work);
    if (trace->partial_sums_bits.size() < trace_count || trace_work.size() < trace_count) {
      throw std::invalid_argument("routed trace capacity is smaller than 6*D");
    }
  }

  const std::size_t h = experts.front().w1.n;
  for (const RoutedExpert& expert : experts) {
    check_matrix_range(expert.w1);
    check_matrix_range(expert.w2);
    check_matrix_range(expert.w3);
    if (expert.w1.n != h) {
      throw std::invalid_argument("routed experts must share one hidden dimension");
    }
    df::quant::detail::validate_expert_parameters(input, expert.w1, expert.w2, expert.w3,
                                                   work.expert, work.expert_output);
  }

  const std::size_t input_groups = gate.k / kGroup;
  const std::size_t hidden_groups = h / kGroup;
  check_range(work.router.linear_bits.first(gate.n));
  check_range(work.router.temp_bits.first(gate.n));
  check_range(work.router.softplus_bits.first(gate.n));
  check_range(work.router.score_bits.first(gate.n));
  check_range(work.router.selection_bits.first(gate.n));
  check_range(work.expert.input_codes.first(gate.k));
  check_range(work.expert.input_scales.first(input_groups));
  check_range(work.expert.gate.first(h));
  check_range(work.expert.up.first(h));
  check_range(work.expert.hidden.first(h));
  check_range(work.expert.hidden_codes.first(h));
  check_range(work.expert.hidden_scales.first(hidden_groups));
  check_range(work.expert_output.first(gate.k));
  check_range(work.accumulator_bits.first(gate.k));
  check_range(out);
  if (trace != nullptr) {
    check_range(trace_work.first(trace_count));
  }
  return Validation{gate.k, h, input_groups, hidden_groups, trace_count};
}

void check_read_against_write(std::span<std::byte> write,
                              std::span<const std::uint16_t> input, GateMatrix gate,
                              std::span<const std::uint32_t> bias,
                              std::span<const RoutedExpert> experts,
                              const std::optional<std::span<const std::uint32_t>>& bias_vl) {
  require_disjoint(std::as_bytes(input), write);
  require_disjoint(std::as_bytes(gate.weight_bf16), write);
  require_disjoint(std::as_bytes(bias), write);
  require_disjoint(std::as_bytes(experts), write);
  if (bias_vl.has_value()) {
    require_disjoint(std::as_bytes(*bias_vl), write);
  }
  for (const RoutedExpert& expert : experts) {
    require_disjoint(std::as_bytes(expert.w1.w_packed), write);
    require_disjoint(std::as_bytes(expert.w1.scale_b), write);
    require_disjoint(std::as_bytes(expert.w2.w_packed), write);
    require_disjoint(std::as_bytes(expert.w2.scale_b), write);
    require_disjoint(std::as_bytes(expert.w3.w_packed), write);
    require_disjoint(std::as_bytes(expert.w3.scale_b), write);
  }
}

void validate_storage(std::span<const std::uint16_t> input, GateMatrix gate,
                      std::span<const std::uint32_t> bias,
                      std::span<const RoutedExpert> experts, RoutedWorkspace work,
                      std::span<std::uint32_t> out, detail::RoutedTrace* trace,
                      std::span<std::uint32_t> trace_work,
                      const std::optional<std::span<const std::uint32_t>>& bias_vl,
                      const Validation& sizes) {
  std::array<std::span<std::byte>, 15> scratch{};
  std::size_t scratch_count = 0;
  scratch[scratch_count++] = std::as_writable_bytes(work.router.linear_bits.first(gate.n));
  scratch[scratch_count++] = std::as_writable_bytes(work.router.temp_bits.first(gate.n));
  scratch[scratch_count++] = std::as_writable_bytes(work.router.softplus_bits.first(gate.n));
  scratch[scratch_count++] = std::as_writable_bytes(work.router.score_bits.first(gate.n));
  scratch[scratch_count++] = std::as_writable_bytes(work.router.selection_bits.first(gate.n));
  scratch[scratch_count++] = std::as_writable_bytes(work.expert.input_codes.first(sizes.d));
  scratch[scratch_count++] =
      std::as_writable_bytes(work.expert.input_scales.first(sizes.input_groups));
  scratch[scratch_count++] = std::as_writable_bytes(work.expert.gate.first(sizes.h));
  scratch[scratch_count++] = std::as_writable_bytes(work.expert.up.first(sizes.h));
  scratch[scratch_count++] = std::as_writable_bytes(work.expert.hidden.first(sizes.h));
  scratch[scratch_count++] = std::as_writable_bytes(work.expert.hidden_codes.first(sizes.h));
  scratch[scratch_count++] =
      std::as_writable_bytes(work.expert.hidden_scales.first(sizes.hidden_groups));
  scratch[scratch_count++] = std::as_writable_bytes(work.expert_output.first(sizes.d));
  scratch[scratch_count++] = std::as_writable_bytes(work.accumulator_bits.first(sizes.d));
  if (trace != nullptr) {
    scratch[scratch_count++] = std::as_writable_bytes(trace_work.first(sizes.trace_count));
  }

  const auto out_full = std::as_writable_bytes(out);
  const auto out_write = std::as_writable_bytes(out.first(sizes.d));
  for (std::size_t i = 0; i < scratch_count; ++i) {
    check_read_against_write(scratch[i], input, gate, bias, experts, bias_vl);
    for (std::size_t j = 0; j < i; ++j) {
      require_disjoint(scratch[j], scratch[i]);
    }
    require_disjoint(scratch[i], out_full);
  }
  check_read_against_write(out_write, input, gate, bias, experts, bias_vl);

  if (trace == nullptr) {
    return;
  }
  auto trace_object =
      std::as_writable_bytes(std::span<detail::RoutedTrace>(trace, static_cast<std::size_t>(1)));
  const auto partial_full = std::as_writable_bytes(trace->partial_sums_bits);
  const auto partial_write =
      std::as_writable_bytes(trace->partial_sums_bits.first(sizes.trace_count));
  check_read_against_write(trace_object, input, gate, bias, experts, bias_vl);
  check_read_against_write(partial_write, input, gate, bias, experts, bias_vl);
  require_disjoint(trace_object, out_full);
  require_disjoint(partial_full, out_full);
  require_disjoint(trace_object, partial_full);
  for (std::size_t i = 0; i < scratch_count; ++i) {
    require_disjoint(scratch[i], trace_object);
    require_disjoint(scratch[i], partial_full);
  }
}

void validate_contents(std::span<const std::uint16_t> input, GateMatrix gate,
                       std::span<const std::uint32_t> bias,
                       std::span<const RoutedExpert> experts,
                       const std::optional<std::span<const std::uint32_t>>& bias_vl) {
  detail::validate_router_contents(input, gate, bias, bias_vl);
  for (const RoutedExpert& expert : experts) {
    df::quant::detail::validate_expert_contents(input, expert.w1, expert.w2, expert.w3, false, 0);
  }
}

struct RoutedPair {
  std::size_t index;
  std::uint32_t weight_bits;
};

void evaluate(std::span<const std::uint16_t> input, GateMatrix gate,
              std::span<const std::uint32_t> bias, std::span<const RoutedExpert> experts,
              RoutedWorkspace work, std::span<std::uint32_t> out, detail::RoutedTrace* trace,
              std::span<std::uint32_t> trace_work, bool image_token,
              std::optional<std::span<const std::uint32_t>> bias_vl) {
  const GateMatrix gate_snapshot = gate;
  const RoutedWorkspace work_snapshot = work;
  const auto bias_vl_snapshot = bias_vl;
  const std::span<std::uint32_t> partial_snapshot =
      trace == nullptr ? std::span<std::uint32_t>{} : trace->partial_sums_bits;

  const Validation sizes = validate_metadata(input, gate_snapshot, bias, experts, work_snapshot,
                                             out, trace, trace_work, bias_vl_snapshot);
  validate_storage(input, gate_snapshot, bias, experts, work_snapshot, out, trace, trace_work,
                   bias_vl_snapshot, sizes);
  validate_contents(input, gate_snapshot, bias, experts, bias_vl_snapshot);

  const RouteResult route_result =
      route(input, gate_snapshot, bias, work_snapshot.router, image_token, bias_vl_snapshot);
  std::array<RoutedPair, kRouterTopK> pairs{};
  for (std::size_t slot = 0; slot < kRouterTopK; ++slot) {
    pairs[slot] = RoutedPair{route_result.indices[slot], route_result.weights_bits[slot]};
  }
  for (std::size_t slot = 1; slot < pairs.size(); ++slot) {
    const RoutedPair candidate = pairs[slot];
    std::size_t position = slot;
    while (position > 0 && candidate.index < pairs[position - 1].index) {
      pairs[position] = pairs[position - 1];
      --position;
    }
    pairs[position] = candidate;
  }

  const auto accumulator = work_snapshot.accumulator_bits.first(sizes.d);
  std::fill(accumulator.begin(), accumulator.end(), 0u);
  std::array<std::size_t, kRouterTopK> expert_indices{};
  for (std::size_t slot = 0; slot < kRouterTopK; ++slot) {
    const RoutedPair pair = pairs[slot];
    expert_indices[slot] = pair.index;
    const RoutedExpert expert_snapshot = experts[pair.index];
    df::quant::expert(input, expert_snapshot.w1, expert_snapshot.w2, expert_snapshot.w3,
                      pair.weight_bits, work_snapshot.expert,
                      work_snapshot.expert_output.first(sizes.d));
    const auto partial = trace == nullptr
                             ? std::span<std::uint32_t>{}
                             : trace_work.subspan(slot * sizes.d, sizes.d);
    detail::accumulate_expert_output(work_snapshot.expert_output.first(sizes.d), accumulator,
                                     partial);
  }

  if (trace != nullptr) {
    std::copy_n(trace_work.begin(), sizes.trace_count, partial_snapshot.begin());
    *trace = detail::RoutedTrace{route_result, expert_indices, partial_snapshot};
  }
  std::copy(accumulator.begin(), accumulator.end(), out.begin());
}

}  // namespace

namespace detail {

void accumulate_expert_output(std::span<const std::uint16_t> expert_output,
                              std::span<std::uint32_t> accumulator,
                              std::span<std::uint32_t> partial_sum) {
  check_range(expert_output);
  check_range(accumulator);
  check_range(partial_sum);
  const std::size_t count = expert_output.size();
  if (accumulator.size() < count || (!partial_sum.empty() && partial_sum.size() < count)) {
    throw std::invalid_argument("routed aggregation capacity is smaller than D");
  }
  const auto accumulator_write = accumulator.first(count);
  require_disjoint(std::as_bytes(expert_output), std::as_writable_bytes(accumulator_write));
  if (!partial_sum.empty()) {
    const auto partial_write = partial_sum.first(count);
    require_disjoint(std::as_bytes(expert_output), std::as_writable_bytes(partial_write));
    require_disjoint(std::as_writable_bytes(accumulator_write),
                     std::as_writable_bytes(partial_sum));
  }
  for (std::size_t column = 0; column < count; ++column) {
    accumulator_write[column] = df::quant::detail::add_fp32(
        accumulator_write[column], static_cast<std::uint32_t>(expert_output[column]) << 16);
  }
  if (!partial_sum.empty()) {
    std::copy(accumulator_write.begin(), accumulator_write.end(), partial_sum.begin());
  }
}

void trace_routed(std::span<const std::uint16_t> input, GateMatrix gate,
                  std::span<const std::uint32_t> bias,
                  std::span<const RoutedExpert> experts, RoutedWorkspace work,
                  std::span<std::uint32_t> out, RoutedTrace& trace,
                  std::span<std::uint32_t> trace_work, bool image_token,
                  std::optional<std::span<const std::uint32_t>> bias_vl) {
  evaluate(input, gate, bias, experts, work, out, &trace, trace_work, image_token, bias_vl);
}

}  // namespace detail

void routed(std::span<const std::uint16_t> input, GateMatrix gate,
            std::span<const std::uint32_t> bias, std::span<const RoutedExpert> experts,
            RoutedWorkspace work, std::span<std::uint32_t> out, bool image_token,
            std::optional<std::span<const std::uint32_t>> bias_vl) {
  evaluate(input, gate, bias, experts, work, out, nullptr, {}, image_token, bias_vl);
}

}  // namespace df::core
