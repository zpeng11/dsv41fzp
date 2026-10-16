#pragma once

#include "df/core/moe.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace df::core::detail {

struct RoutedTrace {
  RouteResult route;
  std::array<std::size_t, kRouterTopK> expert_indices;
  std::span<std::uint32_t> partial_sums_bits;
};

// The same aggregation step used by routed: widen one BF16 expert row and add
// it to the FP32 accumulator in increasing element order. Each nonempty writable
// span holds at least expert_output.size(); only that prefix is written.
void accumulate_expert_output(std::span<const std::uint16_t> expert_output,
                              std::span<std::uint32_t> accumulator,
                              std::span<std::uint32_t> partial_sum = {});

void trace_routed(std::span<const std::uint16_t> input, GateMatrix gate,
                  std::span<const std::uint32_t> bias,
                  std::span<const RoutedExpert> experts, RoutedWorkspace work,
                  std::span<std::uint32_t> out, RoutedTrace& trace,
                  std::span<std::uint32_t> trace_work, bool image_token = false,
                  std::optional<std::span<const std::uint32_t>> bias_vl = std::nullopt);

}  // namespace df::core::detail
