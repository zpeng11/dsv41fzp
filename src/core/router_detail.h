#pragma once

#include "df/core/router.h"

namespace df::core::detail {

struct RouterTrace {
  std::array<std::uint32_t, kRouterTopK> selected_bits;
  std::uint32_t sum_bits;
  std::uint32_t denominator_bits;
  std::array<std::uint32_t, kRouterTopK> normalized_bits;
};

RouteResult trace_route(std::span<const std::uint16_t> input, GateMatrix gate,
                        std::span<const std::uint32_t> bias, RouterWorkspace work,
                        RouterTrace& trace, bool image_token = false,
                        std::optional<std::span<const std::uint32_t>> bias_vl = std::nullopt,
                        unsigned certification_bits = 12288);

std::uint32_t divide_fp32(std::uint32_t numerator, std::uint32_t denominator);
std::uint32_t sqrt_fp32(std::uint32_t bits);
std::uint32_t softplus_fp32(std::uint32_t bits, unsigned max_precision = 12288);

struct SoftplusBounds {
  std::uint32_t lower_bits;
  std::uint32_t upper_bits;
};
SoftplusBounds softplus_bounds(std::uint32_t bits, unsigned precision);

void check_router_counts(std::size_t n, std::size_t k);
void check_router_storage(std::span<const std::span<const std::byte>> reads,
                          std::span<const std::span<std::byte>> writes);
void validate_router_parameters(
    std::span<const std::uint16_t> input, GateMatrix gate,
    std::span<const std::uint32_t> bias, RouterWorkspace work,
    const std::optional<std::span<const std::uint32_t>>& bias_vl = std::nullopt);
void validate_router_contents(
    std::span<const std::uint16_t> input, GateMatrix gate,
    std::span<const std::uint32_t> bias,
    const std::optional<std::span<const std::uint32_t>>& bias_vl = std::nullopt);

}  // namespace df::core::detail
