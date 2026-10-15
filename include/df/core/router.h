#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

// Single-token mainline Gate under docs/ROUTER.md (P0024, D0019). Storage is
// caller-owned; descriptors are copied by value. Metadata and all supplied input
// contents are checked before workspace writes. Parameter rejection changes no
// storage; numeric or certification failure may change workspace prefixes but
// never returns a partial result. Valid calls do not alter the floating environment.
namespace df::core {

inline constexpr std::size_t kRouterTopK = 6;

struct GateMatrix {
  std::size_t n;
  std::size_t k;
  std::span<const std::uint16_t> weight_bf16;
};

struct RouterWorkspace {
  std::span<std::uint32_t> linear_bits;
  std::span<std::uint32_t> temp_bits;
  std::span<std::uint32_t> softplus_bits;
  std::span<std::uint32_t> score_bits;
  std::span<std::uint32_t> selection_bits;
};

struct RouteResult {
  std::array<std::size_t, kRouterTopK> indices;
  std::array<std::uint32_t, kRouterTopK> weights_bits;
};

RouteResult route(std::span<const std::uint16_t> input, GateMatrix gate,
                  std::span<const std::uint32_t> bias, RouterWorkspace work,
                  bool image_token = false,
                  std::optional<std::span<const std::uint32_t>> bias_vl = std::nullopt);

}  // namespace df::core
