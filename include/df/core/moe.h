#pragma once

#include "df/core/router.h"
#include "df/quant/expert.h"

#include <cstdint>
#include <optional>
#include <span>

// Single-token routed branch under docs/MOE.md section 5 (P0026), combining
// the production router and FP4 experts. All caller-owned parameters are validated
// before numeric writes. Workspace prefixes may change after a runtime numeric
// failure, but out is committed only after every selected expert succeeds. The
// numeric path allocates no storage.
namespace df::core {

struct RoutedExpert {
  df::quant::ExpertMatrix w1;
  df::quant::ExpertMatrix w2;
  df::quant::ExpertMatrix w3;
};

struct RoutedWorkspace {
  RouterWorkspace router;
  df::quant::ExpertWorkspace expert;
  std::span<std::uint16_t> expert_output;
  std::span<std::uint32_t> accumulator_bits;
};

void routed(std::span<const std::uint16_t> input, GateMatrix gate,
            std::span<const std::uint32_t> bias, std::span<const RoutedExpert> experts,
            RoutedWorkspace work, std::span<std::uint32_t> out, bool image_token = false,
            std::optional<std::span<const std::uint32_t>> bias_vl = std::nullopt);

}  // namespace df::core
