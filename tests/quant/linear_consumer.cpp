#include "df/quant/linear.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <class Call>
void rejects(Call call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "expected std::invalid_argument");
}

template <class T, std::size_t N>
std::span<std::uint8_t> bytes(std::array<T, N>& storage) {
  return {reinterpret_cast<std::uint8_t*>(storage.data()), sizeof(storage)};
}

// P0019: a public-header-only consumer with independently hand-computed row results.
// Each group has 16 pairs: input (1,2), then (-2,4). Weight pairs by row are
// (1,2), (0.5,-1), (6,1.5), with scale pairs (1,2), (2,0.5), (0.5,4).
// The resulting rows are 80+192=272, -48-40=-88, and 72-384=-312.
template <class Out>
void hand_computed() {
  std::array<std::uint16_t, 64> input{};
  for (std::size_t i = 0; i < input.size(); i += 2) {
    input[i] = i < 32 ? 0x3F80u : 0xC000u;
    input[i + 1] = i < 32 ? 0x4000u : 0x4080u;
  }
  std::array<std::uint8_t, 96> weights{};
  std::fill_n(weights.begin(), 32, 0x42u);
  std::fill_n(weights.begin() + 32, 32, 0xA1u);
  std::fill_n(weights.begin() + 64, 32, 0x37u);
  const std::array<std::uint8_t, 6> scales{127, 128, 128, 126, 126, 129};
  const auto input_before = input;
  const auto weights_before = weights;
  std::array<std::uint8_t, 67> codes{};
  std::array<std::uint8_t, 5> scale_a{};
  std::array<Out, 6> out{};
  codes.fill(0xA5u);
  scale_a.fill(0xA5u);
  out.fill(static_cast<Out>(0xA5A5u));
  df::quant::expert_linear(input, 3, weights, scales, codes, scale_a, std::span<Out>(out));
  const std::array<std::uint32_t, 3> expected{0x43880000u, 0xC2B00000u, 0xC39C0000u};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto value = std::is_same_v<Out, std::uint16_t> ? expected[i] >> 16 : expected[i];
    require(out[i] == value, "hand-computed matrix result mismatch");
  }
  for (std::size_t i = 0; i < 64; ++i) {
    const std::uint8_t expected_code = i % 2 != 0 ? 0x78u : (i < 32 ? 0x70u : 0xF0u);
    require(codes[i] == expected_code, "workspace activation code mismatch");
  }
  require(scale_a[0] == 120 && scale_a[1] == 121, "workspace scale mismatch");
  require(std::all_of(codes.begin() + 64, codes.end(), [](auto x) { return x == 0xA5u; }),
          "activation workspace tail changed");
  require(std::all_of(scale_a.begin() + 2, scale_a.end(), [](auto x) { return x == 0xA5u; }),
          "scale workspace tail changed");
  require(std::all_of(out.begin() + 3, out.end(), [](auto x) { return x == 0xA5A5u; }),
          "output tail changed");
  require(input == input_before && weights == weights_before, "input changed");
}

// Exact-sized buffers cover both endpoints; these sums lie on consecutive BF16
// ties: 1+1/256 rounds down to 1, and 1+1/256+1/128 rounds up to 1+2/128.
template <class Out>
void rounding_boundaries() {
  std::array<std::uint16_t, 32> input{};
  input[0] = 0x3F80u;
  input[1] = 0x3B80u;
  input[2] = 0x3C00u;
  std::array<std::uint8_t, 32> weights{};
  weights[0] = 0x22u;
  weights[16] = 0x22u;
  weights[17] = 0x02u;
  const std::array<std::uint8_t, 2> scales{127, 127};
  std::array<std::uint8_t, 32> codes{};
  std::array<std::uint8_t, 1> scale_a{};
  std::array<Out, 2> out{};
  df::quant::expert_linear(input, 2, weights, scales, codes, scale_a, std::span<Out>(out));
  if constexpr (std::is_same_v<Out, std::uint16_t>) {
    require(out[0] == 0x3F80u && out[1] == 0x3F82u, "BF16 tie-to-even mismatch");
  } else {
    require(out[0] == 0x3F808000u && out[1] == 0x3F818000u, "exact FP32 tie input mismatch");
  }
}

// All invalid calls use genuinely allocated, bounded spans; only scalar N becomes huge.
// Every rejection checks the entire input, output and both workspaces, tails included.
template <class Out>
void failures() {
  std::array<std::uint16_t, 64> input{};
  input.fill(0x3F80u);
  std::array<std::uint8_t, 64> weights{};
  weights.fill(0x22u);
  std::array<std::uint8_t, 4> scales{127, 127, 127, 127};
  std::array<std::uint8_t, 65> codes{};
  std::array<std::uint8_t, 3> scale_a{};
  std::array<Out, 3> out{};
  codes.fill(0xA5u);
  scale_a.fill(0xA5u);
  out.fill(static_cast<Out>(0xA5A5u));
  auto check = [&](std::span<const std::uint16_t> x, std::size_t n,
                   std::span<const std::uint8_t> w, std::span<const std::uint8_t> sb,
                   std::span<std::uint8_t> a, std::span<std::uint8_t> sa, std::span<Out> y) {
    const auto input_before = input;
    const auto weights_before = weights;
    const auto scales_before = scales;
    const auto codes_before = codes;
    const auto scale_a_before = scale_a;
    const auto out_before = out;
    rejects([&] { df::quant::expert_linear(x, n, w, sb, a, sa, y); });
    require(input == input_before && weights == weights_before && scales == scales_before,
            "rejected call changed an input");
    require(codes == codes_before && scale_a == scale_a_before && out == out_before,
            "rejected call changed a destination");
  };
  const auto x = std::span<const std::uint16_t>(input);
  const auto w = std::span<const std::uint8_t>(weights);
  const auto sb = std::span<const std::uint8_t>(scales);
  const auto a = std::span<std::uint8_t>(codes);
  const auto sa = std::span<std::uint8_t>(scale_a);
  const auto y = std::span<Out>(out);
  check(x, 0, w, sb, a, sa, y);
  check(x.first(0), 2, w, sb, a, sa, y);
  check(x.first(63), 2, w, sb, a, sa, y);
  check(x, 2, w.first(63), sb, a, sa, y);
  check(x, 1, w, sb.first(2), a, sa, y);
  check(x, 2, w, sb.first(3), a, sa, y);
  check(x, 1, w.first(32), sb, a, sa, y);
  check(x, 2, w, sb, a.first(63), sa, y);
  check(x, 2, w, sb, a, sa.first(1), y);
  check(x, 2, w, sb, a, sa, y.first(1));
  const auto maximum = std::numeric_limits<std::size_t>::max();
  check(x, maximum / 32 + 1, w, sb, a, sa, y);
  check(x, maximum / sizeof(Out) + 1, w, sb, a, sa, y);
  check(x, maximum / 2 + 1, w, sb, a, sa, y);
  for (const auto nonfinite : {0x7F80u, 0xFF80u, 0x7FC1u, 0xFFFFu}) {
    input.back() = static_cast<std::uint16_t>(nonfinite);
    check(x, 2, w, sb, a, sa, y);
  }
  input.back() = 0x3F80u;
  scales.back() = 0xFFu;
  check(x, 2, w, sb, a, sa, y);
  scales.back() = 127;
  check(x, 2, w, sb, a, a.first(2), y);
}

// Exercise every writable/read-only pairing that can share byte storage, and all
// writable pairs, with real allocations. BF16 also directly aliases the input.
template <class Out>
void overlap_failures() {
  std::array<std::uint16_t, 64> input{};
  input.fill(0x3F80u);
  std::array<std::uint8_t, 64> weights{};
  weights.fill(0x22u);
  std::array<std::uint8_t, 4> scales{127, 127, 127, 127};
  std::array<std::uint8_t, 68> workspace{};
  std::array<Out, 32> out{};
  workspace.fill(0xA5u);
  out.fill(static_cast<Out>(0xA5A5u));
  const auto a = std::span<std::uint8_t>(workspace).first(64);
  const auto sa = std::span<std::uint8_t>(workspace).last(4);
  auto check = [&](std::span<const std::uint8_t> w, std::span<const std::uint8_t> sb,
                   std::span<std::uint8_t> codes, std::span<std::uint8_t> scale_a) {
    const auto input_before = input;
    const auto weights_before = weights;
    const auto scales_before = scales;
    const auto workspace_before = workspace;
    const auto out_before = out;
    rejects([&] {
      df::quant::expert_linear(input, 2, w, sb, codes, scale_a, std::span<Out>(out));
    });
    require(input == input_before && weights == weights_before && scales == scales_before &&
                workspace == workspace_before && out == out_before,
            "overlap rejection changed storage");
  };
  check(weights, scales, bytes(input).first(64), sa);
  check(weights, scales, a, bytes(input).last(2));
  check(weights, scales, weights, sa);
  check(weights, scales, a, std::span<std::uint8_t>(weights).last(2));
  check(weights, a.first(4), a, sa);
  check(weights, scales, a, scales);
  check(bytes(out).first(64), scales, a, sa);
  check(weights, bytes(out).first(4), a, sa);
  check(weights, scales, bytes(out).first(64), sa);
  check(weights, scales, a, bytes(out).first(2));
  check(weights, scales, a, std::span<std::uint8_t>(workspace).subspan(63, 2));
  if constexpr (std::is_same_v<Out, std::uint16_t>) {
    const auto input_before = input;
    const auto workspace_before = workspace;
    rejects([&] {
      df::quant::expert_linear(input, 2, weights, scales, a, sa,
                               std::span<std::uint16_t>(input).last(2));
    });
    require(input == input_before && workspace == workspace_before, "input/output alias wrote");
  }
}

// Effective ranges, not full capacities, determine aliases. Input, weights and
// scale_b may share storage, and each destination may carry unused overlapping tails.
template <class Out>
void allowed_aliases() {
  std::array<std::uint16_t, 64> input_and_output{};
  input_and_output.fill(0);
  const auto input = std::span<const std::uint16_t>(input_and_output).last(32);
  const auto read_bytes = std::span<const std::uint8_t>(bytes(input_and_output)).subspan(64);
  std::array<std::uint8_t, 64> workspace{};
  workspace.fill(0xA5u);
  std::array<Out, 2> out{};
  out.fill(static_cast<Out>(0xA5A5u));
  df::quant::expert_linear(input, 1, read_bytes.first(16), read_bytes.first(1), workspace,
                           std::span<std::uint8_t>(workspace).subspan(32), std::span<Out>(out));
  require(out[0] == 0 && out[1] == 0xA5A5u, "read-only or workspace-tail alias failed");
  require(std::all_of(workspace.begin() + 33, workspace.end(), [](auto x) { return x == 0xA5u; }),
          "shared workspace tail changed");
  require(std::all_of(input_and_output.begin(), input_and_output.end(), [](auto x) { return x == 0; }),
          "shared read-only storage changed");
  workspace.fill(0xA5u);
  df::quant::expert_linear(input, 1, read_bytes.first(16), read_bytes.first(1),
                           std::span<std::uint8_t>(workspace).subspan(1), workspace,
                           std::span<Out>(out));
  require(out[0] == 0 && out[1] == 0xA5A5u, "scale workspace tail alias failed");
  require(std::all_of(workspace.begin() + 33, workspace.end(), [](auto x) { return x == 0xA5u; }),
          "scale workspace unused tail changed");
  if constexpr (std::is_same_v<Out, std::uint16_t>) {
    df::quant::expert_linear(input, 1, read_bytes.first(16), read_bytes.first(1), workspace,
                             std::span<std::uint8_t>(workspace).subspan(32),
                             std::span<std::uint16_t>(input_and_output));
    require(std::all_of(input_and_output.begin(), input_and_output.end(), [](auto x) { return x == 0; }),
            "output tail overlapping inputs changed");
  }
}

}  // namespace

int main() {
  try {
    hand_computed<std::uint32_t>();
    hand_computed<std::uint16_t>();
    rounding_boundaries<std::uint32_t>();
    rounding_boundaries<std::uint16_t>();
    failures<std::uint32_t>();
    failures<std::uint16_t>();
    overlap_failures<std::uint32_t>();
    overlap_failures<std::uint16_t>();
    allowed_aliases<std::uint32_t>();
    allowed_aliases<std::uint16_t>();
    std::cout << "P0019 public linear consumer passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "P0019 public linear consumer: " << error.what() << '\n';
    return 1;
  }
}
