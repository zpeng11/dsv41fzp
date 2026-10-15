#include "df/quant/swiglu.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

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

void ordinary_and_weighted() {
  const std::array<std::uint16_t, 7> gate{0x0000u, 0x8000u, 0x3F80u, 0xBF80u,
                                          0x7F80u, 0xFF80u, 0x7FC1u};
  const std::array<std::uint16_t, 7> up{0x3F80u, 0x3F80u, 0x3F80u, 0x3F80u,
                                        0x0000u, 0x3F80u, 0x3F80u};
  const auto gate_before = gate;
  const auto up_before = up;
  std::array<std::uint16_t, 9> out{};
  out.fill(0xA5A5u);
  df::quant::swiglu(gate, up, out);
  const std::array<std::uint16_t, 7> expected{0x0000u, 0x8000u, 0x3F3Bu, 0xBE8Au,
                                              0x0000u, 0x7FC0u, 0x7FC0u};
  require(std::equal(expected.begin(), expected.end(), out.begin()), "ordinary output mismatch");
  require(out[7] == 0xA5A5u && out[8] == 0xA5A5u, "ordinary output tail changed");
  require(gate == gate_before && up == up_before, "ordinary input changed");

  const std::array<std::uint16_t, 4> weighted_gate{0x3F80u, 0xBF80u, 0x4121u, 0xC121u};
  const std::array<std::uint16_t, 4> weighted_up{0x4000u, 0x4000u, 0x4121u, 0xC121u};
  std::array<std::uint16_t, 4> weighted_out{};
  df::quant::swiglu(weighted_gate, weighted_up, 0x3F000001u, weighted_out);
  require(weighted_out == std::array<std::uint16_t, 4>{0x3F3Bu, 0xBE8Au, 0x4248u, 0x3B0Du},
          "weighted output mismatch");
}

void empty_and_failures() {
  std::array<std::uint16_t, 4> gate{0x3F80u, 0xBF80u, 0x4121u, 0xC121u};
  std::array<std::uint16_t, 4> up{0x4000u, 0x4000u, 0x4121u, 0xC121u};
  std::array<std::uint16_t, 6> out{};
  out.fill(0xA5A5u);
  const auto gate_before = gate;
  const auto up_before = up;
  const auto out_before = out;
  df::quant::swiglu(std::span<const std::uint16_t>{}, std::span<const std::uint16_t>{},
                    std::span<std::uint16_t>{});
  rejects([&] {
    df::quant::swiglu(std::span<const std::uint16_t>{}, std::span<const std::uint16_t>{},
                      0x7F800000u, std::span<std::uint16_t>{});
  });

  const auto check = [&](auto call) {
    rejects(call);
    require(gate == gate_before && up == up_before && out == out_before,
            "rejected call changed storage");
  };
  check([&] { df::quant::swiglu(gate, std::span<const std::uint16_t>(up).first(3), out); });
  check([&] { df::quant::swiglu(gate, up, std::span<std::uint16_t>(out).first(3)); });
  for (const auto weight : {0x7F800000u, 0xFF800000u, 0x7FC00001u}) {
    check([&] { df::quant::swiglu(gate, up, weight, out); });
  }
  check([&] { df::quant::swiglu(gate, up, std::span<std::uint16_t>(gate)); });
  check([&] { df::quant::swiglu(gate, up, std::span<std::uint16_t>(up)); });
}

void allowed_read_overlap() {
  std::array<std::uint16_t, 5> input{0x3F80u, 0x4000u, 0xBF80u, 0xC000u, 0x4121u};
  std::array<std::uint16_t, 3> out{};
  df::quant::swiglu(std::span<const std::uint16_t>(input).first(3),
                    std::span<const std::uint16_t>(input).subspan(1, 3), out);
  require(out == std::array<std::uint16_t, 3>{0x3FBBu, 0xBFE1u, 0x3F0Au},
          "read-only overlap mismatch");
}

}  // namespace

int main() {
  try {
    ordinary_and_weighted();
    empty_and_failures();
    allowed_read_overlap();
    std::cout << "P0021 public SwiGLU consumer passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "P0021 public SwiGLU consumer: " << error.what() << '\n';
    return 1;
  }
}
