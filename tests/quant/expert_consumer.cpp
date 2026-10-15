#include "df/quant/expert.h"

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

struct Fixture {
  static constexpr std::size_t kD = 32;
  static constexpr std::size_t kH = 32;

  std::array<std::uint16_t, kD> input{};
  std::array<std::uint8_t, kH * kD / 2> w1{};
  std::array<std::uint8_t, kD * kH / 2> w2{};
  std::array<std::uint8_t, kH * kD / 2> w3{};
  std::array<std::uint8_t, kH * kD / 32> s1{};
  std::array<std::uint8_t, kD * kH / 32> s2{};
  std::array<std::uint8_t, kH * kD / 32> s3{};
  std::array<std::uint8_t, kD + 3> input_codes{};
  std::array<std::uint8_t, kD / 32 + 2> input_scales{};
  std::array<std::uint16_t, kH + 2> gate{};
  std::array<std::uint16_t, kH + 2> up{};
  std::array<std::uint16_t, kH + 2> hidden{};
  std::array<std::uint8_t, kH + 3> hidden_codes{};
  std::array<std::uint8_t, kH / 32 + 2> hidden_scales{};
  std::array<std::uint16_t, kD + 2> out{};

  Fixture() {
    input.fill(0x3F80u);
    s1.fill(127);
    s2.fill(127);
    s3.fill(127);
    for (std::size_t row = 0; row < kH; ++row) {
      w1[row * (kD / 2)] = 0x02u;
      w3[row * (kD / 2)] = 0x04u;
    }
    for (std::size_t row = 0; row < kD; ++row) {
      w2[row * (kH / 2)] = 0x02u;
    }
    fill_destinations();
  }

  void fill_destinations() {
    input_codes.fill(0xA5u);
    input_scales.fill(0xA5u);
    gate.fill(0xA5A5u);
    up.fill(0xA5A5u);
    hidden.fill(0xA5A5u);
    hidden_codes.fill(0xA5u);
    hidden_scales.fill(0xA5u);
    out.fill(0xA5A5u);
  }

  df::quant::ExpertMatrix matrix1() const { return {kH, kD, w1, s1}; }
  df::quant::ExpertMatrix matrix2() const { return {kD, kH, w2, s2}; }
  df::quant::ExpertMatrix matrix3() const { return {kH, kD, w3, s3}; }
  df::quant::ExpertWorkspace workspace() {
    return {input_codes, input_scales, gate, up, hidden, hidden_codes, hidden_scales};
  }
};

void ordinary_and_weighted() {
  Fixture f;
  const auto input_before = f.input;
  const auto w1_before = f.w1;
  const auto w2_before = f.w2;
  const auto w3_before = f.w3;
  df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), f.workspace(), f.out);
  require(std::all_of(f.out.begin(), f.out.begin() + Fixture::kD,
                      [](auto value) { return value == 0x3FC0u; }),
          "ordinary expert output mismatch");
  require(std::all_of(f.gate.begin(), f.gate.begin() + Fixture::kH,
                      [](auto value) { return value == 0x3F80u; }),
          "gate boundary mismatch");
  require(std::all_of(f.up.begin(), f.up.begin() + Fixture::kH,
                      [](auto value) { return value == 0x4000u; }),
          "up boundary mismatch");
  require(std::all_of(f.hidden.begin(), f.hidden.begin() + Fixture::kH,
                      [](auto value) { return value == 0x3FBBu; }),
          "hidden boundary mismatch");
  require(std::all_of(f.input_codes.begin(), f.input_codes.begin() + Fixture::kD,
                      [](auto value) { return value == 0x78u; }) &&
              f.input_scales[0] == 119,
          "input quantization boundary mismatch");
  require(std::all_of(f.hidden_codes.begin(), f.hidden_codes.begin() + Fixture::kH,
                      [](auto value) { return value == 0x7Cu; }) &&
              f.hidden_scales[0] == 119,
          "hidden quantization boundary mismatch");
  require(f.out[Fixture::kD] == 0xA5A5u && f.out[Fixture::kD + 1] == 0xA5A5u,
          "output tail changed");
  require(f.input_codes[Fixture::kD] == 0xA5u && f.input_scales[1] == 0xA5u &&
              f.gate[Fixture::kH] == 0xA5A5u && f.up[Fixture::kH] == 0xA5A5u &&
              f.hidden[Fixture::kH] == 0xA5A5u && f.hidden_codes[Fixture::kH] == 0xA5u &&
              f.hidden_scales[1] == 0xA5u,
          "workspace tail changed");
  require(f.input == input_before && f.w1 == w1_before && f.w2 == w2_before && f.w3 == w3_before,
          "expert input changed");

  f.fill_destinations();
  df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), 0x3F000000u,
                    f.workspace(), f.out);
  require(std::all_of(f.hidden.begin(), f.hidden.begin() + Fixture::kH,
                      [](auto value) { return value == 0x3F3Bu; }) &&
              std::all_of(f.out.begin(), f.out.begin() + Fixture::kD,
                          [](auto value) { return value == 0x3F40u; }),
          "weighted expert result mismatch");
}

void dimensions_and_content_failures() {
  Fixture f;
  const auto check = [&](auto call) {
    const auto input_before = f.input;
    const auto w1_before = f.w1;
    const auto w2_before = f.w2;
    const auto w3_before = f.w3;
    const auto s1_before = f.s1;
    const auto s2_before = f.s2;
    const auto s3_before = f.s3;
    const auto input_codes_before = f.input_codes;
    const auto input_scales_before = f.input_scales;
    const auto gate_before = f.gate;
    const auto up_before = f.up;
    const auto hidden_before = f.hidden;
    const auto hidden_codes_before = f.hidden_codes;
    const auto hidden_scales_before = f.hidden_scales;
    const auto out_before = f.out;
    rejects(call);
    require(f.input == input_before && f.w1 == w1_before && f.w2 == w2_before &&
                f.w3 == w3_before && f.s1 == s1_before && f.s2 == s2_before &&
                f.s3 == s3_before && f.input_codes == input_codes_before &&
                f.input_scales == input_scales_before && f.gate == gate_before &&
                f.up == up_before && f.hidden == hidden_before &&
                f.hidden_codes == hidden_codes_before && f.hidden_scales == hidden_scales_before &&
                f.out == out_before,
            "parameter rejection changed storage");
  };

  check([&] {
    auto bad = f.matrix1();
    bad.n = 0;
    df::quant::expert(f.input, bad, f.matrix2(), f.matrix3(), f.workspace(), f.out);
  });
  check([&] {
    auto bad = f.matrix2();
    bad.k = 64;
    df::quant::expert(f.input, f.matrix1(), bad, f.matrix3(), f.workspace(), f.out);
  });
  check([&] {
    auto bad = f.matrix3();
    bad.w_packed = bad.w_packed.first(bad.w_packed.size() - 1);
    df::quant::expert(f.input, f.matrix1(), f.matrix2(), bad, f.workspace(), f.out);
  });
  check([&] {
    auto work = f.workspace();
    work.hidden_codes = work.hidden_codes.first(Fixture::kH - 1);
    df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), work, f.out);
  });
  check([&] {
    df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), f.workspace(),
                      std::span<std::uint16_t>(f.out).first(Fixture::kD - 1));
  });
  check([&] {
    df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), 0x7F800000u,
                      f.workspace(), f.out);
  });
  f.input.back() = 0x7F80u;
  check([&] { df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), f.workspace(), f.out); });
  f.input.back() = 0x3F80u;
  f.s2.back() = 0xFFu;
  check([&] { df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), f.workspace(), f.out); });
}

void storage_boundaries() {
  Fixture f;
  const auto check = [&](df::quant::ExpertWorkspace work, std::span<std::uint16_t> out) {
    const auto input_codes_before = f.input_codes;
    const auto input_scales_before = f.input_scales;
    const auto gate_before = f.gate;
    const auto up_before = f.up;
    const auto hidden_before = f.hidden;
    const auto hidden_codes_before = f.hidden_codes;
    const auto hidden_scales_before = f.hidden_scales;
    const auto out_before = f.out;
    rejects([&] { df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), work, out); });
    require(f.input_codes == input_codes_before && f.input_scales == input_scales_before &&
                f.gate == gate_before && f.up == up_before && f.hidden == hidden_before &&
                f.hidden_codes == hidden_codes_before && f.hidden_scales == hidden_scales_before &&
                f.out == out_before,
            "overlap rejection changed destinations");
  };

  auto work = f.workspace();
  work.input_scales = std::span<std::uint8_t>(f.input_codes).subspan(Fixture::kD - 1, 1);
  check(work, f.out);
  {
    std::array<std::uint16_t, Fixture::kH * 2 - 1> shared{};
    shared.fill(0xA5A5u);
    const auto shared_before = shared;
    work = f.workspace();
    work.gate = std::span<std::uint16_t>(shared).first(Fixture::kH);
    work.hidden = std::span<std::uint16_t>(shared).subspan(Fixture::kH - 1, Fixture::kH);
    check(work, f.out);
    require(shared == shared_before, "overlapping gate/hidden storage changed");
  }
  {
    std::array<std::uint16_t, Fixture::kD + Fixture::kH> shared{};
    shared.fill(0xA5A5u);
    const auto shared_before = shared;
    work = f.workspace();
    work.gate = std::span<std::uint16_t>(shared).subspan(Fixture::kD, Fixture::kH);
    check(work, std::span<std::uint16_t>(shared).first(Fixture::kD + 1));
    require(shared == shared_before, "output-tail/scratch overlap changed storage");
  }

  f.fill_destinations();
  std::array<std::uint16_t, Fixture::kD + Fixture::kH + 2> shared_out{};
  shared_out.fill(0xA5A5u);
  const auto output = std::span<std::uint16_t>(shared_out).first(Fixture::kD + 2);
  auto tail_workspace = f.workspace();
  tail_workspace.gate = std::span<std::uint16_t>(shared_out).subspan(Fixture::kD + 2,
                                                                      Fixture::kH);
  df::quant::expert(f.input, f.matrix1(), f.matrix2(), f.matrix3(), tail_workspace, output);
  require(shared_out[Fixture::kD] == 0xA5A5u && shared_out[Fixture::kD + 1] == 0xA5A5u,
          "adjacent full-output/workspace ranges mishandled");
}

}  // namespace

int main() {
  try {
    ordinary_and_weighted();
    dimensions_and_content_failures();
    storage_boundaries();
    std::cout << "P0022 public expert consumer passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "P0022 public expert consumer: " << error.what() << '\n';
    return 1;
  }
}
