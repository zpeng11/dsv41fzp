#include "df/core/router.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::atomic<bool> g_forbid_allocations{false};

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <class Function>
void rejects(Function function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "expected std::invalid_argument");
}

struct Fixture {
  static constexpr std::size_t kN = 8;
  static constexpr std::size_t kK = 1;

  std::array<std::uint16_t, kK> input{};
  std::array<std::uint16_t, kN * kK> gate{};
  std::array<std::uint32_t, kN> text_bias{};
  std::array<std::uint32_t, kN> image_bias{};
  std::array<std::uint32_t, kN + 2> linear{};
  std::array<std::uint32_t, kN + 2> temp{};
  std::array<std::uint32_t, kN + 2> softplus{};
  std::array<std::uint32_t, kN + 2> score{};
  std::array<std::uint32_t, kN + 2> selection{};

  Fixture() {
    input.fill(0);
    gate.fill(0);
    text_bias.fill(0);
    image_bias.fill(0);
    for (std::size_t i = 0; i < 6; ++i) text_bias[i] = 0x3F800000u;
    for (std::size_t i = 2; i < kN; ++i) image_bias[i] = 0x3F800000u;
    fill_work();
  }

  void fill_work() {
    linear.fill(0xA5A5A5A5u);
    temp.fill(0xA5A5A5A5u);
    softplus.fill(0xA5A5A5A5u);
    score.fill(0xA5A5A5A5u);
    selection.fill(0xA5A5A5A5u);
  }

  df::core::GateMatrix matrix() const { return {kN, kK, gate}; }
  df::core::RouterWorkspace workspace() {
    return {linear, temp, softplus, score, selection};
  }
};

void zero_gate_and_bias_selection() {
  Fixture fixture;
  const auto input_before = fixture.input;
  const auto gate_before = fixture.gate;
  const auto text_before = fixture.text_bias;
  const auto image_before = fixture.image_bias;

  g_forbid_allocations = true;
  const auto text = df::core::route(fixture.input, fixture.matrix(), fixture.text_bias,
                                    fixture.workspace(), false, fixture.image_bias);
  g_forbid_allocations = false;
  require(text.indices == std::array<std::size_t, 6>{0, 1, 2, 3, 4, 5},
          "text bias selection mismatch");
  require(text.weights_bits == std::array<std::uint32_t, 6>{
              0x3E7FFFFFu, 0x3E7FFFFFu, 0x3E7FFFFFu,
              0x3E7FFFFFu, 0x3E7FFFFFu, 0x3E7FFFFFu},
          "zero-gate sequential normalization rounding mismatch");

  fixture.fill_work();
  g_forbid_allocations = true;
  const auto image = df::core::route(fixture.input, fixture.matrix(), fixture.text_bias,
                                     fixture.workspace(), true, fixture.image_bias);
  const auto fallback = df::core::route(fixture.input, fixture.matrix(), fixture.text_bias,
                                        fixture.workspace(), true);
  g_forbid_allocations = false;
  require(image.indices == std::array<std::size_t, 6>{2, 3, 4, 5, 6, 7},
          "image bias selection mismatch");
  require(fallback.indices == text.indices && fallback.weights_bits == text.weights_bits,
          "image fallback mismatch");
  require(fixture.input == input_before && fixture.gate == gate_before &&
              fixture.text_bias == text_before && fixture.image_bias == image_before,
          "router changed a read-only input");
  require(fixture.linear[Fixture::kN] == 0xA5A5A5A5u &&
              fixture.temp[Fixture::kN] == 0xA5A5A5A5u &&
              fixture.softplus[Fixture::kN] == 0xA5A5A5A5u &&
              fixture.score[Fixture::kN] == 0xA5A5A5A5u &&
              fixture.selection[Fixture::kN] == 0xA5A5A5A5u,
          "workspace tail changed");
}

void failures_are_atomic() {
  Fixture fixture;
  const auto check = [&](auto call) {
    const auto input = fixture.input;
    const auto gate = fixture.gate;
    const auto bias = fixture.text_bias;
    const auto vl = fixture.image_bias;
    const auto linear = fixture.linear;
    const auto temp = fixture.temp;
    const auto softplus = fixture.softplus;
    const auto score = fixture.score;
    const auto selection = fixture.selection;
    rejects(call);
    require(fixture.input == input && fixture.gate == gate && fixture.text_bias == bias &&
                fixture.image_bias == vl && fixture.linear == linear && fixture.temp == temp &&
                fixture.softplus == softplus && fixture.score == score &&
                fixture.selection == selection,
            "parameter rejection changed storage");
  };

  check([&] {
    df::core::route(fixture.input, fixture.matrix(),
                    std::span<const std::uint32_t>(fixture.text_bias).first(Fixture::kN - 1),
                    fixture.workspace());
  });
  check([&] {
    auto work = fixture.workspace();
    work.score_bits = work.score_bits.first(Fixture::kN - 1);
    df::core::route(fixture.input, fixture.matrix(), fixture.text_bias, work);
  });
  check([&] {
    auto matrix = fixture.matrix();
    matrix.weight_bf16 = matrix.weight_bf16.first(Fixture::kN - 1);
    df::core::route(fixture.input, matrix, fixture.text_bias, fixture.workspace());
  });

  std::array<std::uint32_t, Fixture::kN * 2> shared{};
  shared.fill(0xA5A5A5A5u);
  const auto shared_before = shared;
  auto work = fixture.workspace();
  work.linear_bits = std::span<std::uint32_t>(shared).first(Fixture::kN);
  rejects([&] {
    df::core::route(fixture.input, fixture.matrix(),
                    std::span<const std::uint32_t>(shared).first(Fixture::kN), work);
  });
  require(shared == shared_before, "read/write alias rejection changed storage");
}

}  // namespace

void* operator new(std::size_t size) {
  if (g_forbid_allocations.load(std::memory_order_relaxed)) throw std::bad_alloc();
  if (void* pointer = std::malloc(size)) return pointer;
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

int main() {
  try {
    zero_gate_and_bias_selection();
    failures_are_atomic();
    std::cout << "P0024 public router consumer passed\n";
    return 0;
  } catch (const std::exception& error) {
    g_forbid_allocations = false;
    std::cerr << "P0024 public router consumer: " << error.what() << '\n';
    return 1;
  }
}
