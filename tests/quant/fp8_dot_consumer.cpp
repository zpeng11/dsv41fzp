// P0028 public-header consumer for the FP8 grouped-dot scalar.
#include "df/quant/compute.h"

#include <algorithm>
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

std::atomic<std::size_t> g_allocation_count{0};
std::atomic<bool> g_count_allocations{false};

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
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

template <class Call>
std::size_t count_allocations(Call call) {
  g_allocation_count.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_relaxed);
  try {
    call();
  } catch (...) {
    g_count_allocations.store(false, std::memory_order_relaxed);
    throw;
  }
  g_count_allocations.store(false, std::memory_order_relaxed);
  return g_allocation_count.load(std::memory_order_relaxed);
}

void ordinary_calls() {
  std::array<std::uint8_t, 32> a{};
  std::array<std::uint8_t, 32> w{};
  a[0] = 0x38u;
  w[0] = 0x40u;
  const std::array<std::uint8_t, 1> sa{0x7Fu};
  const std::array<std::uint8_t, 1> sb{0x7Fu};
  std::array<std::uint32_t, 3> fp32{0xA5A5A5A5u, 0xA5A5A5A5u, 0xA5A5A5A5u};
  std::array<std::uint16_t, 3> bf16{0xA5A5u, 0xA5A5u, 0xA5A5u};
  const auto a_before = a;
  const auto w_before = w;

  const std::size_t fp32_allocations = count_allocations(
      [&] { df::quant::fp8_grouped_dot(a, w, sa, sb, fp32); });
  const std::size_t bf16_allocations = count_allocations(
      [&] { df::quant::fp8_grouped_dot(a, w, sa, sb, bf16); });
  require(fp32_allocations == 0 && bf16_allocations == 0,
          "successful public calls allocated storage");
  require(fp32[0] == 0x40000000u && bf16[0] == 0x4000u, "public output mismatch");
  require(fp32[1] == 0xA5A5A5A5u && fp32[2] == 0xA5A5A5A5u &&
              bf16[1] == 0xA5A5u && bf16[2] == 0xA5A5u,
          "public output tail changed");
  require(a == a_before && w == w_before, "public input changed");
}

void overlapping_reads_and_output_tail() {
  std::array<std::uint8_t, 32> codes{};
  codes.fill(0x38u);
  const std::array<std::uint8_t, 1> scale{0x7Fu};
  std::array<std::uint32_t, 3> fp32{0xA5A5A5A5u, 0xA5A5A5A5u, 0xA5A5A5A5u};
  std::array<std::uint16_t, 3> bf16{0xA5A5u, 0xA5A5u, 0xA5A5u};
  df::quant::fp8_grouped_dot(codes, codes, scale, scale, fp32);
  df::quant::fp8_grouped_dot(codes, codes, scale, scale, bf16);
  require(fp32[0] == 0x42000000u && bf16[0] == 0x4200u, "overlapping read result mismatch");
  require(fp32[1] == 0xA5A5A5A5u && fp32[2] == 0xA5A5A5A5u &&
              bf16[1] == 0xA5A5u && bf16[2] == 0xA5A5u,
          "unused output tail changed");

  std::array<std::uint32_t, 10> shared32{};
  auto bytes32 = std::as_writable_bytes(std::span(shared32));
  std::fill(bytes32.begin() + 4, bytes32.begin() + 36, std::byte{0x38});
  const auto input32 = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(bytes32.data() + 4), 32);
  std::array<std::uint8_t, 32> input32_before{};
  std::copy(input32.begin(), input32.end(), input32_before.begin());
  const auto output32 = std::span<std::uint32_t>(shared32).first<3>();
  df::quant::fp8_grouped_dot(input32, codes, scale, scale, output32);
  require(output32[0] == 0x42000000u, "FP32 overlapping-tail result mismatch");
  require(std::equal(input32.begin(), input32.end(), input32_before.begin()),
          "FP32 unused output tail overlapping input changed");

  std::array<std::uint16_t, 18> shared16{};
  auto bytes16 = std::as_writable_bytes(std::span(shared16));
  std::fill(bytes16.begin() + 2, bytes16.begin() + 34, std::byte{0x38});
  const auto input16 = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(bytes16.data() + 2), 32);
  std::array<std::uint8_t, 32> input16_before{};
  std::copy(input16.begin(), input16.end(), input16_before.begin());
  const auto output16 = std::span<std::uint16_t>(shared16).first<3>();
  df::quant::fp8_grouped_dot(input16, codes, scale, scale, output16);
  require(output16[0] == 0x4200u, "BF16 overlapping-tail result mismatch");
  require(std::equal(input16.begin(), input16.end(), input16_before.begin()),
          "BF16 unused output tail overlapping input changed");
}

void rejection_is_atomic() {
  std::array<std::uint8_t, 32> a{};
  std::array<std::uint8_t, 32> w{};
  std::array<std::uint8_t, 1> sa{0x7Fu};
  std::array<std::uint8_t, 1> sb{0x7Fu};
  std::array<std::uint32_t, 2> out{0xA5A5A5A5u, 0xA5A5A5A5u};
  const auto out_before = out;

  rejects([&] { df::quant::fp8_grouped_dot(a, std::span(w).first<31>(), sa, sb, out); });
  require(out == out_before, "length rejection changed output");
  w.back() = 0x7Fu;
  rejects([&] { df::quant::fp8_grouped_dot(a, w, sa, sb, out); });
  require(out == out_before, "content rejection changed output");
  w.back() = 0;
  std::array<std::uint32_t, 8> backing{};
  auto backing_bytes = std::as_writable_bytes(std::span(backing));
  std::fill(backing_bytes.begin(), backing_bytes.end(), std::byte{0x38});
  const auto backing_before = backing;
  const auto aliased_a = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(backing.data()), 32);
  rejects([&] {
    df::quant::fp8_grouped_dot(aliased_a, w, sa, sb, std::span(backing).first<1>());
  });
  require(backing == backing_before, "overlap rejection changed aliased backing");
}

}  // namespace

void* operator new(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed))
    g_allocation_count.fetch_add(1, std::memory_order_relaxed);
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
    ordinary_calls();
    overlapping_reads_and_output_tail();
    rejection_is_atomic();
    std::cout << "P0028 public FP8 grouped-dot consumer passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "P0028 public FP8 grouped-dot consumer: " << error.what() << '\n';
    return 1;
  }
}
