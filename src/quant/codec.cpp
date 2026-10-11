#include "df/quant/codec.h"

#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace df::quant {
namespace {

constexpr std::uint32_t kFp32Sign = 0x80000000u;
constexpr std::uint32_t kFp32QuietNan = 0x7FC00000u;

std::size_t element_count(Shape shape) {
  if (shape.empty()) {
    throw std::invalid_argument("shape must have at least one dimension");
  }
  for (std::size_t dim : shape) {
    if (dim == 0) {
      return 0;
    }
  }
  std::size_t count = 1;
  for (std::size_t dim : shape) {
    if (count > std::numeric_limits<std::size_t>::max() / dim) {
      throw std::overflow_error("shape element count overflows size_t");
    }
    count *= dim;
  }
  return count;
}

// Checks the parts that will actually be read and written; empty ranges never overlap.
template <class In, class Out>
void require_disjoint(std::span<In> in, std::size_t in_count, std::span<Out> out, std::size_t out_count) {
  if (in_count == 0 || out_count == 0) {
    return;
  }
  const auto in_begin = reinterpret_cast<std::uintptr_t>(in.data());
  const auto out_begin = reinterpret_cast<std::uintptr_t>(out.data());
  const auto in_end = in_begin + in_count * sizeof(In);
  const auto out_end = out_begin + out_count * sizeof(Out);
  if (in_begin < out_end && out_begin < in_end) {
    throw std::invalid_argument("input and output buffers overlap");
  }
}

// Shared metadata checks for element-wise maps: exact input length, sufficient output.
template <class In, class Out>
std::size_t check_elementwise(std::span<In> in, Shape shape, std::span<Out> out) {
  const std::size_t count = element_count(shape);
  if (in.size() != count) {
    throw std::invalid_argument("input length does not match shape");
  }
  if (out.size() < count) {
    throw std::invalid_argument("output capacity is smaller than the result");
  }
  require_disjoint(in, count, out, count);
  return count;
}

std::uint32_t e4m3fn_bits(std::uint8_t code) {
  const std::uint32_t sign = (code & 0x80u) ? kFp32Sign : 0u;
  const std::uint32_t exponent = (code >> 3) & 0xFu;
  const std::uint32_t mantissa = code & 0x7u;
  if (exponent == 0xFu && mantissa == 0x7u) {
    return sign | kFp32QuietNan;
  }
  if (exponent != 0) {
    return sign | ((exponent + 120u) << 23) | (mantissa << 20);
  }
  if (mantissa == 0) {
    return sign;
  }
  // Subnormal m * 2^-9: renormalize around the leading set bit of m.
  const auto lead = static_cast<std::uint32_t>(std::bit_width(mantissa) - 1);
  return sign | ((lead + 118u) << 23) | ((mantissa - (1u << lead)) << (23 - lead));
}

}  // namespace

void decode_e4m3fn(std::span<const std::uint8_t> codes, Shape shape, std::span<std::uint32_t> out) {
  const std::size_t count = check_elementwise(codes, shape, out);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = e4m3fn_bits(codes[i]);
  }
}

}  // namespace df::quant
