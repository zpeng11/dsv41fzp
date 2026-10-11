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

// Pack/unpack metadata: even last dimension, exact input and sufficient output lengths.
// Returns the logical nibble count.
template <class In, class Out>
std::size_t check_packing(std::span<In> in, Shape logical_shape, std::span<Out> out, bool packing) {
  const std::size_t count = element_count(logical_shape);
  if (logical_shape.back() % 2 != 0) {
    throw std::invalid_argument("the last logical dimension must be even");
  }
  const std::size_t in_need = packing ? count : count / 2;
  const std::size_t out_need = packing ? count / 2 : count;
  if (in.size() != in_need) {
    throw std::invalid_argument("input length does not match logical shape");
  }
  if (out.size() < out_need) {
    throw std::invalid_argument("output capacity is smaller than the result");
  }
  require_disjoint(in, in_need, out, out_need);
  return count;
}

void require_nibbles(std::span<const std::uint8_t> nibbles) {
  for (std::uint8_t code : nibbles) {
    if (code > 0xFu) {
      throw std::invalid_argument("E2M1 code does not fit in a nibble");
    }
  }
}

std::uint32_t e2m1_bits(std::uint8_t code) {
  const std::uint32_t sign = (code & 0x8u) ? kFp32Sign : 0u;
  const std::uint32_t exponent = (code >> 1) & 0x3u;
  const std::uint32_t mantissa = code & 0x1u;
  if (exponent != 0) {
    return sign | ((exponent + 126u) << 23) | (mantissa << 22);
  }
  return mantissa ? sign | (126u << 23) : sign;
}

std::int8_t e2m1_grid(std::uint8_t code) {
  const int exponent = (code >> 1) & 0x3;
  const int mantissa = code & 0x1;
  const int magnitude = exponent == 0 ? mantissa : (2 + mantissa) << (exponent - 1);
  return static_cast<std::int8_t>((code & 0x8u) ? -magnitude : magnitude);
}

// Returns the non-negative E2M1 code for a grid magnitude, or -1 when off the grid.
int grid_magnitude_code(int magnitude) {
  if (magnitude < 0 || magnitude > 12) {
    return -1;
  }
  if (magnitude < 2) {
    return magnitude;
  }
  const int exponent = std::bit_width(static_cast<unsigned>(magnitude)) - 1;
  const int scaled = magnitude >> (exponent - 1);
  if ((scaled << (exponent - 1)) != magnitude) {
    return -1;
  }
  return (exponent << 1) | (scaled - 2);
}

std::uint32_t e8m0_bits(std::uint8_t code) {
  if (code == 0xFFu) {
    return kFp32QuietNan;
  }
  // 2^(code-127): the FP32 exponent field equals the code; code 0 is the subnormal 2^-127.
  return code == 0 ? 0x00400000u : static_cast<std::uint32_t>(code) << 23;
}

std::uint16_t bf16_rne(std::uint32_t bits) {
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
    return static_cast<std::uint16_t>(((bits >> 16) & 0x8000u) | 0x7FC0u);
  }
  const std::uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<std::uint16_t>((bits + 0x7FFFu + lsb) >> 16);
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

void decode_e2m1(std::span<const std::uint8_t> nibbles, Shape shape, std::span<std::uint32_t> out) {
  const std::size_t count = check_elementwise(nibbles, shape, out);
  require_nibbles(nibbles);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = e2m1_bits(nibbles[i]);
  }
}

void decode_e4m3fn(std::span<const std::uint8_t> codes, Shape shape, std::span<std::uint32_t> out) {
  const std::size_t count = check_elementwise(codes, shape, out);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = e4m3fn_bits(codes[i]);
  }
}

void decode_e8m0(std::span<const std::uint8_t> codes, Shape shape, std::span<std::uint32_t> out) {
  const std::size_t count = check_elementwise(codes, shape, out);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = e8m0_bits(codes[i]);
  }
}

void decode_bf16(std::span<const std::uint16_t> bits, Shape shape, std::span<std::uint32_t> out) {
  const std::size_t count = check_elementwise(bits, shape, out);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = static_cast<std::uint32_t>(bits[i]) << 16;
  }
}

void fp32_to_bf16(std::span<const std::uint32_t> bits, Shape shape, std::span<std::uint16_t> out) {
  const std::size_t count = check_elementwise(bits, shape, out);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = bf16_rne(bits[i]);
  }
}

void pack_e2m1(std::span<const std::uint8_t> nibbles, Shape logical_shape, std::span<std::uint8_t> packed) {
  const std::size_t count = check_packing(nibbles, logical_shape, packed, true);
  require_nibbles(nibbles);
  for (std::size_t i = 0; i < count / 2; ++i) {
    packed[i] = static_cast<std::uint8_t>(nibbles[2 * i] | (nibbles[2 * i + 1] << 4));
  }
}

void unpack_e2m1(std::span<const std::uint8_t> packed, Shape logical_shape, std::span<std::uint8_t> nibbles) {
  const std::size_t count = check_packing(packed, logical_shape, nibbles, false);
  for (std::size_t i = 0; i < count / 2; ++i) {
    nibbles[2 * i] = packed[i] & 0xFu;
    nibbles[2 * i + 1] = packed[i] >> 4;
  }
}

void e2m1_to_grid(std::span<const std::uint8_t> nibbles, Shape shape, std::span<std::int8_t> grid) {
  const std::size_t count = check_elementwise(nibbles, shape, grid);
  require_nibbles(nibbles);
  for (std::size_t i = 0; i < count; ++i) {
    grid[i] = e2m1_grid(nibbles[i]);
  }
}

void grid_to_e2m1(std::span<const std::int8_t> grid, Shape shape, std::span<std::uint8_t> nibbles) {
  const std::size_t count = check_elementwise(grid, shape, nibbles);
  for (std::int8_t value : grid) {
    if (grid_magnitude_code(value < 0 ? -value : value) < 0) {
      throw std::invalid_argument("value is not an E2M1 integer grid point");
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    const int value = grid[i];
    const int code = grid_magnitude_code(value < 0 ? -value : value);
    nibbles[i] = static_cast<std::uint8_t>(value < 0 ? code | 0x8 : code);
  }
}

}  // namespace df::quant
