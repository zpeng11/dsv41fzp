#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// Production scalar codec for the low-precision formats in docs/QUANT.md (P0010, D0007).
//
// Every bulk entry takes a contiguous row-major input view, an explicit logical shape,
// and a caller-owned output buffer whose capacity must cover the result; only the
// result prefix is written. Results are built with integer bit operations, so they do
// not depend on the caller's floating-point environment, which is never read or changed.
//
// Failures: std::invalid_argument for rank 0, length/shape/capacity mismatch, odd last
// dimension where packing applies, overlapping input and output, invalid codes and
// non-grid integers; std::overflow_error when the element count overflows size_t.
// Metadata is validated before any data is read, content before any output is written;
// on failure the output is untouched. Zero-element tensors are accepted.
namespace df::quant {

using Shape = std::span<const std::size_t>;

// E2M1 nibbles (one per byte, 0..15) to FP32 bit patterns; code 8 decodes to -0.
void decode_e2m1(std::span<const std::uint8_t> nibbles, Shape shape, std::span<std::uint32_t> out);

// E4M3FN codes to FP32 bit patterns; 0x7F/0xFF decode to quiet NaN with the code's sign.
void decode_e4m3fn(std::span<const std::uint8_t> codes, Shape shape, std::span<std::uint32_t> out);

// Unsigned E8M0 codes to FP32 bit patterns; 0x00 is the subnormal 2^-127, 0xFF is NaN.
void decode_e8m0(std::span<const std::uint8_t> codes, Shape shape, std::span<std::uint32_t> out);

// BF16 bit patterns to FP32 bit patterns (exact widening).
void decode_bf16(std::span<const std::uint16_t> bits, Shape shape, std::span<std::uint32_t> out);

// FP32 bit patterns to BF16 with round-to-nearest-even; NaN becomes a signed quiet NaN.
void fp32_to_bf16(std::span<const std::uint32_t> bits, Shape shape, std::span<std::uint16_t> out);

// Packs E2M1 nibbles two per byte along the even last logical dimension; the element
// that comes first goes to the low half. `packed` receives count/2 bytes.
void pack_e2m1(std::span<const std::uint8_t> nibbles, Shape logical_shape, std::span<std::uint8_t> packed);

// Inverse of pack_e2m1: every byte value is valid and yields two nibbles.
void unpack_e2m1(std::span<const std::uint8_t> packed, Shape logical_shape, std::span<std::uint8_t> nibbles);

// E2M1 nibbles to the integer grid (twice the decoded value); both zeros map to 0.
void e2m1_to_grid(std::span<const std::uint8_t> nibbles, Shape shape, std::span<std::int8_t> grid);

// Integer grid points to E2M1 nibbles; zero maps to code 0, non-grid values are rejected.
void grid_to_e2m1(std::span<const std::int8_t> grid, Shape shape, std::span<std::uint8_t> nibbles);

}  // namespace df::quant
