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

// E4M3FN codes to FP32 bit patterns; 0x7F/0xFF decode to quiet NaN with the code's sign.
void decode_e4m3fn(std::span<const std::uint8_t> codes, Shape shape, std::span<std::uint32_t> out);

}  // namespace df::quant
