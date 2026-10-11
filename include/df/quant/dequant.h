#pragma once

#include "df/quant/codec.h"

// Given-scale scalar dequantization (P0011, D0007). Data and scales are contiguous
// row-major views with exact, independent storage shapes; packed E2M1 has K/2
// bytes in the last dimension, unlike codec pack/unpack's logical nibble shape.
// Output is caller-owned FP32/BF16 raw bits, with only the result prefix written.
// No broadcasting, transposition, stride, scale inference or FP environment access.
//
// Rank/shape/length/capacity mismatch, invalid data/scales and input/output overlap
// throw std::invalid_argument. Metadata arithmetic overflow throws std::overflow_error;
// numerical overflow produces signed infinity. Metadata precedes all content reads,
// content validation precedes all output writes, and failure leaves output untouched.
// Zero-element tensors are accepted when their rank, grouping and scale shape agree.
namespace df::quant {

// Expert: matrix of packed E2M1, E8M0 scale per 32 logical elements in each row.
void dequantize_expert(std::span<const std::uint8_t> data, Shape data_shape,
                       std::span<const std::uint8_t> scales, Shape scale_shape,
                       std::span<std::uint32_t> out);
void dequantize_expert(std::span<const std::uint8_t> data, Shape data_shape,
                       std::span<const std::uint8_t> scales, Shape scale_shape,
                       std::span<std::uint16_t> out);

// Ordinary FP8 weights: E4M3FN matrix, E8M0 scales covering 32x32 blocks and tails.
void dequantize_fp8_weights(std::span<const std::uint8_t> data, Shape data_shape,
                            std::span<const std::uint8_t> scales, Shape scale_shape,
                            std::span<std::uint32_t> out);
void dequantize_fp8_weights(std::span<const std::uint8_t> data, Shape data_shape,
                            std::span<const std::uint8_t> scales, Shape scale_shape,
                            std::span<std::uint16_t> out);

// Engram: E4M3FN matrix, E8M0 scale per 32 elements in each row.
void dequantize_engram(std::span<const std::uint8_t> data, Shape data_shape,
                       std::span<const std::uint8_t> scales, Shape scale_shape,
                       std::span<std::uint32_t> out);
void dequantize_engram(std::span<const std::uint8_t> data, Shape data_shape,
                       std::span<const std::uint8_t> scales, Shape scale_shape,
                       std::span<std::uint16_t> out);

// SWA: E4M3FN, E8M0 scale per 32 elements along the last dimension, with batch axes.
void dequantize_swa(std::span<const std::uint8_t> data, Shape data_shape,
                    std::span<const std::uint8_t> scales, Shape scale_shape,
                    std::span<std::uint32_t> out);
void dequantize_swa(std::span<const std::uint8_t> data, Shape data_shape,
                    std::span<const std::uint8_t> scales, Shape scale_shape,
                    std::span<std::uint16_t> out);

// Main KV: packed E2M1, strictly positive E4M3FN scale per 16 logical elements.
void dequantize_main_kv(std::span<const std::uint8_t> data, Shape data_shape,
                        std::span<const std::uint8_t> scales, Shape scale_shape,
                        std::span<std::uint32_t> out);
void dequantize_main_kv(std::span<const std::uint8_t> data, Shape data_shape,
                        std::span<const std::uint8_t> scales, Shape scale_shape,
                        std::span<std::uint16_t> out);

// Index K: packed E2M1, E8M0 scale per 32 logical elements, with batch axes.
void dequantize_index_k(std::span<const std::uint8_t> data, Shape data_shape,
                        std::span<const std::uint8_t> scales, Shape scale_shape,
                        std::span<std::uint32_t> out);
void dequantize_index_k(std::span<const std::uint8_t> data, Shape data_shape,
                        std::span<const std::uint8_t> scales, Shape scale_shape,
                        std::span<std::uint16_t> out);

}  // namespace df::quant
