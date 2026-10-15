#include "df/quant/swiglu.h"

#include "detail.h"
#include "df/quant/silu_table.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace df::quant {
namespace {

constexpr std::uint16_t kBf16Sign = 0x8000u;
constexpr std::uint16_t kBf16Magnitude = 0x7FFFu;
constexpr std::uint16_t kBf16Inf = 0x7F80u;
constexpr std::uint16_t kBf16Ten = 0x4120u;
constexpr std::uint32_t kFp32Magnitude = 0x7FFFFFFFu;
constexpr std::uint32_t kFp32Inf = 0x7F800000u;

std::uint16_t clamp_bf16(std::uint16_t code, bool gate) noexcept {
  const std::uint16_t magnitude = code & kBf16Magnitude;
  if (magnitude > kBf16Inf || (gate && (code & kBf16Sign) != 0)) {
    return code;
  }
  return static_cast<std::uint16_t>((code & kBf16Sign) | std::min(magnitude, kBf16Ten));
}

void check_metadata(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
                    bool weighted, std::uint32_t route_weight_bits,
                    std::span<std::uint16_t> out, const detail::SwigluTrace* trace) {
  const std::size_t count = gate.size();
  if (up.size() != count) {
    throw std::invalid_argument("gate and up lengths must match");
  }
  if (weighted && (route_weight_bits & kFp32Magnitude) >= kFp32Inf) {
    throw std::invalid_argument("route weight must be finite FP32");
  }
  detail::check_swiglu_counts(count, trace != nullptr);
  if (out.size() < count) {
    throw std::invalid_argument("output capacity is smaller than the result");
  }
  if (trace != nullptr) {
    if (trace->gate_bits.size() != count || trace->up_bits.size() != count ||
        trace->silu_bits.size() != count || trace->gated_bits.size() != count ||
        trace->weighted_bits.size() != count) {
      throw std::invalid_argument("trace spans do not match the input length");
    }
  }

  std::array<std::span<std::byte>, 6> writes{};
  writes[0] = std::as_writable_bytes(out.first(count));
  if (trace != nullptr) {
    writes[1] = std::as_writable_bytes(trace->gate_bits);
    writes[2] = std::as_writable_bytes(trace->up_bits);
    writes[3] = std::as_writable_bytes(trace->silu_bits);
    writes[4] = std::as_writable_bytes(trace->gated_bits);
    writes[5] = std::as_writable_bytes(trace->weighted_bits);
  }
  detail::check_swiglu_storage(std::as_bytes(gate), std::as_bytes(up), writes);
}

void bridge(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
            bool weighted, std::uint32_t route_weight_bits, std::span<std::uint16_t> out,
            const detail::SwigluTrace* trace) noexcept {
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const std::uint16_t gate_code = clamp_bf16(gate[i], true);
    const std::uint16_t up_code = clamp_bf16(up[i], false);
    const std::uint32_t gate_bits = static_cast<std::uint32_t>(gate_code) << 16;
    const std::uint32_t up_bits = static_cast<std::uint32_t>(up_code) << 16;
    const std::uint32_t silu_bits = generated::kSiluBf16ToFp32[gate_code];
    const std::uint32_t gated_bits = detail::multiply_fp32(silu_bits, up_bits);
    const std::uint32_t weighted_bits =
        weighted ? detail::multiply_fp32(gated_bits, route_weight_bits) : gated_bits;
    out[i] = detail::bf16_rne(weighted_bits);
    if (trace != nullptr) {
      trace->gate_bits[i] = gate_bits;
      trace->up_bits[i] = up_bits;
      trace->silu_bits[i] = silu_bits;
      trace->gated_bits[i] = gated_bits;
      trace->weighted_bits[i] = weighted_bits;
    }
  }
}

void evaluate(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
              bool weighted, std::uint32_t route_weight_bits, std::span<std::uint16_t> out,
              const detail::SwigluTrace* trace) {
  check_metadata(gate, up, weighted, route_weight_bits, out, trace);
  bridge(gate, up, weighted, route_weight_bits, out, trace);
}

}  // namespace

namespace detail {

void check_swiglu_storage(std::span<const std::byte> gate, std::span<const std::byte> up,
                          std::span<const std::span<std::byte>> writes) {
  const std::array reads{gate, up};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) {
      require_disjoint(read, read.size(), writes[i], writes[i].size());
    }
    for (std::size_t j = 0; j < i; ++j) {
      require_disjoint(writes[j], writes[j].size(), writes[i], writes[i].size());
    }
  }
}

void check_swiglu_counts(std::size_t count, bool traced) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
    throw std::invalid_argument("SwiGLU byte count overflows size_t");
  }
  if (traced && count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
    throw std::invalid_argument("SwiGLU trace byte count overflows size_t");
  }
}

void trace_swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
                  std::span<std::uint16_t> out, const SwigluTrace& trace) {
  const SwigluTrace snapshot = trace;
  evaluate(gate, up, false, 0, out, &snapshot);
}

void trace_swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
                  std::uint32_t route_weight_bits, std::span<std::uint16_t> out,
                  const SwigluTrace& trace) {
  const SwigluTrace snapshot = trace;
  evaluate(gate, up, true, route_weight_bits, out, &snapshot);
}

}  // namespace detail

void swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
            std::span<std::uint16_t> out) {
  evaluate(gate, up, false, 0, out, nullptr);
}

void swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
            std::uint32_t route_weight_bits, std::span<std::uint16_t> out) {
  evaluate(gate, up, true, route_weight_bits, out, nullptr);
}

}  // namespace df::quant
