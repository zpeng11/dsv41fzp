#include "df/quant/expert.h"

#include "df/quant/act_quant.h"
#include "df/quant/compute.h"
#include "df/quant/swiglu.h"
#include "detail.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace df::quant {
namespace {

constexpr std::size_t kGroup = 32;
constexpr std::uint32_t kFp32Magnitude = 0x7FFFFFFFu;
constexpr std::uint32_t kFp32Inf = 0x7F800000u;

std::size_t checked_bytes(std::size_t count, std::size_t width) {
  if (count > std::numeric_limits<std::size_t>::max() / width) {
    throw std::invalid_argument("expert element or byte count overflows size_t");
  }
  return count * width;
}

std::size_t checked_product(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument("expert element or byte count overflows size_t");
  }
  return left * right;
}

void check_capacity(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
                    ExpertMatrix w3, ExpertWorkspace work, std::span<std::uint16_t> out,
                    const detail::SwigluTrace* trace, const detail::ExpertCounts& counts) {
  const std::size_t d = input.size();
  const std::size_t h = w1.n;
  if (w1.w_packed.size() != counts.w1_weight_bytes ||
      w1.scale_b.size() != counts.w1_scale_bytes ||
      w2.w_packed.size() != counts.w2_weight_bytes ||
      w2.scale_b.size() != counts.w2_scale_bytes ||
      w3.w_packed.size() != counts.w3_weight_bytes ||
      w3.scale_b.size() != counts.w3_scale_bytes) {
    throw std::invalid_argument("expert matrix payload lengths do not match their shapes");
  }
  if (work.input_codes.size() < d || work.input_scales.size() < counts.input_groups ||
      work.gate.size() < h || work.up.size() < h || work.hidden.size() < h ||
      work.hidden_codes.size() < h || work.hidden_scales.size() < counts.hidden_groups ||
      out.size() < w2.n) {
    throw std::invalid_argument("expert workspace or output capacity is smaller than the result");
  }
  if (trace != nullptr &&
      (trace->gate_bits.size() != h || trace->up_bits.size() != h ||
       trace->silu_bits.size() != h || trace->gated_bits.size() != h ||
       trace->weighted_bits.size() != h)) {
    throw std::invalid_argument("expert trace spans do not match H");
  }
}

void check_storage(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
                   ExpertMatrix w3, ExpertWorkspace work, std::span<std::uint16_t> out,
                   const detail::SwigluTrace* trace, const detail::ExpertCounts& counts) {
  const std::size_t d = input.size();
  const std::size_t h = w1.n;
  const std::array reads{std::as_bytes(input), std::as_bytes(w1.w_packed),
                         std::as_bytes(w1.scale_b), std::as_bytes(w2.w_packed),
                         std::as_bytes(w2.scale_b), std::as_bytes(w3.w_packed),
                         std::as_bytes(w3.scale_b)};
  std::array<std::span<std::byte>, 12> writes{};
  writes[0] = std::as_writable_bytes(work.input_codes.first(d));
  writes[1] = std::as_writable_bytes(work.input_scales.first(counts.input_groups));
  writes[2] = std::as_writable_bytes(work.gate.first(h));
  writes[3] = std::as_writable_bytes(work.up.first(h));
  writes[4] = std::as_writable_bytes(work.hidden.first(h));
  writes[5] = std::as_writable_bytes(work.hidden_codes.first(h));
  writes[6] = std::as_writable_bytes(work.hidden_scales.first(counts.hidden_groups));
  if (trace != nullptr) {
    writes[7] = std::as_writable_bytes(trace->gate_bits);
    writes[8] = std::as_writable_bytes(trace->up_bits);
    writes[9] = std::as_writable_bytes(trace->silu_bits);
    writes[10] = std::as_writable_bytes(trace->gated_bits);
    writes[11] = std::as_writable_bytes(trace->weighted_bits);
  }
  detail::check_expert_storage(reads, writes, std::as_writable_bytes(out),
                               checked_bytes(w2.n, sizeof(std::uint16_t)));
}

void check_contents(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
                    ExpertMatrix w3, bool weighted, std::uint32_t route_weight_bits) {
  if (weighted && (route_weight_bits & kFp32Magnitude) >= kFp32Inf) {
    throw std::invalid_argument("route weight must be finite FP32");
  }
  for (const std::uint16_t code : input) {
    if ((code & 0x7F80u) == 0x7F80u) {
      throw std::invalid_argument("nonfinite BF16 expert input is not accepted");
    }
  }
  for (const auto scales : {w1.scale_b, w2.scale_b, w3.scale_b}) {
    for (const std::uint8_t code : scales) {
      if (code == 0xFFu) {
        throw std::invalid_argument("NaN expert weight scale codes are not accepted");
      }
    }
  }
}

void linear(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales,
            ExpertMatrix matrix, std::span<std::uint16_t> out) {
  const std::size_t weight_stride = matrix.k / 2;
  const std::size_t scale_stride = matrix.k / kGroup;
  for (std::size_t row = 0; row < matrix.n; ++row) {
    expert_grouped_dot(codes, matrix.w_packed.subspan(row * weight_stride, weight_stride), scales,
                       matrix.scale_b.subspan(row * scale_stride, scale_stride),
                       out.subspan(row, 1));
  }
}

void evaluate(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
              ExpertMatrix w3, bool weighted, std::uint32_t route_weight_bits,
              ExpertWorkspace work, std::span<std::uint16_t> out,
              const detail::SwigluTrace* trace) {
  const detail::ExpertCounts counts = trace == nullptr
                                           ? detail::validate_expert_parameters(input, w1, w2, w3,
                                                                                work, out)
                                           : detail::check_expert_counts(
                                                 input.size(), w1.n, w1.k, w2.n, w2.k, w3.n,
                                                 w3.k, out.size(), true);
  if (trace != nullptr) {
    check_capacity(input, w1, w2, w3, work, out, trace, counts);
  }
  check_storage(input, w1, w2, w3, work, out, trace, counts);
  detail::validate_expert_contents(input, w1, w2, w3, weighted, route_weight_bits);

  const auto input_codes = work.input_codes.first(input.size());
  const auto input_scales = work.input_scales.first(counts.input_groups);
  const auto gate = work.gate.first(w1.n);
  const auto up = work.up.first(w1.n);
  const auto hidden = work.hidden.first(w1.n);
  const auto hidden_codes = work.hidden_codes.first(w1.n);
  const auto hidden_scales = work.hidden_scales.first(counts.hidden_groups);
  act_quant(input, input_codes, input_scales);
  linear(input_codes, input_scales, w1, gate);
  linear(input_codes, input_scales, w3, up);
  if (trace == nullptr) {
    if (weighted) {
      swiglu(gate, up, route_weight_bits, hidden);
    } else {
      swiglu(gate, up, hidden);
    }
  } else if (weighted) {
    detail::trace_swiglu(gate, up, route_weight_bits, hidden, *trace);
  } else {
    detail::trace_swiglu(gate, up, hidden, *trace);
  }
  act_quant(hidden, hidden_codes, hidden_scales);
  linear(hidden_codes, hidden_scales, w2, out.first(w2.n));
}

}  // namespace

namespace detail {

ExpertCounts check_expert_counts(std::size_t input_count, std::size_t w1_n, std::size_t w1_k,
                                 std::size_t w2_n, std::size_t w2_k, std::size_t w3_n,
                                 std::size_t w3_k, std::size_t out_count, bool traced) {
  if (input_count == 0 || w1_n == 0 || w1_k == 0 || w2_n == 0 || w2_k == 0 || w3_n == 0 ||
      w3_k == 0 || input_count % kGroup != 0 || w1_n % kGroup != 0 ||
      w1_k % kGroup != 0 || w2_n % kGroup != 0 || w2_k % kGroup != 0 ||
      w3_n % kGroup != 0 || w3_k % kGroup != 0) {
    throw std::invalid_argument("expert dimensions must be positive multiples of 32");
  }
  if (w1_k != input_count || w3_n != w1_n || w3_k != input_count || w2_n != input_count ||
      w2_k != w1_n) {
    throw std::invalid_argument("expected W1/W3 H-by-D and W2 D-by-H");
  }

  const ExpertCounts counts{input_count / kGroup,
                            w1_n / kGroup,
                            checked_product(w1_n, w1_k / 2),
                            checked_product(w1_n, w1_k / kGroup),
                            checked_product(w2_n, w2_k / 2),
                            checked_product(w2_n, w2_k / kGroup),
                            checked_product(w3_n, w3_k / 2),
                            checked_product(w3_n, w3_k / kGroup)};
  checked_bytes(input_count, sizeof(std::uint16_t));
  checked_bytes(w1_n, sizeof(std::uint16_t));
  checked_bytes(out_count, sizeof(std::uint16_t));
  checked_bytes(input_count, sizeof(std::uint8_t));
  checked_bytes(counts.input_groups, sizeof(std::uint8_t));
  checked_bytes(w1_n, sizeof(std::uint8_t));
  checked_bytes(counts.hidden_groups, sizeof(std::uint8_t));
  if (traced) {
    checked_bytes(w1_n, sizeof(std::uint32_t));
  }
  return counts;
}

void check_expert_storage(std::span<const std::span<const std::byte>> reads,
                          std::span<const std::span<std::byte>> scratch_writes,
                          std::span<std::byte> out, std::size_t out_write_bytes) {
  if (out_write_bytes > out.size()) {
    throw std::invalid_argument("expert output capacity is smaller than the result");
  }
  for (std::size_t i = 0; i < scratch_writes.size(); ++i) {
    for (const auto read : reads) {
      require_disjoint(read, read.size(), scratch_writes[i], scratch_writes[i].size());
    }
    for (std::size_t j = 0; j < i; ++j) {
      require_disjoint(scratch_writes[j], scratch_writes[j].size(), scratch_writes[i],
                       scratch_writes[i].size());
    }
    require_disjoint(out, out.size(), scratch_writes[i], scratch_writes[i].size());
  }
  const auto out_write = out.first(out_write_bytes);
  for (const auto read : reads) {
    require_disjoint(read, read.size(), out_write, out_write.size());
  }
}

ExpertCounts validate_expert_parameters(std::span<const std::uint16_t> input, ExpertMatrix w1,
                                        ExpertMatrix w2, ExpertMatrix w3,
                                        ExpertWorkspace work, std::span<std::uint16_t> out) {
  const ExpertCounts counts =
      check_expert_counts(input.size(), w1.n, w1.k, w2.n, w2.k, w3.n, w3.k, out.size(), false);
  check_capacity(input, w1, w2, w3, work, out, nullptr, counts);
  return counts;
}

void validate_expert_contents(std::span<const std::uint16_t> input, ExpertMatrix w1,
                              ExpertMatrix w2, ExpertMatrix w3, bool weighted,
                              std::uint32_t route_weight_bits) {
  check_contents(input, w1, w2, w3, weighted, route_weight_bits);
}

void trace_expert(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
                  ExpertMatrix w3, ExpertWorkspace work, std::span<std::uint16_t> out,
                  const SwigluTrace& trace) {
  const SwigluTrace snapshot = trace;
  evaluate(input, w1, w2, w3, false, 0, work, out, &snapshot);
}

void trace_expert(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
                  ExpertMatrix w3, std::uint32_t route_weight_bits, ExpertWorkspace work,
                  std::span<std::uint16_t> out, const SwigluTrace& trace) {
  const SwigluTrace snapshot = trace;
  evaluate(input, w1, w2, w3, true, route_weight_bits, work, out, &snapshot);
}

}  // namespace detail

void expert(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
            ExpertMatrix w3, ExpertWorkspace work, std::span<std::uint16_t> out) {
  evaluate(input, w1, w2, w3, false, 0, work, out, nullptr);
}

void expert(std::span<const std::uint16_t> input, ExpertMatrix w1, ExpertMatrix w2,
            ExpertMatrix w3, std::uint32_t route_weight_bits, ExpertWorkspace work,
            std::span<std::uint16_t> out) {
  evaluate(input, w1, w2, w3, true, route_weight_bits, work, out, nullptr);
}

}  // namespace df::quant
