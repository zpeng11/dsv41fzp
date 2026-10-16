// Test executor for the production quant library (P0010 codec, P0011 dequant, P0013
// compute, P0017 activation). Not a production format.
//
// Reads one request per stdin line and writes one response per line:
//   codec:    <op> shape=<d0,d1,...> cap=<n> [alias] : <hex token> ...
//   dequant:  <op> shape=<d0,d1,...> scale_shape=<d0,d1,...> cap=<n> [alias|scale_alias]
//             : <data hex token> ... ; <scale hex token> ...
//   compute:  <op> cap=<n> [overlap=a|w|sa|sb [in_offset=<b>] [out_offset=<b>]] [poison]
//             : <a codes> ... ; <w bytes> ... ; <scale_a codes> ... ; <scale_b codes> ...
//   add_fp32|multiply_fp32: <op> cap=<n> : <left fp32 bits> ... ; <right fp32 bits> ...
//   activation: act_quant[_trace] cap=<a capacity> scale_cap=<scale capacity>
//             [overlap=input_a|input_scale|outputs [in_offset=<b>] [out_offset=<b>]]
//             [trace_alias=<buffer>,<buffer>|trace_object_alias] [poison]
//             [trace_shape=<six lengths>]
//             : <BF16 bits> ...
//   combination: act_quant_dot_fp32|act_quant_dot_bf16 cap=<n>
//             : <BF16 bits> ... ; <packed weights> ... ; <scale_b> ...
//   P0028: fp8_grouped_dot_fp32|fp8_grouped_dot_bf16 cap=<n>
//             [overlap=a|w|sa|sb [in_offset=<b>] [out_offset=<b>]] [poison]
//             : <a codes> ... ; <w codes> ... ; <scale_a codes> ... ; <scale_b codes> ...
//   P0028 trace: fp8_grouped_dot_trace_fp32|fp8_grouped_dot_trace_bf16 cap=<response words>
//             [out_cap=<output capacity>]
//             [trace_shape=<six lengths>] [trace_alias=<buffer>,<buffer>]
//             [in_offset=<b>] [out_offset=<b>] [poison]
//   P0028 combination: act_quant_fp8_dot_fp32|act_quant_fp8_dot_bf16 cap=<n>
//             : <BF16 bits> ... ; <FP8 weight codes> ... ; <scale_b> ...
//   P0019: expert_linear_fp32|expert_linear_bf16 shape=N cap=<out capacity>
//             [a_cap=K scale_cap=K/32] [workspace] [overlap=<buffer>,<buffer>]
//             [in_offset=<bytes> out_offset=<bytes>] [poison|poison_writes]
//             : <BF16 bits> ... ; <row-major packed weights> ... ; <scale_b> ...
//   P0021: swiglu[_trace] cap=<out capacity> [route=<FP32 bits>]
//             [overlap=<buffer>,<buffer> in_offset=<bytes> out_offset=<bytes>]
//             [trace_alias=<buffer>,<buffer>] [trace_shape=<five lengths>] [poison]
//             : <gate BF16 bits> ... ; <up BF16 bits> ...
//   P0021 counts: swiglu_counts[_trace] cap=<element count> : (empty payload)
//   Linear buffers are input,w,sb,a,sa,out. Defaults are exact workspace sizes;
//   workspace responses are uint32 words: full out, full a, full sa capacities.
//   poison makes inputs unreadable; poison_writes makes destinations inaccessible.
//   Activation ok responses contain uint32 words: both complete output buffers,
//   then (trace only) amax, bounded_amax, product, exponent, quotient, clamped.
//   response: ok <hex output element> ...   (the whole output buffer, cap elements)
//             err <invalid_argument|overflow_error> changed=<0|1>
// Tokens are range-checked against the input element width before narrowing. The
// payload is split at every ';' and each op requires its exact section count; fields
// belonging to another request family are protocol faults, and any protocol or
// executor fault exits non-zero instead of producing an `err` response.
//
// Compute ops are expert_grouped_dot_fp32/_bf16 and the private expert_grouped_dot_trace,
// which answers products[K], partial_sums[K], dot, after_a, block, acc (K/32 each) and
// the FP32 result through df::quant::detail::trace_expert_grouped_dot; its cap must be
// exactly 2K + 4K/32 + 1 for valid K. Every buffer is an exact-length allocation so
// ASan sees any access past it. `overlap=` places the named input at byte in_offset
// and the output at byte out_offset (a multiple of the output element size) of one
// exact raw allocation. `poison` marks all four inputs unreadable for the call under
// ASan (a no-op otherwise; the `asan` op reports which), so a request with bad
// metadata proves no input is read before metadata validation. The private add_fp32
// op runs df::quant::detail::add_fp32 on arbitrary bit-pattern pairs and needs
// cap >= the pair count. Buffers are pre-filled with kFill bytes and sized
// to their exact element counts so out-of-bounds access is visible to ASan; `alias`
// makes data and output share one raw allocation, `scale_alias` scales and output.
// Dequant ops carry the suffix _fp32 or _bf16. The private `multiply_finite` op pairs
// finite binary32 bit patterns, one pair per input element, through the production
// helper df::quant::detail::multiply_finite; non-finite tokens are protocol faults.
//
// Every production call, codec and dequant alike, runs inside a full MXCSR
// invariance window measured with _mm_getcsr immediately around the call, exception
// paths included; stdio runs outside the window and cannot blur the verdict.
//
// Options: --mxcsr=default|up|down|zero|ftz|daz|ftz_daz|overflow_unmasked sets the
// SSE control word before the first request and verifies the control bits at exit;
// overflow_unmasked also clears pending status flags. --status=clear|set starts with
// all six status flags clear (default) or set, so a call that clears or raises a flag
// breaks the full-MXCSR window either way. The `mxcsr` op reports the register.
// --fault=crash|exit|garbage|mismatch|drop deliberately breaks the executor;
// input-write|shared-gap-write mutates activation storage after a successful call
// but before snapshot checks, proving that write-contract failures reach CTest.

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <xmmintrin.h>

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#endif

#include "detail.h"
#include "df/quant/act_quant.h"
#include "df/quant/codec.h"
#include "df/quant/compute.h"
#include "df/quant/dequant.h"
#include "df/quant/expert.h"
#include "df/quant/linear.h"
#include "df/quant/swiglu.h"

namespace {

namespace q = df::quant;

constexpr unsigned char kFill = 0xA5;
constexpr int kOk = 0;
constexpr int kOverflow = 1;
constexpr int kInvalid = 2;

[[noreturn]] void die(const std::string& msg) {
  std::cerr << "quant_exec: " << msg << '\n';
  std::exit(2);
}

struct Request {
  std::string op;
  std::vector<std::size_t> shape;
  std::vector<std::size_t> scale_shape;
  std::size_t scale_cap = 0;
  bool has_scale_cap = false;
  std::size_t out_cap = 0;
  bool has_out_cap = false;
  std::size_t input_count = 0;
  bool has_input_count = false;
  std::vector<std::size_t> trace_shape;
  bool has_trace_shape = false;
  std::size_t cap = 0;
  bool has_cap = false;
  bool has_shape = false;
  bool alias = false;
  bool scale_alias = false;
  bool has_scale_shape = false;
  bool has_scale_section = false;
  bool has_overlap = false;
  bool has_trace_alias = false;
  bool trace_object_alias = false;
  std::size_t overlap = 0;
  std::size_t act_first = 0;
  std::size_t act_second = 1;
  bool has_in_offset = false;
  bool has_out_offset = false;
  std::size_t in_offset = 0;
  std::size_t out_offset = 0;
  bool poison = false;
  bool poison_writes = false;
  bool runtime_failure = false;
  bool workspace = false;
  std::vector<std::size_t> work_shape;
  bool has_work_shape = false;
  bool has_route = false;
  std::uint32_t route = 0;
  bool has_a_cap = false;
  std::size_t a_cap = 0;
  std::vector<std::vector<std::uint64_t>> sections;
  std::vector<std::uint64_t> tokens;
  std::vector<std::uint64_t> scale_tokens;
};

constexpr std::string_view kLinearBuffers[] = {"input", "w", "sb", "a", "sa", "out"};
constexpr std::string_view kComputeInputs[] = {"a", "w", "sa", "sb"};
constexpr std::string_view kFp8TraceBuffers[] = {"a", "w", "sa", "sb", "out", "products",
    "partial_sums", "dot", "after_a", "block", "acc"};
constexpr std::string_view kSwigluBuffers[] = {"gate", "up", "out", "gate_bits", "up_bits",
                                               "silu_bits", "gated_bits", "weighted_bits"};
constexpr std::string_view kExpertBuffers[] = {
    "input", "w1packed", "w1scale", "w2packed", "w2scale", "w3packed", "w3scale",
    "input_codes", "input_scales", "gate", "up", "hidden", "hidden_codes",
    "hidden_scales", "out", "gate_bits", "up_bits", "silu_bits", "gated_bits",
    "weighted_bits"};
constexpr std::string_view kExpertStorageBuffers[] = {"read", "scratch", "out"};

bool is_linear(const Request& req) {
  return req.op == "expert_linear_fp32" || req.op == "expert_linear_bf16";
}

bool is_swiglu(const Request& req) {
  return req.op == "swiglu" || req.op == "swiglu_trace";
}

bool is_expert(const Request& req) {
  return req.op == "expert_bf16" || req.op == "expert_trace";
}

bool is_fp8_trace(const Request& req) {
  return req.op == "fp8_grouped_dot_trace_fp32" || req.op == "fp8_grouped_dot_trace_bf16";
}
constexpr std::string_view kActBuffers[] = {"input", "a", "scale", "amax_bits",
    "bounded_amax_bits", "product_bits", "exponent", "quotient_bits", "clamped_bits"};

std::uint64_t parse_number(std::string_view text, int base) {
  std::uint64_t value = 0;
  const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (text.empty() || ec != std::errc() || ptr != text.data() + text.size()) {
    die("malformed number '" + std::string(text) + "'");
  }
  return value;
}

std::vector<std::string_view> split(std::string_view text, char sep) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find(sep, start);
    const std::size_t stop = end == std::string_view::npos ? text.size() : end;
    if (stop > start) {
      parts.push_back(text.substr(start, stop - start));
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return parts;
}

std::vector<std::size_t> parse_shape(std::string_view text) {
  std::vector<std::size_t> dims;
  if (text.empty()) {
    return dims;
  }
  std::size_t start = 0;
  while (true) {
    const std::size_t end = text.find(',', start);
    const std::size_t stop = end == std::string_view::npos ? text.size() : end;
    dims.push_back(parse_number(text.substr(start, stop - start), 10));
    if (end == std::string_view::npos) {
      return dims;
    }
    start = end + 1;
  }
}

Request parse_request(std::string_view line) {
  const std::size_t colon = line.find(':');
  if (colon == std::string_view::npos) {
    die("request lacks ':' separator");
  }
  const auto head = split(line.substr(0, colon), ' ');
  if (head.empty()) {
    die("request lacks an op");
  }
  Request req;
  req.op = std::string(head[0]);
  for (std::size_t i = 1; i < head.size(); ++i) {
    const std::string_view field = head[i];
    if (field.starts_with("route=")) {
      if ((!is_swiglu(req) && !is_expert(req)) || req.has_route) die("invalid or duplicate route");
      const std::uint64_t value = parse_number(field.substr(6), 16);
      if (value > std::numeric_limits<std::uint32_t>::max()) die("route exceeds FP32 width");
      req.route = static_cast<std::uint32_t>(value);
      req.has_route = true;
    } else if (field.starts_with("a_cap=")) {
      if (!is_linear(req) || req.has_a_cap) die("invalid or duplicate a_cap");
      req.a_cap = parse_number(field.substr(6), 10);
      req.has_a_cap = true;
    } else if (field == "workspace") {
      if (!is_linear(req) || req.workspace) die("invalid or duplicate workspace");
      req.workspace = true;
    } else if (field == "poison_writes") {
      if ((!is_linear(req) && !is_expert(req)) || req.poison_writes) die("invalid or duplicate poison_writes");
      req.poison_writes = true;
    } else if (field == "runtime_failure") {
      if (!is_expert(req) || req.runtime_failure) die("invalid or duplicate runtime_failure");
      req.runtime_failure = true;
    } else if (field.starts_with("input_count=")) {
      if (!req.op.starts_with("expert_counts") || req.has_input_count)
        die("invalid or duplicate input_count");
      req.input_count = parse_number(field.substr(12), 10);
      req.has_input_count = true;
    } else if (field.starts_with("scale_cap=")) {
      if (req.has_scale_cap) die("duplicate scale_cap field");
      req.scale_cap = parse_number(field.substr(10), 10);
      req.has_scale_cap = true;
    } else if (field.starts_with("out_cap=")) {
      if (!is_fp8_trace(req) || req.has_out_cap) die("invalid or duplicate out_cap");
      req.out_cap = parse_number(field.substr(8), 10);
      req.has_out_cap = true;
    } else if (field.starts_with("work_shape=")) {
      if ((!is_expert(req) && req.op != "expert_storage") || req.has_work_shape)
        die("invalid or duplicate work_shape");
      req.work_shape = parse_shape(field.substr(11));
      req.has_work_shape = true;
    } else if (field.starts_with("trace_shape=")) {
      if (req.has_trace_shape) die("duplicate trace_shape field");
      req.trace_shape = parse_shape(field.substr(12));
      req.has_trace_shape = true;
    } else if (field.starts_with("scale_shape=")) {
      if (req.has_scale_shape) die("duplicate scale_shape field");
      req.scale_shape = parse_shape(field.substr(12));
      req.has_scale_shape = true;
    } else if (field.starts_with("shape=")) {
      if (req.has_shape) die("duplicate shape field");
      req.shape = parse_shape(field.substr(6));
      req.has_shape = true;
    } else if (field.starts_with("cap=")) {
      if (req.has_cap) die("duplicate cap field");
      req.cap = parse_number(field.substr(4), 10);
      req.has_cap = true;
    } else if (field == "alias") {
      if (req.alias) die("duplicate alias field");
      req.alias = true;
    } else if (field == "scale_alias") {
      if (req.scale_alias) die("duplicate scale_alias field");
      req.scale_alias = true;
    } else if (field.starts_with("overlap=")) {
      if (req.has_overlap) die("duplicate overlap field");
      const auto name = field.substr(8);
      if (is_linear(req) || is_swiglu(req) || is_expert(req) || is_fp8_trace(req) ||
          req.op == "expert_storage") {
        const auto comma = name.find(',');
        if (comma == std::string_view::npos) die("overlap needs two buffers");
        const auto index = [&](std::string_view value) {
          if (is_linear(req)) {
            const auto* hit = std::find(std::begin(kLinearBuffers), std::end(kLinearBuffers), value);
            if (hit == std::end(kLinearBuffers)) die("unknown linear buffer");
            return static_cast<std::size_t>(hit - std::begin(kLinearBuffers));
          }
          if (is_expert(req)) {
            const auto* hit = std::find(std::begin(kExpertBuffers), std::end(kExpertBuffers), value);
            if (hit == std::end(kExpertBuffers)) die("unknown expert buffer");
            return static_cast<std::size_t>(hit - std::begin(kExpertBuffers));
          }
          if (req.op == "expert_storage") {
            const auto* hit = std::find(std::begin(kExpertStorageBuffers),
                                        std::end(kExpertStorageBuffers), value);
            if (hit == std::end(kExpertStorageBuffers)) die("unknown expert storage buffer");
            return static_cast<std::size_t>(hit - std::begin(kExpertStorageBuffers));
          }
          if (is_fp8_trace(req)) {
            const auto* hit = std::find(std::begin(kFp8TraceBuffers),
                                        std::end(kFp8TraceBuffers), value);
            if (hit == std::end(kFp8TraceBuffers)) die("unknown FP8 trace buffer");
            return static_cast<std::size_t>(hit - std::begin(kFp8TraceBuffers));
          }
          const auto* hit = std::find(std::begin(kSwigluBuffers), std::end(kSwigluBuffers), value);
          if (hit == std::end(kSwigluBuffers)) die("unknown SwiGLU buffer");
          return static_cast<std::size_t>(hit - std::begin(kSwigluBuffers));
        };
        req.act_first = index(name.substr(0, comma));
        req.act_second = index(name.substr(comma + 1));
        if (req.act_first == req.act_second) die("overlap needs distinct buffers");
      } else if (req.op == "act_quant" || req.op == "act_quant_trace") {
        if (name == "input_a") { req.act_first = 0; req.act_second = 1; }
        else if (name == "input_scale") { req.act_first = 0; req.act_second = 2; }
        else if (name == "outputs") { req.act_first = 1; req.act_second = 2; }
        else die("unknown activation overlap");
      } else {
        const auto* hit = std::find(std::begin(kComputeInputs), std::end(kComputeInputs), name);
        if (hit == std::end(kComputeInputs)) die("unknown overlap input '" + std::string(name) + "'");
        req.overlap = static_cast<std::size_t>(hit - std::begin(kComputeInputs));
      }
      req.has_overlap = true;
    } else if (field == "trace_object_alias") {
      if (req.op != "act_quant_trace") die("trace_object_alias requires activation trace entry");
      if (req.trace_object_alias) die("duplicate trace_object_alias field");
      req.trace_object_alias = true;
    } else if (field.starts_with("trace_alias=")) {
      const bool activation = req.op == "act_quant_trace";
      const bool swiglu = req.op == "swiglu_trace";
      const bool fp8 = is_fp8_trace(req);
      if (!activation && !swiglu && !fp8) die("trace_alias requires trace entry");
      if (req.has_trace_alias) die("duplicate trace_alias field");
      const auto pair = field.substr(12);
      const auto comma = pair.find(',');
      if (comma == std::string_view::npos) die("trace_alias requires two buffer names");
      const auto index = [&](std::string_view buffer) {
        if (activation) {
          const auto* hit = std::find(std::begin(kActBuffers), std::end(kActBuffers), buffer);
          if (hit == std::end(kActBuffers)) die("unknown activation buffer");
          return static_cast<std::size_t>(hit - std::begin(kActBuffers));
        }
        if (fp8) {
          const auto* hit = std::find(std::begin(kFp8TraceBuffers),
                                      std::end(kFp8TraceBuffers), buffer);
          if (hit == std::end(kFp8TraceBuffers)) die("unknown FP8 trace buffer");
          return static_cast<std::size_t>(hit - std::begin(kFp8TraceBuffers));
        }
        const auto* hit = std::find(std::begin(kSwigluBuffers), std::end(kSwigluBuffers), buffer);
        if (hit == std::end(kSwigluBuffers)) die("unknown SwiGLU buffer");
        return static_cast<std::size_t>(hit - std::begin(kSwigluBuffers));
      };
      req.act_first = index(pair.substr(0, comma));
      req.act_second = index(pair.substr(comma + 1));
      const std::size_t writable_start = activation ? 3 : fp8 ? 4 : 2;
      if (req.act_first == req.act_second ||
          (req.act_first < writable_start && req.act_second < writable_start)) {
        die("trace_alias needs distinct buffers including a writable range");
      }
      req.has_trace_alias = true;
    } else if (field.starts_with("in_offset=")) {
      if (req.has_in_offset) die("duplicate in_offset field");
      req.in_offset = parse_number(field.substr(10), 10);
      req.has_in_offset = true;
    } else if (field.starts_with("out_offset=")) {
      if (req.has_out_offset) die("duplicate out_offset field");
      req.out_offset = parse_number(field.substr(11), 10);
      req.has_out_offset = true;
    } else if (field == "poison") {
      if (req.poison) die("duplicate poison field");
      req.poison = true;
    } else {
      die("unknown field '" + std::string(field) + "'");
    }
  }
  std::string_view payload = line.substr(colon + 1);
  while (true) {
    const std::size_t sep = payload.find(';');
    auto& section = req.sections.emplace_back();
    for (const auto token : split(payload.substr(0, sep), ' ')) {
      section.push_back(parse_number(token, 16));
    }
    if (sep == std::string_view::npos) {
      break;
    }
    payload = payload.substr(sep + 1);
  }
  req.has_scale_section = req.sections.size() > 1;
  req.tokens = req.sections[0];
  if (req.has_scale_section) {
    req.scale_tokens = req.sections[1];
  }
  return req;
}

bool has_compute_fields(const Request& req) {
  return req.has_overlap || req.has_in_offset || req.has_out_offset || req.poison;
}

void require_sections(const Request& req, std::size_t count, std::string_view family) {
  if (req.sections.size() != count) {
    die(std::string(family) + " request needs exactly " + std::to_string(count) + " sections");
  }
}

template <class T>
void append_hex(std::string& out, T value) {
  char buf[2 * sizeof(T) + 1];
  std::snprintf(buf, sizeof buf, "%0*llx", static_cast<int>(2 * sizeof(T)),
                static_cast<unsigned long long>(static_cast<std::make_unsigned_t<T>>(value)));
  out += ' ';
  out += buf;
}

template <class T>
T narrow_token(std::uint64_t token) {
  using U = std::make_unsigned_t<T>;
  if (token > std::numeric_limits<U>::max()) {
    die("token exceeds the input element width");
  }
  return static_cast<T>(static_cast<U>(token));
}

constexpr bool finite_bits(std::uint32_t bits) {
  return (bits & 0x7F800000u) != 0x7F800000u;
}

std::size_t buffer_bytes(std::size_t count, std::size_t width) {
  if (count > std::numeric_limits<std::size_t>::max() / width) {
    die("executor buffer byte length overflows size_t");
  }
  return count * width;
}

bool buffer_changed(const unsigned char* before, const void* out, std::size_t bytes) {
  return bytes != 0 && std::memcmp(before, out, bytes) != 0;
}

std::string err_response(int outcome, bool changed) {
  std::string response = "err ";
  response += outcome == kOverflow ? "overflow_error" : "invalid_argument";
  response += changed ? " changed=1" : " changed=0";
  return response;
}

template <class Out>
std::string ok_response(const Out* out, std::size_t cap) {
  std::string response = "ok";
  for (std::size_t i = 0; i < cap; ++i) {
    append_hex(response, out[i]);
  }
  return response;
}

// Full-MXCSR invariance window around one production call, exception paths included.
template <class Fn>
int call_guarded(Fn fn) {
  const unsigned before = _mm_getcsr();
  int outcome = kOk;
  try {
    fn();
  } catch (const std::overflow_error&) {
    outcome = kOverflow;
  } catch (const std::invalid_argument&) {
    outcome = kInvalid;
  }
  if (_mm_getcsr() != before) {
    die("production call changed MXCSR");
  }
  return outcome;
}

// Runs one codec library call. In alias mode input and output share one raw allocation.
template <class In, class Out, class Fn>
std::string run(const Request& req, Fn fn) {
  if (!req.has_cap) {
    die("request lacks cap=");
  }
  if (req.has_scale_shape || req.has_scale_section || req.scale_alias || has_compute_fields(req)) {
    die("codec request carries dequant- or compute-only fields");
  }
  const std::size_t n_in = req.tokens.size();
  const std::size_t in_bytes = buffer_bytes(n_in, sizeof(In));
  const std::size_t out_bytes = buffer_bytes(req.cap, sizeof(Out));

  std::unique_ptr<In[]> in_owner;
  std::unique_ptr<Out[]> out_owner;
  std::unique_ptr<std::uint64_t[]> shared;
  In* in_ptr = nullptr;
  Out* out_ptr = nullptr;
  if (req.alias) {
    const std::size_t bytes = std::max(in_bytes, out_bytes);
    shared = std::make_unique<std::uint64_t[]>(bytes / 8 + (bytes % 8 != 0));
    std::memset(shared.get(), kFill, bytes);
    in_ptr = reinterpret_cast<In*>(shared.get());
    out_ptr = reinterpret_cast<Out*>(shared.get());
  } else {
    in_owner = std::make_unique<In[]>(n_in);
    out_owner = std::make_unique<Out[]>(req.cap);
    std::memset(out_owner.get(), kFill, out_bytes);
    in_ptr = in_owner.get();
    out_ptr = out_owner.get();
  }
  for (std::size_t i = 0; i < n_in; ++i) {
    in_ptr[i] = narrow_token<In>(req.tokens[i]);
  }
  const std::vector<unsigned char> before(reinterpret_cast<unsigned char*>(out_ptr),
                                          reinterpret_cast<unsigned char*>(out_ptr) + out_bytes);

  const int outcome = call_guarded([&] {
    fn(std::span<const In>(in_ptr, n_in), q::Shape(req.shape), std::span<Out>(out_ptr, req.cap));
  });
  if (outcome != kOk) {
    return err_response(outcome, buffer_changed(before.data(), out_ptr, out_bytes));
  }
  return ok_response(out_ptr, req.cap);
}

// Runs one dequantization library call with exact, separately allocated data, scale
// and output buffers. `alias` shares one raw allocation between data and output,
// `scale_alias` between scales and output, so overlap rejection stays observable.
template <class Out, class Fn>
std::string run_dequant(const Request& req, Fn fn) {
  if (!req.has_cap) {
    die("request lacks cap=");
  }
  if (!req.has_shape || !req.has_scale_shape || !req.has_scale_section) {
    die("dequant request lacks shape=, scale_shape= or ';'");
  }
  if (req.alias && req.scale_alias) {
    die("alias and scale_alias are mutually exclusive");
  }
  if (has_compute_fields(req)) {
    die("dequant request carries compute-only fields");
  }
  require_sections(req, 2, "dequant");
  const std::size_t n_data = req.tokens.size();
  const std::size_t n_scale = req.scale_tokens.size();
  const std::size_t out_bytes = buffer_bytes(req.cap, sizeof(Out));

  std::unique_ptr<std::uint8_t[]> data_owner;
  std::unique_ptr<std::uint8_t[]> scale_owner;
  std::unique_ptr<std::uint64_t[]> shared;
  std::unique_ptr<Out[]> out_owner;
  std::uint8_t* data_ptr = nullptr;
  std::uint8_t* scale_ptr = nullptr;
  Out* out_ptr = nullptr;
  if (req.alias || req.scale_alias) {
    const std::size_t bytes = std::max(req.alias ? n_data : n_scale, out_bytes);
    shared = std::make_unique<std::uint64_t[]>(bytes / 8 + (bytes % 8 != 0));
    std::memset(shared.get(), kFill, bytes);
    if (req.alias) {
      data_ptr = reinterpret_cast<std::uint8_t*>(shared.get());
      scale_owner = std::make_unique<std::uint8_t[]>(n_scale);
      scale_ptr = scale_owner.get();
    } else {
      scale_ptr = reinterpret_cast<std::uint8_t*>(shared.get());
      data_owner = std::make_unique<std::uint8_t[]>(n_data);
      data_ptr = data_owner.get();
    }
    out_ptr = reinterpret_cast<Out*>(shared.get());
  } else {
    data_owner = std::make_unique<std::uint8_t[]>(n_data);
    scale_owner = std::make_unique<std::uint8_t[]>(n_scale);
    out_owner = std::make_unique<Out[]>(req.cap);
    std::memset(out_owner.get(), kFill, out_bytes);
    data_ptr = data_owner.get();
    scale_ptr = scale_owner.get();
    out_ptr = out_owner.get();
  }
  for (std::size_t i = 0; i < n_data; ++i) {
    data_ptr[i] = narrow_token<std::uint8_t>(req.tokens[i]);
  }
  for (std::size_t i = 0; i < n_scale; ++i) {
    scale_ptr[i] = narrow_token<std::uint8_t>(req.scale_tokens[i]);
  }
  const std::vector<unsigned char> before(reinterpret_cast<unsigned char*>(out_ptr),
                                          reinterpret_cast<unsigned char*>(out_ptr) + out_bytes);
  const q::Shape data_shape(req.shape);
  const q::Shape scale_shape(req.scale_shape);

  const int outcome = call_guarded([&] {
    fn(std::span<const std::uint8_t>(data_ptr, n_data), data_shape,
       std::span<const std::uint8_t>(scale_ptr, n_scale), scale_shape,
       std::span<Out>(out_ptr, req.cap));
  });
  if (outcome != kOk) {
    return err_response(outcome, buffer_changed(before.data(), out_ptr, out_bytes));
  }
  return ok_response(out_ptr, req.cap);
}

// Private probe of the shared multiply helper; the protocol accepts only valid
// finite requests, every other metadata combination is a protocol fault.
std::string run_multiply(const Request& req) {
  if (!req.has_cap || !req.has_shape || !req.has_scale_shape || !req.has_scale_section) {
    die("multiply_finite request lacks cap=, shape=, scale_shape= or ';'");
  }
  if (req.alias || req.scale_alias || has_compute_fields(req)) {
    die("multiply_finite takes no alias flags");
  }
  require_sections(req, 2, "multiply_finite");
  if (req.scale_shape != req.shape || req.tokens.size() != req.scale_tokens.size() ||
      req.cap < req.tokens.size()) {
    die("multiply_finite requires matching shapes and one pair per input element");
  }
  const std::size_t n = req.tokens.size();
  try {
    if (q::detail::element_count(req.shape) != n) {
      die("multiply_finite input count does not match shape");
    }
  } catch (const std::exception&) {
    die("multiply_finite has invalid shape metadata");
  }
  const auto data = std::make_unique<std::uint32_t[]>(n);
  const auto scale = std::make_unique<std::uint32_t[]>(n);
  const std::size_t out_bytes = buffer_bytes(req.cap, sizeof(std::uint32_t));
  const auto out = std::make_unique<std::uint32_t[]>(req.cap);
  std::memset(out.get(), kFill, out_bytes);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t left = narrow_token<std::uint32_t>(req.tokens[i]);
    const std::uint32_t right = narrow_token<std::uint32_t>(req.scale_tokens[i]);
    if (!finite_bits(left) || !finite_bits(right)) {
      die("multiply_finite token is not a finite binary32 pattern");
    }
    data[i] = left;
    scale[i] = right;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const unsigned before = _mm_getcsr();
    out[i] = q::detail::multiply_finite(data[i], scale[i]);
    if (_mm_getcsr() != before) {
      die("multiply_finite changed MXCSR");
    }
  }
  return ok_response(out.get(), req.cap);
}

// Exact-length buffers for the four compute inputs; with overlap= the named input
// lives at byte in_offset of the shared raw allocation instead of owning one.
struct ComputeInputs {
  std::unique_ptr<std::uint8_t[]> owners[4];
  std::span<const std::uint8_t> views[4];
};

ComputeInputs compute_inputs(const Request& req, std::uint8_t* shared) {
  ComputeInputs in;
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& tokens = req.sections[i];
    std::uint8_t* ptr = nullptr;
    if (req.has_overlap && req.overlap == i) {
      ptr = shared + req.in_offset;
    } else {
      in.owners[i] = std::make_unique<std::uint8_t[]>(tokens.size());
      ptr = in.owners[i].get();
    }
    for (std::size_t t = 0; t < tokens.size(); ++t) {
      ptr[t] = narrow_token<std::uint8_t>(tokens[t]);
    }
    in.views[i] = std::span<const std::uint8_t>(ptr, tokens.size());
  }
  return in;
}

void require_compute_fields(const Request& req, bool fp8 = false) {
  if (!req.has_cap) {
    die("request lacks cap=");
  }
  if (req.has_shape || req.has_scale_shape || req.alias || req.scale_alias) {
    die("compute request carries codec or dequant fields");
  }
  if ((req.has_in_offset || req.has_out_offset) && !req.has_overlap) {
    die("in_offset= and out_offset= require overlap=");
  }
  if (req.poison && req.has_overlap && !fp8) {
    die("poison and overlap= are mutually exclusive");
  }
  require_sections(req, 4, "compute");
}

constexpr bool kAsan =
#if defined(__SANITIZE_ADDRESS__)
    true;
#else
    false;
#endif

void set_poison(const ComputeInputs& in, bool poisoned) {
#if defined(__SANITIZE_ADDRESS__)
  for (const auto& view : in.views) {
    if (poisoned) {
      ASAN_POISON_MEMORY_REGION(view.data(), view.size());
    } else {
      ASAN_UNPOISON_MEMORY_REGION(view.data(), view.size());
    }
  }
#else
  (void)in;
  (void)poisoned;
#endif
}

// Runs one public expert_grouped_dot overload on a cap-element output buffer.
template <class Out>
std::string run_compute(const Request& req) {
  require_compute_fields(req);
  const std::size_t out_bytes = buffer_bytes(req.cap, sizeof(Out));
  if (req.out_offset % sizeof(Out) != 0) {
    die("out_offset must be a multiple of the output element size");
  }
  std::size_t bytes = req.out_offset + out_bytes;
  if (req.has_overlap) {
    bytes = std::max(bytes, req.in_offset + req.sections[req.overlap].size());
  }
  const auto raw = std::make_unique<std::uint8_t[]>(bytes);
  std::memset(raw.get(), kFill, bytes);
  const ComputeInputs in = compute_inputs(req, raw.get());
  Out* out_ptr = reinterpret_cast<Out*>(raw.get() + req.out_offset);
  const std::vector<unsigned char> before(reinterpret_cast<unsigned char*>(out_ptr),
                                          reinterpret_cast<unsigned char*>(out_ptr) + out_bytes);
  if (req.poison) {
    set_poison(in, true);
  }
  const int outcome = call_guarded([&] {
    q::expert_grouped_dot(in.views[0], in.views[1], in.views[2], in.views[3],
                          std::span<Out>(out_ptr, req.cap));
  });
  if (req.poison) {
    set_poison(in, false);
  }
  if (outcome != kOk) {
    return err_response(outcome, buffer_changed(before.data(), out_ptr, out_bytes));
  }
  return ok_response(out_ptr, req.cap);
}

std::size_t extent(std::size_t offset, std::size_t bytes);

// P0028 public FP8 grouped dot with input snapshots and exact write-prefix checks.
template <class Out>
std::string run_fp8_compute(const Request& req, std::string_view fault) {
  require_compute_fields(req, true);
  if (req.has_trace_shape || req.has_trace_alias) die("ordinary FP8 dot carries trace fields");
  const std::size_t out_bytes = buffer_bytes(req.cap, sizeof(Out));
  if (req.out_offset % sizeof(Out) != 0) {
    die("out_offset must be a multiple of the output element size");
  }
  std::size_t bytes = extent(req.out_offset, out_bytes);
  if (req.has_overlap) bytes = std::max(bytes, extent(req.in_offset, req.sections[req.overlap].size()));
  const auto raw = std::make_unique<std::uint8_t[]>(bytes);
  std::memset(raw.get(), kFill, bytes);
  const ComputeInputs in = compute_inputs(req, raw.get());
  Out* const out_ptr = reinterpret_cast<Out*>(raw.get() + req.out_offset);
  std::vector<std::vector<unsigned char>> input_before(4);
  for (std::size_t i = 0; i < 4; ++i) {
    input_before[i].assign(in.views[i].begin(), in.views[i].end());
  }
  const std::vector<unsigned char> before(raw.get(), raw.get() + bytes);
  if (req.poison) set_poison(in, true);
  const int outcome = call_guarded([&] {
    q::fp8_grouped_dot(in.views[0], in.views[1], in.views[2], in.views[3],
                       std::span<Out>(out_ptr, req.cap));
  });
  if (req.poison) {
    set_poison(in, false);
#if defined(__SANITIZE_ADDRESS__)
    ASAN_UNPOISON_MEMORY_REGION(raw.get(), bytes);
#endif
  }
  if (outcome != kOk) {
    if (fault == "reject-input-write" && !in.views[0].empty())
      const_cast<std::uint8_t*>(in.views[0].data())[0] ^= 1;
    bool changed = buffer_changed(before.data(), raw.get(), bytes);
    for (std::size_t i = 0; i < 4; ++i)
      changed |= !std::equal(input_before[i].begin(), input_before[i].end(), in.views[i].begin());
    return err_response(outcome, changed);
  }
  if (fault == "input-write" && !in.views[0].empty()) {
    const_cast<std::uint8_t*>(in.views[0].data())[0] ^= 1;
  }
  for (std::size_t i = 0; i < 4; ++i) {
    if (!std::equal(input_before[i].begin(), input_before[i].end(), in.views[i].begin()))
      die("FP8 dot input changed on success");
  }
  const std::size_t written = req.cap == 0 ? 0 : sizeof(Out);
  for (std::size_t i = 0; i < bytes; ++i) {
    const bool output_byte = i >= req.out_offset && i - req.out_offset < written;
    if (!output_byte && raw[i] != before[i]) die("FP8 dot bytes outside write range changed on success");
  }
  return ok_response(out_ptr, req.cap);
}

// Private trace connection: one contiguous response buffer carved into the stage spans.
std::string run_trace(const Request& req) {
  require_compute_fields(req);
  if (has_compute_fields(req)) {
    die("expert_grouped_dot_trace takes no overlap=, offsets or poison");
  }
  const std::size_t k = req.sections[0].size();
  const std::size_t blocks = k / 32;
  if (req.cap != 2 * k + 4 * blocks + 1) {
    die("expert_grouped_dot_trace cap must be 2K + 4K/32 + 1");
  }
  const std::size_t out_bytes = buffer_bytes(req.cap, sizeof(std::uint32_t));
  const auto out = std::make_unique<std::uint32_t[]>(req.cap);
  std::memset(out.get(), kFill, out_bytes);
  const ComputeInputs in = compute_inputs(req, nullptr);
  std::uint32_t* cursor = out.get();
  const auto take = [&cursor](std::size_t n) {
    const std::span<std::uint32_t> span(cursor, n);
    cursor += n;
    return span;
  };
  q::detail::DotTrace trace;
  trace.products = take(k);
  trace.partial_sums = take(k);
  trace.dot = take(blocks);
  trace.after_a = take(blocks);
  trace.block = take(blocks);
  trace.acc = take(blocks);
  std::uint32_t* const result = cursor;
  const std::vector<unsigned char> before(reinterpret_cast<unsigned char*>(out.get()),
                                          reinterpret_cast<unsigned char*>(out.get()) + out_bytes);
  const int outcome = call_guarded([&] {
    *result = q::detail::trace_expert_grouped_dot(in.views[0], in.views[1], in.views[2],
                                                  in.views[3], trace);
  });
  if (outcome != kOk) {
    return err_response(outcome, buffer_changed(before.data(), out.get(), out_bytes));
  }
  return ok_response(out.get(), req.cap);
}

// P0028 trace responses are uint32 words: six stage spans followed by the full output buffer.
template <class Out>
std::string run_fp8_trace(const Request& req, std::string_view fault) {
  if (!req.has_cap || req.has_shape || req.has_scale_shape || req.alias || req.scale_alias ||
      req.has_overlap || req.trace_object_alias || req.has_scale_cap || req.has_a_cap ||
      req.workspace || req.poison_writes || req.runtime_failure || req.has_route ||
      req.has_input_count || req.has_work_shape) {
    die("FP8 trace requires cap= and optional trace_shape=, trace_alias=, offsets or poison");
  }
  if ((req.has_in_offset || req.has_out_offset) && !req.has_trace_alias)
    die("FP8 trace offsets require trace_alias=");
  require_sections(req, 4, "FP8 trace");
  const std::size_t k = req.sections[0].size();
  const std::size_t groups = k / 32;
  const std::vector<std::size_t> shape = req.has_trace_shape
      ? req.trace_shape : std::vector<std::size_t>{k, k, groups, groups, groups, groups};
  if (shape.size() != 6) die("FP8 trace_shape requires six lengths");
  const std::size_t out_cap = req.has_out_cap ? req.out_cap : 1;
  if (req.cap != shape[0] + shape[1] + shape[2] + shape[3] + shape[4] + shape[5] + out_cap)
    die("FP8 trace cap must equal trace elements plus out_cap");
  const std::size_t sizes[] = {req.sections[0].size(), req.sections[1].size(),
      req.sections[2].size(), req.sections[3].size(), buffer_bytes(out_cap, sizeof(Out)),
      buffer_bytes(shape[0], 4), buffer_bytes(shape[1], 4), buffer_bytes(shape[2], 4),
      buffer_bytes(shape[3], 4), buffer_bytes(shape[4], 4), buffer_bytes(shape[5], 4)};
  const std::size_t alignments[] = {1, 1, 1, 1, sizeof(Out), 4, 4, 4, 4, 4, 4};
  std::unique_ptr<std::uint8_t[]> byte_owners[11];
  std::unique_ptr<std::uint16_t[]> half_owners[11];
  std::unique_ptr<std::uint32_t[]> word_owners[11];
  std::unique_ptr<std::uint64_t[]> shared_owner;
  unsigned char* buffers[11] = {};
  unsigned char* shared = nullptr;
  std::size_t shared_size = 0;
  if (req.has_trace_alias) {
    for (const auto& [index, offset] : {std::pair{req.act_first, req.in_offset},
                                       std::pair{req.act_second, req.out_offset}}) {
      if (index >= 11 || offset % alignments[index]) die("unaligned FP8 trace alias offset");
      shared_size = std::max(shared_size, extent(offset, sizes[index]));
    }
    shared_owner = std::make_unique<std::uint64_t[]>(shared_size / 8 + (shared_size % 8 != 0));
    shared = reinterpret_cast<unsigned char*>(shared_owner.get());
    std::memset(shared, kFill, shared_size);
    buffers[req.act_first] = shared + req.in_offset;
    buffers[req.act_second] = shared + req.out_offset;
  }
  for (std::size_t i = 0; i < 11; ++i) {
    if (buffers[i]) continue;
    if (alignments[i] == 1) {
      byte_owners[i] = std::make_unique<std::uint8_t[]>(sizes[i]);
      buffers[i] = byte_owners[i].get();
    } else if (alignments[i] == 2) {
      half_owners[i] = std::make_unique<std::uint16_t[]>(sizes[i] / 2);
      buffers[i] = reinterpret_cast<unsigned char*>(half_owners[i].get());
    } else {
      word_owners[i] = std::make_unique<std::uint32_t[]>(sizes[i] / 4);
      buffers[i] = reinterpret_cast<unsigned char*>(word_owners[i].get());
    }
    std::memset(buffers[i], kFill, sizes[i]);
  }
  for (std::size_t section = 0; section < 4; ++section) {
    for (std::size_t i = 0; i < req.sections[section].size(); ++i)
      buffers[section][i] = narrow_token<std::uint8_t>(req.sections[section][i]);
  }
  std::vector<std::vector<unsigned char>> before(11);
  for (std::size_t i = 0; i < 11; ++i) before[i].assign(buffers[i], buffers[i] + sizes[i]);
  std::vector<unsigned char> before_shared;
  if (shared) before_shared.assign(shared, shared + shared_size);
  const std::span<const std::uint8_t> inputs[] = {
      {buffers[0], req.sections[0].size()}, {buffers[1], req.sections[1].size()},
      {buffers[2], req.sections[2].size()}, {buffers[3], req.sections[3].size()}};
  q::detail::DotTrace trace{{reinterpret_cast<std::uint32_t*>(buffers[5]), shape[0]},
      {reinterpret_cast<std::uint32_t*>(buffers[6]), shape[1]},
      {reinterpret_cast<std::uint32_t*>(buffers[7]), shape[2]},
      {reinterpret_cast<std::uint32_t*>(buffers[8]), shape[3]},
      {reinterpret_cast<std::uint32_t*>(buffers[9]), shape[4]},
      {reinterpret_cast<std::uint32_t*>(buffers[10]), shape[5]}};
#if defined(__SANITIZE_ADDRESS__)
  if (req.poison) for (std::size_t i = 0; i < 4; ++i) ASAN_POISON_MEMORY_REGION(buffers[i], sizes[i]);
#endif
  const int outcome = call_guarded([&] {
    q::detail::trace_fp8_grouped_dot(inputs[0], inputs[1], inputs[2], inputs[3],
        std::span<Out>(reinterpret_cast<Out*>(buffers[4]), out_cap), trace);
  });
#if defined(__SANITIZE_ADDRESS__)
  if (req.poison) for (std::size_t i = 0; i < 4; ++i) ASAN_UNPOISON_MEMORY_REGION(buffers[i], sizes[i]);
  if (shared) ASAN_UNPOISON_MEMORY_REGION(shared, shared_size);
#endif
  if (outcome != kOk) {
    if (fault == "reject-input-write" && sizes[0]) buffers[0][0] ^= 1;
    bool changed = shared && buffer_changed(before_shared.data(), shared, shared_size);
    for (std::size_t i = 0; i < 11; ++i) {
      if (shared && (i == req.act_first || i == req.act_second)) continue;
      changed |= buffer_changed(before[i].data(), buffers[i], sizes[i]);
    }
    return err_response(outcome, changed);
  }
  if (fault == "input-write" && sizes[0]) buffers[0][0] ^= 1;
  for (std::size_t i = 0; i < 4; ++i)
    if (buffer_changed(before[i].data(), buffers[i], sizes[i])) die("FP8 trace input changed on success");
  const std::size_t used[] = {0, 0, 0, 0, sizeof(Out), sizes[5], sizes[6], sizes[7],
                              sizes[8], sizes[9], sizes[10]};
  for (std::size_t i = 4; i < 11; ++i) {
    if (shared && (i == req.act_first || i == req.act_second)) continue;
    if (buffer_changed(before[i].data() + std::min(used[i], sizes[i]),
                       buffers[i] + std::min(used[i], sizes[i]),
                       sizes[i] - std::min(used[i], sizes[i])))
      die("FP8 trace bytes outside write ranges changed on success");
  }
  if (shared) {
    for (std::size_t offset = 0; offset < shared_size; ++offset) {
      bool writable = false;
      for (const std::size_t index : {req.act_first, req.act_second}) {
        if (index < 4) continue;
        const std::size_t start = static_cast<std::size_t>(buffers[index] - shared);
        writable |= offset >= start && offset - start < std::min(used[index], sizes[index]);
      }
      if (!writable && shared[offset] != before_shared[offset])
        die("FP8 trace bytes outside write ranges changed on success");
    }
  }
  std::string response = "ok";
  for (std::size_t index = 5; index < 11; ++index) {
    for (std::size_t offset = 0; offset < sizes[index]; offset += 4) {
      std::uint32_t value = 0;
      std::memcpy(&value, buffers[index] + offset, 4);
      append_hex(response, value);
    }
  }
  for (std::size_t i = 0; i < out_cap; ++i) {
    Out value = 0;
    std::memcpy(&value, buffers[4] + i * sizeof(Out), sizeof(Out));
    append_hex(response, static_cast<std::uint32_t>(value));
  }
  return response;
}

// Private probes of production FP32 helpers; every bit pattern is a valid operand.
std::string run_binary_fp32(const Request& req, bool multiply) {
  if (!req.has_cap || req.has_shape || req.has_scale_shape || req.alias || req.scale_alias ||
      has_compute_fields(req)) {
    die("binary FP32 request takes only cap=");
  }
  require_sections(req, 2, multiply ? "multiply_fp32" : "add_fp32");
  const std::size_t n = req.sections[0].size();
  if (req.sections[1].size() != n || req.cap < n) {
    die("binary FP32 request requires one right operand per left operand and cap >= pairs");
  }
  const std::size_t out_bytes = buffer_bytes(req.cap, sizeof(std::uint32_t));
  const auto out = std::make_unique<std::uint32_t[]>(req.cap);
  std::memset(out.get(), kFill, out_bytes);
  std::vector<std::uint32_t> left(n);
  std::vector<std::uint32_t> right(n);
  for (std::size_t i = 0; i < n; ++i) {
    left[i] = narrow_token<std::uint32_t>(req.sections[0][i]);
    right[i] = narrow_token<std::uint32_t>(req.sections[1][i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    const unsigned before = _mm_getcsr();
    out[i] = multiply ? q::detail::multiply_fp32(left[i], right[i])
                      : q::detail::add_fp32(left[i], right[i]);
    if (_mm_getcsr() != before) {
      die("binary FP32 helper changed MXCSR");
    }
  }
  return ok_response(out.get(), req.cap);
}

// P0017 activation calls own exact-length buffers; every output, including trace
// stages, is snapshotted before rejection, and poisoned inputs prove metadata order.
std::size_t extent(std::size_t offset, std::size_t bytes) {
  if (offset > std::numeric_limits<std::size_t>::max() - bytes) die("buffer extent overflow");
  return offset + bytes;
}

void poison_activation(std::span<const std::uint16_t> input, bool poisoned) {
#if defined(__SANITIZE_ADDRESS__)
  if (poisoned) ASAN_POISON_MEMORY_REGION(input.data(), input.size_bytes());
  else ASAN_UNPOISON_MEMORY_REGION(input.data(), input.size_bytes());
#else
  (void)input;
  (void)poisoned;
#endif
}

std::string run_act_quant(const Request& req, std::string_view fault) {
  const bool traced = req.op == "act_quant_trace";
  if (!req.has_cap || !req.has_scale_cap || req.has_shape || req.has_scale_shape ||
      req.alias || req.scale_alias || (!traced && req.has_trace_shape)) {
    die("activation request needs cap= and scale_cap=, and no foreign fields");
  }
  require_sections(req, 1, "activation");
  if (req.has_overlap && req.has_trace_alias) die("overlap and trace_alias are mutually exclusive");
  const bool shared_pair = req.has_overlap || req.has_trace_alias;
  if ((req.has_in_offset || req.has_out_offset) && !shared_pair) die("offsets require an alias pair");
  const std::size_t k = req.tokens.size();
  if (req.trace_object_alias && (shared_pair || req.has_trace_shape || k != 64 ||
                                req.cap != k || req.cap > sizeof(q::detail::ActQuantTrace))) {
    die("trace_object_alias requires K=cap=64 and independent stage buffers");
  }
  std::vector<std::size_t> shape = {k / 32, k / 32, k / 32, k / 32, k, k};
  if (req.has_trace_shape) {
    if (req.trace_shape.size() != 6) die("trace_shape requires six lengths");
    shape = req.trace_shape;
  }
  const std::size_t count = traced ? 9 : 3;
  std::size_t lengths[9] = {buffer_bytes(k, 2), req.cap, req.scale_cap};
  for (std::size_t i = 3; i < count; ++i) lengths[i] = buffer_bytes(shape[i - 3], 4);
  std::unique_ptr<std::uint16_t[]> input_owner;
  std::unique_ptr<std::uint8_t[]> byte_owner[2];
  std::unique_ptr<std::uint32_t[]> stage_owner[6];
  unsigned char* buffers[9] = {};
  const auto delete_shared = [](void* ptr) { ::operator delete(ptr); };
  std::unique_ptr<void, decltype(delete_shared)> shared(nullptr, delete_shared);
  std::size_t shared_bytes = 0;
  if (shared_pair) {
    for (const auto& [index, offset] : {std::pair{req.act_first, req.in_offset},
                                     std::pair{req.act_second, req.out_offset}}) {
      const std::size_t alignment = index == 0 ? 2 : index < 3 ? 1 : 4;
      if (offset % alignment) die("unaligned activation alias offset");
      shared_bytes = std::max(shared_bytes, extent(offset, lengths[index]));
    }
    shared.reset(::operator new(shared_bytes));
    std::memset(shared.get(), kFill, shared_bytes);
    auto* base = reinterpret_cast<unsigned char*>(shared.get());
    buffers[req.act_first] = base + req.in_offset;
    buffers[req.act_second] = base + req.out_offset;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (buffers[i] != nullptr) continue;
    if (i == 0) {
      input_owner = std::make_unique<std::uint16_t[]>(k);
      buffers[i] = reinterpret_cast<unsigned char*>(input_owner.get());
    } else if (i < 3) {
      byte_owner[i - 1] = std::make_unique<std::uint8_t[]>(lengths[i]);
      buffers[i] = byte_owner[i - 1].get();
    } else {
      stage_owner[i - 3] = std::make_unique<std::uint32_t[]>(shape[i - 3]);
      buffers[i] = reinterpret_cast<unsigned char*>(stage_owner[i - 3].get());
    }
  }
  for (std::size_t i = 1; i < count; ++i) std::memset(buffers[i], kFill, lengths[i]);
  for (std::size_t i = 0; i < k; ++i) {
    const auto value = narrow_token<std::uint16_t>(req.tokens[i]);
    std::memcpy(buffers[0] + i * 2, &value, 2);
  }
  auto* base = reinterpret_cast<unsigned char*>(shared.get());
  auto* input = reinterpret_cast<std::uint16_t*>(buffers[0]);
  auto* scale = buffers[2];
  q::detail::ActQuantTrace trace;
  if (traced) {
    trace.amax_bits = {reinterpret_cast<std::uint32_t*>(buffers[3]), shape[0]};
    trace.bounded_amax_bits = {reinterpret_cast<std::uint32_t*>(buffers[4]), shape[1]};
    trace.product_bits = {reinterpret_cast<std::uint32_t*>(buffers[5]), shape[2]};
    trace.exponent = {reinterpret_cast<std::int32_t*>(buffers[6]), shape[3]};
    trace.quotient_bits = {reinterpret_cast<std::uint32_t*>(buffers[7]), shape[4]};
    trace.clamped_bits = {reinterpret_cast<std::uint32_t*>(buffers[8]), shape[5]};
  }
  if (req.trace_object_alias) buffers[1] = reinterpret_cast<unsigned char*>(&trace);
  auto* a = buffers[1];
  std::vector<unsigned char> before[9];
  for (std::size_t i = 0; i < count; ++i) {
    before[i].assign(buffers[i], buffers[i] + lengths[i]);
  }
  std::vector<unsigned char> before_shared;
  if (shared_pair) before_shared.assign(base, base + shared_bytes);
  const auto* object_bytes = reinterpret_cast<const unsigned char*>(&trace);
  const std::vector<unsigned char> before_object(object_bytes, object_bytes + sizeof(trace));
  const std::span<const std::uint16_t> in(input, k);
  if (req.poison) poison_activation(in, true);
#if defined(__SANITIZE_ADDRESS__)
  if (req.poison && shared_pair) ASAN_POISON_MEMORY_REGION(base, shared_bytes);
#endif
  const int outcome = call_guarded([&] {
    if (traced) q::detail::trace_act_quant(in, {a, req.cap}, {scale, req.scale_cap}, trace);
    else q::act_quant(in, {a, req.cap}, {scale, req.scale_cap});
  });
  if (req.poison) poison_activation(in, false);
#if defined(__SANITIZE_ADDRESS__)
  if (shared_pair) ASAN_UNPOISON_MEMORY_REGION(base, shared_bytes);
#endif
  if (outcome != kOk) {
    bool changed = shared_pair && buffer_changed(before_shared.data(), base, shared_bytes);
    for (std::size_t i = 0; i < count; ++i) {
      changed |= buffer_changed(before[i].data(), buffers[i], lengths[i]);
    }
    if (req.trace_object_alias) {
      changed |= buffer_changed(before_object.data(), &trace, sizeof(trace));
    }
    return err_response(outcome, changed);
  }
  const auto written_bytes = [&](std::size_t i) {
    return i == 1 ? k : i == 2 ? k / 32 : lengths[i];
  };
  const auto shared_writable = [&](std::size_t offset) {
    for (std::size_t i : {req.act_first, req.act_second}) {
      if (i == 0) continue;
      const std::size_t start = static_cast<std::size_t>(buffers[i] - base);
      if (offset >= start && offset - start < written_bytes(i)) return true;
    }
    return false;
  };
  if (fault == "input-write") buffers[0][0] ^= 1;
  if (fault == "shared-gap-write" && shared_pair) {
    for (std::size_t i = 0; i < shared_bytes; ++i) {
      if (!shared_writable(i)) {
        base[i] ^= 1;
        break;
      }
    }
  }
  if (buffer_changed(before[0].data(), buffers[0], lengths[0])) {
    die("activation input changed on success");
  }
  for (std::size_t i = 1; i < count; ++i) {
    if (shared_pair && (i == req.act_first || i == req.act_second)) continue;
    const std::size_t used = written_bytes(i);
    if (buffer_changed(before[i].data() + used, buffers[i] + used, lengths[i] - used)) {
      die("activation bytes outside write ranges changed on success");
    }
  }
  if (shared_pair) {
    for (std::size_t i = 0; i < shared_bytes; ++i) {
      if (!shared_writable(i) && base[i] != before_shared[i]) {
        die("activation bytes outside write ranges changed on success");
      }
    }
  }
  if (req.trace_object_alias && buffer_changed(before_object.data() + k,
                                               object_bytes + k, sizeof(trace) - k)) {
    die("activation trace descriptor tail changed on success");
  }
  std::string response = "ok";
  for (std::size_t i = 0; i < req.cap; ++i) append_hex(response, static_cast<std::uint32_t>(a[i]));
  for (std::size_t i = 0; i < req.scale_cap; ++i) append_hex(response, static_cast<std::uint32_t>(scale[i]));
  if (traced) {
    for (std::size_t i = 0; i < 6; ++i) {
      for (std::size_t j = 0; j < shape[i]; ++j) {
        std::uint32_t value = 0;
        std::memcpy(&value, buffers[i + 3] + j * 4, 4);
        append_hex(response, value);
      }
    }
  }
  return response;
}

void poison_swiglu(std::span<const std::uint16_t> gate, std::span<const std::uint16_t> up,
                    bool poisoned) {
#if defined(__SANITIZE_ADDRESS__)
  for (const auto input : {gate, up}) {
    if (poisoned) ASAN_POISON_MEMORY_REGION(input.data(), input.size_bytes());
    else ASAN_UNPOISON_MEMORY_REGION(input.data(), input.size_bytes());
  }
#else
  (void)gate;
  (void)up;
  (void)poisoned;
#endif
}

std::string run_swiglu(const Request& req, std::string_view fault) {
  const bool traced = req.op == "swiglu_trace";
  if (!req.has_cap || req.has_shape || req.has_scale_shape || req.has_scale_cap ||
      req.has_a_cap || req.workspace || req.poison_writes || req.alias || req.scale_alias ||
      req.trace_object_alias || (!traced && (req.has_trace_shape || req.has_trace_alias))) {
    die("SwiGLU requires cap= and no foreign fields");
  }
  require_sections(req, 2, "SwiGLU");
  if (req.has_overlap && req.has_trace_alias) die("overlap and trace_alias are mutually exclusive");
  const bool shared_pair = req.has_overlap || req.has_trace_alias;
  if ((req.has_in_offset || req.has_out_offset) && !shared_pair) die("SwiGLU offsets require an alias pair");
  if (req.has_overlap && traced && (req.act_first >= 3 || req.act_second >= 3))
    die("trace buffers require trace_alias");
  if (req.has_overlap && !traced && (req.act_first >= 3 || req.act_second >= 3))
    die("ordinary SwiGLU has no trace buffers");

  const std::size_t gate_count = req.sections[0].size();
  const std::size_t up_count = req.sections[1].size();
  std::vector<std::size_t> shape(5, gate_count);
  if (req.has_trace_shape) {
    if (req.trace_shape.size() != 5) die("SwiGLU trace_shape requires five lengths");
    shape = req.trace_shape;
  }
  const std::size_t count = traced ? 8 : 3;
  std::size_t lengths[8] = {buffer_bytes(gate_count, 2), buffer_bytes(up_count, 2),
                            buffer_bytes(req.cap, 2)};
  for (std::size_t i = 3; i < count; ++i) lengths[i] = buffer_bytes(shape[i - 3], 4);
  const std::size_t widths[8] = {2, 2, 2, 4, 4, 4, 4, 4};
  const std::size_t alignments[8] = {2, 2, 2, 4, 4, 4, 4, 4};
  const auto deleter = [](void* pointer) { ::operator delete(pointer); };
  using Owner = std::unique_ptr<void, decltype(deleter)>;
  Owner owners[] = {Owner(nullptr, deleter), Owner(nullptr, deleter), Owner(nullptr, deleter),
                    Owner(nullptr, deleter), Owner(nullptr, deleter), Owner(nullptr, deleter),
                    Owner(nullptr, deleter), Owner(nullptr, deleter)};
  Owner shared(nullptr, deleter);
  unsigned char* buffers[8] = {};
  unsigned char* shared_base = nullptr;
  std::size_t shared_size = 0;
  if (shared_pair) {
    for (const auto& [index, offset] : {std::pair{req.act_first, req.in_offset},
                                       std::pair{req.act_second, req.out_offset}}) {
      if (index >= count) die("SwiGLU alias names an unavailable buffer");
      if (offset % alignments[index]) die("unaligned SwiGLU alias offset");
      shared_size = std::max(shared_size, extent(offset, lengths[index]));
    }
    shared.reset(::operator new(shared_size));
    shared_base = static_cast<unsigned char*>(shared.get());
    std::memset(shared_base, kFill, shared_size);
    buffers[req.act_first] = shared_base + req.in_offset;
    buffers[req.act_second] = shared_base + req.out_offset;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (!buffers[i]) {
      owners[i].reset(::operator new(lengths[i]));
      buffers[i] = static_cast<unsigned char*>(owners[i].get());
    }
    std::memset(buffers[i], kFill, lengths[i]);
  }
  for (std::size_t section = 0; section < 2; ++section) {
    for (std::size_t i = 0; i < req.sections[section].size(); ++i) {
      const auto value = narrow_token<std::uint16_t>(req.sections[section][i]);
      std::memcpy(buffers[section] + 2 * i, &value, 2);
    }
  }
  std::vector<unsigned char> before[8];
  for (std::size_t i = 0; i < count; ++i) before[i].assign(buffers[i], buffers[i] + lengths[i]);
  std::vector<unsigned char> before_shared;
  if (shared_pair) before_shared.assign(shared_base, shared_base + shared_size);

  const auto overlaps = [](std::size_t first, std::size_t first_size,
                           std::size_t second, std::size_t second_size) {
    return first_size != 0 && second_size != 0 && first < extent(second, second_size) &&
           second < extent(first, first_size);
  };
  const bool mixed_overlap = shared_pair && widths[req.act_first] != widths[req.act_second] &&
      overlaps(req.in_offset, lengths[req.act_first], req.out_offset, lengths[req.act_second]);
  const std::span<const std::uint16_t> gate(reinterpret_cast<const std::uint16_t*>(buffers[0]), gate_count);
  const std::span<const std::uint16_t> up(reinterpret_cast<const std::uint16_t*>(buffers[1]), up_count);
  const std::span<std::uint16_t> out(reinterpret_cast<std::uint16_t*>(buffers[2]), req.cap);
  q::detail::SwigluTrace trace;
  if (traced && !mixed_overlap) {
    trace.gate_bits = {reinterpret_cast<std::uint32_t*>(buffers[3]), shape[0]};
    trace.up_bits = {reinterpret_cast<std::uint32_t*>(buffers[4]), shape[1]};
    trace.silu_bits = {reinterpret_cast<std::uint32_t*>(buffers[5]), shape[2]};
    trace.gated_bits = {reinterpret_cast<std::uint32_t*>(buffers[6]), shape[3]};
    trace.weighted_bits = {reinterpret_cast<std::uint32_t*>(buffers[7]), shape[4]};
  }
  if (req.poison && !mixed_overlap) poison_swiglu(gate, up, true);
  const int outcome = call_guarded([&] {
    if (mixed_overlap) {
      const bool trace_shape_valid = !traced ||
          std::all_of(shape.begin(), shape.end(), [&](std::size_t value) { return value == gate_count; });
      if (gate_count != up_count || req.cap < gate_count || !trace_shape_valid ||
          (req.has_route && !finite_bits(req.route))) {
        die("mixed SwiGLU alias probe requires otherwise valid metadata");
      }
      std::array<std::span<std::byte>, 6> writes{};
      writes[0] = {reinterpret_cast<std::byte*>(buffers[2]), 2 * gate_count};
      if (traced) {
        for (std::size_t i = 1; i < writes.size(); ++i)
          writes[i] = {reinterpret_cast<std::byte*>(buffers[i + 2]), 4 * shape[i - 1]};
      }
      q::detail::check_swiglu_storage(
          {reinterpret_cast<const std::byte*>(buffers[0]), 2 * gate_count},
          {reinterpret_cast<const std::byte*>(buffers[1]), 2 * up_count}, writes);
    } else if (traced) {
      if (req.has_route) q::detail::trace_swiglu(gate, up, req.route, out, trace);
      else q::detail::trace_swiglu(gate, up, out, trace);
    } else if (req.has_route) {
      q::swiglu(gate, up, req.route, out);
    } else {
      q::swiglu(gate, up, out);
    }
  });
  if (req.poison && !mixed_overlap) poison_swiglu(gate, up, false);
  if (outcome != kOk) {
    if (fault == "reject-input-write" && gate_count) buffers[0][0] ^= 1;
    bool changed = shared_pair && buffer_changed(before_shared.data(), shared_base, shared_size);
    for (std::size_t i = 0; i < count; ++i)
      changed |= buffer_changed(before[i].data(), buffers[i], lengths[i]);
    return err_response(outcome, changed);
  }
  if (gate_count != up_count || req.cap < gate_count) die("SwiGLU accepted invalid metadata");
  const std::size_t used[8] = {0, 0, buffer_bytes(gate_count, 2),
                               buffer_bytes(shape[0], 4), buffer_bytes(shape[1], 4),
                               buffer_bytes(shape[2], 4), buffer_bytes(shape[3], 4),
                               buffer_bytes(shape[4], 4)};
  const auto shared_writable = [&](std::size_t offset) {
    if (!shared_pair) return false;
    for (std::size_t index : {req.act_first, req.act_second}) {
      if (index < 2) continue;
      const std::size_t start = static_cast<std::size_t>(buffers[index] - shared_base);
      if (offset >= start && offset - start < used[index]) return true;
    }
    return false;
  };
  if (fault == "input-write" && gate_count) buffers[0][0] ^= 1;
  if (fault == "shared-gap-write" && shared_pair) {
    for (std::size_t i = 0; i < shared_size; ++i) {
      if (!shared_writable(i)) { shared_base[i] ^= 1; break; }
    }
  }
  for (std::size_t i = 0; i < 2; ++i) {
    if (buffer_changed(before[i].data(), buffers[i], lengths[i])) die("SwiGLU input changed on success");
  }
  for (std::size_t i = 2; i < count; ++i) {
    if (shared_pair && (i == req.act_first || i == req.act_second)) continue;
    if (buffer_changed(before[i].data() + used[i], buffers[i] + used[i], lengths[i] - used[i]))
      die("SwiGLU bytes outside write ranges changed on success");
  }
  if (shared_pair) {
    for (std::size_t i = 0; i < shared_size; ++i) {
      if (!shared_writable(i) && shared_base[i] != before_shared[i])
        die("SwiGLU bytes outside write ranges changed on success");
    }
  }
  std::string response = "ok";
  for (std::size_t i = 0; i < req.cap; ++i) append_hex(response, static_cast<std::uint32_t>(out[i]));
  if (traced) {
    for (const auto stage : {trace.gate_bits, trace.up_bits, trace.silu_bits,
                             trace.gated_bits, trace.weighted_bits}) {
      for (const auto value : stage) append_hex(response, value);
    }
  }
  return response;
}

// Private scalar probe of the SwiGLU count checks shared by every production
// entry; huge element counts stay testable because no span is constructed.
std::string run_swiglu_counts(const Request& req, bool traced) {
  if (!req.has_cap || req.has_shape || req.has_scale_shape || req.has_scale_cap ||
      req.has_a_cap || req.workspace || req.alias || req.scale_alias || req.poison ||
      req.poison_writes || req.has_overlap || req.has_trace_alias || req.trace_object_alias ||
      req.has_route || req.has_trace_shape || req.has_in_offset || req.has_out_offset) {
    die("swiglu_counts takes only cap=");
  }
  require_sections(req, 1, "swiglu_counts");
  if (!req.sections[0].empty()) die("swiglu_counts takes no payload tokens");
  const int outcome = call_guarded([&] { q::detail::check_swiglu_counts(req.cap, traced); });
  if (outcome != kOk) {
    return err_response(outcome, false);
  }
  return "ok";
}

// P0017 composition exists only here: production activation buffers are passed
// directly to the production grouped dot, without a driver round-trip or public API.
template <class Out>
std::string run_act_combination(const Request& req, bool fp8) {
  if (!req.has_cap || req.has_scale_cap || req.has_trace_shape || req.has_shape ||
      req.has_scale_shape || req.alias || req.scale_alias || has_compute_fields(req)) {
    die("activation combination takes only cap=");
  }
  require_sections(req, 3, "activation combination");
  const std::size_t k = req.sections[0].size();
  const auto input = std::make_unique<std::uint16_t[]>(k);
  for (std::size_t i = 0; i < k; ++i) input[i] = narrow_token<std::uint16_t>(req.sections[0][i]);
  const auto a = std::make_unique<std::uint8_t[]>(k);
  const auto scale = std::make_unique<std::uint8_t[]>(k / 32);
  Request compute;
  compute.sections = {{}, req.sections[1], {}, req.sections[2]};
  auto in = compute_inputs(compute, nullptr);
  const auto out = std::make_unique<Out[]>(req.cap);
  const std::size_t bytes = buffer_bytes(req.cap, sizeof(Out));
  std::memset(out.get(), kFill, bytes);
  const std::vector<unsigned char> before(reinterpret_cast<unsigned char*>(out.get()),
                                         reinterpret_cast<unsigned char*>(out.get()) + bytes);
  const int outcome = call_guarded([&] {
    q::act_quant({input.get(), k}, {a.get(), k}, {scale.get(), k / 32});
    if (fp8) {
      q::fp8_grouped_dot({a.get(), k}, in.views[1], {scale.get(), k / 32}, in.views[3],
                         std::span<Out>(out.get(), req.cap));
    } else {
      q::expert_grouped_dot({a.get(), k}, in.views[1], {scale.get(), k / 32}, in.views[3],
                            std::span<Out>(out.get(), req.cap));
    }
  });
  if (outcome != kOk) return err_response(outcome, buffer_changed(before.data(), out.get(), bytes));
  return ok_response(out.get(), req.cap);
}

// P0019 linear requests own exact allocations, snapshot all storage, and call only
// the public matrix API. Workspace responses encode every element as a uint32 word.
// The executor rejects intersecting full FP32-output/BF16-input spans: differently
// typed arrays cannot share their object lifetimes, even in unused output tails.
// This limits legal C++ test construction, not the production overlap contract.
template <class Out>
std::string run_linear(const Request& req, std::string_view fault) {
  if (!req.has_cap || !req.has_shape || req.shape.size() != 1 || req.has_scale_shape ||
      req.alias || req.scale_alias || req.has_trace_shape || req.has_trace_alias ||
      req.trace_object_alias) die("linear requires shape=N cap= and no foreign fields");
  require_sections(req, 3, "linear");
  if ((req.has_in_offset || req.has_out_offset) && !req.has_overlap) die("linear offsets need overlap");
  const std::size_t k = req.sections[0].size(), n = req.shape[0];
  const std::size_t a_cap = req.has_a_cap ? req.a_cap : k;
  const std::size_t sa_cap = req.has_scale_cap ? req.scale_cap : k / 32;
  const std::size_t sizes[] = {buffer_bytes(k, 2), req.sections[1].size(),
      req.sections[2].size(), a_cap, sa_cap, buffer_bytes(req.cap, sizeof(Out))};
  const std::size_t alignments[] = {2, 1, 1, 1, 1, sizeof(Out)};
  const auto deleter = [](void* p) { ::operator delete(p); };
  using Owner = std::unique_ptr<void, decltype(deleter)>;
  Owner owners[] = {Owner(nullptr, deleter), Owner(nullptr, deleter), Owner(nullptr, deleter),
                    Owner(nullptr, deleter), Owner(nullptr, deleter), Owner(nullptr, deleter)};
  Owner shared(nullptr, deleter);
  std::unique_ptr<std::uint16_t[]> typed_shared;
  const bool input_output_pair = req.has_overlap &&
      ((req.act_first == 0 && req.act_second == 5) ||
       (req.act_first == 5 && req.act_second == 0));
  unsigned char* shared_base = nullptr;
  unsigned char* buffers[6] = {};
  std::size_t shared_size = 0;
  if (req.has_overlap) {
    for (const auto& [i, offset] : {std::pair{req.act_first, req.in_offset},
                                   std::pair{req.act_second, req.out_offset}}) {
      if (offset % alignments[i]) die("unaligned linear offset");
      shared_size = std::max(shared_size, extent(offset, sizes[i]));
    }
    if constexpr (std::is_same_v<Out, std::uint32_t>) {
      if (input_output_pair) {
        if (sizes[req.act_first] != 0 && sizes[req.act_second] != 0 &&
            req.in_offset < extent(req.out_offset, sizes[req.act_second]) &&
            req.out_offset < extent(req.in_offset, sizes[req.act_first])) {
          die("linear FP32 output and BF16 input full spans cannot intersect");
        }
      }
    }
    if constexpr (std::is_same_v<Out, std::uint16_t>) {
      if (input_output_pair) {
        typed_shared = std::make_unique<std::uint16_t[]>(shared_size / 2);
        shared_base = reinterpret_cast<unsigned char*>(typed_shared.get());
      }
    }
    if (!shared_base) {
      shared.reset(::operator new(shared_size));
      shared_base = static_cast<unsigned char*>(shared.get());
    }
    std::memset(shared_base, kFill, shared_size);
    buffers[req.act_first] = shared_base + req.in_offset;
    buffers[req.act_second] = shared_base + req.out_offset;
  }
  for (std::size_t i = 0; i < 6; ++i) {
    if (!buffers[i]) {
      owners[i].reset(::operator new(sizes[i]));
      buffers[i] = static_cast<unsigned char*>(owners[i].get());
    }
    std::memset(buffers[i], kFill, sizes[i]);
  }
  for (std::size_t i = 0; i < k; ++i) {
    const auto value = narrow_token<std::uint16_t>(req.sections[0][i]);
    std::memcpy(buffers[0] + 2 * i, &value, 2);
  }
  for (std::size_t b = 1; b < 3; ++b) {
    for (std::size_t i = 0; i < sizes[b]; ++i) buffers[b][i] = narrow_token<std::uint8_t>(req.sections[b][i]);
  }
  std::vector<unsigned char> before[6];
  for (std::size_t i = 0; i < 6; ++i) before[i].assign(buffers[i], buffers[i] + sizes[i]);
  auto* base = shared_base;
  std::vector<unsigned char> before_shared;
  if (req.has_overlap) before_shared.assign(base, base + shared_size);
#if defined(__SANITIZE_ADDRESS__)
  for (std::size_t i = 0; i < 6; ++i) {
    if ((req.poison && i < 3) || (req.poison_writes && i >= 3))
      ASAN_POISON_MEMORY_REGION(buffers[i], sizes[i]);
  }
  if (req.poison && req.has_overlap) ASAN_POISON_MEMORY_REGION(base, shared_size);
#endif
  const int outcome = call_guarded([&] {
    if constexpr (std::is_same_v<Out, std::uint16_t>) {
      if (input_output_pair) {
        const std::span<std::uint16_t> storage(typed_shared.get(), shared_size / 2);
        const std::size_t input_offset = req.act_first == 0 ? req.in_offset : req.out_offset;
        const std::size_t output_offset = req.act_first == 5 ? req.in_offset : req.out_offset;
        q::expert_linear(storage.subspan(input_offset / 2, k), n,
                         {buffers[1], sizes[1]}, {buffers[2], sizes[2]},
                         {buffers[3], a_cap}, {buffers[4], sa_cap},
                         storage.subspan(output_offset / 2, req.cap));
        return;
      }
    }
    q::expert_linear({reinterpret_cast<const std::uint16_t*>(buffers[0]), k}, n,
                     {buffers[1], sizes[1]}, {buffers[2], sizes[2]},
                     {buffers[3], a_cap}, {buffers[4], sa_cap},
                     std::span<Out>(reinterpret_cast<Out*>(buffers[5]), req.cap));
  });
#if defined(__SANITIZE_ADDRESS__)
  for (std::size_t i = 0; i < 6; ++i) ASAN_UNPOISON_MEMORY_REGION(buffers[i], sizes[i]);
  if (req.has_overlap) ASAN_UNPOISON_MEMORY_REGION(base, shared_size);
#endif
  if (outcome != kOk) {
    bool changed = req.has_overlap && buffer_changed(before_shared.data(), base, shared_size);
    for (std::size_t i = 0; i < 6; ++i) changed |= buffer_changed(before[i].data(), buffers[i], sizes[i]);
    return err_response(outcome, changed);
  }
  if (k > a_cap || k / 32 > sa_cap || n > req.cap) die("linear accepted undersized output");
  const std::size_t used[] = {0, 0, 0, k, k / 32, buffer_bytes(n, sizeof(Out))};
  const auto writable = [&](std::size_t offset) {
    for (std::size_t i : {req.act_first, req.act_second}) {
      if (i < 3) continue;
      const std::size_t start = static_cast<std::size_t>(buffers[i] - base);
      if (offset >= start && offset - start < used[i]) return true;
    }
    return false;
  };
  if (fault == "input-write" && k) buffers[0][0] ^= 1;
  if (fault == "shared-gap-write" && req.has_overlap) {
    for (std::size_t i = 0; i < shared_size; ++i) {
      if (!writable(i)) { base[i] ^= 1; break; }
    }
  }
  for (std::size_t i = 0; i < 6; ++i) {
    if (i < 3 && buffer_changed(before[i].data(), buffers[i], sizes[i])) die("linear input changed on success");
    if (req.has_overlap && (i == req.act_first || i == req.act_second)) continue;
    if (buffer_changed(before[i].data() + used[i], buffers[i] + used[i], sizes[i] - used[i]))
      die("linear bytes outside write ranges changed on success");
  }
  if (req.has_overlap) {
    for (std::size_t i = 0; i < shared_size; ++i) {
      if (!writable(i) && base[i] != before_shared[i]) die("linear bytes outside write ranges changed on success");
    }
  }
  std::string response = "ok";
  for (std::size_t i = 0; i < req.cap; ++i) {
    Out value = 0;
    std::memcpy(&value, buffers[5] + i * sizeof(Out), sizeof(value));
    if (req.workspace) append_hex(response, static_cast<std::uint32_t>(value));
    else append_hex(response, value);
  }
  if (req.workspace) {
    for (std::size_t b : {3u, 4u}) {
      for (std::size_t i = 0; i < sizes[b]; ++i) append_hex(response, static_cast<std::uint32_t>(buffers[b][i]));
    }
  }
  return response;
}

std::string run_expert_storage(const Request& req) {
  if (!req.has_cap || !req.has_work_shape || req.work_shape.size() != 4 || req.has_shape ||
      req.has_scale_shape || req.has_scale_cap || req.has_a_cap || req.workspace || req.alias ||
      req.scale_alias || req.poison || req.poison_writes || req.runtime_failure || !req.has_overlap ||
      req.has_trace_alias || req.trace_object_alias || req.has_route || req.has_trace_shape ||
      !req.has_in_offset || !req.has_out_offset || req.has_input_count) {
    die("expert_storage requires cap=, work_shape=read,write,out,out_write and overlap=");
  }
  require_sections(req, 1, "expert_storage");
  if (!req.sections[0].empty()) die("expert_storage takes no payload");
  const std::size_t read_size = req.work_shape[0];
  const std::size_t write_size = req.work_shape[1];
  const std::size_t out_size = req.work_shape[2];
  const std::size_t out_write = req.work_shape[3];
  if (req.cap != out_size) die("expert_storage cap must equal out byte size");
  const std::size_t sizes[] = {read_size, write_size, out_size};
  if (req.act_first >= 3 || req.act_second >= 3) die("expert_storage overlap uses input_codes,input_scales,gate");
  const std::size_t total = std::max(extent(req.in_offset, sizes[req.act_first]),
                                     extent(req.out_offset, sizes[req.act_second]));
  const auto raw = std::make_unique<std::byte[]>(total);
  std::unique_ptr<std::byte[]> owners[3];
  std::span<std::byte> buffers[3];
  buffers[req.act_first] = {raw.get() + req.in_offset, sizes[req.act_first]};
  buffers[req.act_second] = {raw.get() + req.out_offset, sizes[req.act_second]};
  for (std::size_t i = 0; i < 3; ++i) {
    if (i == req.act_first || i == req.act_second) continue;
    owners[i] = std::make_unique<std::byte[]>(sizes[i]);
    buffers[i] = {owners[i].get(), sizes[i]};
  }
  const std::array<std::span<const std::byte>, 1> reads{buffers[0]};
  const std::array<std::span<std::byte>, 1> writes{buffers[1]};
  const int outcome = call_guarded([&] {
    q::detail::check_expert_storage(reads, writes, buffers[2], out_write);
  });
  if (outcome != kOk) return err_response(outcome, false);
  return "ok";
}

std::string run_expert_counts(const Request& req, bool traced) {
  if (!req.has_cap || !req.has_input_count || !req.has_shape || req.shape.size() != 6 || req.has_work_shape ||
      req.has_scale_shape || req.has_scale_cap || req.has_a_cap || req.workspace || req.alias ||
      req.scale_alias || req.poison || req.poison_writes || req.runtime_failure || req.has_overlap ||
      req.has_trace_alias || req.trace_object_alias || req.has_route || req.has_trace_shape ||
      req.has_in_offset || req.has_out_offset) {
    die("expert_counts requires shape=w1N,w1K,w2N,w2K,w3N,w3K and cap=out_count");
  }
  require_sections(req, 1, "expert_counts");
  if (!req.sections[0].empty()) die("expert_counts takes no payload");
  q::detail::ExpertCounts counts{};
  const int outcome = call_guarded([&] {
    counts = q::detail::check_expert_counts(req.input_count, req.shape[0], req.shape[1],
                                            req.shape[2], req.shape[3], req.shape[4],
                                            req.shape[5], req.cap, traced);
  });
  if (outcome != kOk) return err_response(outcome, false);
  const std::size_t values[] = {counts.input_groups, counts.hidden_groups,
      counts.w1_weight_bytes, counts.w1_scale_bytes, counts.w2_weight_bytes,
      counts.w2_scale_bytes, counts.w3_weight_bytes, counts.w3_scale_bytes};
  return ok_response(values, std::size(values));
}

std::string run_expert(const Request& req, std::string_view fault) {
  const bool traced = req.op == "expert_trace";
  if (!req.has_cap || !req.has_shape || req.shape.size() != 6 || !req.has_work_shape ||
      req.work_shape.size() != 7 || req.has_scale_shape || req.has_scale_cap || req.has_a_cap ||
      req.workspace || req.alias || req.scale_alias || req.has_trace_alias ||
      req.trace_object_alias || (!traced && req.has_trace_shape)) {
    die("expert requires six matrix dimensions, seven workspace capacities and no foreign fields");
  }
  require_sections(req, 7, "expert");
  if ((req.has_in_offset || req.has_out_offset) && !req.has_overlap)
    die("expert offsets require overlap");
  if (traced && req.has_trace_shape && req.trace_shape.size() != 5)
    die("expert trace_shape requires five lengths");

  const std::size_t input_count = req.sections[0].size();
  const std::size_t trace_count = traced ? 5 : 0;
  const std::size_t count = 15 + trace_count;
  std::vector<std::size_t> sizes(count);
  sizes[0] = buffer_bytes(input_count, 2);
  for (std::size_t i = 1; i < 7; ++i) sizes[i] = req.sections[i].size();
  sizes[7] = req.work_shape[0];
  sizes[8] = req.work_shape[1];
  sizes[9] = buffer_bytes(req.work_shape[2], 2);
  sizes[10] = buffer_bytes(req.work_shape[3], 2);
  sizes[11] = buffer_bytes(req.work_shape[4], 2);
  sizes[12] = req.work_shape[5];
  sizes[13] = req.work_shape[6];
  sizes[14] = buffer_bytes(req.cap, 2);
  if (traced) {
    const std::vector<std::size_t> shape = req.has_trace_shape ? req.trace_shape
                                                               : std::vector<std::size_t>(5, req.shape[0]);
    for (std::size_t i = 0; i < 5; ++i) sizes[15 + i] = buffer_bytes(shape[i], 4);
  }
  const std::size_t widths[] = {2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2,
                                4, 4, 4, 4, 4};
  std::vector<std::unique_ptr<std::uint8_t[]>> byte_owners(count);
  std::vector<std::unique_ptr<std::uint16_t[]>> half_owners(count);
  std::vector<std::unique_ptr<std::uint32_t[]>> word_owners(count);
  std::unique_ptr<std::uint8_t[]> shared_bytes;
  std::unique_ptr<std::uint16_t[]> shared_halves;
  std::unique_ptr<std::uint32_t[]> shared_words;
  std::vector<unsigned char*> buffers(count, nullptr);
  unsigned char* shared_base = nullptr;
  std::size_t shared_size = 0;
  if (req.has_overlap) {
    if (req.act_first >= count || req.act_second >= count)
      die("expert overlap names an unavailable buffer");
    if (widths[req.act_first] != widths[req.act_second])
      die("mixed expert overlap requires expert_storage byte probe");
    const std::size_t width = widths[req.act_first];
    for (const auto& [index, offset] : {std::pair{req.act_first, req.in_offset},
                                       std::pair{req.act_second, req.out_offset}}) {
      if (offset % width) die("unaligned expert overlap offset");
      shared_size = std::max(shared_size, extent(offset, sizes[index]));
    }
    if (shared_size % width) die("expert shared extent is not element aligned");
    if (width == 1) {
      shared_bytes = std::make_unique<std::uint8_t[]>(shared_size);
      shared_base = shared_bytes.get();
    } else if (width == 2) {
      shared_halves = std::make_unique<std::uint16_t[]>(shared_size / 2);
      shared_base = reinterpret_cast<unsigned char*>(shared_halves.get());
    } else {
      shared_words = std::make_unique<std::uint32_t[]>(shared_size / 4);
      shared_base = reinterpret_cast<unsigned char*>(shared_words.get());
    }
    std::memset(shared_base, kFill, shared_size);
    buffers[req.act_first] = shared_base + req.in_offset;
    buffers[req.act_second] = shared_base + req.out_offset;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (!buffers[i]) {
      if (widths[i] == 1) {
        byte_owners[i] = std::make_unique<std::uint8_t[]>(sizes[i]);
        buffers[i] = byte_owners[i].get();
      } else if (widths[i] == 2) {
        half_owners[i] = std::make_unique<std::uint16_t[]>(sizes[i] / 2);
        buffers[i] = reinterpret_cast<unsigned char*>(half_owners[i].get());
      } else {
        word_owners[i] = std::make_unique<std::uint32_t[]>(sizes[i] / 4);
        buffers[i] = reinterpret_cast<unsigned char*>(word_owners[i].get());
      }
      std::memset(buffers[i], kFill, sizes[i]);
    }
  }
  for (std::size_t i = 0; i < input_count; ++i) {
    const auto value = narrow_token<std::uint16_t>(req.sections[0][i]);
    std::memcpy(buffers[0] + 2 * i, &value, 2);
  }
  for (std::size_t section = 1; section < 7; ++section) {
    for (std::size_t i = 0; i < req.sections[section].size(); ++i)
      buffers[section][i] = narrow_token<std::uint8_t>(req.sections[section][i]);
  }
  std::vector<std::vector<unsigned char>> before(count);
  for (std::size_t i = 0; i < count; ++i)
    before[i].assign(buffers[i], buffers[i] + sizes[i]);
  std::vector<unsigned char> before_shared;
  if (req.has_overlap) before_shared.assign(shared_base, shared_base + shared_size);

  q::ExpertMatrix matrices[] = {
      {req.shape[0], req.shape[1], {buffers[1], sizes[1]}, {buffers[2], sizes[2]}},
      {req.shape[2], req.shape[3], {buffers[3], sizes[3]}, {buffers[4], sizes[4]}},
      {req.shape[4], req.shape[5], {buffers[5], sizes[5]}, {buffers[6], sizes[6]}}};
  q::ExpertWorkspace work{{buffers[7], req.work_shape[0]}, {buffers[8], req.work_shape[1]},
      {reinterpret_cast<std::uint16_t*>(buffers[9]), req.work_shape[2]},
      {reinterpret_cast<std::uint16_t*>(buffers[10]), req.work_shape[3]},
      {reinterpret_cast<std::uint16_t*>(buffers[11]), req.work_shape[4]},
      {buffers[12], req.work_shape[5]}, {buffers[13], req.work_shape[6]}};
  const std::span<const std::uint16_t> input(reinterpret_cast<const std::uint16_t*>(buffers[0]),
                                             input_count);
  const std::span<std::uint16_t> output(reinterpret_cast<std::uint16_t*>(buffers[14]), req.cap);
  q::detail::SwigluTrace trace;
  if (traced) {
    const std::vector<std::size_t> shape = req.has_trace_shape ? req.trace_shape
                                                               : std::vector<std::size_t>(5, req.shape[0]);
    trace = {{reinterpret_cast<std::uint32_t*>(buffers[15]), shape[0]},
             {reinterpret_cast<std::uint32_t*>(buffers[16]), shape[1]},
             {reinterpret_cast<std::uint32_t*>(buffers[17]), shape[2]},
             {reinterpret_cast<std::uint32_t*>(buffers[18]), shape[3]},
             {reinterpret_cast<std::uint32_t*>(buffers[19]), shape[4]}};
  }
#if defined(__SANITIZE_ADDRESS__)
  if (req.poison) for (std::size_t i = 0; i < 7; ++i) ASAN_POISON_MEMORY_REGION(buffers[i], sizes[i]);
  if (req.poison_writes) for (std::size_t i = 7; i < count; ++i) ASAN_POISON_MEMORY_REGION(buffers[i], sizes[i]);
#endif
  const std::size_t used[] = {0, 0, 0, 0, 0, 0, 0, req.shape[1], req.shape[1] / 32,
      buffer_bytes(req.shape[0], 2), buffer_bytes(req.shape[0], 2), buffer_bytes(req.shape[0], 2),
      req.shape[0], req.shape[0] / 32, buffer_bytes(req.shape[2], 2),
      buffer_bytes(req.shape[0], 4), buffer_bytes(req.shape[0], 4), buffer_bytes(req.shape[0], 4),
      buffer_bytes(req.shape[0], 4), buffer_bytes(req.shape[0], 4)};
  const auto shared_byte = [&](std::size_t index, std::size_t offset) {
    if (!req.has_overlap || (index != req.act_first && index != req.act_second)) return false;
    const std::size_t start = static_cast<std::size_t>(buffers[index] - shared_base);
    return offset >= start && offset - start < std::min(used[index], sizes[index]);
  };
  const auto allowed_shared = [&](std::size_t offset, bool runtime) {
    for (std::size_t index : {req.act_first, req.act_second}) {
      if (index >= count || index < 7) continue;
      if (runtime && (index == 12 || index == 13 || index == 14)) continue;
      if (shared_byte(index, offset)) return true;
    }
    return false;
  };
  const int outcome = call_guarded([&] {
    if (fault == "expert-early-reject") throw std::invalid_argument("injected early expert rejection");
    if (traced) {
      if (req.has_route) q::detail::trace_expert(input, matrices[0], matrices[1], matrices[2],
                                                 req.route, work, output, trace);
      else q::detail::trace_expert(input, matrices[0], matrices[1], matrices[2], work, output, trace);
    } else if (req.has_route) {
      q::expert(input, matrices[0], matrices[1], matrices[2], req.route, work, output);
    } else {
      q::expert(input, matrices[0], matrices[1], matrices[2], work, output);
    }
  });
#if defined(__SANITIZE_ADDRESS__)
  for (std::size_t i = 0; i < count; ++i) ASAN_UNPOISON_MEMORY_REGION(buffers[i], sizes[i]);
  if (req.has_overlap) ASAN_UNPOISON_MEMORY_REGION(shared_base, shared_size);
#endif
  if (outcome != kOk) {
    if (fault == "reject-input-write" && input_count) buffers[0][0] ^= 1;
    if (fault == "shared-gap-write" && req.has_overlap) {
      for (std::size_t i = 0; i < shared_size; ++i) {
        if (!allowed_shared(i, req.runtime_failure)) { shared_base[i] ^= 1; break; }
      }
    }
    bool changed = false;
    if (req.runtime_failure) {
      for (std::size_t i = 0; i < 7; ++i)
        if (!req.has_overlap || (i != req.act_first && i != req.act_second))
          changed |= buffer_changed(before[i].data(), buffers[i], sizes[i]);
      if (!req.has_overlap || (req.act_first != 14 && req.act_second != 14))
        changed |= buffer_changed(before[14].data(), buffers[14], sizes[14]);
      for (std::size_t i = 7; i < count; ++i) {
        if (i == 14 || (req.has_overlap && (i == req.act_first || i == req.act_second))) continue;
        const std::size_t prefix = (i == 12 || i == 13) ? 0 : std::min(used[i], sizes[i]);
        changed |= buffer_changed(before[i].data() + prefix, buffers[i] + prefix,
                                  sizes[i] - prefix);
      }
      if (req.has_overlap) {
        for (std::size_t i = 0; i < shared_size; ++i)
          if (!allowed_shared(i, true) && shared_base[i] != before_shared[i]) changed = true;
      }
      const auto prefix_changed = [&](std::size_t index) {
        const std::size_t prefix = std::min(used[index], sizes[index]);
        return prefix != 0 && buffer_changed(before[index].data(), buffers[index], prefix);
      };
      for (std::size_t index : {7u, 8u, 9u, 10u, 11u}) {
        if (!prefix_changed(index)) die("runtime expert rejection did not reach every bridge boundary");
      }
      if (traced) {
        for (std::size_t index = 15; index < count; ++index)
          if (!prefix_changed(index)) die("runtime expert rejection did not populate trace stages");
      }
      bool hidden_all_nan = true;
      for (std::size_t i = 0; i < req.shape[0]; ++i) {
        std::uint16_t value = 0;
        std::memcpy(&value, buffers[11] + 2 * i, 2);
        hidden_all_nan &= (value & 0x7FFFu) > 0x7F80u;
      }
      if (!hidden_all_nan || prefix_changed(12) || prefix_changed(13))
        die("runtime expert rejection was not the expected hidden quantization boundary");
    } else {
      for (std::size_t i = 0; i < count; ++i)
        if (!req.has_overlap || (i != req.act_first && i != req.act_second))
          changed |= buffer_changed(before[i].data(), buffers[i], sizes[i]);
      if (req.has_overlap)
        changed |= buffer_changed(before_shared.data(), shared_base, shared_size);
    }
    return err_response(outcome, changed);
  }
  if (req.runtime_failure) die("runtime_failure request unexpectedly succeeded");
  if (fault == "input-write" && input_count) buffers[0][0] ^= 1;
  if (fault == "shared-gap-write" && req.has_overlap) {
    for (std::size_t i = 0; i < shared_size; ++i) {
      if (!allowed_shared(i, false)) { shared_base[i] ^= 1; break; }
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (req.has_overlap && (i == req.act_first || i == req.act_second)) continue;
    if (i < 7 && buffer_changed(before[i].data(), buffers[i], sizes[i]))
      die("expert input changed on success");
    const std::size_t prefix = std::min(used[i], sizes[i]);
    if (buffer_changed(before[i].data() + prefix, buffers[i] + prefix, sizes[i] - prefix))
      die("expert bytes outside write ranges changed on success");
  }
  if (req.has_overlap) {
    for (std::size_t i = 0; i < shared_size; ++i)
      if (!allowed_shared(i, false) && shared_base[i] != before_shared[i])
        die("expert bytes outside write ranges changed on success");
  }
  std::string response = "ok";
  const auto append_buffer = [&](std::size_t index, std::size_t width) {
    for (std::size_t offset = 0; offset < sizes[index]; offset += width) {
      std::uint32_t value = 0;
      std::memcpy(&value, buffers[index] + offset, width);
      append_hex(response, value);
    }
  };
  append_buffer(7, 1);
  append_buffer(8, 1);
  append_buffer(9, 2);
  append_buffer(10, 2);
  if (traced) for (std::size_t i = 15; i < 20; ++i) append_buffer(i, 4);
  append_buffer(11, 2);
  append_buffer(12, 1);
  append_buffer(13, 1);
  append_buffer(14, 2);
  return response;
}

using DequantFn32 = void (*)(std::span<const std::uint8_t>, q::Shape,
                             std::span<const std::uint8_t>, q::Shape, std::span<std::uint32_t>);
using DequantFn16 = void (*)(std::span<const std::uint8_t>, q::Shape,
                             std::span<const std::uint8_t>, q::Shape, std::span<std::uint16_t>);

struct DequantEntry {
  std::string_view base;
  DequantFn32 fp32;
  DequantFn16 bf16;
};

constexpr DequantEntry kDequantOps[] = {
    {"dequantize_expert", q::dequantize_expert, q::dequantize_expert},
    {"dequantize_fp8_weights", q::dequantize_fp8_weights, q::dequantize_fp8_weights},
    {"dequantize_engram", q::dequantize_engram, q::dequantize_engram},
    {"dequantize_swa", q::dequantize_swa, q::dequantize_swa},
    {"dequantize_main_kv", q::dequantize_main_kv, q::dequantize_main_kv},
    {"dequantize_index_k", q::dequantize_index_k, q::dequantize_index_k},
};

std::string dispatch(const Request& req, std::string_view fault) {
  using u8 = std::uint8_t;
  using u16 = std::uint16_t;
  using u32 = std::uint32_t;
  using i8 = std::int8_t;
  if (req.has_out_cap && !is_fp8_trace(req)) die("out_cap field on another request family");
  if (req.op == "expert_bf16" || req.op == "expert_trace") return run_expert(req, fault);
  if (req.op == "expert_counts") return run_expert_counts(req, false);
  if (req.op == "expert_counts_trace") return run_expert_counts(req, true);
  if (req.op == "expert_storage") return run_expert_storage(req);
  if (req.has_input_count || req.has_work_shape || req.runtime_failure)
    die("expert fields on another request family");
  if (req.op == "expert_linear_fp32") return run_linear<u32>(req, fault);
  if (req.op == "expert_linear_bf16") return run_linear<u16>(req, fault);
  if (req.op == "swiglu" || req.op == "swiglu_trace") return run_swiglu(req, fault);
  if (req.op == "swiglu_counts") return run_swiglu_counts(req, false);
  if (req.op == "swiglu_counts_trace") return run_swiglu_counts(req, true);
  if (req.has_route) die("route field on another request family");
  if (req.op == "act_quant" || req.op == "act_quant_trace") return run_act_quant(req, fault);
  if (req.op == "act_quant_dot_fp32") return run_act_combination<u32>(req, false);
  if (req.op == "act_quant_dot_bf16") return run_act_combination<u16>(req, false);
  if (req.op == "act_quant_fp8_dot_fp32") return run_act_combination<u32>(req, true);
  if (req.op == "act_quant_fp8_dot_bf16") return run_act_combination<u16>(req, true);
  if ((req.has_scale_cap || req.has_trace_shape) && !is_fp8_trace(req))
    die("activation fields on another request family");
  if (req.op == "asan") {
    const std::uint32_t value = kAsan ? 1 : 0;
    return ok_response(&value, 1);
  }
  if (req.op == "version" || req.op == "mxcsr") {
    std::uint32_t value = 0;
    call_guarded([&] {
      value = req.op == "version" ? q::format_contract_version()
                                  : static_cast<std::uint32_t>(_mm_getcsr());
    });
    return ok_response(&value, 1);
  }
  if (req.op == "decode_e2m1") return run<u8, u32>(req, q::decode_e2m1);
  if (req.op == "decode_e4m3fn") return run<u8, u32>(req, q::decode_e4m3fn);
  if (req.op == "decode_e8m0") return run<u8, u32>(req, q::decode_e8m0);
  if (req.op == "decode_bf16") return run<u16, u32>(req, q::decode_bf16);
  if (req.op == "fp32_to_bf16") return run<u32, u16>(req, q::fp32_to_bf16);
  if (req.op == "pack_e2m1") return run<u8, u8>(req, q::pack_e2m1);
  if (req.op == "unpack_e2m1") return run<u8, u8>(req, q::unpack_e2m1);
  if (req.op == "e2m1_to_grid") return run<u8, i8>(req, q::e2m1_to_grid);
  if (req.op == "grid_to_e2m1") return run<i8, u8>(req, q::grid_to_e2m1);
  if (req.op == "multiply_finite") return run_multiply(req);
  if (req.op == "multiply_fp32") return run_binary_fp32(req, true);
  if (req.op == "add_fp32") return run_binary_fp32(req, false);
  if (req.op == "expert_grouped_dot_fp32") return run_compute<u32>(req);
  if (req.op == "expert_grouped_dot_bf16") return run_compute<u16>(req);
  if (req.op == "expert_grouped_dot_trace") return run_trace(req);
  if (req.op == "fp8_grouped_dot_fp32") return run_fp8_compute<u32>(req, fault);
  if (req.op == "fp8_grouped_dot_bf16") return run_fp8_compute<u16>(req, fault);
  if (req.op == "fp8_grouped_dot_trace_fp32") return run_fp8_trace<u32>(req, fault);
  if (req.op == "fp8_grouped_dot_trace_bf16") return run_fp8_trace<u16>(req, fault);
  for (const auto& entry : kDequantOps) {
    if (!req.op.starts_with(entry.base)) {
      continue;
    }
    const std::string_view tail = std::string_view(req.op).substr(entry.base.size());
    if (tail == "_fp32") return run_dequant<std::uint32_t>(req, entry.fp32);
    if (tail == "_bf16") return run_dequant<std::uint16_t>(req, entry.bf16);
  }
  die("unknown op '" + req.op + "'");
}

constexpr unsigned kMxcsrRoundingMask = 0x6000u;
constexpr unsigned kMxcsrRoundUp = 0x4000u;
constexpr unsigned kMxcsrRoundDown = 0x2000u;
constexpr unsigned kMxcsrRoundZero = 0x6000u;
constexpr unsigned kMxcsrFtz = 0x8000u;
constexpr unsigned kMxcsrDaz = 0x0040u;
constexpr unsigned kMxcsrStatusFlags = 0x003Fu;
constexpr unsigned kMxcsrOverflowMask = 0x0400u;

unsigned mxcsr_for(std::string_view env) {
  const unsigned csr = (_mm_getcsr() & ~(kMxcsrRoundingMask | kMxcsrFtz | kMxcsrDaz |
                                         kMxcsrStatusFlags)) | 0x1F80u;
  if (env == "default") return csr;
  if (env == "up") return (csr & ~kMxcsrRoundingMask) | kMxcsrRoundUp;
  if (env == "down") return (csr & ~kMxcsrRoundingMask) | kMxcsrRoundDown;
  if (env == "zero") return (csr & ~kMxcsrRoundingMask) | kMxcsrRoundZero;
  if (env == "ftz") return csr | kMxcsrFtz;
  if (env == "daz") return csr | kMxcsrDaz;
  if (env == "ftz_daz") return csr | kMxcsrFtz | kMxcsrDaz;
  if (env == "overflow_unmasked") return csr & ~(kMxcsrStatusFlags | kMxcsrOverflowMask);
  die("unknown --mxcsr value '" + std::string(env) + "'");
}

// Changes the last hex digit of the first output element of an ok response.
std::string corrupt(std::string response) {
  if (response.starts_with("ok ")) {
    char& digit = response[std::min(response.find(' ', 3), response.size()) - 1];
    digit = digit == '0' ? '1' : '0';
  }
  return response;
}

}  // namespace

int main(int argc, char** argv) {
  std::string env = "default";
  std::string fault = "none";
  std::string status = "clear";
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg.starts_with("--mxcsr=")) {
      env = arg.substr(8);
    } else if (arg.starts_with("--fault=")) {
      fault = arg.substr(8);
    } else if (arg.starts_with("--status=")) {
      status = arg.substr(9);
    } else {
      die("unknown argument '" + std::string(arg) + "'");
    }
  }
  if (fault != "none" && fault != "crash" && fault != "exit" && fault != "garbage" &&
      fault != "mismatch" && fault != "drop" && fault != "input-write" &&
      fault != "reject-input-write" && fault != "shared-gap-write" &&
      fault != "expert-early-reject") {
    die("unknown --fault value '" + fault + "'");
  }
  if (status != "clear" && status != "set") {
    die("unknown --status value '" + status + "'");
  }
  const unsigned csr = mxcsr_for(env) | (status == "set" ? kMxcsrStatusFlags : 0u);
  _mm_setcsr(csr);

  std::ios::sync_with_stdio(false);
  std::string line;
  std::size_t served = 0;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    if (fault == "crash") std::abort();
    const Request req = parse_request(line);
    std::string response = dispatch(req, fault);
    if (served == 0 && fault == "garbage") response = "zzz";
    if (fault == "mismatch" && req.op != "mxcsr") response = corrupt(std::move(response));
    if (!(fault == "drop" && std::cin.peek() == EOF)) {
      std::cout << response << '\n';
    }
    if (fault == "exit") {
      std::cout.flush();
      std::exit(3);
    }
    ++served;
  }
  std::cout.flush();
  if ((_mm_getcsr() & ~kMxcsrStatusFlags) != (csr & ~kMxcsrStatusFlags)) {
    die("MXCSR control bits changed while serving requests");
  }
  return std::cout ? 0 : 2;
}
