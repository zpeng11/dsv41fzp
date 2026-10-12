// Test executor for the production quant library (P0010 codec, P0011 dequant, P0013
// compute). Not a production format.
//
// Reads one request per stdin line and writes one response per line:
//   codec:    <op> shape=<d0,d1,...> cap=<n> [alias] : <hex token> ...
//   dequant:  <op> shape=<d0,d1,...> scale_shape=<d0,d1,...> cap=<n> [alias|scale_alias]
//             : <data hex token> ... ; <scale hex token> ...
//   compute:  <op> cap=<n> [overlap=a|w|sa|sb [in_offset=<b>] [out_offset=<b>]] [poison]
//             : <a codes> ... ; <w bytes> ... ; <scale_a codes> ... ; <scale_b codes> ...
//   add_fp32: add_fp32 cap=<n> : <left fp32 bits> ... ; <right fp32 bits> ...
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
// breaks the full-MXCSR window either way. The `mxcsr` op reports the register. --fault=crash|exit|garbage|mismatch|drop deliberately breaks the executor
// so the driver can prove that every such failure reaches CTest.

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <xmmintrin.h>

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#endif

#include "detail.h"
#include "df/quant/codec.h"
#include "df/quant/compute.h"
#include "df/quant/dequant.h"

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
  std::size_t cap = 0;
  bool has_cap = false;
  bool has_shape = false;
  bool alias = false;
  bool scale_alias = false;
  bool has_scale_shape = false;
  bool has_scale_section = false;
  bool has_overlap = false;
  std::size_t overlap = 0;
  bool has_in_offset = false;
  bool has_out_offset = false;
  std::size_t in_offset = 0;
  std::size_t out_offset = 0;
  bool poison = false;
  std::vector<std::vector<std::uint64_t>> sections;
  std::vector<std::uint64_t> tokens;
  std::vector<std::uint64_t> scale_tokens;
};

constexpr std::string_view kComputeInputs[] = {"a", "w", "sa", "sb"};

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
    if (field.starts_with("scale_shape=")) {
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
      const auto* hit = std::find(std::begin(kComputeInputs), std::end(kComputeInputs), name);
      if (hit == std::end(kComputeInputs)) die("unknown overlap input '" + std::string(name) + "'");
      req.overlap = static_cast<std::size_t>(hit - std::begin(kComputeInputs));
      req.has_overlap = true;
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

void require_compute_fields(const Request& req) {
  if (!req.has_cap) {
    die("request lacks cap=");
  }
  if (req.has_shape || req.has_scale_shape || req.alias || req.scale_alias) {
    die("compute request carries codec or dequant fields");
  }
  if ((req.has_in_offset || req.has_out_offset) && !req.has_overlap) {
    die("in_offset= and out_offset= require overlap=");
  }
  if (req.poison && req.has_overlap) {
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

// Private probe of the production FP32 adder; every bit pattern is a valid operand.
std::string run_add(const Request& req) {
  if (!req.has_cap || req.has_shape || req.has_scale_shape || req.alias || req.scale_alias ||
      has_compute_fields(req)) {
    die("add_fp32 request takes only cap=");
  }
  require_sections(req, 2, "add_fp32");
  const std::size_t n = req.sections[0].size();
  if (req.sections[1].size() != n || req.cap < n) {
    die("add_fp32 requires one right operand per left operand and cap >= pairs");
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
    out[i] = q::detail::add_fp32(left[i], right[i]);
    if (_mm_getcsr() != before) {
      die("add_fp32 changed MXCSR");
    }
  }
  return ok_response(out.get(), req.cap);
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

std::string dispatch(const Request& req) {
  using u8 = std::uint8_t;
  using u16 = std::uint16_t;
  using u32 = std::uint32_t;
  using i8 = std::int8_t;
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
  if (req.op == "add_fp32") return run_add(req);
  if (req.op == "expert_grouped_dot_fp32") return run_compute<u32>(req);
  if (req.op == "expert_grouped_dot_bf16") return run_compute<u16>(req);
  if (req.op == "expert_grouped_dot_trace") return run_trace(req);
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
      fault != "mismatch" && fault != "drop") {
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
    std::string response = dispatch(parse_request(line));
    if (served == 0 && fault == "garbage") response = "zzz";
    if (fault == "mismatch") response = corrupt(std::move(response));
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
