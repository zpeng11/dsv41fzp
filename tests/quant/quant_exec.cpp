// Test executor for the production quant library (P0010 codec, P0011 dequant). Not a
// production format.
//
// Reads one request per stdin line and writes one response per line:
//   codec:    <op> shape=<d0,d1,...> cap=<n> [alias] : <hex token> ...
//   dequant:  <op> shape=<d0,d1,...> scale_shape=<d0,d1,...> cap=<n> [alias|scale_alias]
//             : <data hex token> ... ; <scale hex token> ...
//   response: ok <hex output element> ...   (the whole output buffer, cap elements)
//             err <invalid_argument|overflow_error> changed=<0|1>
// Tokens are range-checked against the input element width before narrowing. A
// dequant request without scale_shape= or ';' is a protocol fault, as is dequant-only
// metadata on a codec request; any protocol or executor fault exits non-zero instead
// of producing an `err` response. Buffers are pre-filled with kFill bytes and sized
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
// overflow_unmasked also clears pending status flags. The `mxcsr` op reports the
// register. --fault=crash|exit|garbage|mismatch|drop deliberately breaks the executor
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

#include "detail.h"
#include "df/quant/codec.h"
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
  std::vector<std::uint64_t> tokens;
  std::vector<std::uint64_t> scale_tokens;
};

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
    } else {
      die("unknown field '" + std::string(field) + "'");
    }
  }
  const std::string_view payload = line.substr(colon + 1);
  const std::size_t sep = payload.find(';');
  req.has_scale_section = sep != std::string_view::npos;
  const std::string_view data_part =
      sep == std::string_view::npos ? payload : payload.substr(0, sep);
  const std::string_view scale_part =
      sep == std::string_view::npos ? std::string_view() : payload.substr(sep + 1);
  for (const auto token : split(data_part, ' ')) {
    req.tokens.push_back(parse_number(token, 16));
  }
  for (const auto token : split(scale_part, ' ')) {
    req.scale_tokens.push_back(parse_number(token, 16));
  }
  return req;
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
  if (req.has_scale_shape || req.has_scale_section || req.scale_alias) {
    die("codec request carries dequant-only fields");
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
  if (req.alias || req.scale_alias) {
    die("multiply_finite takes no alias flags");
  }
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
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg.starts_with("--mxcsr=")) {
      env = arg.substr(8);
    } else if (arg.starts_with("--fault=")) {
      fault = arg.substr(8);
    } else {
      die("unknown argument '" + std::string(arg) + "'");
    }
  }
  if (fault != "none" && fault != "crash" && fault != "exit" && fault != "garbage" &&
      fault != "mismatch" && fault != "drop") {
    die("unknown --fault value '" + fault + "'");
  }
  const unsigned csr = mxcsr_for(env);
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
