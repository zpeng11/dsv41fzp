// Test executor for the production quant library (P0010). Not a production format.
//
// Reads one request per stdin line and writes one response per line:
//   request:  <op> shape=<d0,d1,...> cap=<n> [alias] : <hex token> ...
//   response: ok <hex output element> ...        (the whole output buffer, cap elements)
//             err <invalid_argument|overflow_error> changed=<0|1>
// Tokens are range-checked against the input element width before narrowing; any
// protocol or executor fault exits non-zero instead of producing an `err` response.
// Output buffers are pre-filled with kFill bytes; inputs and outputs are allocated
// with their exact element counts so out-of-bounds access is visible to ASan.

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

#include "df/quant/codec.h"

namespace {

constexpr unsigned char kFill = 0xA5;

[[noreturn]] void die(const std::string& msg) {
  std::cerr << "quant_exec: " << msg << '\n';
  std::exit(2);
}

struct Request {
  std::string op;
  std::vector<std::size_t> shape;
  std::size_t cap = 0;
  bool has_cap = false;
  bool alias = false;
  std::vector<std::uint64_t> tokens;
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
    if (field.starts_with("shape=")) {
      for (const auto dim : split(field.substr(6), ',')) {
        req.shape.push_back(parse_number(dim, 10));
      }
    } else if (field.starts_with("cap=")) {
      req.cap = parse_number(field.substr(4), 10);
      req.has_cap = true;
    } else if (field == "alias") {
      req.alias = true;
    } else {
      die("unknown field '" + std::string(field) + "'");
    }
  }
  for (const auto token : split(line.substr(colon + 1), ' ')) {
    req.tokens.push_back(parse_number(token, 16));
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

// Runs one library call. In alias mode input and output share one raw allocation.
template <class In, class Out, class Fn>
std::string run(const Request& req, Fn fn) {
  if (!req.has_cap) {
    die("request lacks cap=");
  }
  const std::size_t n_in = req.tokens.size();
  const std::size_t in_bytes = n_in * sizeof(In);
  const std::size_t out_bytes = req.cap * sizeof(Out);

  std::unique_ptr<In[]> in_owner;
  std::unique_ptr<Out[]> out_owner;
  std::unique_ptr<std::uint64_t[]> shared;
  In* in_ptr = nullptr;
  Out* out_ptr = nullptr;
  if (req.alias) {
    const std::size_t bytes = std::max(in_bytes, out_bytes);
    shared = std::make_unique<std::uint64_t[]>((bytes + 7) / 8);
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
  const auto changed = [&] {
    return std::memcmp(before.data(), out_ptr, out_bytes) != 0 ? "1" : "0";
  };

  try {
    fn(std::span<const In>(in_ptr, n_in), df::quant::Shape(req.shape), std::span<Out>(out_ptr, req.cap));
  } catch (const std::overflow_error&) {
    return std::string("err overflow_error changed=") + changed();
  } catch (const std::invalid_argument&) {
    return std::string("err invalid_argument changed=") + changed();
  }
  std::string response = "ok";
  for (std::size_t i = 0; i < req.cap; ++i) {
    append_hex(response, out_ptr[i]);
  }
  return response;
}

std::string dispatch(const Request& req) {
  namespace q = df::quant;
  using u8 = std::uint8_t;
  using u16 = std::uint16_t;
  using u32 = std::uint32_t;
  using i8 = std::int8_t;
  if (req.op == "version") {
    std::string response = "ok";
    append_hex(response, q::format_contract_version());
    return response;
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
  die("unknown op '" + req.op + "'");
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    std::cout << dispatch(parse_request(line)) << '\n';
  }
  std::cout.flush();
  return std::cout ? 0 : 2;
}
