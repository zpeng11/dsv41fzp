// P0024 test executor for the production scalar router. This is not a production protocol.

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
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

#include "df/core/router.h"
#include "router_detail.h"

namespace {

namespace core = df::core;
constexpr std::uint32_t kFill = 0xA5A5A5A5u;
constexpr unsigned kCertificationBits = 12288;
constexpr unsigned kStatusMask = 0x3Fu;

[[noreturn]] void die(const std::string& message) {
  std::cerr << "core_exec: " << message << '\n';
  std::exit(2);
}

std::string_view trim(std::string_view text) {
  while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
  while (!text.empty() && text.back() == ' ') text.remove_suffix(1);
  return text;
}

std::vector<std::string_view> split(std::string_view text, char separator, bool keep_empty = false) {
  std::vector<std::string_view> result;
  std::size_t start = 0;
  while (true) {
    const std::size_t end = text.find(separator, start);
    const std::size_t stop = end == std::string_view::npos ? text.size() : end;
    const auto item = trim(text.substr(start, stop - start));
    if (keep_empty || !item.empty()) result.push_back(item);
    if (end == std::string_view::npos) return result;
    start = end + 1;
  }
}

std::uint64_t number(std::string_view text, int base) {
  std::uint64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (text.empty() || error != std::errc() || end != text.data() + text.size()) {
    die("malformed number '" + std::string(text) + "'");
  }
  return value;
}

std::vector<std::uint64_t> tokens(std::string_view text) {
  std::vector<std::uint64_t> result;
  for (const auto item : split(text, ' ')) result.push_back(number(item, 16));
  return result;
}

template <class T>
T narrow(std::uint64_t value) {
  if (value > std::numeric_limits<std::make_unsigned_t<T>>::max()) die("token exceeds width");
  return static_cast<T>(value);
}

struct Request {
  std::string op;
  std::size_t n = 0;
  std::size_t k = 0;
  bool has_n = false;
  bool has_k = false;
  bool image = false;
  bool has_image = false;
  bool vl = false;
  bool has_vl = false;
  bool trace = false;
  bool has_trace = false;
  bool vl_present = false;
  bool has_vl_present = false;
  std::array<std::size_t, 5> caps{};
  bool has_caps = false;
  unsigned precision = kCertificationBits;
  bool has_precision = false;
  bool poison = false;
  bool has_overlap = false;
  std::string first;
  std::string second;
  std::size_t in_offset = 0;
  std::size_t out_offset = 0;
  bool has_in_offset = false;
  bool has_out_offset = false;
  std::size_t read_size = 0;
  std::size_t write_size = 0;
  std::size_t read_offset = 0;
  std::size_t write_offset = 0;
  bool has_read_size = false;
  bool has_write_size = false;
  bool has_read_offset = false;
  bool has_write_offset = false;
  std::vector<std::vector<std::uint64_t>> sections;
};

std::array<std::size_t, 5> parse_caps(std::string_view text) {
  const auto values = split(text, ',');
  if (values.size() != 5) die("caps needs five counts");
  std::array<std::size_t, 5> result{};
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = number(values[i], 10);
  return result;
}

Request parse(std::string_view line) {
  const std::size_t colon = line.find(':');
  if (colon == std::string_view::npos) die("request lacks ':'");
  const auto head = split(line.substr(0, colon), ' ');
  if (head.empty()) die("request lacks op");
  Request request;
  request.op = std::string(head.front());
  for (std::size_t i = 1; i < head.size(); ++i) {
    const auto field = head[i];
    const auto assign_bool = [&](std::string_view value, bool& target, bool& present) {
      if (present || (value != "0" && value != "1")) die("invalid or duplicate boolean field");
      target = value == "1";
      present = true;
    };
    if (field.starts_with("n=")) {
      if (request.has_n) die("duplicate n");
      request.n = number(field.substr(2), 10); request.has_n = true;
    } else if (field.starts_with("k=")) {
      if (request.has_k) die("duplicate k");
      request.k = number(field.substr(2), 10); request.has_k = true;
    } else if (field.starts_with("image=")) {
      assign_bool(field.substr(6), request.image, request.has_image);
    } else if (field.starts_with("vl_present=")) {
      assign_bool(field.substr(11), request.vl_present, request.has_vl_present);
    } else if (field.starts_with("vl=")) {
      assign_bool(field.substr(3), request.vl, request.has_vl);
    } else if (field.starts_with("trace=")) {
      assign_bool(field.substr(6), request.trace, request.has_trace);
    } else if (field.starts_with("caps=")) {
      if (request.has_caps) die("duplicate caps");
      request.caps = parse_caps(field.substr(5)); request.has_caps = true;
    } else if (field.starts_with("precision=")) {
      if (request.has_precision) die("duplicate precision");
      const auto value = number(field.substr(10), 10);
      if (value > std::numeric_limits<unsigned>::max()) die("precision exceeds unsigned");
      request.precision = static_cast<unsigned>(value); request.has_precision = true;
    } else if (field.starts_with("overlap=")) {
      if (request.has_overlap) die("duplicate overlap");
      const auto pair = split(field.substr(8), ',', true);
      if (pair.size() != 2 || pair[0].empty() || pair[1].empty() || pair[0] == pair[1])
        die("overlap needs two distinct names");
      request.first = pair[0]; request.second = pair[1]; request.has_overlap = true;
    } else if (field.starts_with("in_offset=")) {
      if (request.has_in_offset) die("duplicate in_offset");
      request.in_offset = number(field.substr(10), 10); request.has_in_offset = true;
    } else if (field.starts_with("out_offset=")) {
      if (request.has_out_offset) die("duplicate out_offset");
      request.out_offset = number(field.substr(11), 10); request.has_out_offset = true;
    } else if (field.starts_with("read_size=")) {
      if (request.has_read_size) die("duplicate read_size");
      request.read_size = number(field.substr(10), 10); request.has_read_size = true;
    } else if (field.starts_with("write_size=")) {
      if (request.has_write_size) die("duplicate write_size");
      request.write_size = number(field.substr(11), 10); request.has_write_size = true;
    } else if (field.starts_with("read_offset=")) {
      if (request.has_read_offset) die("duplicate read_offset");
      request.read_offset = number(field.substr(12), 10); request.has_read_offset = true;
    } else if (field.starts_with("write_offset=")) {
      if (request.has_write_offset) die("duplicate write_offset");
      request.write_offset = number(field.substr(13), 10); request.has_write_offset = true;
    } else if (field == "poison") {
      if (request.poison) die("duplicate poison");
      request.poison = true;
    } else {
      die("unknown field '" + std::string(field) + "'");
    }
  }
  for (const auto section : split(line.substr(colon + 1), ';', true)) {
    request.sections.push_back(tokens(section));
  }
  return request;
}

void append_hex(std::string& output, std::uint32_t value) {
  char buffer[9];
  std::snprintf(buffer, sizeof buffer, "%08x", value);
  output += output.back() == '=' ? "" : " ";
  output += buffer;
}

void append_hex64(std::string& output, std::uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(value));
  output += output.back() == '=' ? "" : " ";
  output += buffer;
}

template <class Range>
void segment(std::string& output, std::string_view name, const Range& values) {
  output += " ; "; output += name; output += '=';
  for (const auto value : values) append_hex(output, static_cast<std::uint32_t>(value));
}

void scalar(std::string& output, std::string_view name, std::uint32_t value) {
  const std::array values{value};
  segment(output, name, values);
}

struct Outcome {
  enum Kind { ok, invalid, overflow, runtime } kind = ok;
};

template <class Function>
Outcome guarded(Function function) {
  const unsigned before = _mm_getcsr();
  Outcome result;
  try {
    function();
  } catch (const std::overflow_error&) {
    result.kind = Outcome::overflow;
  } catch (const std::invalid_argument&) {
    result.kind = Outcome::invalid;
  } catch (const std::runtime_error&) {
    result.kind = Outcome::runtime;
  }
  if (_mm_getcsr() != before) die("production call changed MXCSR");
  return result;
}

std::string error_response(Outcome outcome, bool changed, bool input_changed, bool guard_changed,
                           bool trace_changed = false) {
  std::string result = "err kind=";
  result += outcome.kind == Outcome::invalid ? "invalid_argument" :
            outcome.kind == Outcome::overflow ? "overflow_error" : "runtime_error";
  result += " changed=" + std::to_string(changed);
  result += " inputchanged=" + std::to_string(input_changed);
  result += " guardchanged=" + std::to_string(guard_changed);
  result += " tracechanged=" + std::to_string(trace_changed);
  return result;
}

bool differs(const std::vector<unsigned char>& before, const void* after, std::size_t bytes) {
  return bytes != 0 && std::memcmp(before.data(), after, bytes) != 0;
}

std::size_t checked_add(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) die("executor extent overflow");
  return left + right;
}

constexpr std::array<std::string_view, 7> kNames = {
    "bias", "bias_vl", "linear", "temp", "softplus", "score", "selection"};

std::size_t name_index(std::string_view name) {
  const auto found = std::find(kNames.begin(), kNames.end(), name);
  if (found == kNames.end()) die("unknown overlap name");
  return static_cast<std::size_t>(found - kNames.begin());
}

void poison_bytes(const void* pointer, std::size_t bytes, bool poison) {
#if defined(__SANITIZE_ADDRESS__)
  if (poison) ASAN_POISON_MEMORY_REGION(pointer, bytes);
  else ASAN_UNPOISON_MEMORY_REGION(pointer, bytes);
#else
  (void)pointer; (void)bytes; (void)poison;
#endif
}

std::string run_route(const Request& request) {
  if (!request.has_n || !request.has_k || !request.has_image || !request.has_vl ||
      !request.has_trace || !request.has_caps || request.sections.size() != 4)
    die("route requires n,k,image,vl,trace,caps and four payload sections");
  if (request.has_precision && !request.trace) die("precision requires trace=1");
  if ((request.has_in_offset || request.has_out_offset) && !request.has_overlap)
    die("offsets require overlap");
  if (!request.vl && !request.sections[3].empty()) die("vl=0 requires empty fourth section");
  const bool vl_present = request.has_vl_present ? request.vl_present : request.vl;
  if (!vl_present && !request.sections[3].empty()) die("absent vision bias requires empty section");

  std::vector<std::uint16_t> input(request.sections[0].size());
  std::vector<std::uint16_t> gate(request.sections[1].size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = narrow<std::uint16_t>(request.sections[0][i]);
  for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = narrow<std::uint16_t>(request.sections[1][i]);

  const std::size_t lengths[7] = {request.sections[2].size(), request.sections[3].size(),
                                  request.caps[0], request.caps[1], request.caps[2],
                                  request.caps[3], request.caps[4]};
  std::vector<std::uint32_t> owners[7];
  std::uint32_t* buffers[7]{};
  std::vector<std::uint32_t> shared;
  std::size_t shared_size = 0;
  std::size_t first_index = 0, second_index = 0;
  if (request.has_overlap) {
    first_index = name_index(request.first); second_index = name_index(request.second);
    shared_size = std::max(checked_add(request.in_offset, lengths[first_index]),
                           checked_add(request.out_offset, lengths[second_index]));
    shared.assign(shared_size, kFill);
    buffers[first_index] = shared.data() + request.in_offset;
    buffers[second_index] = shared.data() + request.out_offset;
  }
  for (std::size_t i = 0; i < 7; ++i) {
    if (!buffers[i]) {
      owners[i].assign(lengths[i], kFill);
      buffers[i] = owners[i].data();
    }
  }
  for (std::size_t section = 2; section < 4; ++section) {
    const std::size_t index = section - 2;
    for (std::size_t i = 0; i < request.sections[section].size(); ++i)
      buffers[index][i] = narrow<std::uint32_t>(request.sections[section][i]);
  }

  std::vector<unsigned char> before[7];
  for (std::size_t i = 0; i < 7; ++i) {
    const auto* bytes = reinterpret_cast<unsigned char*>(buffers[i]);
    before[i].assign(bytes, bytes + lengths[i] * 4);
  }
  std::vector<unsigned char> input_before(reinterpret_cast<unsigned char*>(input.data()),
                                          reinterpret_cast<unsigned char*>(input.data()) + input.size() * 2);
  std::vector<unsigned char> gate_before(reinterpret_cast<unsigned char*>(gate.data()),
                                         reinterpret_cast<unsigned char*>(gate.data()) + gate.size() * 2);
  std::vector<unsigned char> shared_before;
  if (request.has_overlap) {
    const auto* bytes = reinterpret_cast<unsigned char*>(shared.data());
    shared_before.assign(bytes, bytes + shared.size() * 4);
  }

  const core::GateMatrix matrix{request.n, request.k, {gate.data(), gate.size()}};
  core::RouterWorkspace work{{buffers[2], lengths[2]}, {buffers[3], lengths[3]},
                             {buffers[4], lengths[4]}, {buffers[5], lengths[5]},
                             {buffers[6], lengths[6]}};
  const std::span<const std::uint32_t> bias(buffers[0], lengths[0]);
  std::optional<std::span<const std::uint32_t>> vision;
  if (vl_present) vision = std::span<const std::uint32_t>(buffers[1], lengths[1]);
  core::RouteResult result{};
  core::detail::RouterTrace trace;
  std::memset(&trace, 0xA5, sizeof(trace));
  const std::array<unsigned char, sizeof(core::detail::RouterTrace)> trace_before = [&] {
    std::array<unsigned char, sizeof(core::detail::RouterTrace)> bytes{};
    std::memcpy(bytes.data(), &trace, sizeof(trace));
    return bytes;
  }();

  if (request.poison) {
    poison_bytes(input.data(), input.size() * 2, true);
    poison_bytes(gate.data(), gate.size() * 2, true);
    poison_bytes(buffers[0], lengths[0] * 4, true);
    if (vl_present) poison_bytes(buffers[1], lengths[1] * 4, true);
  }
  const Outcome outcome = guarded([&] {
    if (request.trace) {
      result = core::detail::trace_route(input, matrix, bias, work, trace, request.image,
                                         vision, request.precision);
    } else {
      result = core::route(input, matrix, bias, work, request.image, vision);
    }
  });
  if (request.poison) {
    poison_bytes(input.data(), input.size() * 2, false);
    poison_bytes(gate.data(), gate.size() * 2, false);
    poison_bytes(buffers[0], lengths[0] * 4, false);
    if (vl_present) poison_bytes(buffers[1], lengths[1] * 4, false);
  }

  const bool input_changed = differs(input_before, input.data(), input.size() * 2) ||
                             differs(gate_before, gate.data(), gate.size() * 2) ||
                             differs(before[0], buffers[0], lengths[0] * 4) ||
                             differs(before[1], buffers[1], lengths[1] * 4);
  const bool trace_changed = request.trace &&
      std::memcmp(trace_before.data(), &trace, sizeof(trace)) != 0;
  bool work_changed = false;
  bool guard_changed = false;
  for (std::size_t i = 2; i < 7; ++i) {
    work_changed |= differs(before[i], buffers[i], lengths[i] * 4);
    const bool shared_buffer = request.has_overlap && (i == first_index || i == second_index);
    const std::size_t prefix = std::min(request.n, lengths[i]);
    if (!shared_buffer && lengths[i] > prefix) {
      guard_changed |= std::memcmp(before[i].data() + prefix * 4, buffers[i] + prefix,
                                   (lengths[i] - prefix) * 4) != 0;
    }
  }
  if (request.has_overlap) {
    const auto* bytes = reinterpret_cast<unsigned char*>(shared.data());
    for (std::size_t offset = 0; offset < shared.size(); ++offset) {
      bool writable = false;
      for (const std::size_t index : {first_index, second_index}) {
        if (index < 2) continue;
        const std::size_t start = static_cast<std::size_t>(buffers[index] - shared.data());
        writable |= offset >= start && offset - start < std::min(request.n, lengths[index]);
      }
      if (!writable && std::memcmp(shared_before.data() + offset * 4, bytes + offset * 4, 4))
        guard_changed = true;
    }
  }
  if (outcome.kind != Outcome::ok) {
    return error_response(outcome, input_changed || work_changed || trace_changed || guard_changed,
                          input_changed, guard_changed, trace_changed);
  }
  if (input_changed) die("router input changed on success");
  if (guard_changed) die("router bytes outside write prefixes changed on success");

  std::string response = "ok";
  for (std::size_t i = 0; i < 5; ++i)
    segment(response, kNames[i + 2], std::span<const std::uint32_t>(buffers[i + 2], lengths[i + 2]));
  response += " ; indices=";
  for (const std::size_t index : result.indices) append_hex64(response, index);
  if (request.trace) {
    segment(response, "selected", trace.selected_bits);
    scalar(response, "sum", trace.sum_bits);
    scalar(response, "denominator", trace.denominator_bits);
    segment(response, "normalized", trace.normalized_bits);
  }
  segment(response, "weights", result.weights_bits);
  response += " ; input=";
  for (const auto value : input) append_hex(response, value);
  response += " ; gate=";
  for (const auto value : gate) append_hex(response, value);
  segment(response, "bias", bias);
  response += " ; bias_vl=";
  if (vl_present) for (const auto value : *vision) append_hex(response, value);
  return response;
}

std::string run_counts(const Request& request) {
  if (!request.has_n || !request.has_k || request.sections.size() != 1 ||
      !request.sections[0].empty()) die("counts requires n,k and no payload");
  const Outcome outcome = guarded([&] { core::detail::check_router_counts(request.n, request.k); });
  if (outcome.kind != Outcome::ok) return error_response(outcome, false, false, false);
  return "ok";
}

std::string run_storage(const Request& request) {
  if (!request.has_read_size || !request.has_write_size || !request.has_read_offset ||
      !request.has_write_offset || request.sections.size() != 1 || !request.sections[0].empty())
    die("storage requires sizes, offsets and no payload");
  const std::size_t total = std::max(checked_add(request.read_offset, request.read_size),
                                     checked_add(request.write_offset, request.write_size));
  std::vector<std::byte> bytes(total, std::byte{0xA5});
  const std::vector<std::byte> before = bytes;
  const std::array reads{std::span<const std::byte>(bytes.data() + request.read_offset,
                                                    request.read_size)};
  const std::array writes{std::span<std::byte>(bytes.data() + request.write_offset,
                                               request.write_size)};
  const Outcome outcome = guarded([&] { core::detail::check_router_storage(reads, writes); });
  const bool changed = bytes != before;
  const bool input_changed = !std::equal(reads[0].begin(), reads[0].end(),
                                         before.begin() + request.read_offset);
  bool guard_changed = false;
  for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
    const bool writable = offset >= request.write_offset &&
                          offset - request.write_offset < request.write_size;
    guard_changed |= !writable && bytes[offset] != before[offset];
  }
  if (outcome.kind != Outcome::ok)
    return error_response(outcome, changed, input_changed, guard_changed);
  if (changed) die("router storage check changed backing");
  return "ok";
}

std::string run_math(const Request& request) {
  std::uint32_t value = 0;
  core::detail::SoftplusBounds bounds{};
  Outcome outcome;
  if (request.op == "div") {
    if (request.sections.size() != 2 || request.sections[0].size() != 1 ||
        request.sections[1].size() != 1) die("div requires one pair");
    outcome = guarded([&] { value = core::detail::divide_fp32(
        narrow<std::uint32_t>(request.sections[0][0]), narrow<std::uint32_t>(request.sections[1][0])); });
  } else if (request.op == "sqrt") {
    if (request.sections.size() != 1 || request.sections[0].size() != 1) die("sqrt requires one token");
    outcome = guarded([&] { value = core::detail::sqrt_fp32(narrow<std::uint32_t>(request.sections[0][0])); });
  } else if (request.op == "softplus") {
    if (!request.has_precision || request.sections.size() != 1 || request.sections[0].size() != 1)
      die("softplus requires precision and one token");
    outcome = guarded([&] { value = core::detail::softplus_fp32(
        narrow<std::uint32_t>(request.sections[0][0]), request.precision); });
  } else {
    if (!request.has_precision || request.sections.size() != 1 || request.sections[0].size() != 1)
      die("bounds requires precision and one token");
    outcome = guarded([&] { bounds = core::detail::softplus_bounds(
        narrow<std::uint32_t>(request.sections[0][0]), request.precision); });
  }
  if (outcome.kind != Outcome::ok) return error_response(outcome, false, false, false);
  std::string response = "ok";
  if (request.op == "bounds") {
    scalar(response, "lower", bounds.lower_bits); scalar(response, "upper", bounds.upper_bits);
  } else {
    scalar(response, "value", value);
  }
  return response;
}

std::string dispatch(const Request& request) {
  if (request.op == "route") return run_route(request);
  if (request.op == "counts") return run_counts(request);
  if (request.op == "storage") return run_storage(request);
  if (request.op == "div" || request.op == "sqrt" || request.op == "softplus" ||
      request.op == "bounds") return run_math(request);
  if (request.op == "poison_probe") {
    if (request.sections.size() != 1 || !request.sections[0].empty()) die("probe takes no payload");
#if defined(__SANITIZE_ADDRESS__)
    const auto byte = std::make_unique<unsigned char>(0x5Au);
    ASAN_POISON_MEMORY_REGION(byte.get(), 1);
    const volatile unsigned char value = *byte;
    (void)value;
    ASAN_UNPOISON_MEMORY_REGION(byte.get(), 1);
#endif
    return "ok";
  }
  if (request.op == "max_precision" || request.op == "mxcsr" || request.op == "asan") {
    if (request.sections.size() != 1 || !request.sections[0].empty()) die("probe takes no payload");
    std::uint32_t value = request.op == "max_precision" ? kCertificationBits :
        request.op == "mxcsr" ? _mm_getcsr() :
#if defined(__SANITIZE_ADDRESS__)
        1u;
#else
        0u;
#endif
    std::string response = "ok"; scalar(response, "value", value); return response;
  }
  die("unknown op '" + request.op + "'");
}

constexpr unsigned kRoundingMask = 0x6000u;
constexpr unsigned kFtz = 0x8000u;
constexpr unsigned kDaz = 0x40u;
constexpr unsigned kOverflowMask = 0x400u;

unsigned mxcsr_for(std::string_view name) {
  unsigned value = (_mm_getcsr() & ~(kRoundingMask | kFtz | kDaz | kStatusMask)) | 0x1F80u;
  if (name == "default") return value;
  if (name == "up") return value | 0x4000u;
  if (name == "down") return value | 0x2000u;
  if (name == "zero") return value | 0x6000u;
  if (name == "ftz") return value | kFtz;
  if (name == "daz") return value | kDaz;
  if (name == "ftz_daz") return value | kFtz | kDaz;
  if (name == "overflow_unmasked") return value & ~kOverflowMask;
  die("unknown MXCSR environment");
}

std::string wrong(std::string response) {
  for (char& character : response) {
    if (character >= '0' && character <= '9') { character = character == '0' ? '1' : '0'; break; }
  }
  return response;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mxcsr = "default";
  std::string fault = "none";
  unsigned status = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];
    if (argument.starts_with("--mxcsr=")) mxcsr = argument.substr(8);
    else if (argument.starts_with("--status=")) {
      const auto value = number(argument.substr(9), 16);
      if (value & ~kStatusMask) die("status exceeds MXCSR flags");
      status = static_cast<unsigned>(value);
    } else if (argument.starts_with("--fault=")) fault = argument.substr(8);
    else die("unknown argument");
  }
  if (fault != "none" && fault != "crash" && fault != "exit" && fault != "malformed" &&
      fault != "missing" && fault != "wrong") die("unknown fault");
  const unsigned initial = mxcsr_for(mxcsr) | status;
  _mm_setcsr(initial);

  std::ios::sync_with_stdio(false);
  std::string line;
  std::size_t served = 0;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    if (fault == "crash") std::abort();
    std::string response = dispatch(parse(line));
    if (served == 0 && fault == "malformed") response = "not-a-response";
    if (fault == "wrong") response = wrong(std::move(response));
    if (!(fault == "missing" && std::cin.peek() == EOF)) std::cout << response << '\n';
    if (fault == "exit") { std::cout.flush(); std::exit(3); }
    ++served;
  }
  std::cout.flush();
  if (_mm_getcsr() != initial) die("MXCSR changed while serving requests");
  return std::cout ? 0 : 2;
}
