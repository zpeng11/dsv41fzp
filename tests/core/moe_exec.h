// P0026 routed-branch executor support, included inside core_exec's anonymous namespace.

struct MoeMatrixStorage {
  std::size_t n = 0;
  std::size_t k = 0;
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;

  df::quant::ExpertMatrix view() const { return {n, k, packed, scales}; }
};

struct MoeExpertStorage {
  std::array<MoeMatrixStorage, 3> matrices;

  core::RoutedExpert view() const {
    return {matrices[0].view(), matrices[1].view(), matrices[2].view()};
  }
};

template <class T>
std::vector<unsigned char> byte_snapshot(const std::vector<T>& values) {
  const auto* begin = reinterpret_cast<const unsigned char*>(values.data());
  return values.empty() ? std::vector<unsigned char>{}
                        : std::vector<unsigned char>(begin, begin + values.size() * sizeof(T));
}

template <class T>
bool vector_changed(const std::vector<unsigned char>& before, const std::vector<T>& values) {
  return differs(before, values.data(), values.size() * sizeof(T));
}

std::string moe_error_response(Outcome outcome, bool input_changed, bool guard_changed,
                               bool trace_changed, bool work_changed, bool out_changed) {
  std::string result = "err kind=";
  result += outcome.kind == Outcome::invalid ? "invalid_argument" :
            outcome.kind == Outcome::overflow ? "overflow_error" : "runtime_error";
  result += " changed=" + std::to_string(input_changed || guard_changed || trace_changed ||
                                          work_changed || out_changed);
  result += " inputchanged=" + std::to_string(input_changed);
  result += " guardchanged=" + std::to_string(guard_changed);
  result += " tracechanged=" + std::to_string(trace_changed);
  result += " workchanged=" + std::to_string(work_changed);
  result += " outchanged=" + std::to_string(out_changed);
  return result;
}

struct ContractFixture {
  static constexpr std::size_t d = 32;
  static constexpr std::size_t h = 32;
  static constexpr std::size_t n = 6;
  std::array<std::uint16_t, d> input{};
  std::array<std::uint16_t, n * d> gate{};
  std::array<std::uint32_t, n> bias{};
  std::array<MoeExpertStorage, n> storage{};
  std::array<core::RoutedExpert, n> experts{};
  std::array<std::uint32_t, n + 1> linear{}, temp{}, softplus{}, score{}, selection{};
  std::array<std::uint8_t, d + 1> input_codes{};
  std::array<std::uint8_t, d / 32 + 1> input_scales{};
  std::array<std::uint16_t, h + 1> expert_gate{}, up{}, hidden{};
  std::array<std::uint8_t, h + 1> hidden_codes{};
  std::array<std::uint8_t, h / 32 + 1> hidden_scales{};
  std::array<std::uint16_t, d + 1> expert_output{};
  std::array<std::uint32_t, d + 1> accumulator{}, out{};

  ContractFixture() {
    input.fill(0x3F80u);
    bias.fill(0x3F800000u);
    for (std::size_t id = 0; id < n; ++id) {
      auto& matrices = storage[id].matrices;
      matrices[0].n = h; matrices[0].k = d;
      matrices[1].n = d; matrices[1].k = h;
      matrices[2].n = h; matrices[2].k = d;
      for (auto& matrix : matrices) {
        matrix.packed.assign(matrix.n * matrix.k / 2, 0);
        matrix.scales.assign(matrix.n * matrix.k / 32, 127);
      }
      experts[id] = storage[id].view();
    }
    fill();
  }

  void fill() {
    linear.fill(kFill); temp.fill(kFill); softplus.fill(kFill); score.fill(kFill);
    selection.fill(kFill); input_codes.fill(0xA5u); input_scales.fill(0xA5u);
    expert_gate.fill(0xA5A5u); up.fill(0xA5A5u); hidden.fill(0xA5A5u);
    hidden_codes.fill(0xA5u); hidden_scales.fill(0xA5u);
    expert_output.fill(0xA5A5u); accumulator.fill(kFill); out.fill(kFill);
  }

  core::GateMatrix matrix() const { return {n, d, gate}; }
  core::RoutedWorkspace work() {
    return {{linear, temp, softplus, score, selection},
            {input_codes, input_scales, expert_gate, up, hidden, hidden_codes, hidden_scales},
            expert_output, accumulator};
  }
};

std::string run_moe_contract(const Request& request) {
  if (request.sections.size() != 1 || !request.sections[0].empty() || request.has_n ||
      request.has_k || request.has_trace || request.has_out_cap || request.has_trace_cap ||
      request.has_caps || request.has_expert_caps || request.poison)
    die("moe_contract takes no fields or payload");
  ContractFixture fixture;
  const auto reject = [&](auto call, std::string_view label) {
    const auto out = fixture.out;
    const auto accumulator = fixture.accumulator;
    const auto expert_output = fixture.expert_output;
    const Outcome outcome = guarded(call);
    if (outcome.kind != Outcome::invalid) die(std::string(label) + " did not reject invalid_argument");
    if (fixture.out != out || fixture.accumulator != accumulator ||
        fixture.expert_output != expert_output)
      die(std::string(label) + " changed formal output or ordinary work on rejection");
  };

  std::array<std::uint32_t, 6 * ContractFixture::d + 2> partial{};
  std::array<std::uint32_t, 6 * ContractFixture::d + 2> trace_work{};
  partial.fill(kFill); trace_work.fill(kFill);
  core::detail::RoutedTrace trace{};
  trace.partial_sums_bits = partial;
  reject([&] {
    auto short_trace = trace;
    short_trace.partial_sums_bits = std::span<std::uint32_t>(partial).first(6 * ContractFixture::d - 1);
    core::detail::trace_routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                               fixture.work(), fixture.out, short_trace, trace_work);
  }, "short partial");
  reject([&] {
    core::detail::trace_routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                               fixture.work(), fixture.out, trace,
                               std::span<std::uint32_t>(trace_work).first(6 * ContractFixture::d - 1));
  }, "short trace work");

  reject([&] {
    auto alias = trace;
    alias.partial_sums_bits = std::span<std::uint32_t>(trace_work).first(6 * ContractFixture::d);
    core::detail::trace_routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                               fixture.work(), fixture.out, alias, trace_work);
  }, "partial/trace work alias");
  reject([&] {
    std::array<std::uint32_t, 6 * ContractFixture::d + 2> shared{};
    shared.fill(kFill);
    auto alias = trace;
    alias.partial_sums_bits = std::span<std::uint32_t>(shared).first(6 * ContractFixture::d);
    core::detail::trace_routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                               fixture.work(), std::span<std::uint32_t>(shared).first(ContractFixture::d + 1),
                               alias, trace_work);
  }, "partial/out alias");
  reject([&] {
    std::array<std::uint32_t, 6 * ContractFixture::d> shared{};
    auto alias = trace;
    alias.partial_sums_bits = shared;
    core::detail::trace_routed(fixture.input, fixture.matrix(),
                               std::span<const std::uint32_t>(shared).first(ContractFixture::n),
                               fixture.experts, fixture.work(), fixture.out, alias, trace_work);
  }, "partial/readonly alias");

  fixture.fill();
  partial.fill(kFill); trace_work.fill(kFill);
  core::detail::trace_routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                             fixture.work(), fixture.out, trace, trace_work);
  if (fixture.out[ContractFixture::d] != kFill || partial[6 * ContractFixture::d] != kFill ||
      trace_work[6 * ContractFixture::d] != kFill)
    die("trace success changed a destination tail");

  fixture.fill();
  fixture.input.fill(0x7F7Fu);
  fixture.storage[1].matrices[0].packed.assign(ContractFixture::h * ContractFixture::d / 2, 0xEEu);
  fixture.storage[1].matrices[0].scales.assign(ContractFixture::h * ContractFixture::d / 32, 254u);
  fixture.storage[1].matrices[2].packed.assign(ContractFixture::h * ContractFixture::d / 2, 0x22u);
  fixture.storage[1].matrices[2].scales.assign(ContractFixture::h * ContractFixture::d / 32, 127u);
  fixture.experts[1] = fixture.storage[1].view();
  partial.fill(kFill); trace_work.fill(kFill);
  trace = {}; trace.partial_sums_bits = partial;
  const auto failed_out = fixture.out;
  const auto failed_partial = partial;
  const auto failed_trace = trace;
  const Outcome failed = guarded([&] {
    core::detail::trace_routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                               fixture.work(), fixture.out, trace, trace_work);
  });
  if (failed.kind != Outcome::invalid || fixture.out != failed_out || partial != failed_partial ||
      trace.route.indices != failed_trace.route.indices ||
      trace.route.weights_bits != failed_trace.route.weights_bits ||
      trace.expert_indices != failed_trace.expert_indices ||
      trace.partial_sums_bits.data() != failed_trace.partial_sums_bits.data() ||
      trace.partial_sums_bits.size() != failed_trace.partial_sums_bits.size())
    die("runtime routed failure committed formal output or trace");
  return "ok";
}

std::string run_moe_accumulate(const Request& request) {
  if (!request.has_k || !request.has_trace || !request.has_out_cap || !request.has_trace_cap ||
      request.sections.size() != 2 || request.has_n || request.has_image || request.has_vl ||
      request.has_vl_present || request.has_caps || request.has_expert_caps ||
      request.has_expert_out_cap || request.has_accumulator_cap || request.has_precision ||
      request.has_overlap || request.has_in_offset || request.has_out_offset ||
      request.has_read_size || request.has_write_size || request.has_read_offset ||
      request.has_write_offset || request.poison)
    die("moe_accumulate requires only k,trace,out_cap,trace_cap and two sections");
  if (request.sections[0].size() != request.k || request.sections[1].size() != request.k)
    die("moe_accumulate sections must both have k elements");
  std::vector<std::uint32_t> accumulator(request.out_cap, kFill);
  std::vector<std::uint16_t> expert_output(request.sections[1].size());
  std::vector<std::uint32_t> partial(request.trace_cap, kFill);
  if (request.sections[0].size() > accumulator.size()) die("initial accumulator exceeds out_cap");
  for (std::size_t i = 0; i < request.sections[0].size(); ++i)
    accumulator[i] = narrow<std::uint32_t>(request.sections[0][i]);
  for (std::size_t i = 0; i < expert_output.size(); ++i)
    expert_output[i] = narrow<std::uint16_t>(request.sections[1][i]);
  const auto input_before = byte_snapshot(expert_output);
  const auto out_before = byte_snapshot(accumulator);
  const auto partial_before = byte_snapshot(partial);
  const Outcome outcome = guarded([&] {
    core::detail::accumulate_expert_output(expert_output, accumulator,
        request.trace ? std::span<std::uint32_t>(partial) : std::span<std::uint32_t>{});
  });
  const bool input_changed = vector_changed(input_before, expert_output);
  const bool out_changed = vector_changed(out_before, accumulator);
  const bool trace_changed = vector_changed(partial_before, partial);
  const auto tail_differs = [](const auto& before, const auto& after, std::size_t used) {
    using Value = typename std::decay_t<decltype(after)>::value_type;
    const std::size_t prefix = std::min(used, after.size());
    const std::size_t offset = prefix * sizeof(Value);
    const std::size_t bytes = (after.size() - prefix) * sizeof(Value);
    return bytes && std::memcmp(before.data() + offset, after.data() + prefix, bytes) != 0;
  };
  const bool guard_changed = tail_differs(out_before, accumulator, request.k) ||
      tail_differs(partial_before, partial, request.trace ? request.k : 0);
  if (outcome.kind != Outcome::ok)
    return moe_error_response(outcome, input_changed, guard_changed, trace_changed, false,
                              out_changed);
  if (input_changed || guard_changed) die("moe accumulator changed forbidden storage");
  std::string response = "ok ; partial=";
  if (request.trace) for (const auto value : partial) append_hex(response, value);
  segment(response, "out", accumulator);
  return response;
}

std::string run_moe(const Request& request) {
  if (!request.has_n || !request.has_k || !request.has_image || !request.has_vl ||
      !request.has_trace || !request.has_caps || !request.has_expert_caps ||
      !request.has_expert_out_cap || !request.has_accumulator_cap || !request.has_out_cap ||
      !request.has_trace_cap || request.has_precision || request.has_overlap ||
      request.has_in_offset || request.has_out_offset || request.has_read_size ||
      request.has_write_size || request.has_read_offset || request.has_write_offset) {
    die("moe requires n,k,image,vl,trace,caps,expert_caps and all routed capacities");
  }
  if (request.n > (std::numeric_limits<std::size_t>::max() - 4) / 9)
    die("moe section count overflows size_t");
  const std::size_t expected_sections = 4 + request.n * 9;
  if (request.sections.size() != expected_sections)
    die("moe needs four common sections plus nine sections per expert");
  if (!request.vl && !request.sections[3].empty()) die("vl=0 requires empty fourth section");
  const bool vl_present = request.has_vl_present ? request.vl_present : request.vl;
  if (!vl_present && !request.sections[3].empty()) die("absent vision bias requires empty section");

  std::vector<std::uint16_t> input(request.sections[0].size());
  std::vector<std::uint16_t> gate(request.sections[1].size());
  std::vector<std::uint32_t> bias(request.sections[2].size());
  std::vector<std::uint32_t> vision(request.sections[3].size());
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = narrow<std::uint16_t>(request.sections[0][i]);
  for (std::size_t i = 0; i < gate.size(); ++i)
    gate[i] = narrow<std::uint16_t>(request.sections[1][i]);
  for (std::size_t i = 0; i < bias.size(); ++i)
    bias[i] = narrow<std::uint32_t>(request.sections[2][i]);
  for (std::size_t i = 0; i < vision.size(); ++i)
    vision[i] = narrow<std::uint32_t>(request.sections[3][i]);

  std::vector<MoeExpertStorage> expert_storage(request.n);
  std::size_t section = 4;
  for (auto& expert : expert_storage) {
    for (auto& matrix : expert.matrices) {
      const auto& dimensions = request.sections[section++];
      if (dimensions.size() != 2) die("each moe matrix descriptor needs n and k");
      matrix.n = narrow<std::size_t>(dimensions[0]);
      matrix.k = narrow<std::size_t>(dimensions[1]);
      matrix.packed.resize(request.sections[section].size());
      for (std::size_t i = 0; i < matrix.packed.size(); ++i)
        matrix.packed[i] = narrow<std::uint8_t>(request.sections[section][i]);
      ++section;
      matrix.scales.resize(request.sections[section].size());
      for (std::size_t i = 0; i < matrix.scales.size(); ++i)
        matrix.scales[i] = narrow<std::uint8_t>(request.sections[section][i]);
      ++section;
    }
  }
  std::vector<core::RoutedExpert> experts;
  experts.reserve(expert_storage.size());
  for (const auto& expert : expert_storage) experts.push_back(expert.view());

  std::array<std::vector<std::uint32_t>, 5> router_buffers;
  for (std::size_t i = 0; i < router_buffers.size(); ++i)
    router_buffers[i].assign(request.caps[i], kFill);
  std::vector<std::uint8_t> input_codes(request.expert_caps[0], 0xA5u);
  std::vector<std::uint8_t> input_scales(request.expert_caps[1], 0xA5u);
  std::vector<std::uint16_t> expert_gate(request.expert_caps[2], 0xA5A5u);
  std::vector<std::uint16_t> expert_up(request.expert_caps[3], 0xA5A5u);
  std::vector<std::uint16_t> hidden(request.expert_caps[4], 0xA5A5u);
  std::vector<std::uint8_t> hidden_codes(request.expert_caps[5], 0xA5u);
  std::vector<std::uint8_t> hidden_scales(request.expert_caps[6], 0xA5u);
  std::vector<std::uint16_t> expert_output(request.expert_out_cap, 0xA5A5u);
  std::vector<std::uint32_t> accumulator(request.accumulator_cap, kFill);
  std::vector<std::uint32_t> out(request.out_cap, kFill);
  std::vector<std::uint32_t> partial(request.trace_cap, kFill);
  std::vector<std::uint32_t> trace_work(request.trace_cap, kFill);

  const auto input_before = byte_snapshot(input);
  const auto gate_before = byte_snapshot(gate);
  const auto bias_before = byte_snapshot(bias);
  const auto vision_before = byte_snapshot(vision);
  std::vector<std::vector<unsigned char>> expert_before;
  for (const auto& expert : expert_storage) {
    for (const auto& matrix : expert.matrices) {
      expert_before.push_back(byte_snapshot(matrix.packed));
      expert_before.push_back(byte_snapshot(matrix.scales));
    }
  }
  std::array<std::vector<unsigned char>, 5> router_before;
  for (std::size_t i = 0; i < router_buffers.size(); ++i)
    router_before[i] = byte_snapshot(router_buffers[i]);
  const auto input_codes_before = byte_snapshot(input_codes);
  const auto input_scales_before = byte_snapshot(input_scales);
  const auto expert_gate_before = byte_snapshot(expert_gate);
  const auto expert_up_before = byte_snapshot(expert_up);
  const auto hidden_before = byte_snapshot(hidden);
  const auto hidden_codes_before = byte_snapshot(hidden_codes);
  const auto hidden_scales_before = byte_snapshot(hidden_scales);
  const auto expert_output_before = byte_snapshot(expert_output);
  const auto accumulator_before = byte_snapshot(accumulator);
  const auto out_before = byte_snapshot(out);
  const auto partial_before = byte_snapshot(partial);
  const auto trace_work_before = byte_snapshot(trace_work);

  const core::GateMatrix matrix{request.n, request.k, gate};
  const core::RouterWorkspace router_work{router_buffers[0], router_buffers[1], router_buffers[2],
                                          router_buffers[3], router_buffers[4]};
  const df::quant::ExpertWorkspace expert_work{input_codes, input_scales, expert_gate, expert_up,
                                               hidden, hidden_codes, hidden_scales};
  const core::RoutedWorkspace work{router_work, expert_work, expert_output, accumulator};
  std::optional<std::span<const std::uint32_t>> bias_vl;
  if (vl_present) bias_vl = vision;
  core::detail::RoutedTrace trace{};
  trace.partial_sums_bits = partial;
  const std::array<unsigned char, sizeof(trace)> trace_before = [&] {
    std::array<unsigned char, sizeof(trace)> bytes{};
    std::memcpy(bytes.data(), &trace, sizeof(trace));
    return bytes;
  }();

  if (request.poison) {
    poison_bytes(input.data(), input.size() * sizeof(input[0]), true);
    poison_bytes(gate.data(), gate.size() * sizeof(gate[0]), true);
    poison_bytes(bias.data(), bias.size() * sizeof(bias[0]), true);
    if (vl_present) poison_bytes(vision.data(), vision.size() * sizeof(vision[0]), true);
    for (const auto& expert : expert_storage) for (const auto& item : expert.matrices) {
      poison_bytes(item.packed.data(), item.packed.size(), true);
      poison_bytes(item.scales.data(), item.scales.size(), true);
    }
  }
  const Outcome outcome = guarded([&] {
    if (request.trace) {
      core::detail::trace_routed(input, matrix, bias, experts, work, out, trace, trace_work,
                                 request.image, bias_vl);
    } else {
      core::routed(input, matrix, bias, experts, work, out, request.image, bias_vl);
    }
  });
  if (request.poison) {
    poison_bytes(input.data(), input.size() * sizeof(input[0]), false);
    poison_bytes(gate.data(), gate.size() * sizeof(gate[0]), false);
    poison_bytes(bias.data(), bias.size() * sizeof(bias[0]), false);
    if (vl_present) poison_bytes(vision.data(), vision.size() * sizeof(vision[0]), false);
    for (const auto& expert : expert_storage) for (const auto& item : expert.matrices) {
      poison_bytes(item.packed.data(), item.packed.size(), false);
      poison_bytes(item.scales.data(), item.scales.size(), false);
    }
  }

  bool input_changed = vector_changed(input_before, input) || vector_changed(gate_before, gate) ||
                       vector_changed(bias_before, bias) || vector_changed(vision_before, vision);
  std::size_t before_index = 0;
  for (const auto& expert : expert_storage) for (const auto& item : expert.matrices) {
    input_changed |= vector_changed(expert_before[before_index++], item.packed);
    input_changed |= vector_changed(expert_before[before_index++], item.scales);
  }
  bool work_changed = vector_changed(trace_work_before, trace_work);
  for (std::size_t i = 0; i < router_buffers.size(); ++i)
    work_changed |= vector_changed(router_before[i], router_buffers[i]);
  work_changed |= vector_changed(input_codes_before, input_codes) ||
                  vector_changed(input_scales_before, input_scales) ||
                  vector_changed(expert_gate_before, expert_gate) ||
                  vector_changed(expert_up_before, expert_up) ||
                  vector_changed(hidden_before, hidden) ||
                  vector_changed(hidden_codes_before, hidden_codes) ||
                  vector_changed(hidden_scales_before, hidden_scales) ||
                  vector_changed(expert_output_before, expert_output) ||
                  vector_changed(accumulator_before, accumulator);
  const bool out_changed = vector_changed(out_before, out);
  const bool trace_changed = request.trace &&
      (std::memcmp(trace_before.data(), &trace, sizeof(trace)) != 0 ||
       vector_changed(partial_before, partial));

  const std::size_t d = request.k;
  const std::size_t h = expert_storage.empty() ? 0 : expert_storage.front().matrices[0].n;
  bool guard_changed = false;
  const auto tail_changed = [&](const auto& before, const auto& after, std::size_t used) {
    using Value = typename std::decay_t<decltype(after)>::value_type;
    const std::size_t prefix = std::min(used, after.size());
    const std::size_t offset = prefix * sizeof(Value);
    const std::size_t bytes = (after.size() - prefix) * sizeof(Value);
    return bytes != 0 && std::memcmp(before.data() + offset, after.data() + prefix, bytes) != 0;
  };
  for (std::size_t i = 0; i < router_buffers.size(); ++i)
    guard_changed |= tail_changed(router_before[i], router_buffers[i], request.n);
  guard_changed |= tail_changed(input_codes_before, input_codes, d);
  guard_changed |= tail_changed(input_scales_before, input_scales, d / 32);
  guard_changed |= tail_changed(expert_gate_before, expert_gate, h);
  guard_changed |= tail_changed(expert_up_before, expert_up, h);
  guard_changed |= tail_changed(hidden_before, hidden, h);
  guard_changed |= tail_changed(hidden_codes_before, hidden_codes, h);
  guard_changed |= tail_changed(hidden_scales_before, hidden_scales, h / 32);
  guard_changed |= tail_changed(expert_output_before, expert_output, d);
  guard_changed |= tail_changed(accumulator_before, accumulator, d);
  guard_changed |= tail_changed(out_before, out, d);
  if (request.trace) {
    guard_changed |= tail_changed(partial_before, partial, 6 * d);
    guard_changed |= tail_changed(trace_work_before, trace_work, 6 * d);
  } else {
    guard_changed |= vector_changed(partial_before, partial) || vector_changed(trace_work_before, trace_work);
  }

  if (outcome.kind != Outcome::ok) {
    return moe_error_response(outcome, input_changed, guard_changed, trace_changed, work_changed,
                              out_changed);
  }
  if (input_changed) die("routed input changed on success");
  if (guard_changed) die("routed bytes outside write prefixes changed on success");

  std::string response = "ok ; indices=";
  if (request.trace) for (const auto index : trace.route.indices) append_hex64(response, index);
  response += " ; weights=";
  if (request.trace) for (const auto value : trace.route.weights_bits) append_hex(response, value);
  response += " ; experts=";
  if (request.trace) for (const auto index : trace.expert_indices) append_hex64(response, index);
  response += " ; partial=";
  if (request.trace) for (std::size_t i = 0; i < std::min(6 * d, partial.size()); ++i)
    append_hex(response, partial[i]);
  segment(response, "out", out);
  response += " ; expert_output=";
  for (const auto value : expert_output) {
    char buffer[5];
    std::snprintf(buffer, sizeof buffer, "%04x", static_cast<unsigned>(value));
    response += response.back() == '=' ? "" : " ";
    response += buffer;
  }
  return response;
}
