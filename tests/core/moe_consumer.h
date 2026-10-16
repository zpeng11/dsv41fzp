// P0026 public routed-branch consumer checks, included inside router_consumer's namespace.

struct MoeExpertData {
  static constexpr std::size_t kD = 32;
  static constexpr std::size_t kH = 32;
  std::array<std::uint8_t, kH * kD / 2> w1{};
  std::array<std::uint8_t, kD * kH / 2> w2{};
  std::array<std::uint8_t, kH * kD / 2> w3{};
  std::array<std::uint8_t, kH * kD / 32> s1{};
  std::array<std::uint8_t, kD * kH / 32> s2{};
  std::array<std::uint8_t, kH * kD / 32> s3{};

  MoeExpertData() { s1.fill(127); s2.fill(127); s3.fill(127); }

  df::core::RoutedExpert view() const {
    return {{kH, kD, w1, s1}, {kD, kH, w2, s2}, {kH, kD, w3, s3}};
  }
};

struct MoeFixture {
  static constexpr std::size_t kD = 32;
  static constexpr std::size_t kH = 32;
  static constexpr std::size_t kN = 8;

  std::array<std::uint16_t, kD> input{};
  std::array<std::uint16_t, kN * kD> gate_weights{};
  std::array<std::uint32_t, kN> bias{};
  std::array<std::uint32_t, kN> bias_vl{};
  std::array<MoeExpertData, kN> expert_data{};
  std::array<df::core::RoutedExpert, kN> experts{};
  std::array<std::uint32_t, kN + 2> linear{}, temp{}, softplus{}, score{}, selection{};
  std::array<std::uint8_t, kD + 2> input_codes{};
  std::array<std::uint8_t, kD / 32 + 2> input_scales{};
  std::array<std::uint16_t, kH + 2> expert_gate{}, up{}, hidden{};
  std::array<std::uint8_t, kH + 2> hidden_codes{};
  std::array<std::uint8_t, kH / 32 + 2> hidden_scales{};
  std::array<std::uint16_t, kD + 2> expert_output{};
  std::array<std::uint32_t, kD + 2> accumulator{};
  std::array<std::uint32_t, kD + 2> out{};

  MoeFixture() {
    input.fill(0x3F80u);
    for (std::size_t i = 0; i < 6; ++i) bias[i] = 0x3F800000u;
    for (std::size_t i = 2; i < kN; ++i) bias_vl[i] = 0x3F800000u;
    refresh_experts();
    fill_destinations();
  }

  void refresh_experts() {
    for (std::size_t i = 0; i < kN; ++i) experts[i] = expert_data[i].view();
  }

  void fill_destinations() {
    linear.fill(0xA5A5A5A5u); temp.fill(0xA5A5A5A5u); softplus.fill(0xA5A5A5A5u);
    score.fill(0xA5A5A5A5u); selection.fill(0xA5A5A5A5u);
    input_codes.fill(0xA5u); input_scales.fill(0xA5u);
    expert_gate.fill(0xA5A5u); up.fill(0xA5A5u); hidden.fill(0xA5A5u);
    hidden_codes.fill(0xA5u); hidden_scales.fill(0xA5u);
    expert_output.fill(0xA5A5u); accumulator.fill(0xA5A5A5A5u); out.fill(0xA5A5A5A5u);
  }

  df::core::GateMatrix matrix() const { return {kN, kD, gate_weights}; }
  df::core::RoutedWorkspace workspace() {
    return {{linear, temp, softplus, score, selection},
            {input_codes, input_scales, expert_gate, up, hidden, hidden_codes, hidden_scales},
            expert_output, accumulator};
  }
};

struct MoeSnapshot {
  std::array<std::uint16_t, MoeFixture::kD> input;
  std::array<std::uint16_t, MoeFixture::kN * MoeFixture::kD> gate_weights;
  std::array<std::uint32_t, MoeFixture::kN> bias, bias_vl;
  std::array<MoeExpertData, MoeFixture::kN> expert_data;
  std::array<df::core::RoutedExpert, MoeFixture::kN> experts;
  std::array<std::uint32_t, MoeFixture::kN + 2> linear, temp, softplus, score, selection;
  std::array<std::uint8_t, MoeFixture::kD + 2> input_codes;
  std::array<std::uint8_t, MoeFixture::kD / 32 + 2> input_scales;
  std::array<std::uint16_t, MoeFixture::kH + 2> expert_gate, up, hidden;
  std::array<std::uint8_t, MoeFixture::kH + 2> hidden_codes;
  std::array<std::uint8_t, MoeFixture::kH / 32 + 2> hidden_scales;
  std::array<std::uint16_t, MoeFixture::kD + 2> expert_output;
  std::array<std::uint32_t, MoeFixture::kD + 2> accumulator, out;
};

MoeSnapshot snapshot(const MoeFixture& f) {
  return {f.input, f.gate_weights, f.bias, f.bias_vl, f.expert_data, f.experts,
          f.linear, f.temp, f.softplus, f.score, f.selection, f.input_codes, f.input_scales,
          f.expert_gate, f.up, f.hidden, f.hidden_codes, f.hidden_scales, f.expert_output,
          f.accumulator, f.out};
}

bool operator==(const MoeExpertData& left, const MoeExpertData& right) {
  return left.w1 == right.w1 && left.w2 == right.w2 && left.w3 == right.w3 &&
         left.s1 == right.s1 && left.s2 == right.s2 && left.s3 == right.s3;
}

bool same_matrix(df::quant::ExpertMatrix left, df::quant::ExpertMatrix right) {
  return left.n == right.n && left.k == right.k && left.w_packed.data() == right.w_packed.data() &&
         left.w_packed.size() == right.w_packed.size() && left.scale_b.data() == right.scale_b.data() &&
         left.scale_b.size() == right.scale_b.size();
}

bool same_expert(df::core::RoutedExpert left, df::core::RoutedExpert right) {
  return same_matrix(left.w1, right.w1) && same_matrix(left.w2, right.w2) &&
         same_matrix(left.w3, right.w3);
}

bool operator==(const MoeSnapshot& left, const MoeSnapshot& right) {
  const bool descriptors = std::equal(left.experts.begin(), left.experts.end(),
                                      right.experts.begin(), same_expert);
  return descriptors && left.input == right.input && left.gate_weights == right.gate_weights &&
         left.bias == right.bias && left.bias_vl == right.bias_vl &&
         left.expert_data == right.expert_data && left.linear == right.linear &&
         left.temp == right.temp && left.softplus == right.softplus &&
         left.score == right.score && left.selection == right.selection &&
         left.input_codes == right.input_codes && left.input_scales == right.input_scales &&
         left.expert_gate == right.expert_gate && left.up == right.up &&
         left.hidden == right.hidden && left.hidden_codes == right.hidden_codes &&
         left.hidden_scales == right.hidden_scales && left.expert_output == right.expert_output &&
         left.accumulator == right.accumulator && left.out == right.out;
}

template <class Call>
void rejects_unchanged(MoeFixture& fixture, Call call, std::string_view label) {
  const MoeSnapshot before = snapshot(fixture);
  rejects(call);
  require(snapshot(fixture) == before,
          std::string(label) + " changed storage on parameter rejection");
}

void moe_success_allocation_tail_and_environment() {
  MoeFixture fixture;
  const auto input = fixture.input;
  const auto gate = fixture.gate_weights;
  const auto bias = fixture.bias;
  const auto bias_vl = fixture.bias_vl;
  const unsigned saved = _mm_getcsr();
  constexpr unsigned kControl = 0xE040u;
  for (const unsigned mode : {0u, 0x4000u, 0x2000u, 0x6000u, 0x8000u, 0x40u, 0x8040u,
                              0xFFFFFFFFu}) {
    for (const unsigned status : {0u, 0x3Fu}) {
      fixture.fill_destinations();
      unsigned setting = (saved & ~kControl & ~0x3Fu) | 0x1F80u | status;
      if (mode == 0xFFFFFFFFu) setting &= ~0x400u;
      else setting |= mode;
      _mm_setcsr(setting);
      g_forbid_allocations = true;
      df::core::routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                       fixture.workspace(), fixture.out, false, fixture.bias_vl);
      g_forbid_allocations = false;
      require(_mm_getcsr() == setting, "routed changed MXCSR");
      require(std::all_of(fixture.out.begin(), fixture.out.begin() + MoeFixture::kD,
                          [](auto value) { return value == 0; }),
              "zero routed result mismatch");
      require(fixture.out[MoeFixture::kD] == 0xA5A5A5A5u &&
                  fixture.expert_output[MoeFixture::kD] == 0xA5A5u &&
                  fixture.accumulator[MoeFixture::kD] == 0xA5A5A5A5u &&
                  fixture.linear[MoeFixture::kN] == 0xA5A5A5A5u &&
                  fixture.input_codes[MoeFixture::kD] == 0xA5u,
              "routed destination tail changed");
    }
  }
  _mm_setcsr(saved);
  require(fixture.input == input && fixture.gate_weights == gate && fixture.bias == bias &&
              fixture.bias_vl == bias_vl, "routed changed read-only input");
}

void moe_readonly_sharing_and_adjacent_writes_are_accepted() {
  MoeFixture fixture;
  auto shared_reads = fixture.experts;
  shared_reads[1].w1 = shared_reads[0].w1;
  std::array<std::uint32_t, 2 * MoeFixture::kD + 2> adjacent{};
  adjacent.fill(0xA5A5A5A5u);
  auto work = fixture.workspace();
  work.accumulator_bits = std::span<std::uint32_t>(adjacent).first(MoeFixture::kD);
  const auto out = std::span<std::uint32_t>(adjacent).subspan(MoeFixture::kD, MoeFixture::kD + 2);
  df::core::routed(fixture.input, fixture.matrix(), fixture.bias, shared_reads, work, out);
  require(adjacent.back() == 0xA5A5A5A5u, "adjacent output tail changed");
}

void moe_parameter_rejections_are_atomic() {
  MoeFixture fixture;
  rejects_unchanged(fixture, [&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias,
                     std::span<const df::core::RoutedExpert>(fixture.experts).first(7),
                     fixture.workspace(), fixture.out);
  }, "expert count");
  rejects_unchanged(fixture, [&] {
    auto work = fixture.workspace();
    work.accumulator_bits = work.accumulator_bits.first(MoeFixture::kD - 1);
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts, work,
                     fixture.out);
  }, "accumulator capacity");

  auto bad_unselected = fixture.experts;
  bad_unselected[7].w1.w_packed = bad_unselected[7].w1.w_packed.first(
      bad_unselected[7].w1.w_packed.size() - 1);
  rejects_unchanged(fixture, [&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, bad_unselected,
                     fixture.workspace(), fixture.out);
  }, "unselected expert packed length");
  bad_unselected = fixture.experts;
  bad_unselected[7].w2.k += 32;
  rejects_unchanged(fixture, [&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, bad_unselected,
                     fixture.workspace(), fixture.out);
  }, "unselected expert matrix relation");

  auto bad_scale = fixture.experts;
  fixture.expert_data[7].s2.back() = 0xFFu;
  fixture.refresh_experts();
  bad_scale = fixture.experts;
  rejects_unchanged(fixture, [&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, bad_scale,
                     fixture.workspace(), fixture.out);
  }, "unselected expert payload");
  fixture.expert_data[7].s2.back() = 127;
  fixture.refresh_experts();

  auto invalid_text = fixture.bias;
  invalid_text.back() = 0x7F800000u;
  rejects_unchanged(fixture, [&] {
    df::core::routed(fixture.input, fixture.matrix(), invalid_text, fixture.experts,
                     fixture.workspace(), fixture.out, true, fixture.bias_vl);
  }, "unused text bias");
  auto invalid_vision = fixture.bias_vl;
  invalid_vision.back() = 0x7F800000u;
  rejects_unchanged(fixture, [&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                     fixture.workspace(), fixture.out, false, invalid_vision);
  }, "unused vision bias");
}

void moe_aliases_reject_and_runtime_keeps_output() {
  MoeFixture fixture;
  const auto out_before = fixture.out;
  const auto packed_before = fixture.expert_data[0].w1;
  auto work = fixture.workspace();
  work.expert.input_codes = std::span<std::uint8_t>(fixture.expert_data[0].w1).first(MoeFixture::kD);
  rejects([&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts, work,
                     fixture.out);
  });
  require(fixture.out == out_before && fixture.expert_data[0].w1 == packed_before,
          "expert payload/workspace alias rejection changed storage");

  fixture.fill_destinations();
  const auto shared_out_before = fixture.out;
  work = fixture.workspace();
  work.accumulator_bits = fixture.out;
  rejects([&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts, work,
                     fixture.out);
  });
  require(fixture.out == shared_out_before, "out/work alias rejection changed out");

  fixture.fill_destinations();
  fixture.input.fill(0x7F7Fu);
  fixture.expert_data[1].w1.fill(0xEEu);
  fixture.expert_data[1].s1.fill(254u);
  fixture.expert_data[1].w3.fill(0x22u);
  fixture.expert_data[1].s3.fill(127u);
  fixture.refresh_experts();
  const auto runtime_out = fixture.out;
  rejects([&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, fixture.experts,
                     fixture.workspace(), fixture.out);
  });
  require(fixture.out == runtime_out, "expert runtime failure changed formal output");
}

void moe_metadata_precedes_poisoned_contents() {
#if defined(__SANITIZE_ADDRESS__)
  MoeFixture fixture;
  const MoeSnapshot before = snapshot(fixture);
  auto invalid = fixture.experts;
  invalid[7].w1.w_packed = invalid[7].w1.w_packed.first(invalid[7].w1.w_packed.size() - 1);
  const auto poison = [](const void* pointer, std::size_t bytes) {
    ASAN_POISON_MEMORY_REGION(pointer, bytes);
  };
  poison(fixture.input.data(), sizeof(fixture.input));
  poison(fixture.gate_weights.data(), sizeof(fixture.gate_weights));
  poison(fixture.bias.data(), sizeof(fixture.bias));
  for (const auto& item : fixture.expert_data) {
    poison(item.w1.data(), item.w1.size()); poison(item.s1.data(), item.s1.size());
    poison(item.w2.data(), item.w2.size()); poison(item.s2.data(), item.s2.size());
    poison(item.w3.data(), item.w3.size()); poison(item.s3.data(), item.s3.size());
  }
  rejects([&] {
    df::core::routed(fixture.input, fixture.matrix(), fixture.bias, invalid,
                     fixture.workspace(), fixture.out);
  });
  ASAN_UNPOISON_MEMORY_REGION(fixture.input.data(), sizeof(fixture.input));
  ASAN_UNPOISON_MEMORY_REGION(fixture.gate_weights.data(), sizeof(fixture.gate_weights));
  ASAN_UNPOISON_MEMORY_REGION(fixture.bias.data(), sizeof(fixture.bias));
  for (const auto& item : fixture.expert_data) {
    ASAN_UNPOISON_MEMORY_REGION(item.w1.data(), item.w1.size());
    ASAN_UNPOISON_MEMORY_REGION(item.s1.data(), item.s1.size());
    ASAN_UNPOISON_MEMORY_REGION(item.w2.data(), item.w2.size());
    ASAN_UNPOISON_MEMORY_REGION(item.s2.data(), item.s2.size());
    ASAN_UNPOISON_MEMORY_REGION(item.w3.data(), item.w3.size());
    ASAN_UNPOISON_MEMORY_REGION(item.s3.data(), item.s3.size());
  }
  require(snapshot(fixture) == before, "poisoned metadata rejection changed storage");
#endif
}

void run_moe_consumer() {
  moe_success_allocation_tail_and_environment();
  moe_readonly_sharing_and_adjacent_writes_are_accepted();
  moe_parameter_rejections_are_atomic();
  moe_aliases_reject_and_runtime_keeps_output();
  moe_metadata_precedes_poisoned_contents();
}
