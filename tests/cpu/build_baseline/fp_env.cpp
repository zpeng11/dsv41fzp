// Runtime numeric preconditions for df_strict_fp targets (P0009, D0005).
//
// The default process environment must satisfy every check. Each check is
// then run against a deliberately broken environment (upward rounding, FTZ,
// DAZ) and must fail there, proving it can detect the violation.

#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>

#include <xmmintrin.h>

#include "fp_probe.h"

static_assert(std::numeric_limits<float>::is_iec559, "float must be IEEE 754 binary32");
static_assert(std::numeric_limits<float>::digits == 24, "float must have a 24-bit significand");
static_assert(FLT_EVAL_METHOD == 0, "float expressions must evaluate in float precision");

namespace {

constexpr unsigned kMxcsrFtz = 0x8000;
constexpr unsigned kMxcsrDaz = 0x0040;

using bb::probe_add;
using bb::probe_mul;

// Operands and expected values are hex literals: computing them at run time
// would happen inside the broken environment and hide the violation.
bool check_round_to_nearest_even() {
  if (std::fegetround() != FE_TONEAREST) return false;
  return probe_add(1.0f, 0x1p-24f) == 1.0f && probe_add(0x1.000002p0f, 0x1p-24f) == 0x1.000004p0f &&
         probe_add(-1.0f, -0x1p-24f) == -1.0f;
}

bool check_signed_zero() {
  return std::signbit(probe_mul(-1.0f, 0.0f)) && !std::signbit(probe_add(0.0f, -0.0f)) &&
         std::signbit(probe_add(-0.0f, -0.0f));
}

bool check_subnormal_output() {
  return probe_mul(FLT_MIN, 0.5f) == 0x1p-127f;
}

bool check_subnormal_input() {
  return probe_mul(0x1p-140f, 0x1p30f) == 0x1p-110f;
}

bool check_mxcsr_clear() {
  return (_mm_getcsr() & (kMxcsrFtz | kMxcsrDaz)) == 0;
}

struct Check {
  const char* name;
  bool (*fn)();
};

constexpr Check kChecks[] = {
    {"round_to_nearest_even", check_round_to_nearest_even},
    {"signed_zero", check_signed_zero},
    {"subnormal_output(no FTZ)", check_subnormal_output},
    {"subnormal_input(no DAZ)", check_subnormal_input},
    {"mxcsr_ftz_daz_clear", check_mxcsr_clear},
};

bool expect_detected(const char* env, bool (*fn)()) {
  const bool passed = fn();
  std::printf("%s %s: check %s\n", passed ? "FAIL" : "PASS", env,
              passed ? "did not detect the violation" : "detected the violation");
  return !passed;
}

}  // namespace

int main() {
  bool ok = true;
  for (const Check& c : kChecks) {
    const bool passed = c.fn();
    std::printf("%s default env: %s\n", passed ? "PASS" : "FAIL", c.name);
    ok = ok && passed;
  }

  const unsigned csr = _mm_getcsr();

  std::fesetround(FE_UPWARD);
  ok = expect_detected("FE_UPWARD", check_round_to_nearest_even) && ok;
  std::fesetround(FE_TONEAREST);

  _mm_setcsr(csr | kMxcsrFtz);
  ok = expect_detected("FTZ", check_subnormal_output) && ok;
  _mm_setcsr(csr);

  _mm_setcsr(csr | kMxcsrDaz);
  ok = expect_detected("DAZ", check_subnormal_input) && ok;
  _mm_setcsr(csr);

  return ok ? 0 : 1;
}
