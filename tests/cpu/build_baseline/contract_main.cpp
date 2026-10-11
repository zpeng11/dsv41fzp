// FMA contraction probe driver (P0009, D0005).
//
// a = 1 + 2^-12, c = -(1 + 2^-11): the exact a*a + c is 2^-24, while rounding
// a*a to binary32 first yields 0. "--expect fused|separate" selects which
// result this build must produce.

#include <cmath>
#include <cstdio>
#include <cstring>

namespace bb {
float probe_mul_add(float a, float b, float c);
}

int main(int argc, char** argv) {
  if (argc != 3 || std::strcmp(argv[1], "--expect") != 0 ||
      (std::strcmp(argv[2], "fused") != 0 && std::strcmp(argv[2], "separate") != 0)) {
    std::fprintf(stderr, "usage: %s --expect fused|separate\n", argv[0]);
    return 2;
  }
  if (!__builtin_cpu_supports("fma")) {
    std::fprintf(stderr, "FAIL host CPU lacks FMA; the contraction probe cannot run\n");
    return 1;
  }

  const bool expect_fused = std::strcmp(argv[2], "fused") == 0;
  volatile float a = 0x1.001p0f;
  volatile float c = -0x1.002p0f;
  const float got = bb::probe_mul_add(a, a, c);
  const float want = expect_fused ? 0x1p-24f : 0.0f;

  const bool ok = got == want && !std::signbit(got);
  std::printf("%s expect=%s got=%a want=%a\n", ok ? "PASS" : "FAIL", argv[2],
              static_cast<double>(got), static_cast<double>(want));
  return ok ? 0 : 1;
}
