#include <cstdio>
#include <cstdlib>

#include "bb/smoke.h"

int main(int argc, char** argv) {
  std::int32_t values[] = {1, 2, 3, 4};
  if (argc > 1) values[0] = static_cast<std::int32_t>(std::atoi(argv[1]));

  std::int32_t sum = 0;
  if (!bb::checked_sum(values, 4, &sum) || sum != values[0] + 9) {
    std::fprintf(stderr, "FAIL checked_sum: got %d\n", static_cast<int>(sum));
    return 1;
  }

  const std::int32_t overflow[] = {INT32_MAX, 1};
  if (bb::checked_sum(overflow, 2, &sum)) {
    std::fprintf(stderr, "FAIL checked_sum: overflow not detected\n");
    return 1;
  }

  std::puts("PASS smoke");
  return 0;
}
