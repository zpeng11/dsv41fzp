#include <cstdio>

#include "bb/version.h"

int main() {
  std::printf("base=%s\n", bb::version_base());
  std::printf("sha=%s\n", bb::version_git_sha());
  std::printf("dirty=%d\n", bb::version_dirty() ? 1 : 0);
  std::printf("display=%s\n", bb::version_string());
  return 0;
}
