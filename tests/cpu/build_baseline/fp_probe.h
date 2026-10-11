#pragma once

namespace bb {

// Out-of-line binary32 operations; callers cannot constant-fold through them.
float probe_add(float a, float b);
float probe_mul(float a, float b);

}  // namespace bb
