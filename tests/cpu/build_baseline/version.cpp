#include "bb/version.h"

#include "df/version_info.h"

namespace bb {

const char* version_base() { return DF_VERSION_BASE; }
const char* version_git_sha() { return DF_GIT_SHA; }
bool version_dirty() { return DF_GIT_DIRTY != 0; }
const char* version_string() { return DF_VERSION_STRING; }

}  // namespace bb
