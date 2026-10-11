#pragma once

namespace bb {

// Engine version metadata captured at the most recent build.
const char* version_base();
const char* version_git_sha();
bool version_dirty();
const char* version_string();

}  // namespace bb
