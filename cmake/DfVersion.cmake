# Build-time engine version metadata (AGENTS.md 引擎版本, D0005).
#
# df_version_info is an always-run target that regenerates
# <binary>/generated/df/version_info.h from VERSION and the Git work tree
# on every build; the header is only rewritten when its content changes.
# Consumers call df_use_version_info(<target>) and should include the
# header from a single translation unit.

find_package(Git REQUIRED)

set(DF_VERSION_INCLUDE_DIR "${PROJECT_BINARY_DIR}/generated")
set(DF_VERSION_HEADER "${DF_VERSION_INCLUDE_DIR}/df/version_info.h")

add_custom_target(df_version_info
  COMMAND "${CMAKE_COMMAND}"
          -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
          -DOUTPUT=${DF_VERSION_HEADER}
          -P "${CMAKE_CURRENT_LIST_DIR}/df_version_gen.cmake"
  BYPRODUCTS "${DF_VERSION_HEADER}"
  COMMENT "Refreshing engine version metadata"
  VERBATIM)

function(df_use_version_info target)
  add_dependencies(${target} df_version_info)
  target_include_directories(${target} PRIVATE "${DF_VERSION_INCLUDE_DIR}")
endfunction()
